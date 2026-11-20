"""Layer-sharded storage for gradient salience tensors.

Large models cannot keep a full set of weight-shaped gradients in host memory
while also serializing the same tensors. This store writes each decoder layer
atomically and loads one layer at a time during quantization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch


SCHEMA_VERSION = 1


def gradient_shard_dir(cache_path: str | os.PathLike[str]) -> Path:
    return Path(f"{cache_path}.layers")


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def release_file_cache(path: Path) -> None:
    """Ask Linux to evict a completed shard from the page cache."""

    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def release_model_source_cache(model_path: str | os.PathLike[str]) -> None:
    """Release cached source checkpoint pages after model construction."""

    root = Path(model_path)
    if not root.is_dir():
        return
    for pattern in ("*.safetensors", "*.bin"):
        for path in root.glob(pattern):
            if path.is_file():
                release_file_cache(path)


def release_tree_file_cache(
    root: str | os.PathLike[str],
    *,
    sync: bool = False,
) -> None:
    """Release cached pages for completed artifacts under one path."""

    path = Path(root)
    if not path.exists():
        return
    if sync and hasattr(os, "sync"):
        os.sync()
    if path.is_file():
        release_file_cache(path)
        return
    for child in path.rglob("*"):
        if child.is_file():
            release_file_cache(child)


class LayerGradientStore:
    def __init__(
        self,
        cache_path: str | os.PathLike[str],
        *,
        num_layers: int,
        module_names: Sequence[str],
    ) -> None:
        self.cache_path = Path(cache_path)
        self.root = gradient_shard_dir(cache_path)
        self.num_layers = int(num_layers)
        self.module_names = tuple(str(name) for name in module_names)
        self.manifest_path = self.root / "manifest.json"

    @classmethod
    def from_manifest(
        cls,
        cache_path: str | os.PathLike[str],
    ) -> "LayerGradientStore":
        manifest_path = gradient_shard_dir(cache_path) / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported gradient store schema: {payload.get('schema_version')}"
            )
        return cls(
            cache_path,
            num_layers=int(payload["num_layers"]),
            module_names=payload["module_names"],
        )

    @classmethod
    def open_if_present(
        cls,
        cache_path: str | os.PathLike[str],
    ) -> "LayerGradientStore | None":
        manifest_path = gradient_shard_dir(cache_path) / "manifest.json"
        if not manifest_path.is_file():
            return None
        return cls.from_manifest(cache_path)

    def layer_path(self, layer_index: int) -> Path:
        return self.root / f"layer_{int(layer_index):03d}.pt"

    def completed_layers(self) -> list[int]:
        return [
            layer_index
            for layer_index in range(self.num_layers)
            if self.layer_path(layer_index).is_file()
        ]

    def is_complete(self) -> bool:
        if not self.manifest_path.is_file():
            return False
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return (
            payload.get("status") == "complete"
            and len(self.completed_layers()) == self.num_layers
        )

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_manifest(status="partial")

    def _write_manifest(self, *, status: str) -> None:
        _write_json_atomic(
            self.manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "format": "layer_sharded_gradient_salience",
                "status": status,
                "num_layers": self.num_layers,
                "module_names": list(self.module_names),
                "completed_layers": self.completed_layers(),
            },
        )

    def save_layer(
        self,
        layer_index: int,
        gradients: Mapping[str, torch.Tensor | None],
    ) -> None:
        layer_index = int(layer_index)
        missing = set(self.module_names) - set(gradients)
        if missing:
            raise ValueError(
                f"Layer {layer_index} is missing gradient modules: {sorted(missing)}"
            )
        empty = [name for name in self.module_names if gradients[name] is None]
        if empty:
            raise ValueError(
                f"Layer {layer_index} has empty gradient modules: {empty}"
            )
        payload = {
            name: gradients[name].detach().to(device="cpu").contiguous()
            for name in self.module_names
        }
        destination = self.layer_path(layer_index)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        release_file_cache(destination)
        self._write_manifest(status="partial")

    def mark_complete(self) -> None:
        completed = self.completed_layers()
        if len(completed) != self.num_layers:
            raise RuntimeError(
                "Cannot complete gradient store: "
                f"{len(completed)}/{self.num_layers} layers are present"
            )
        self._write_manifest(status="complete")

    def load_layer(self, layer_index: int) -> dict[str, torch.Tensor | None]:
        path = self.layer_path(layer_index)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        release_file_cache(path)
        if set(payload) != set(self.module_names):
            raise ValueError(
                f"Gradient shard {path} has unexpected modules: {sorted(payload)}"
            )
        return {name: payload[name] for name in self.module_names}

    def __getitem__(self, layer_index: int) -> dict[str, torch.Tensor | None]:
        return self.load_layer(layer_index)

    def __len__(self) -> int:
        return self.num_layers
