"""Router-only checkpoints for quality-threshold training."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class ThresholdMapCheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_epochs: list[int], router_config: dict[str, Any]):
        self.checkpoint_epochs = sorted(
            {int(epoch) for epoch in checkpoint_epochs if int(epoch) > 0}
        )
        self.router_config = dict(router_config)
        self.saved_epochs: set[int] = set()

    @staticmethod
    def _copy_if_exists(source: Path, destination: Path) -> None:
        if source.exists():
            shutil.copy2(source, destination)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.epoch is None:
            return control
        epoch = int(round(float(state.epoch)))
        if (
            epoch <= 0
            or abs(float(state.epoch) - epoch) > 0.05
            or epoch not in self.checkpoint_epochs
            or epoch in self.saved_epochs
        ):
            return control
        if not hasattr(model, "router") or model.router is None:
            raise RuntimeError("Cannot save checkpoint: model has no Router")
        checkpoint_dir = Path(args.output_dir) / "checkpoints" / f"epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.router.state_dict(), checkpoint_dir / "router_weights.pt")
        (checkpoint_dir / "router_config.json").write_text(
            json.dumps(self.router_config, indent=2), encoding="utf-8"
        )
        metadata = {
            "checkpoint_epoch": epoch,
            "state_epoch": float(state.epoch),
            "global_step": int(state.global_step),
            "output_dir": str(args.output_dir),
            "router_architecture": self.router_config.get(
                "router_architecture",
                self.router_config.get("router_config", {}).get(
                    "router_type", "online_ttt"
                ),
            ),
            "online_ttt_lr": self.router_config.get(
                "online_ttt_lr",
                self.router_config.get("router_config", {}).get(
                    "ttt_lr_linear"
                ),
            ),
            "online_ttt_concept_dim": self.router_config.get(
                "online_ttt_concept_dim",
                self.router_config.get("router_config", {}).get(
                    "concept_dim"
                ),
            ),
            "router_parameter_count": self.router_config.get("parameter_count"),
            "router_trainable_parameter_count": self.router_config.get(
                "trainable_parameter_count"
            ),
            "model_signature": self.router_config.get("model_signature"),
        }
        (checkpoint_dir / "router_checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        for filename in ("quick_config.yaml", "router_training_config.yaml"):
            self._copy_if_exists(
                Path(args.output_dir) / filename,
                checkpoint_dir / filename,
            )
        try:
            state.save_to_json(str(checkpoint_dir / "trainer_state.json"))
        except Exception as exc:
            logger.warning("Could not save trainer state for epoch %s: %s", epoch, exc)
        self.saved_epochs.add(epoch)
        logger.info("Router checkpoint saved: epoch=%s path=%s", epoch, checkpoint_dir)
        return control
