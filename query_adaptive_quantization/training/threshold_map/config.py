"""Configuration for threshold-supervised precision routing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROUTER_ARCHITECTURE: dict[str, Any] = {
    "precision_levels": [3, 4, 5, 6, 7, 8],
    "router_type": "online_ttt",
    "router_hidden_size": 256,
    "temperature": 1.0,
    "use_gumbel_softmax": True,
    "dropout_rate": 0.1,
    "pooling_method": "mean",
    "router_output_granularity": "block",
    "concept_dim": 352,
    "logit_dim": 64,
    "ttt_lr_linear": 0.1,
    "ttt_lr_ln": 0.1,
    "router_ttt_layers": 2,
    "router_ttt_heads": 1,
    "router_ttt_residual": "none",
    "router_input_source": "prelude_hidden",
    "router_prelude_layers": 2,
    "router_prelude_precision": 5,
    "router_segment_count": 8,
    "router_readout": "segment_attention",
    "router_layer_decoder": "layer_gru",
}

def build_router_architecture(
    online_ttt_lr: float = 0.1,
) -> dict[str, Any]:
    if online_ttt_lr < 0.0:
        raise ValueError("online_ttt_lr must be non-negative")
    architecture = deepcopy(ROUTER_ARCHITECTURE)
    architecture["ttt_lr_linear"] = float(online_ttt_lr)
    architecture["ttt_lr_ln"] = float(online_ttt_lr)
    return architecture


def validate_router_architecture(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a serialized config against the canonical Router shape."""
    architecture = dict(config)
    expected = {
        key: value
        for key, value in ROUTER_ARCHITECTURE.items()
        if key not in {"ttt_lr_linear", "ttt_lr_ln"}
    }
    mismatches = {
        key: {"expected": value, "observed": architecture.get(key)}
        for key, value in expected.items()
        if architecture.get(key, value) != value
    }
    linear_lr = float(architecture.get("ttt_lr_linear", 0.1))
    layer_norm_lr = float(architecture.get("ttt_lr_ln", linear_lr))
    if linear_lr < 0.0 or layer_norm_lr < 0.0 or linear_lr != layer_norm_lr:
        mismatches["online_ttt_lr"] = {
            "expected": "equal non-negative linear/LN rates",
            "observed": [linear_lr, layer_norm_lr],
        }
    if mismatches:
        details = ", ".join(
            f"{key}={value['observed']!r}" for key, value in mismatches.items()
        )
        raise ValueError(
            "The public VAMP workflow supports only the canonical Online TTT "
            f"architecture; incompatible fields: {details}"
        )
    canonical = deepcopy(ROUTER_ARCHITECTURE)
    canonical["ttt_lr_linear"] = linear_lr
    canonical["ttt_lr_ln"] = layer_norm_lr
    return canonical


def build_training_config(
    *,
    student_model: str,
    label_dataset: str,
    output_dir: str,
    epochs: int = 10,
    max_length: int = 1024,
    train_batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.05,
    min_lr_rate: float = 0.1,
    max_grad_norm: float = 1.0,
    label_smoothing: float = 0.05,
    checkpoint_epochs: list[int] | None = None,
    report_to: str = "none",
    logging_steps_fraction: float = 0.05,
    gradient_checkpointing: bool = True,
    flash_attention: bool = False,
    log_level: str = "INFO",
    log_dir: str = "./logs",
    log_to_file: bool = True,
    online_ttt_lr: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if not 0.0 <= min_lr_rate <= 1.0:
        raise ValueError("min_lr_rate must be in [0, 1]")
    if online_ttt_lr < 0.0:
        raise ValueError("online_ttt_lr must be non-negative")
    checkpoints = checkpoint_epochs or list(range(1, epochs + 1))
    checkpoints = sorted({int(epoch) for epoch in checkpoints if int(epoch) > 0})
    architecture = build_router_architecture(online_ttt_lr)
    router_config = {
        "router_architecture": "online_ttt",
        "online_ttt_lr": float(online_ttt_lr),
        "online_ttt_concept_dim": 352,
        "map_label_smoothing": float(label_smoothing),
        "checkpoint_epochs": checkpoints,
        "loss_mode": "threshold_precision_map_ce",
        "save_complete_model": False,
        "learning_rate": float(learning_rate),
        "enabled": True,
        "only_training": True,
        "router_config": architecture,
        "weight_decay": 0.01,
        "use_separate_optimizer": True,
        "requires_teacher": False,
    }
    return {
        "experiment_name": "threshold-supervised-precision-routing",
        "project_name": "query-adaptive-quantization",
        "logging": {
            "level": log_level,
            "to_file": bool(log_to_file),
            "dir": log_dir,
        },
        "models": {"teacher": None, "student": student_model},
        "dataset": {
            "name": "threshold_precision_map_labels",
            "label_dataset_path": label_dataset,
            "packing": False,
        },
        "tokenizer": {"max_length": int(max_length)},
        "training": {
            "per_device_train_batch_size": int(train_batch_size),
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": int(grad_accum),
            "learning_rate": float(learning_rate),
            "weight_decay": 0.01,
            "warmup_ratio": float(warmup_ratio),
            "lr_scheduler_type": "cosine_with_min_lr",
            "lr_scheduler_kwargs": {"min_lr_rate": float(min_lr_rate)},
            "max_grad_norm": float(max_grad_norm),
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": bool(gradient_checkpointing),
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "eval_strategy": "no",
            "save_strategy": "no",
            "remove_unused_columns": False,
            "report_to": report_to,
            "optim": "adamw_torch",
            "output_dir": output_dir,
            "seed": int(seed),
            "data_seed": int(seed),
        },
        "training_aux": {
            "num_train_epochs": int(epochs),
            "logging_steps_fraction": float(logging_steps_fraction),
        },
        "model_config": {
            "use_flash_attention": bool(flash_attention),
            "torch_dtype": "bfloat16",
            "router": router_config,
            "weight_freezing": {"strategy": "router_only"},
        },
    }


def load_training_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    return config


def save_training_config(path: str | Path, config: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
