"""End-to-end Router training from offline threshold precision-map labels."""

from __future__ import annotations

import gc
import json
import logging
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, TrainingArguments, default_data_collator

from query_adaptive_quantization.modules import QueryAdaptiveQuantizationForCausalLM

from .checkpointing import ThresholdMapCheckpointCallback
from .config import (
    load_training_config,
    save_training_config,
    validate_router_architecture,
)
from .data import load_precision_map_dataset
from .trainer import ThresholdMapTrainer

logger = logging.getLogger(__name__)


def _clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ThresholdMapTrainingPipeline:
    model_class = QueryAdaptiveQuantizationForCausalLM

    def _validate_router_architecture(self, architecture: dict) -> dict:
        return validate_router_architecture(architecture)

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = load_training_config(self.config_path)
        self.output_dir = Path(self.config["training"]["output_dir"])
        self.router_config = self.config["model_config"]["router"]
        self.router_config["router_config"] = self._validate_router_architecture(
            self.router_config["router_config"]
        )
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _setup_logging(self) -> None:
        log_config = self.config.get("logging", {})
        level = getattr(logging, str(log_config.get("level", "INFO")).upper())
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if log_config.get("to_file", True):
            log_dir = Path(log_config.get("dir", "./logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(
                logging.FileHandler(log_dir / f"{self.output_dir.name}.log")
            )
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=handlers,
            force=True,
        )


    def _load_model(self) -> None:
        student_path = self.config["models"]["student"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            student_path,
            add_eos_token=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.model_class.from_quantized(
            student_path,
            enable_router=True,
            router_config=self.router_config["router_config"],
        ).cuda()
        if not set(range(3, 9)).issubset(self.model.supported_bits):
            raise ValueError("The packed model must support all Int3–Int8 precisions")
        frozen = 0
        trainable = 0
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = "router" in name.lower()
            if parameter.requires_grad:
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
        if trainable == 0:
            raise RuntimeError("No trainable Router parameters found")
        self.router_config["parameter_count"] = int(trainable)
        self.router_config["trainable_parameter_count"] = int(trainable)
        self.router_config["frozen_parameter_count"] = int(frozen)
        self.router_config["model_signature"] = {
            "num_layers": int(len(self.model.layer_to_aplinears)),
            "hidden_size": int(self.model.config.hidden_size),
            "num_branches": int(self.model.router.num_precision_blocks),
            "precision_levels": [
                int(bit) for bit in self.model.router.precision_levels
            ],
        }
        save_training_config(
            self.output_dir / "router_training_config.yaml",
            self.config,
        )
        save_training_config(
            self.output_dir / "quick_config.yaml",
            self.config,
        )
        self.model.router.train()
        logger.info(
            "Loaded packed student; canonical Online TTT Router "
            "parameters=%s frozen=%s online_lr=%s concept_dim=352 depth=2",
            f"{trainable:,}",
            f"{frozen:,}",
            self.router_config.get("online_ttt_lr"),
        )

    def _training_arguments(self, train_size: int) -> TrainingArguments:
        config = dict(self.config["training"])
        epochs = int(self.config["training_aux"]["num_train_epochs"])
        batch_size = int(config["per_device_train_batch_size"])
        grad_accum = int(config["gradient_accumulation_steps"])
        world_size = 1  # The training interface uses one visible GPU.
        batches_per_epoch = math.ceil(train_size / (batch_size * world_size))
        optimizer_steps = math.ceil(batches_per_epoch / grad_accum) * epochs
        logging_fraction = float(
            self.config["training_aux"].get("logging_steps_fraction", 0.05)
        )
        logging_steps = max(1, int(optimizer_steps * logging_fraction))
        return TrainingArguments(
            **config,
            max_steps=max(1, optimizer_steps),
            num_train_epochs=epochs,
            logging_steps=logging_steps,
            run_name=self.output_dir.name,
            logging_dir=str(Path("./logs") / self.output_dir.name),
            ddp_find_unused_parameters=False,
        )

    def _save_router(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.model.router.state_dict(),
            self.output_dir / "router_weights.pt",
        )
        (self.output_dir / "router_config.json").write_text(
            json.dumps(self.router_config, indent=2),
            encoding="utf-8",
        )
        try:
            stats = self.model.router.get_statistics()
            if isinstance(stats, dict) and "message" not in stats:
                (self.output_dir / "router_statistics.json").write_text(
                    json.dumps(stats, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            logger.warning("Could not save Router statistics: %s", exc)

    def run(self) -> None:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Select one CUDA GPU with CUDA_VISIBLE_DEVICES")
        self._setup_logging()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_model()
        train_dataset, validation_dataset = load_precision_map_dataset(
            self.config["dataset"]["label_dataset_path"]
        )
        training_args = self._training_arguments(len(train_dataset))
        callbacks = [
            ThresholdMapCheckpointCallback(
                self.router_config.get("checkpoint_epochs", []),
                self.router_config,
            )
        ]
        self.trainer = ThresholdMapTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            processing_class=self.tokenizer,
            data_collator=default_data_collator,
            callbacks=callbacks,
            label_smoothing=float(
                self.router_config.get("map_label_smoothing", 0.05)
            ),
        )
        if self.config["training"].get("gradient_checkpointing", False):
            self.model.config.use_cache = False
        logger.info(
            "Starting threshold-map training: train=%s validation=%s epochs=%s",
            len(train_dataset),
            len(validation_dataset),
            self.config["training_aux"]["num_train_epochs"],
        )
        self.trainer.train()
        self._save_router()
        _clear_cuda_memory()
