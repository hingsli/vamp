"""Hugging Face Trainer integration for precision-map supervision."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from transformers import Trainer

from .loss import precision_map_cross_entropy


def _metric_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


class ThresholdMapTrainer(Trainer):
    def __init__(self, *args, label_smoothing: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = float(label_smoothing)
        self._last_map_metrics: dict[str, float] = {}

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        target_bits = torch.as_tensor(
            inputs["precision_map"],
            dtype=torch.long,
            device=input_ids.device,
        )
        expected_target_shape = (
            int(input_ids.shape[0]),
            int(model.router.num_layers),
            2,
        )
        if tuple(target_bits.shape) != expected_target_shape:
            raise ValueError(
                f"Precision-map target shape {tuple(target_bits.shape)} does not "
                f"match expected {expected_target_shape}"
            )
        hidden_states = model.get_router_input_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            detach=True,
        )
        router_output = model.router(
            hidden_states,
            attention_mask,
            hard_selection=False,
        )
        precision_logits = router_output["precision_logits"]
        loss_output = precision_map_cross_entropy(
            precision_logits,
            target_bits,
            [int(bit) for bit in model.router.precision_levels],
            label_smoothing=self.label_smoothing,
        )
        if not loss_output.loss.requires_grad:
            raise RuntimeError("Precision-map loss must require gradients")
        self._last_map_metrics = {
            key: _metric_scalar(value) for key, value in loss_output.metrics.items()
        }
        if return_outputs:
            return (
                loss_output.loss,
                SimpleNamespace(loss=loss_output.loss, logits=precision_logits),
            )
        return loss_output.loss

    def log(self, logs: dict[str, float], *args, **kwargs):
        values = dict(logs)
        for key, value in self._last_map_metrics.items():
            values.setdefault(f"router/{key}", value)
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None and optimizer.param_groups:
            values.setdefault(
                "router/effective_learning_rate",
                float(optimizer.param_groups[0].get("lr", 0.0)),
            )
        return super().log(values, *args, **kwargs)
