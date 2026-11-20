"""Direct blockwise precision-map classification loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ThresholdMapLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    target_indices: torch.Tensor


def precision_bits_to_indices(
    target_bits: torch.Tensor,
    precision_levels: list[int],
) -> torch.Tensor:
    target_indices = torch.full_like(target_bits, -1, dtype=torch.long)
    for level_index, bit in enumerate(precision_levels):
        target_indices = torch.where(
            target_bits == int(bit),
            torch.full_like(target_indices, level_index),
            target_indices,
        )
    if torch.any(target_indices < 0):
        invalid = torch.unique(target_bits[target_indices < 0]).detach().cpu().tolist()
        raise ValueError(
            f"Precision-map labels contain unsupported bits {invalid}; "
            f"expected {precision_levels}"
        )
    return target_indices


def precision_map_cross_entropy(
    precision_logits: torch.Tensor,
    target_bits: torch.Tensor,
    precision_levels: list[int],
    *,
    label_smoothing: float,
) -> ThresholdMapLossOutput:
    expected_shape = (*target_bits.shape, len(precision_levels))
    if tuple(precision_logits.shape) != expected_shape:
        raise ValueError(
            f"Router logits shape {tuple(precision_logits.shape)} does not "
            f"match expected {expected_shape}"
        )
    target_indices = precision_bits_to_indices(target_bits, precision_levels)
    loss = F.cross_entropy(
        precision_logits.reshape(-1, len(precision_levels)),
        target_indices.reshape(-1),
        label_smoothing=float(label_smoothing),
    )
    with torch.no_grad():
        predicted_indices = precision_logits.argmax(dim=-1)
        levels = torch.tensor(
            precision_levels,
            device=precision_logits.device,
            dtype=torch.long,
        )
        predicted_bits = levels[predicted_indices]
        equality = predicted_bits == target_bits
        branch_predicted = predicted_bits.float().mean(dim=(0, 1))
        branch_target = target_bits.float().mean(dim=(0, 1))
        metrics = {
            "total_loss": loss.detach(),
            "map_ce": loss.detach(),
            "map_slot_accuracy": equality.float().mean(),
            "map_exact_accuracy": equality.reshape(target_bits.shape[0], -1)
            .all(dim=-1)
            .float()
            .mean(),
            "map_predicted_avg_bit": predicted_bits.float().mean(),
            "map_target_avg_bit": target_bits.float().mean(),
            "map_predicted_mha_bit": branch_predicted[0],
            "map_predicted_mlp_bit": branch_predicted[1],
            "map_target_mha_bit": branch_target[0],
            "map_target_mlp_bit": branch_target[1],
        }
    return ThresholdMapLossOutput(loss, metrics, target_indices)
