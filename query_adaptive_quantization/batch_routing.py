"""Shared inference precision maps for batches using one set of weights."""

import torch


def shared_precision_map(values: torch.Tensor, policy: str | None = None) -> torch.Tensor:
    """Reduce [batch, layer, branch] predictions without ignoring any prompt.

    Batch-one keeps its original map. Larger batches require an explicit
    policy; ``max`` never lowers a slot relative to any member's prediction.
    This is a precision rule, not a guarantee about downstream quality.
    """
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[2] != 2:
        raise ValueError("Expected nonempty [batch, layer, 2] precision predictions")
    if policy not in (None, "max"):
        raise ValueError(f"Unknown batch_map_policy: {policy}")
    if values.shape[0] == 1:
        return values[0]
    if policy is None:
        raise ValueError("Batch routing requires explicit batch_map_policy='max'")
    return values.amax(dim=0)
