"""Numerically stable helpers for gradient-based quantization salience."""

from __future__ import annotations

import torch


GRADIENT_PRE_SQUARE_SCALE = 256.0


def scaled_squared_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Return squared-gradient salience without BF16 underflow.

    Quantization only uses relative salience within each weight row, so the
    shared scale factor does not change the weighted clustering objective.
    Scaling before squaring keeps small BF16 gradients in the normal range.
    """

    scaled = gradient.float() * GRADIENT_PRE_SQUARE_SCALE
    return scaled.square().to(dtype=gradient.dtype)
