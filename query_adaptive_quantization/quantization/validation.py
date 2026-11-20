"""Validation helpers for parent-precision MSQ weights."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def parent_reconstruction_metrics(
    original_weight: np.ndarray,
    parent_labels: np.ndarray,
    parent_lut: np.ndarray,
    *,
    row_chunk_size: int = 128,
) -> dict[str, Any]:
    """Measure reconstruction quality without materializing a full copy."""

    original = np.asarray(original_weight, dtype=np.float32)
    labels = np.asarray(parent_labels, dtype=np.uint8)
    lut = np.asarray(parent_lut, dtype=np.float32)
    rows, groups, group_size = labels.shape
    if original.shape != (rows, groups * group_size):
        raise ValueError(
            f"Weight shape {original.shape} does not match labels {labels.shape}"
        )
    if lut.shape[:2] != (rows, groups):
        raise ValueError(f"LUT shape {lut.shape} does not match labels {labels.shape}")

    squared_error = 0.0
    squared_weight = 0.0
    max_row_relative_rmse = 0.0
    degenerate_rows = 0
    for start in range(0, rows, row_chunk_size):
        stop = min(rows, start + row_chunk_size)
        chunk_labels = labels[start:stop].astype(np.intp, copy=False)
        chunk_lut = lut[start:stop]
        row_indices = np.arange(stop - start)[:, None, None]
        group_indices = np.arange(groups)[None, :, None]
        reconstructed = chunk_lut[row_indices, group_indices, chunk_labels]
        chunk_original = original[start:stop].reshape(
            stop - start, groups, group_size
        )
        difference = reconstructed.astype(np.float64) - chunk_original.astype(
            np.float64
        )
        squared_error += float(np.square(difference).sum())
        squared_weight += float(
            np.square(chunk_original.astype(np.float64)).sum()
        )

        row_error = np.square(difference).mean(axis=(1, 2))
        row_weight = np.square(chunk_original.astype(np.float64)).mean(
            axis=(1, 2)
        )
        row_relative = np.sqrt(row_error / np.maximum(row_weight, 1e-30))
        max_row_relative_rmse = max(
            max_row_relative_rmse, float(row_relative.max(initial=0.0))
        )
        lut_span = np.ptp(chunk_lut, axis=2)
        weight_span = np.ptp(chunk_original, axis=2)
        degenerate_rows += int(((lut_span == 0) & (weight_span != 0)).sum())

    relative_rmse = math.sqrt(squared_error / max(squared_weight, 1e-30))
    return {
        "relative_rmse": float(relative_rmse),
        "max_row_relative_rmse": float(max_row_relative_rmse),
        "degenerate_rows": int(degenerate_rows),
        "finite": bool(
            np.isfinite(original).all()
            and np.isfinite(labels).all()
            and np.isfinite(lut).all()
        ),
    }


def validate_parent_reconstruction(
    original_weight: np.ndarray,
    parent_labels: np.ndarray,
    parent_lut: np.ndarray,
    *,
    layer_index: int,
    module_name: str,
    max_relative_rmse: float,
) -> dict[str, Any]:
    metrics = parent_reconstruction_metrics(
        original_weight,
        parent_labels,
        parent_lut,
    )
    metrics.update({"layer": int(layer_index), "module": module_name})
    if not metrics["finite"]:
        raise RuntimeError(
            f"Non-finite parent reconstruction at layer={layer_index} "
            f"module={module_name}"
        )
    if metrics["degenerate_rows"]:
        raise RuntimeError(
            f"Degenerate parent LUT rows at layer={layer_index} "
            f"module={module_name}: {metrics['degenerate_rows']}"
        )
    if metrics["relative_rmse"] >= max_relative_rmse:
        raise RuntimeError(
            f"Parent reconstruction RMSE exceeds limit at layer={layer_index} "
            f"module={module_name}: {metrics['relative_rmse']:.6f} "
            f">= {max_relative_rmse:.6f}"
        )
    return metrics
