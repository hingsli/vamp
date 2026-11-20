#!/usr/bin/env python3
"""Offline quantization entrypoint for query-adaptive quantization."""

import argparse

import numpy as np
import torch

try:
    _NUMPY_RECONSTRUCT = np._core.multiarray._reconstruct
except AttributeError:
    _NUMPY_RECONSTRUCT = np.core.multiarray._reconstruct


torch.serialization.add_safe_globals(
    [
        np.ndarray,
        _NUMPY_RECONSTRUCT,
        np.dtype,
        np.float16,
        np.float32,
        np.uint8,
        np.int32,
        np.int64,
    ]
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantize a model with query-adaptive precision."
    )
    parser.add_argument("model", type=str, help="Model name or local checkpoint path.")
    parser.add_argument(
        "--seed-precision",
        "--seed_precision",
        dest="seed_precision",
        type=int,
        help="Seed precision used for the initial bitplane layout.",
    )
    parser.add_argument(
        "--parent-precision",
        "--parent_precision",
        dest="parent_precision",
        type=int,
        help="Parent precision used for the packed representation.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gradients", "quantize", "pack"],
        default="pack",
        help="Quantization mode. Use 'pack' for gradients -> quantize -> pack.",
    )
    parser.add_argument("--yaml-path", "--yaml_path", dest="yaml_path", type=str)
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=str)
    parser.add_argument("--dataset", type=str, help="Calibration dataset name.")
    parser.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int)
    parser.add_argument(
        "--num-examples", "--num_examples", dest="num_examples", type=int
    )
    parser.add_argument("--cpu-count", "--cpu_count", dest="cpu_count", type=int)
    parser.add_argument("--overwrite-gradients", action="store_true")
    parser.add_argument("--overwrite-quantize", action="store_true")
    parser.add_argument("--overwrite-pack", action="store_true")
    parser.add_argument(
        "--random-state", "--random_state", dest="random_state", type=int
    )
    parser.add_argument("--group-count", "--group_count", dest="group_count", type=int)
    parser.add_argument("--use-optimized-gradients", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Use non-reentrant gradient checkpointing in the memory-limited "
            "collector. This requires all model dropout probabilities to be zero."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--gradient_accumulation_steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--memory-check-interval",
        "--memory_check_interval",
        dest="memory_check_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-memory-gb", "--max_memory_gb", dest="max_memory_gb", type=float
    )
    parser.add_argument(
        "--max-parent-relative-rmse",
        "--max_parent_relative_rmse",
        dest="max_parent_relative_rmse",
        type=float,
        default=0.05,
        help="Fail before packing when an Int8 parent module exceeds this RMSE.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if v is not None}
    from query_adaptive_quantization.quantization import query_adaptive_quantize

    query_adaptive_quantize(**options)


if __name__ == "__main__":
    main()
