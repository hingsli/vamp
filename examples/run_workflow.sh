#!/usr/bin/env bash
set -euo pipefail

: "${PACKED_MODEL:?Set PACKED_MODEL to a local Int3-Int8 packed model directory}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
artifact_dir="${ARTIFACT_DIR:-artifacts/workflow}"

vamp-search \
  --packed-model "$PACKED_MODEL" \
  --data examples/calibration.jsonl \
  --output-dir "$artifact_dir/labels" \
  --num-samples 8 --max-length 128 --validation-samples 2 \
  --threshold 0.001

vamp-train \
  --packed-model "$PACKED_MODEL" \
  --labels "$artifact_dir/labels" \
  --output-dir "$artifact_dir/router" \
  --epochs 1 --grad-accum 1

vamp-generate \
  --packed-model "$PACKED_MODEL" \
  --router "$artifact_dir/router" \
  --prompt 'Explain why the sky appears blue.' \
  --max-new-tokens 64
