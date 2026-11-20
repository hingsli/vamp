# VAMP

**Verified, Amortized Mixed-Precision quantization** selects a precision map
for each prompt before generating an answer. Offline search finds low-cost
maps under a KL tolerance; a Router learns those maps and predicts one at
inference time. Each transformer layer has two decisions: attention and
feed-forward precision, chosen from Int3 through Int8.

This repository provides precision-map search, Router training and
single-GPU inference. The workflow is:

```text
Packed model + calibration text
    → verified precision maps
    → supervised Router training
    → prompt-only map prediction
    → generation with a fixed map
```

## Contents

| Location | Purpose |
| --- | --- |
| `query_adaptive_quantization/threshold_maps/` | Parameter-weighted cost, KL verification and precision-map search |
| `query_adaptive_quantization/modules/Router.py` | Online TTT features, segment attention and layer-wise map prediction |
| `query_adaptive_quantization/training/threshold_map/` | Map-supervised loss, training and Router checkpoints |
| `query_adaptive_quantization/inference.py` | Resident, prompt-only inference |
| `query_adaptive_quantization/modules/kernels/` | Bitplane CUDA kernels |
| `query_adaptive_quantization/quantization/` | Optional model quantization and packing |
| `examples/` | Example inputs and a workflow script |
| `tests/` | CPU checks for search, training and inference contracts |

Model weights, trained Router checkpoints and the paper's label-set manifests
are not bundled. The instructions below explain how to prepare a model,
build training labels and train your own Router.

## Installation

The GPU workflow has been tested on Linux with Python 3.12, one RTX 4090,
CUDA toolkit 12.6 and PyTorch 2.8.0+cu128. Install the recorded dependencies,
then compile the extension in the same environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.lock
python -m pip install -e . --no-deps
python -m pip install --no-build-isolation ./query_adaptive_quantization/modules/kernels
```

The CUDA toolkit and a compatible C++ compiler must already be installed on
the GPU machine. The Python wheel contains kernel sources, not a compiled
CUDA binary.

For CPU-only inspection and unit tests, Python 3.11 is also supported. Create
a virtual environment and run `python -m pip install -e '.[dev]'`; skip the CUDA
extension. See [validation notes](docs/validation.md) for the tested scope.

## Getting started

Start with a local Qwen3 model packed by this codebase, with all Int3–Int8
bitplanes and lookup tables. An ordinary Hugging Face checkpoint or an AWQ/GPTQ
checkpoint is not interchangeable with this format. If you do not have a
packed model, follow [model preparation](docs/prepare_model.md) first.

```bash
export CUDA_VISIBLE_DEVICES=0
export PACKED_MODEL=/path/to/packed-model
bash examples/run_workflow.sh
```

The script runs search, training and generation, as detailed below, and writes
to `artifacts/workflow/`. Set `ARTIFACT_DIR` to choose another output directory.
The example uses eight short calibration sequences and one training epoch
for a quick first run. For quality evaluation, use a larger, representative
training corpus and a separate evaluation set; the example settings differ
from those used in the paper.

### 1. Search for verified maps

```bash
vamp-search \
  --packed-model "$PACKED_MODEL" \
  --data examples/calibration.jsonl \
  --output-dir artifacts/workflow/labels \
  --num-samples 8 --max-length 128 --validation-samples 2 \
  --threshold 0.001
```

Input JSONL has one `{"text": "..."}` object per line. JSON and Parquet with a
`text` column also work. Text is tokenized, separated by EOS and packed into
fixed-length sequences. Provide at least `num-samples × max-length` tokens.
The final two sequences form a held-out split; the training command uses only
`train.parquet`.

Search uses the same packed model at uniform Int8 as its reference. It accepts
maps whose mean next-token `KL(reference || candidate)` is at most the
threshold, and minimizes parameter-weighted bit width. Output includes
`labels.jsonl`, `train.parquet`, `validation.parquet` and a `manifest.json`
recording settings and input/configuration hashes.

### 2. Train the Router

```bash
vamp-train \
  --packed-model "$PACKED_MODEL" \
  --labels artifacts/workflow/labels \
  --output-dir artifacts/workflow/router \
  --epochs 1 --grad-accum 1
```

The base model is frozen. The Router learns a `[layers, 2]` map with supervised
cross entropy. `--online-ttt-lr` controls its token-local fast-state update
step size; `--learning-rate` controls the outer optimizer. The held-out split
is saved for subsequent evaluation; this command saves the Router after the
requested training epochs without selecting a checkpoint on that split.

The output directory contains `router_weights.pt`, `router_config.json` and
the resolved training configuration. Use the same packed model for search,
training and generation, including the same packed weights and configuration.

### 3. Generate with one prompt-only map

```bash
vamp-generate \
  --packed-model "$PACKED_MODEL" \
  --router artifacts/workflow/router \
  --prompt 'Explain why the sky appears blue.' \
  --max-new-tokens 64
```

This prints the continuation, selected map and parameter-weighted bit width.
The Router sees only the prompt. Its feature pass does not supply a KV cache
to generation; the chosen map is fixed for both prefill and decoding.
The inference command accepts one unpadded prompt at a time and keeps weights
resident on the GPU. The paper's on-demand loading experiments require
additional scheduling and hardware profiling beyond this command.

The KL tolerance verifies searched labels. A trained Router's predictions on
new inputs must be evaluated separately for answer quality and tolerance
compliance.

## Development

```bash
python -m pytest -q
python -m build
```

These tests use small tensors and a synthetic packed-model stand-in where
needed. They do not download models or run GPU benchmarks. The
[implementation guide](docs/implementation.md) identifies the main algorithm
entry points and describes the artifact formats.

`requirements-dev.lock` records the Python packages used for local validation,
including the development and preparation extras. It contains no local paths.
CUDA toolkit and driver versions must be recorded separately for GPU runs.

## License and acknowledgements

The existing MIT license is preserved in [LICENSE](LICENSE). The packed-model
backend builds on [Any-Precision LLM](https://github.com/SNU-ARC/any-precision-llm).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution. Base-model
weights and external datasets retain their own licenses; neither is bundled.
