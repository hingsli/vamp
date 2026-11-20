# Validation status

The public workflow passed on **one NVIDIA RTX 4090** on 2026-11-20.
It used the existing Qwen3-4B packed weights, the recorded Python environment and
an unchanged export of this repository's runtime code. The CUDA extension was
compiled from the included sources.

## GPU environment

- Linux x86_64, Python 3.12.9
- PyTorch 2.8.0+cu128, CUDA runtime 12.8, CUDA toolkit 12.6.85
- Transformers 4.57.6, Accelerate 1.13.0
- Datasets 2.16.0, PyArrow 19.0.1

The installed package versions are recorded in `requirements-cuda.lock`.
The [validation summary](validation-rtx4090.json) records the checked runtime
source hashes without machine paths or account details.

## Results

| Check | Result |
| --- | --- |
| Fresh environment and package installation | Passed |
| CUDA extension compilation and import | Passed |
| CPU tests on Linux | 42 passed |
| examples/run_workflow.sh | Passed |
| Search | 8 sequences of 128 tokens; all 8 labels satisfied KL ≤ 0.001 |
| Router training | 6 training sequences, 1 epoch, 6 optimizer steps; finite logged losses and gradient norms |
| Generation | Produced a nonempty continuation and a 36 × 2 precision map |
| GPU test with the newly trained Router | Passed |
| GPU test with an existing trained Router | Passed |

The remaining two label sequences were held out. They were not used for
training or checkpoint selection. The public workflow completed in 81.01
seconds, including model loading. This is the duration of a small smoke run,
not an inference latency benchmark.

The GPU tests check that the Router is called once, its map has the expected
shape and supported bit widths, generation returns tokens, and prior precision
settings are restored. The CPU tests additionally cover verified search and
fallback behavior, masked KL scoring, parameter-weighted cost, label
serialization, the supervised loss, Router shapes and chunking, failure
cleanup, and independence from experiment modules.

The same 42 CPU tests passed locally with Python 3.11.15, PyTorch 2.10.0 and
Transformers 4.57.6 on macOS arm64. Editable installation, all four CLI help
commands, preparation-module imports, shell syntax, and source/wheel packaging
also passed. `requirements-dev.lock` records that local development environment.

## Repeating the smoke check

After installing the package and CUDA extension, set `PACKED_MODEL` and select
one GPU with `CUDA_VISIBLE_DEVICES`, then run:

```bash
bash examples/run_workflow.sh
export ROUTER_ARTIFACT=artifacts/workflow/router
python -m pytest -q tests/test_cuda_runtime.py
```

Use a fresh `ARTIFACT_DIR` if the default output directory already contains a
run, and point `ROUTER_ARTIFACT` at its `router` subdirectory.

## Scope

This run starts from an existing packed model. Base-model quantization and
packing were not rerun. It validates the small resident workflow, including
loading an existing Router; it does not establish answer quality, reproduce
the paper's tables, or validate on-demand execution or multiple GPUs.

The saved Qwen3 tokenizer was used unchanged. Transformers 4.57.6 emitted a
Mistral-regex warning for this local artifact's version metadata; no Mistral
regex patch was applied to the Qwen2Tokenizer used by Qwen3.
