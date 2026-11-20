# Third-party notices

The existing project license is retained verbatim in `LICENSE`:
Copyright (c) 2025 MohismLab, MIT License.

The packed weight representation, incremental quantization and CUDA backend
build on [Any-Precision LLM](https://github.com/SNU-ARC/any-precision-llm),
Copyright (c) 2024 SNU ARC Lab. Its MIT license is reproduced in
`licenses/any-precision-llm-MIT.txt` and applies to the inherited portions of
the analyzer, quantization and packed linear/kernel implementation.

PyTorch, Transformers, Accelerate, Datasets, Flash1DKMeans and the other
installed dependencies retain their respective licenses. They are installed
as dependencies rather than vendored here. Model weights and third-party
datasets are not part of this source distribution.
