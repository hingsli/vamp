# Prepare a packed model

The workflow uses the shared bitplane representation used by VAMP.
Skip this step if you already have a compatible packed model and tokenizer.

Install the preparation dependencies in the same Python/CUDA environment:

```bash
python -m pip install -e '.[quantize]'
```

Run the existing gradient collection, incremental quantization and packing
pipeline on a local Qwen3-4B checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=0
python quantize.py /path/to/Qwen3-4B \
  --mode pack --seed-precision 3 --parent-precision 8 \
  --dataset wikitext2 --num-examples 64 --seq-len 512 \
  --use-optimized-gradients --gradient-checkpointing \
  --cpu-count 4 --random-state 42 --cache-dir cache
```

This uses WikiText-2 for salience calibration and may download that dataset.
The model path is supplied by the caller. Model preparation requires memory
for the original model and intermediate tensors; fitting the final packed
model does not imply that preparation fits on the same device.

For a local model directory named `Qwen3-4B`, the command writes:

```text
cache/packed/qaqprec-(Qwen3-4B)-w8_orig3-gc1-wikitext2_s64_blk512/
```

Point `PACKED_MODEL` at that directory. It contains the tokenizer, a
`config.json` with an `anyprec` field and packed weight tensors. The seed and
parent precision must cover the Router's complete Int3–Int8 range.

Packing is independent of the eight-example demonstration in `examples/`.
For an experiment, keep the quantization calibration data, Router label data
and final evaluation data separate, and retain the preparation settings with
the resulting artifacts. The repository does not include trained
weights or claim to reproduce the paper's tables from its sample text.
