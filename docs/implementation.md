# Implementation guide

## Search

`ThresholdPrecisionMapSearcher` in `threshold_maps/search.py` first computes
the uniform reference logits. Group-level probes and a pruned additive
frontier propose maps; complete packed-model forwards verify the candidates.
Refinement lowers individual slots or exchanges precision between slots.
`BlockWeightCostModel` weights each branch by its number of quantized weight
parameters. The objective is a proxy for storage/computation, not a measured
latency guarantee.

The public search command fixes the reference at Int8 and the candidate set
at Int3–Int8. Its `--threshold` is the allowed mean next-token KL on a
calibration sequence. The lower-level search class also supports other
reference settings.

## Router

`OnlineLearningPrecisionRouter` in `modules/Router.py` uses two Online TTT
blocks, eight-segment attention and a layer GRU. Two initial transformer
layers at Int5 supply prompt features. Fast states are temporary and local
to each token; inference does not train persistent parameters or carry fast
states between requests.

The logits have shape `[batch, layers, 2, 6]`, with branches ordered as
attention (`mha`) and feed-forward (`mlp`). The six classes represent
`[3, 4, 5, 6, 7, 8]` bits. Evaluation selects the argmax. The training loss in
`training/threshold_map/loss.py` compares these logits with verified map labels.

## Inference

`inference.predict_map` reads the prompt features once. `fixed_map` applies
the result and restores previous precision settings even if generation
raises an exception. `QAQRuntime.generate` runs a fresh prefill under the
chosen map and holds that map through decoding. Requests on one runtime
instance are serial.

The inference command uses resident execution, without a hardware-specific cost
profile. The packed backend retains its transfer/cache primitives to preserve
its module structure, but their configuration and performance experiments are
not part of the inference interface.

## Artifacts

Each label row contains token IDs, an attention mask, a `[layers, 2]`
`precision_map`, measured `kl_to_reference`, weighted bit width and search
statistics. `manifest.json` records configuration and data hashes and changes
to `status: complete` only after all labels and splits have been written.
Interrupted runs may leave partial labels; choose a fresh output directory
when rerunning the search command.

A Router artifact consists of `router_weights.pt` (a state dictionary),
`router_config.json` and the saved training configuration. Generation requires
the first two files and an explicitly supplied packed model. The state
dictionary is loaded strictly, without replacing missing parameters.

The import namespace and checkpoint layout retain the original QAQ names.
VAMP is the method name used in the public documentation.
