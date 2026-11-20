"""Optional GPU smoke check using caller-supplied packed weights and Router."""

import os

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("PACKED_MODEL")
    or not os.environ.get("ROUTER_ARTIFACT"),
    reason="Requires CUDA, PACKED_MODEL and ROUTER_ARTIFACT",
)
def test_resident_generation_uses_one_valid_map():
    from query_adaptive_quantization import load_runtime

    runtime = load_runtime(
        packed_model=os.environ["PACKED_MODEL"],
        router_artifact=os.environ["ROUTER_ARTIFACT"],
    )
    inputs = runtime.tokenizer("Explain how rain forms.", return_tensors="pt").to(runtime.model.device)
    calls = []
    handle = runtime.model.router.register_forward_hook(lambda *_: calls.append(1))
    previous = runtime.model._snapshot_precisions()
    try:
        tokens = runtime.generate(**inputs, max_new_tokens=8, do_sample=False)
    finally:
        handle.remove()
    assert len(calls) == 1
    assert tokens.shape[1] > inputs.input_ids.shape[1]
    precision = runtime.last_precision_map
    assert precision.shape == (len(runtime.model.layer_to_aplinears), 2)
    assert set(precision.flatten().tolist()) <= set(range(3, 9))
    assert runtime.model._snapshot_precisions() == previous
