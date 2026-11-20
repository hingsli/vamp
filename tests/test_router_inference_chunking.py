from unittest.mock import patch

import pytest
import torch

from query_adaptive_quantization.modules.Router import OnlineLearningPrecisionRouter


@pytest.fixture
def router():
    torch.manual_seed(42)
    return OnlineLearningPrecisionRouter(hidden_size=64, num_layers=4).eval()


def reference_forward(router, hidden, mask):
    """Pre-chunking forward, including the original full-sequence readout."""
    inputs = hidden.to(router.task_qkv_proj.weight.dtype)
    projections = [router.task_qkv_proj, *router.extra_ttt_qkv_projs]
    for index, projection in enumerate(projections):
        query, key, value = projection(inputs).chunk(3, dim=-1)
        if mask is not None:
            token_mask = mask.to(query.dtype).unsqueeze(-1)
            query, key, value = (x * token_mask for x in (query, key, value))
        state = router._init_fast_weights(
            query.shape[0], query.shape[1], query.device, query.dtype
        )
        state = router._learn(key, value, state, index)
        inputs = router._predict(query, state, index)
    return router._selection_outputs(router._decode(inputs, mask), True)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 19])
@pytest.mark.parametrize("masked", [False, True])
def test_chunking_matches_original_forward(router, length, masked):
    # Double precision isolates chunking equivalence from GEMM accumulation differences.
    router = router.double()
    router.INFERENCE_TOKEN_CHUNK_SIZE = 8
    hidden = torch.randn(2, length, 64)
    mask = None
    if masked:
        mask = torch.ones(2, length, dtype=torch.long)
        mask[0, length // 2:] = 0
        mask[1, :length // 3] = 0
    with torch.no_grad():
        expected = reference_forward(router, hidden, mask)
        with patch.object(router, "_apply_ttt_tokens", wraps=router._apply_ttt_tokens) as apply:
            actual = router(hidden, mask, hard_selection=True)
    for call in apply.call_args_list:
        assert call.args[0].shape[1] <= 8
    torch.testing.assert_close(
        actual["precision_logits"], expected["precision_logits"], rtol=1e-9, atol=1e-10
    )
    assert torch.equal(actual["precision_values"], expected["precision_values"])
    assert router.total_forwards.item() == 4
    assert router.precision_usage_stats.sum().item() == 4 * 4 * 2


def test_default_chunk_bound_and_state_dict_compatibility(router):
    state = {key: value.clone() for key, value in router.state_dict().items()}
    assert router.INFERENCE_TOKEN_CHUNK_SIZE == 128
    with torch.inference_mode(), patch.object(
        router, "_apply_ttt_tokens", wraps=router._apply_ttt_tokens
    ) as apply:
        output = router(torch.randn(1, 129, 64), hard_selection=True)
    assert [call.args[0].shape[1] for call in apply.call_args_list] == [128, 1, 128, 1]
    assert torch.isfinite(output["precision_logits"]).all()
    missing, unexpected = router.load_state_dict(state, strict=True)
    assert not missing and not unexpected


@pytest.mark.parametrize("training,grad_enabled", [(True, False), (True, True), (False, True)])
def test_training_and_autograd_keep_original_path(router, training, grad_enabled):
    router.train(training)
    router.INFERENCE_TOKEN_CHUNK_SIZE = 2
    embedding = torch.nn.Embedding(16, 64).requires_grad_(False)
    hidden = embedding(torch.tensor([[1, 2, 3, 4, 5]]))
    with torch.set_grad_enabled(grad_enabled), patch.object(
        router, "_apply_ttt_tokens", wraps=router._apply_ttt_tokens
    ) as apply:
        output = router(hidden, hard_selection=True)
        loss = output["precision_logits"].square().mean()
        assert torch.isfinite(loss)
        if grad_enabled:
            loss.backward()
            assert embedding.weight.grad is None
            assert router.task_qkv_proj.weight.grad is not None
            assert all(torch.isfinite(p.grad).all() for p in router.parameters() if p.grad is not None)
    assert [call.args[0].shape[1] for call in apply.call_args_list] == [5, 5]
