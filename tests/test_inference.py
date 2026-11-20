"""The runtime must route once, hold the map fixed, and restore it on failure."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from query_adaptive_quantization.inference import QAQRuntime


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden, attention_mask):
        self.calls += 1
        return {"precision_values": torch.tensor([[[3, 4], [5, 6]]])}

    def precision_values_to_schedule(self, value):
        return value.tolist()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = Router()
        self.on_demand_enabled = False
        self.current = [[8, 8], [8, 8]]
        self.fail = False
        self.features = []
        self.generated_maps = []

    def get_router_input_hidden_states(self, input_ids, attention_mask, detach=True):
        self.features.append(input_ids.clone())
        return input_ids.float().unsqueeze(-1)

    def _snapshot_precisions(self):
        return [row[:] for row in self.current]

    def _restore_precisions(self, value):
        self.current = value

    def set_precision(self, value):
        self.current = value

    def generate(self, input_ids, *, attention_mask, use_router, **kwargs):
        assert use_router is False
        assert "past_key_values" not in kwargs
        self.generated_maps.append(self.current)
        if self.fail:
            raise RuntimeError("generation failed")
        # Stand in for multiple decoding steps; all see the same schedule.
        result = input_ids
        for _ in range(3):
            assert self.current == [[3, 4], [5, 6]]
            result = torch.cat([result, torch.tensor([[9]])], dim=1)
        return result


def test_prompt_only_map_is_reused_and_previous_precision_restored():
    model = Model()
    runtime = QAQRuntime(model, SimpleNamespace())
    prompt = torch.tensor([[1, 2, 3, 4]])
    result = runtime.generate(prompt)
    assert model.router.calls == 1
    assert torch.equal(model.features[0], prompt)
    assert result.tolist() == [[1, 2, 3, 4, 9, 9, 9]]
    assert model.current == [[8, 8], [8, 8]]
    assert runtime.last_precision_map.tolist() == [[3, 4], [5, 6]]


def test_generation_failure_restores_map_and_releases_lock():
    model = Model()
    runtime = QAQRuntime(model, None)
    prompt = torch.tensor([[1, 2]])
    model.fail = True
    with pytest.raises(RuntimeError, match="generation failed"):
        runtime.generate(prompt)
    assert model.current == [[8, 8], [8, 8]]
    model.fail = False
    runtime.generate(prompt)
    assert model.router.calls == 2


@pytest.mark.parametrize("kwargs", [
    {"past_key_values": object()}, {"use_router": True}, {"precision": 8},
    {"attention_mask": torch.tensor([[0, 1]])},
])
def test_runtime_rejects_overrides_and_padded_prompts(kwargs):
    model = Model()
    runtime = QAQRuntime(model, None)
    with pytest.raises(ValueError):
        runtime.generate(torch.tensor([[1, 2]]), **kwargs)
    assert model.router.calls == 0


def test_runtime_rejects_batches_and_concurrent_requests():
    runtime = QAQRuntime(Model(), None)
    with pytest.raises(ValueError):
        runtime.generate(torch.ones(2, 4, dtype=torch.long))
    runtime._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="concurrent"):
            runtime.generate(torch.ones(1, 4, dtype=torch.long))
    finally:
        runtime._lock.release()
