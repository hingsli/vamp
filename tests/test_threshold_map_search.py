from types import SimpleNamespace

import torch
import torch.nn as nn

from query_adaptive_quantization.threshold_maps import (
    BlockWeightCostModel,
    ThresholdPrecisionMapSearcher,
)


class DummyAPLinear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features


class DummyPackedModel(nn.Module):
    def __init__(self, num_layers=4, vocab_size=16):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(
            anyprec={
                "arch_config": {
                    "module_names": ["self_attn.q_proj", "mlp.up_proj"]
                }
            }
        )
        self.layer_to_aplinears = [
            [DummyAPLinear(4, 4), DummyAPLinear(4, 8)]
            for _ in range(num_layers)
        ]
        self.current = {
            layer_idx: {"mha": 8, "mlp": 8} for layer_idx in range(num_layers)
        }

    def _set_precision_silent(self, schedule):
        self.current = {
            int(layer_idx): {
                "mha": int(values["mha"]),
                "mlp": int(values["mlp"]),
            }
            for layer_idx, values in schedule.items()
        }

    def _snapshot_precisions(self):
        return {layer: dict(values) for layer, values in self.current.items()}

    def _restore_precisions(self, snapshot):
        self._set_precision_silent(snapshot)

    def forward(self, input_ids, **_):
        batch_size, sequence_length = input_ids.shape
        token_axis = torch.arange(
            self.vocab_size, device=input_ids.device, dtype=torch.float32
        )
        centers = (input_ids % self.vocab_size).float().unsqueeze(-1)
        logits = -0.05 * (token_axis.view(1, 1, -1) - centers).pow(2)
        weighted_bit = sum(
            values["mha"] + 2.0 * values["mlp"]
            for values in self.current.values()
        ) / (3.0 * len(self.current))
        degradation = (8.0 - weighted_bit) * 0.08
        logits[..., 0] -= degradation
        logits[..., 1] += degradation
        return SimpleNamespace(logits=logits.view(batch_size, sequence_length, -1))


def test_weight_cost_uses_branch_parameter_counts():
    model = DummyPackedModel(num_layers=4)
    cost = BlockWeightCostModel.from_model(model)
    assert cost.manifest()["branch_weight_counts"] == {"mha": 64, "mlp": 128}
    precision_map = torch.tensor([[4, 5]] * 4)
    assert cost.effective_bit(precision_map) == (64 * 4 + 128 * 5) / 192


def test_reference_search_returns_feasible_blockwise_map():
    model = DummyPackedModel(num_layers=4)
    searcher = ThresholdPrecisionMapSearcher(
        model,
        num_groups=2,
        max_dp_candidates=4,
        refinement_passes=1,
        max_swap_candidates=2,
    )
    input_ids = torch.arange(16).view(1, 16) % model.vocab_size
    attention_mask = torch.ones_like(input_ids)
    reference = searcher.build_reference(input_ids, attention_mask)
    reference_map = searcher.full_map(8, input_ids.device)
    reference_eval = searcher.evaluate_map(
        input_ids, attention_mask, reference_map, reference
    )
    assert reference_eval.kl_to_reference < 1e-7

    threshold = 0.01
    result = searcher.search_selected_threshold(
        input_ids, attention_mask, threshold
    )
    precision_map = torch.tensor(result["precision_map"])
    assert tuple(precision_map.shape) == (4, 2)
    assert set(precision_map.reshape(-1).tolist()).issubset({3, 4, 5, 6, 7, 8})
    assert result["kl_to_reference"] <= threshold + 1e-6
    assert result["weighted_effective_bit"] <= 8.0

    sweep = searcher.group_sweep(
        input_ids,
        attention_mask,
        thresholds=[0.005, threshold],
    )
    for threshold_result in sweep["threshold_results"].values():
        assert (
            threshold_result["kl_to_reference"]
            <= threshold_result["threshold"] + 1e-6
        )


def test_kl_uses_only_valid_next_token_positions():
    model = DummyPackedModel(num_layers=4)
    searcher = ThresholdPrecisionMapSearcher(model, num_groups=2)
    input_ids = torch.arange(8).view(1, 8) % model.vocab_size
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
    reference = searcher.build_reference(input_ids, attention_mask)
    assert reference.token_count == 3
    reference_evaluation = searcher.evaluate_map(
        input_ids,
        attention_mask,
        searcher.full_map(8, input_ids.device),
        reference,
    )
    int3_evaluation = searcher.evaluate_map(
        input_ids,
        attention_mask,
        searcher.full_map(3, input_ids.device),
        reference,
    )
    assert reference_evaluation.kl_to_reference < 1e-7
    assert int3_evaluation.kl_to_reference > reference_evaluation.kl_to_reference


def test_int4_reference_supports_higher_bits_and_never_exceeds_reference_cost():
    model = DummyPackedModel(num_layers=4)
    searcher = ThresholdPrecisionMapSearcher(
        model,
        reference_bit=4,
        precision_levels=[3, 4, 5, 6, 7, 8],
        num_groups=2,
        max_dp_candidates=4,
        refinement_passes=1,
        max_swap_candidates=4,
    )
    input_ids = torch.arange(16).view(1, 16) % model.vocab_size
    attention_mask = torch.ones_like(input_ids)
    reference = searcher.build_reference(input_ids, attention_mask)
    reference_map = searcher.full_map(4, input_ids.device)
    reference_evaluation = searcher.evaluate_map(
        input_ids, attention_mask, reference_map, reference
    )
    assert reference_evaluation.kl_to_reference < 1e-7

    sweep = searcher.group_sweep(
        input_ids,
        attention_mask,
        thresholds=[0.01],
        reference=reference,
    )
    assert set(sweep["uniform"]) == {"3", "4", "5", "6", "7", "8"}

    result = searcher.search_selected_threshold(
        input_ids,
        attention_mask,
        threshold=0.01,
    )
    precision_map = torch.tensor(result["precision_map"])
    assert tuple(precision_map.shape) == (4, 2)
    assert set(precision_map.flatten().tolist()).issubset({3, 4, 5, 6, 7, 8})
    assert result["kl_to_reference"] <= 0.01 + 1e-6
    assert result["weighted_cost"] <= reference_evaluation.weighted_cost


def test_search_falls_back_to_uniform_reference_when_no_lower_map_is_feasible():
    model = DummyPackedModel(num_layers=4)
    searcher = ThresholdPrecisionMapSearcher(
        model,
        reference_bit=4,
        precision_levels=[3, 4],
        num_groups=2,
        refinement_passes=1,
    )
    input_ids = torch.arange(16).view(1, 16) % model.vocab_size
    attention_mask = torch.ones_like(input_ids)
    result = searcher.search_selected_threshold(
        input_ids,
        attention_mask,
        threshold=0.0,
    )
    assert result["kl_to_reference"] == 0.0
    assert result["precision_map"] == [[4, 4]] * 4
