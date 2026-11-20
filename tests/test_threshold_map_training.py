import torch
import torch.nn as nn

from query_adaptive_quantization.training.threshold_map.config import (
    build_training_config,
)
from query_adaptive_quantization.training.threshold_map.loss import (
    precision_map_cross_entropy,
    precision_bits_to_indices,
)
from query_adaptive_quantization.training.threshold_map.trainer import (
    ThresholdMapTrainer,
)
from query_adaptive_quantization.cli.train import build_parser


class _ToyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 2
        self.precision_levels = [3, 4, 5, 6, 7, 8]
        self.projection = nn.Linear(4, self.num_layers * 2 * 6)

    def forward(self, hidden_states, attention_mask, hard_selection=False):
        pooled = hidden_states.mean(dim=1)
        return {
            "precision_logits": self.projection(pooled).reshape(-1, 2, 2, 6)
        }


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.embedding.requires_grad_(False)
        self.router = _ToyRouter()

    def get_router_input_hidden_states(
        self, input_ids, attention_mask, detach=True
    ):
        hidden_states = self.embedding(input_ids)
        return hidden_states.detach() if detach else hidden_states


def test_precision_bits_to_indices():
    target = torch.tensor([[[3, 8], [5, 6]]])
    indices = precision_bits_to_indices(target, [3, 4, 5, 6, 7, 8])
    assert indices.tolist() == [[[0, 5], [2, 3]]]


def test_map_cross_entropy_is_finite_and_differentiable():
    logits = torch.randn(2, 4, 2, 6, requires_grad=True)
    target = torch.randint(3, 9, (2, 4, 2))
    output = precision_map_cross_entropy(
        logits,
        target,
        [3, 4, 5, 6, 7, 8],
        label_smoothing=0.05,
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_new_training_config_uses_generic_method_names():
    config = build_training_config(
        student_model="packed-model",
        label_dataset="labels",
        output_dir="router",
        epochs=1,
        online_ttt_lr=0.003,
    )
    assert config["experiment_name"] == "threshold-supervised-precision-routing"
    assert config["dataset"]["name"] == "threshold_precision_map_labels"
    assert (
        config["model_config"]["router"]["loss_mode"]
        == "threshold_precision_map_ce"
    )
    router = config["model_config"]["router"]
    assert router["online_ttt_lr"] == 0.003
    assert router["router_config"]["ttt_lr_linear"] == 0.003
    assert router["router_config"]["ttt_lr_ln"] == 0.003


def test_training_defaults_to_validated_online_ttt_configuration():
    config = build_training_config(
        student_model="packed-model",
        label_dataset="labels",
        output_dir="router",
        epochs=1,
    )
    router = config["model_config"]["router"]
    assert router["router_architecture"] == "online_ttt"
    assert router["online_ttt_lr"] == 0.1
    assert router["online_ttt_concept_dim"] == 352
    assert router["router_config"]["ttt_lr_linear"] == 0.1
    assert router["router_config"]["ttt_lr_ln"] == 0.1


def test_public_training_cli_only_exposes_online_ttt_controls():
    option_strings = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert "--online-ttt-lr" in option_strings
    assert "--online-ttt-concept-dim" not in option_strings
    assert "--online-ttt-depth" not in option_strings
    assert "--architecture-variant" not in option_strings
    assert ("--router-" + "architecture") not in option_strings
    assert ("--layer-" + "branch-query-dim") not in option_strings


def test_trainer_loss_updates_only_router_parameters():
    model = _ToyModel()
    trainer_stub = type(
        "TrainerStub",
        (),
        {"label_smoothing": 0.05, "_last_map_metrics": {}},
    )()
    loss = ThresholdMapTrainer.compute_loss(
        trainer_stub,
        model,
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "precision_map": torch.tensor([[[4, 5], [6, 7]]]),
        },
    )
    loss.backward()
    assert model.embedding.weight.grad is None
    assert model.router.projection.weight.grad is not None
    assert torch.isfinite(model.router.projection.weight.grad).all()
