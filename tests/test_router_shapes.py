import gc
import pytest
import torch
from query_adaptive_quantization.modules.Router import OnlineLearningPrecisionRouter


@pytest.mark.parametrize("num_layers", (28, 36, 40))
def test_canonical_router_supports_model_specific_layer_counts(
    num_layers: int,
) -> None:
    router = OnlineLearningPrecisionRouter(
        hidden_size=64,
        num_layers=num_layers,
    ).eval()
    with torch.no_grad():
        output = router(
            torch.randn(1, 3, 64),
            torch.ones(1, 3, dtype=torch.long),
            hard_selection=True,
        )
    assert list(output["precision_logits"].shape) == [
        1,
        num_layers,
        2,
        6,
    ]
    assert list(output["precision_values"].shape) == [1, num_layers, 2]
    del router
    gc.collect()


def test_canonical_router_rejects_research_architectures() -> None:
    with pytest.raises(ValueError, match="canonical VAMP Router"):
        OnlineLearningPrecisionRouter(
            hidden_size=64,
            num_layers=4,
            router_readout="mean",
        )

