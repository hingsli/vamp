"""Resident inference: select one map from the prompt, then hold it fixed."""

from contextlib import contextmanager
import json
from pathlib import Path
import threading

import torch

from .batch_routing import shared_precision_map


@torch.no_grad()
def predict_map(model, input_ids, attention_mask=None):
    """Read only prompt features; the Router returns [layers, 2] bit widths."""
    if model.training or model.router.training:
        raise ValueError("Call eval() on the model and Router before prediction")
    hidden = model.get_router_input_hidden_states(input_ids, attention_mask, detach=True)
    output = model.router(hidden, attention_mask)
    return shared_precision_map(output["precision_values"]).detach()


@contextmanager
def fixed_map(model, precision_map):
    """Apply a map for prefill and decode, restoring the previous map on exit."""
    previous = model._snapshot_precisions()
    try:
        model.set_precision(model.router.precision_values_to_schedule(precision_map))
        yield
    finally:
        model._restore_precisions(previous)


class QAQRuntime:
    """Serial generation with resident packed weights and a frozen Router.

    The Python class name is retained for compatibility with existing code.
    This wrapper accepts one unpadded prompt per call and does not reuse a
    prelude KV cache for generation.
    """

    def __init__(self, model, tokenizer):
        if model.on_demand_enabled:
            raise ValueError("This entry point requires resident weights")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.last_precision_map = None
        self._lock = threading.Lock()

    @torch.no_grad()
    def generate(self, input_ids, *, attention_mask=None, **kwargs):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Provide one nonempty prompt with shape [1, tokens]")
        forbidden = {"use_router", "precision", "precision_map", "inputs_embeds", "past_key_values"}
        if forbidden.intersection(kwargs):
            raise ValueError("The runtime selects the precision map and creates a fresh generation cache")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or not torch.all(attention_mask == 1):
            raise ValueError("The runtime accepts one unpadded prompt")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Use separate runtime instances for concurrent requests")
        try:
            self.last_precision_map = None
            precision = predict_map(self.model, input_ids, attention_mask)
            self.last_precision_map = precision.cpu()
            with fixed_map(self.model, precision):
                return self.model.generate(
                    input_ids, attention_mask=attention_mask, use_router=False, **kwargs
                )
        finally:
            self._lock.release()


def load_runtime(*, packed_model, router_artifact, device="cuda:0"):
    """Load a packed model and a strictly matching Router state dictionary."""
    from transformers import AutoTokenizer
    from .modules import QueryAdaptiveQuantizationForCausalLM
    from .training.threshold_map.config import validate_router_architecture

    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Packed-model inference requires an NVIDIA GPU and the CUDA extension")
    if torch.cuda.device_count() != 1 or torch.device(device).index not in (None, 0):
        raise RuntimeError("Select one CUDA GPU with CUDA_VISIBLE_DEVICES and use device='cuda:0'")
    artifact = Path(router_artifact)
    outer = json.loads((artifact / "router_config.json").read_text())
    config = validate_router_architecture(outer.get("router_config", outer))
    model = QueryAdaptiveQuantizationForCausalLM.from_quantized(
        str(packed_model), trust_remote_code=False, enable_router=True, router_config=config,
    )
    if not set(range(3, 9)).issubset(model.supported_bits):
        raise ValueError("The packed model must support all Int3–Int8 precisions")
    expected = outer.get("model_signature")
    observed = {
        "num_layers": len(model.layer_to_aplinears),
        "hidden_size": int(model.config.hidden_size),
        "num_branches": int(model.router.num_precision_blocks),
        "precision_levels": list(model.router.precision_levels),
    }
    if expected is not None and expected != observed:
        raise ValueError(f"Router and packed model signatures differ: expected {expected}, got {observed}")
    state = torch.load(artifact / "router_weights.pt", map_location="cpu", weights_only=True)
    model.router.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.router_feature_extraction = "early_exit"
    tokenizer = AutoTokenizer.from_pretrained(str(packed_model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return QAQRuntime(model, tokenizer)
