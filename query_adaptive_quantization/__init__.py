"""VAMP public API, retaining the original import namespace for compatibility."""

from importlib import import_module

__all__ = ["QueryAdaptiveQuantizationForCausalLM", "QAQRuntime", "load_runtime"]


def __getattr__(name):
    if name in {"QAQRuntime", "load_runtime"}:
        return getattr(import_module(".inference", __name__), name)
    if name == "QueryAdaptiveQuantizationForCausalLM":
        return import_module(".modules.QueryAdaptiveQuantizationForCausalLM", __name__).QueryAdaptiveQuantizationForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
