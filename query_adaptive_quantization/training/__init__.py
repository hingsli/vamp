"""Training namespace for the quality-threshold Router release."""

from importlib import import_module

__all__ = [
    "ThresholdMapTrainingPipeline",
]


def __getattr__(name):
    if name == "ThresholdMapTrainingPipeline":
        return import_module(
            ".threshold_map", __name__
        ).ThresholdMapTrainingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
