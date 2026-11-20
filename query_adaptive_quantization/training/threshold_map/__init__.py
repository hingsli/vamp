"""Direct training from quality-threshold precision maps."""

from .config import (
    ROUTER_ARCHITECTURE,
    build_router_architecture,
    build_training_config,
    validate_router_architecture,
)
from .pipeline import ThresholdMapTrainingPipeline

__all__ = [
    "ThresholdMapTrainingPipeline",
    "ROUTER_ARCHITECTURE",
    "build_router_architecture",
    "build_training_config",
    "validate_router_architecture",
]
