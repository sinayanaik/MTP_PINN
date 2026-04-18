"""Torque prediction models (physics-regularized + residual only)."""

from Neural_Networks.models.torque_fnns import (
    PhysicsRegularizedFNN,
    ResidualCorrectionFNN,
    UnifiedTorqueFNN,
    _reduce_physics_to_total,
)

MODEL_REGISTRY: dict[str, type] = {
    "PhysicsRegularizedFNN": PhysicsRegularizedFNN,
    "ResidualCorrectionFNN": ResidualCorrectionFNN,
}

MODEL_CATEGORIES: dict[str, list[str]] = {
    "Physics-regularized": ["PhysicsRegularizedFNN"],
    "Residual correction": ["ResidualCorrectionFNN"],
}

FNN_MODELS: set[str] = set(MODEL_REGISTRY)
PHYSICS_WEIGHT_MODELS: set[str] = {"PhysicsRegularizedFNN"}
PHYSICS_INPUT_MODELS: set[str] = {"ResidualCorrectionFNN"}
MODEL_SAVE_DIRS: dict[str, str] = {name: name for name in MODEL_REGISTRY}

__all__ = [
    "PhysicsRegularizedFNN",
    "ResidualCorrectionFNN",
    "UnifiedTorqueFNN",
    "_reduce_physics_to_total",
    "MODEL_REGISTRY",
    "MODEL_CATEGORIES",
    "MODEL_SAVE_DIRS",
    "FNN_MODELS",
    "PHYSICS_WEIGHT_MODELS",
    "PHYSICS_INPUT_MODELS",
]
