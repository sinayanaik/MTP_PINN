"""Torque ``nn.Module`` definitions and training entrypoints.

Run from repo root with ``PYTHONPATH=.``:

- ``python3 -m Neural_Networks.models.run_fnn``
- ``python3 -m Neural_Networks.models.run_physics_regularized``
- ``python3 -m Neural_Networks.models.run_physics_residual``

Shared training code (metrics, pipeline, checkpointing) lives in
``Neural_Networks.models.shared``.
"""

from Neural_Networks.models.torque_models import (
    ACTIVATION_MAP,
    BlackBoxFNN,
    PhysicsRegularizedFNN,
    ResidualCorrectionFNN,
    build_mlp,
    reduce_physics_to_total,
)

__all__ = [
    "ACTIVATION_MAP",
    "BlackBoxFNN",
    "PhysicsRegularizedFNN",
    "ResidualCorrectionFNN",
    "build_mlp",
    "reduce_physics_to_total",
]
