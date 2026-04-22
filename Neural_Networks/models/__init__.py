"""Torque ``nn.Module`` definitions and training entrypoints.

Run from repo root with ``PYTHONPATH=.``:

- ``python3 -m Neural_Networks.models.run_fnn``
- ``python3 -m Neural_Networks.models.run_physics_regularized``
- ``python3 -m Neural_Networks.models.run_physics_residual``

Shared training code (metrics, pipeline, checkpointing) lives in
``Neural_Networks.models.shared``.

``torch`` is imported lazily (see ``__getattr__``) so auxiliary modules
under this package (e.g. the grid-search driver) can configure
``CUDA_VISIBLE_DEVICES`` in a multiprocessing initializer *before* CUDA
initialises in worker processes.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ACTIVATION_MAP",
    "BlackBoxFNN",
    "PhysicsRegularizedFNN",
    "ResidualCorrectionFNN",
    "build_mlp",
    "reduce_physics_to_total",
]

_TORQUE_MOD = "Neural_Networks.models.torque_models"


def __getattr__(name: str) -> Any:
    if name in __all__:
        _tm = importlib.import_module(_TORQUE_MOD)
        return getattr(_tm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # noqa: D401
    return sorted(__all__)
