"""HPC grid — wide sweep for a dedicated A100/H100-class GPU.

**Serious** (``BASE`` / ``AXES`` / ``MODELS``): long runs, full Cartesian
product, no early stopping by default.

**Trial** (``BASE_TRIAL`` / ``AXES_TRIAL`` / ``MODELS_TRIAL``): quick smoke
through a tiny subset; separate ``STUDY_NAME_TRIAL`` so batch folders do not
overlap serious studies.  When you run ``python -m Neural_Networks.apps.grid_search``,
choose trial vs serious in the prompt (TTY) or set ``NN_GRID_MODE`` / ``--mode``.

Parallel training is enabled by default (``MAX_PARALLEL_TRIALS = "auto"``).

Run:
  python -m Neural_Networks.apps.grid_search
  python -m Neural_Networks.apps.grid_report
"""
from __future__ import annotations

STUDY_NAME = "hpc_sweep"

SEEDS_PER_CELL = 2

RUN_DIR: str | None = None

# "auto" → detect from VRAM at runtime.  Set to 1 for sequential.
MAX_PARALLEL_TRIALS: int | str = "auto"

# Save lightweight ``snapshots/snapshot_ep*.pt`` every N epochs (0 = off).
SNAPSHOT_EVERY: int = 250

# Fixed hp applied to every cell — only keys NOT in AXES belong here.
BASE: dict = {
    "early_stopping":  False,
    "data_train_seed": 0,
    "epochs":            3000,
}

AXES: dict = {
    "learning_rate":       [1e-4, 5e-4],
    # Single schedule: fewer cells; warmup_cosine is the default sweet spot.
    "lr_scheduler":        ["warmup_cosine"],
    # One batch size on HPC avoids an extra axis with little gain vs LR/WD.
    "batch_size":          [1024],
    "weight_decay":        [2.0e-3, 8.0e-3],
    "dropout":             [0.0, 0.10],
    "feature_noise_std":   [0.01],
    "grad_clip_norm":      [5.0],
    # Only PhysicsRegularized + EquationConstrained sweep this (see grid_search).
    "physics_weight":      [0.05, 0.25, 0.35, 0.45],
    # One strong MLP block for flat models; width ablation is secondary here.
    "hidden_layers":       [[512, 1024, 512],  [512, 512, 512, 512], [256, 256, 256],],
    # Drop 0.3: 0.7 and full data already span low-data vs full-data regimes.
    "data_train_fraction": [1.0],
}

MODELS: list[str] = [
    "BlackBoxFNN",
    "ResidualCorrectionFNN",
    "PhysicsRegularizedFNN",
    "LagrangianStructuredFNN",
    "DecomposedStructuredPINNFNN",
    "EquationConstrainedPINNFNN",
]

# ---------------------------------------------------------------------------
# Trial preset — quick smoke (selected at runtime or via NN_GRID_MODE=trial)
# ---------------------------------------------------------------------------

STUDY_NAME_TRIAL = "hpc_sweep_trial"

SEEDS_TRIAL = 1

SNAPSHOT_TRIAL: int = 0

BASE_TRIAL: dict = {
    "early_stopping":  True,
    "patience":        5,
    "data_train_seed": 0,
    "epochs":          100,
}

AXES_TRIAL: dict = {
    "learning_rate":       [1e-4],
    "lr_scheduler":        ["warmup_cosine"],
    "batch_size":          [1024],
    "weight_decay":        [2.0e-3],
    "dropout":             [0.10],
    "feature_noise_std":   [0.01],
    "grad_clip_norm":      [5.0],
    "physics_weight":      [0.25],
    "hidden_layers":       [[512, 1024, 512]],
    "data_train_fraction": [1.0],
}

MODELS_TRIAL: list[str] = [
    "BlackBoxFNN",
    "ResidualCorrectionFNN",
    "PhysicsRegularizedFNN",
    "LagrangianStructuredFNN",
    "DecomposedStructuredPINNFNN",
    "EquationConstrainedPINNFNN",
]
