"""Laptop grid — small, fast sweep for local smoke-testing.

Purpose: verify the full pipeline works end-to-end on a local GPU/CPU in
minutes, not hours.  Use HPC grid for real tuning.

Edit any list below and re-run:  python -m Neural_Networks.apps.grid_search

Axis relevance (auto-filtered by grid_search):
  physics_weight  → only PhysicsRegularizedFNN, EquationConstrainedPINNFNN.
                    For all other models alpha is a no-op (physics lives in the
                    architecture) and this axis is projected out.
  hidden_layers   → only flat-MLP models.  Structured Lagrangian/Decomposed
                    ignore it (they use inertia/coriolis/gravity/friction_layers).

Parallel ETA seed caps (``NN_GRID_ETA_SEED_CAP_S``, ``NN_GRID_ETA_SEED_K``) are
documented on ``Neural_Networks.apps.grid_search`` when using large epoch counts.
"""
from __future__ import annotations

STUDY_NAME = "laptop_sweep"

SEEDS_PER_CELL = 1

# None → auto-pick the newest preprocessed run under Neural_Networks/train_data/
RUN_DIR: str | None = None

# "auto" detects from VRAM (will resolve to 1 on a 4 GB laptop GPU).
MAX_PARALLEL_TRIALS: int | str = "auto"

# Lightweight snapshots every N epochs (0 = off).
SNAPSHOT_EVERY: int = 10

# Fixed hp applied to every cell (overlaid on model defaults).
# Only keys that are NOT swept belong here.
BASE: dict = {
    "learning_rate":  3.0e-4,
    "early_stopping": False,
    "data_train_seed": 0,
    "epochs":           30,
}

# Sweep axes — Cartesian product.  Single-element lists are included so
# they show up in grid_log.jsonl for reproducibility even on the laptop
# preset; expand any list to run a real sweep.
AXES: dict = {
    "lr_scheduler":        ["warmup_cosine"],
    "batch_size":          [512],
    "weight_decay":        [5.0e-3],
    "dropout":             [0.10],
    "feature_noise_std":   [0.02],
    "grad_clip_norm":      [5.0],
    "physics_weight":      [0.10],
    "data_train_fraction": [0.3, 1.0],
}

MODELS: list[str] = [
    "BlackBoxFNN",
    "ResidualCorrectionFNN",
    "PhysicsRegularizedFNN",
    "LagrangianStructuredFNN",
    "DecomposedStructuredPINNFNN",
    "EquationConstrainedPINNFNN",
]
