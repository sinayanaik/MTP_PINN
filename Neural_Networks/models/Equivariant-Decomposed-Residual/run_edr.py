#!/usr/bin/env python3
"""EDR training entry point.

Equivariant-Decomposed-Residual model: learns four structurally-constrained
corrections (gravity, inertia, Coriolis, friction) on top of the nominal
RNEA + friction physics model.  Default δ-nets are wider (~10–35k params)
than the guide’s minimal sketch but still far smaller than BlackBoxFNN.

Usage (from repository root)::

    PYTHONPATH=. python3 Neural_Networks/models/Equivariant-Decomposed-Residual/run_edr.py

The script adds the EDR directory to sys.path so that sibling imports
(edr_model, edr_corrections, edr_strategy) resolve correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — must happen before importing EDR siblings.
# Add the EDR directory (this file's parent) to sys.path if not already there.
# This is necessary because the directory name contains a hyphen, which makes
# it an invalid Python package identifier.
# ---------------------------------------------------------------------------
_EDR_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _EDR_DIR.parent.parent.parent   # MTP_PINN/

if str(_EDR_DIR) not in sys.path:
    sys.path.insert(0, str(_EDR_DIR))

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports — sibling EDR modules and shared pipeline.
# ---------------------------------------------------------------------------
from edr_strategy import EDR_STRATEGY, DEFAULT_EXHAUSTIVE_EDR  # noqa: E402

from Neural_Networks.models.shared.pipeline import TrainJob, main_cli  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset path — edit this to point at your preprocessed run directory.
# ---------------------------------------------------------------------------
_NN_ROOT = _REPO_ROOT / "Neural_Networks"

TRAIN_DATA_RUN_DIR: str = str(
    _NN_ROOT / "train_data"
    / "run_abl_q0_qd91_qdd91_tau51_lk_20260515_1837"
)

MODELS_DIR:    str = str(_NN_ROOT / "Trained_Models" / "Journal_Comparison" / "EDR")
REGISTRY_FILE: str = str(_NN_ROOT / "Trained_Models" / "Journal_Comparison" / "models_registry.yaml")

# ---------------------------------------------------------------------------
# Hyperparameters
# All keys in DEFAULT_EXHAUSTIVE_EDR are valid here; override as needed.
# ---------------------------------------------------------------------------
HP: dict[str, Any] = {
    # ── Training schedule ──────────────────────────────────────────────────
    "epochs":              1090,
    "batch_size":          256,
    "learning_rate":       3e-4,
    "weight_decay":        2e-3,
    "optimizer":           "adamw",
    "lr_scheduler":        "cosine_warm_restarts",
    "warm_restart_T_0":    15,
    "warm_restart_T_mult": 1,
    "warm_restart_eta_min": 3e-6,
    "early_stopping":      True,
    "early_stop_metric":   "val_loss",    # val_loss tracks training objective; val_rmse has dist-shift.
    "patience":            120,           # Allow ~150 epochs total (same as old best).
    "min_delta":           1e-5,          # val_loss scale is much smaller; use tight threshold.
    "grad_clip_norm":      1.0,
    "feature_noise_std":   0.02,
    "print_every":         2,
    "seed":                42,
    "data_train_fraction": 1.0,
    "data_train_seed":     0,
    "stride":              1,
    "snapshot_every":      0,
    "torch_compile":       False,
    "torch_compile_mode":  "default",
    # ── EDR network architecture ───────────────────────────────────────────
    "activation":          "silu",
    "gravity_hidden":      [64, 64],      # 2× old capacity; keeps generalisation.
    "inertia_hidden":      [64, 64],
    "coriolis_hidden":     [64, 64],
    "friction_hidden":     [32, 32],
    "correction_dropout":  0.15,
    # ── EDR curriculum (adaptive phase-2 transition) ──────────────────────
    "phase2_start_epoch":     None,
    "phase2_plateau_window":  5,
    "phase2_plateau_threshold": 5e-3,
    "phase2_min_epoch":       3,
    "phase2_max_epoch":       25,         # Force phase-2 early; strong reg keeps corrections small.
    # ── EDR loss weights ──────────────────────────────────────────────────
    "lambda_correction_reg":  5e-2,       # Strong regularisation — same as old best model.
    "correction_reg_inertia_normalize": True,
    "enable_passivity_loss":  False,
    "lambda_passivity":       1e-2,
    "frozen_lr_ratio":        1.0,
}

MODEL_TYPE  = "EDR"
SAVE_SUBDIR = "EDR"


def main() -> None:
    """Construct and launch the EDR training job."""
    # Load normalization stats so the gravity network gets sin/cos trig features.
    _meta_path = Path(TRAIN_DATA_RUN_DIR) / "metadata.json"
    if _meta_path.exists():
        with open(_meta_path) as f:
            _meta = json.load(f)
        _norm = _meta.get("normalisation", {})
        if "mean_q" in _norm and "std_q" in _norm:
            HP["_q_mean"] = _norm["mean_q"]
            HP["_q_std"]  = _norm["std_q"]

    job = TrainJob(
        run_dir=TRAIN_DATA_RUN_DIR,
        models_dir=MODELS_DIR,
        registry_file=REGISTRY_FILE,
        model_type=MODEL_TYPE,
        save_subdir=SAVE_SUBDIR,
        hp=HP,
        strategy=EDR_STRATEGY,
        run_help=(
            "Neural_Networks/models/Equivariant-Decomposed-Residual/run_edr.py"
        ),
    )
    main_cli(job, log_name="run_edr")


if __name__ == "__main__":
    main()
