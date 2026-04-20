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
    / "run_0419_1338_qraw_d25p3i_ddL_mraw_a1_R_70v15t15_f1p0t1p0_f6a3df"
)

MODELS_DIR:    str = str(_NN_ROOT / "Trained_Models" / "EDR")
REGISTRY_FILE: str = str(_NN_ROOT / "Trained_Models" / "models_registry.yaml")

# ---------------------------------------------------------------------------
# Hyperparameters
# All keys in DEFAULT_EXHAUSTIVE_EDR are valid here; override as needed.
# ---------------------------------------------------------------------------
HP: dict[str, Any] = {
    # ── Training schedule ──────────────────────────────────────────────────
    "epochs":              300,
    "batch_size":          256,
    "learning_rate":       3e-4,
    "weight_decay":        1e-3,
    "optimizer":           "adamw",
    "lr_scheduler":        "reduce_on_plateau",
    "early_stopping":      True,
    "early_stop_metric":   "val_rmse",
    "patience":            80,
    "min_delta":           1e-4,
    "grad_clip_norm":      1.0,
    "feature_noise_std":   0.01,
    "print_every":         2,
    "seed":                42,
    "data_train_fraction": 1.0,
    "data_train_seed":     0,
    "stride":              1,
    "snapshot_every":      0,
    "torch_compile":       False,   # Do not enable — incompatible with Jacobian.
    "torch_compile_mode":  "default",
    # ── EDR network architecture ───────────────────────────────────────────
    "activation":          "silu",
    "gravity_hidden":      [32, 32],
    "inertia_hidden":      [32, 32],
    "coriolis_hidden":     [32, 32],
    "friction_hidden":     [16, 16],
    "correction_dropout":  0.20,
    # ── EDR curriculum ─────────────────────────────────────────────────────
    # Phase 1: gravity + friction only. Phase 2: all four corrections.
    # Phase 1 alone doesn't improve val_rmse much; shorten to let inertia/Coriolis
    # activate early (they carry most of the correction signal).
    "phase2_start_epoch":  10,
    # ── EDR loss weights ──────────────────────────────────────────────────
    "lambda_correction_reg":  5e-2,
    "correction_reg_inertia_normalize": False,
    # Passivity loss: disabled by default (expensive; recommended for v2).
    "enable_passivity_loss":  False,
    "lambda_passivity":       1e-2,
    # Learning-rate multiplier for inertia/Coriolis param group.
    # 1.0 = full LR when unfrozen; Adam’s adaptive denominator provides
    # naturally conservative initial steps with cold momentum buffers.
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
