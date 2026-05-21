#!/usr/bin/env python3
"""A1 — Diagnostic baseline: train EDR + PhysReg side-by-side with per-epoch
test eval and NO early stopping, then report where val_rmse and test_rmse
minimize.

The cheapest test of the working hypothesis "EDR's val<test gap is
early-stop selection on a noisy val curve": if best-test epoch >> best-val
epoch (or the curves diverge), the gap is selection-driven and a smoother
selection (EMA, train-aware ES, last-epoch) closes it.

Cost target: ≤ 30 min per run on a local RTX 3050.

From repository root::

    PYTHONPATH=. python3 Neural_Networks/diagnostics/run_a1_baseline.py
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

# ── Path setup ─────────────────────────────────────────────────────────────
_DIAG_DIR  = Path(__file__).resolve().parent
_NN_ROOT   = _DIAG_DIR.parent
_REPO_ROOT = _NN_ROOT.parent
_EDR_DIR   = _NN_ROOT / "models" / "Equivariant-Decomposed-Residual"

for _p in (_REPO_ROOT, _EDR_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Imports ────────────────────────────────────────────────────────────────
import json

from Neural_Networks.models.shared.pipeline import TrainJob, run_training       # noqa: E402
from Neural_Networks.models.shared.strategies import PHYSICS_REG_STRATEGY        # noqa: E402
from edr_strategy import EDR_STRATEGY                                            # noqa: E402

# ── Dataset / output paths ─────────────────────────────────────────────────
TRAIN_DATA_RUN_DIR = str(
    _NN_ROOT / "train_data" / "run_abl_q0_qd91_qdd91_tau51_lk_20260515_1837"
)
RESULTS_DIR = _DIAG_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = str(RESULTS_DIR / "_diag_registry.yaml")

# ── Shared diagnostic knobs ────────────────────────────────────────────────
# Why this config: data_train_fraction=0.3 keeps signal but cuts per-epoch
# cost ~3×.  early_stopping=False lets the run go the full epoch budget so
# we can see whether test_rmse keeps decreasing past best-val.  patience is
# still set (large) so the code path matches production.
import os as _os
_SMOKE = _os.environ.get("DIAG_SMOKE", "").strip() == "1"
_LONG  = _os.environ.get("DIAG_LONG",  "").strip() == "1"

if _SMOKE:
    _EPOCHS, _FRAC = 3, 0.05
elif _LONG:
    # Long training on small data fraction — accelerates overfitting so the
    # val<test gap observed on HPC can be reproduced locally.
    _EPOCHS, _FRAC = 250, 0.15
else:
    _EPOCHS, _FRAC = 50, 0.25

COMMON_DIAGNOSTIC_HP: dict[str, Any] = {
    "epochs":              _EPOCHS,
    "batch_size":          512,
    "data_train_fraction": _FRAC,
    "data_train_seed":     0,
    "stride":              1,
    "seed":                42,
    "early_stopping":      False,
    "patience":            10_000,
    "min_delta":           1e-5,
    "print_every":         5,
    "diagnostic_test_eval": True,
    "torch_compile":       False,
    "snapshot_every":      0,
}


# ── EDR baseline HP (matches FIXED_HP_EDR from run_journal_grid_3model.py) ──
EDR_HP: dict[str, Any] = {
    **COMMON_DIAGNOSTIC_HP,
    "learning_rate":       3e-4,
    "weight_decay":        1e-5,
    "optimizer":           "adamw",
    "lr_scheduler":        "warmup_cosine",
    "warmup_cosine_min_factor": 0.05,
    "grad_clip_norm":      1.0,
    "feature_noise_std":   0.02,
    "ema_decay":           0.9,
    "early_stop_metric":   "val_rmse",
    # δ-net architecture (journal best width [48,48])
    "activation":          "silu",
    "gravity_hidden":      [48, 48],
    "inertia_hidden":      [48, 48],
    "coriolis_hidden":     [48, 48],
    "friction_hidden":     [24, 24],
    "correction_dropout":  0.30,
    # Adaptive curriculum / γ-gate (matches journal)
    "correction_gain_ramp_frac": 0.30,
    "phase2_start_epoch":     None,
    # Regularisers
    "lambda_correction_reg":  1e-2,
    "correction_reg_inertia_normalize": True,
    "enable_passivity_loss":  False,
    "lambda_passivity":       1e-2,
    # Structural priors — OFF (journal empirical winner)
    "coriolis_structural":  False,
    "inertia_psd":          False,
    "spectral_norm":        False,
    "friction_form":        "mlp",
    "use_friction_qdd":     True,
    "use_phys_cond":        True,
    "coriolis_matrix_form": False,
}

# ── PhysReg baseline HP (matches FIXED_HP_PHYSREG) ─────────────────────────
PHYSREG_HP: dict[str, Any] = {
    **COMMON_DIAGNOSTIC_HP,
    "learning_rate":       3e-4,
    "weight_decay":        5e-3,
    "dropout":             0.30,
    "activation":          "silu",
    "hidden_layers":       [256, 512, 256],
    "optimizer":           "adamw",
    "lr_scheduler":        "warmup_cosine",
    "early_stop_metric":   "val_rmse",
    "grad_clip_norm":      5.0,
    "feature_noise_std":   0.02,
    "physics_weight":      0.5,
    "physics_warmup_fraction": 0.05,
    "phi_lr_ratio":        0.1,
}


def _attach_q_norm_for_edr(hp: dict[str, Any]) -> None:
    """EDR uses sin/cos trig features; needs q normalization stats."""
    meta_path = Path(TRAIN_DATA_RUN_DIR) / "metadata.json"
    if not meta_path.is_file():
        return
    with open(meta_path) as f:
        meta = json.load(f)
    norm = meta.get("normalisation", {})
    if "mean_q" in norm and "std_q" in norm:
        hp["_q_mean"] = norm["mean_q"]
        hp["_q_std"]  = norm["std_q"]


def _run_one(name: str, strategy, hp: dict[str, Any]) -> dict[str, Any]:
    """Run one training job, return summary {best_val_epoch, best_test_epoch, etc.}."""
    save_subdir = name
    models_dir = str(RESULTS_DIR / name)
    job = TrainJob(
        run_dir=TRAIN_DATA_RUN_DIR,
        models_dir=models_dir,
        registry_file=REGISTRY_FILE,
        model_type=name,
        save_subdir=save_subdir,
        hp=hp,
        strategy=strategy,
        run_help=f"Neural_Networks/diagnostics/run_a1_baseline.py ({name})",
    )
    log = logging.getLogger(f"diag.{name}")
    test_rmse = run_training(job, log=log)
    if test_rmse is None:
        raise RuntimeError(f"{name} run failed — see logs")

    # Locate the most recently created subdir of models_dir — that's our run
    run_dir = max(Path(models_dir).iterdir(), key=lambda p: p.stat().st_mtime)
    hist_csv = run_dir / "training_history.csv"
    if not hist_csv.is_file():
        raise RuntimeError(f"{name}: training_history.csv missing in {run_dir}")

    rows = []
    with open(hist_csv, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)

    def _f(r, k):
        v = r.get(k, "")
        return float(v) if v not in ("", None) else float("nan")

    val_curve  = [_f(r, "val_rmse")  for r in rows]
    test_curve = [_f(r, "test_rmse") for r in rows]
    train_curve = [_f(r, "train_rmse") for r in rows]
    ema_curve  = [_f(r, "ema_val_rmse") for r in rows]

    def _argmin(xs):
        return min(range(len(xs)), key=lambda i: xs[i]) + 1  # 1-indexed epoch

    best_val_ep  = _argmin(val_curve)
    best_test_ep = _argmin(test_curve)
    best_train_ep = _argmin(train_curve)
    best_ema_ep   = _argmin(ema_curve) if any(e == e for e in ema_curve) else None

    return {
        "name":              name,
        "run_dir":           str(run_dir),
        "headline_test_rmse": float(test_rmse),
        "epochs":            len(rows),
        "best_val_epoch":    best_val_ep,
        "best_val_rmse":     val_curve[best_val_ep - 1],
        "test_rmse_at_best_val": test_curve[best_val_ep - 1],
        "best_test_epoch":   best_test_ep,
        "best_test_rmse":    test_curve[best_test_ep - 1],
        "last_test_rmse":    test_curve[-1],
        "last_val_rmse":     val_curve[-1],
        "best_train_epoch":  best_train_ep,
        "best_train_rmse":   train_curve[best_train_ep - 1],
        "best_ema_epoch":    best_ema_ep,
        "best_ema_val_rmse": ema_curve[best_ema_ep - 1] if best_ema_ep else None,
        "test_rmse_at_best_ema": test_curve[best_ema_ep - 1] if best_ema_ep else None,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("diag.a1")

    # EDR needs q-normalization stats for trig features
    edr_hp = dict(EDR_HP)
    _attach_q_norm_for_edr(edr_hp)

    tag = _os.environ.get("DIAG_TAG", "A1").strip() or "A1"

    log.info("=" * 78)
    log.info("%s baseline — EDR  (epochs=%d frac=%.2f)", tag, _EPOCHS, _FRAC)
    log.info("=" * 78)
    edr_summary = _run_one(f"EDR_{tag}", EDR_STRATEGY, edr_hp)

    log.info("=" * 78)
    log.info("%s baseline — PhysReg", tag)
    log.info("=" * 78)
    phys_summary = _run_one(f"PhysReg_{tag}", PHYSICS_REG_STRATEGY, PHYSREG_HP)

    # ── Combined report ────────────────────────────────────────────────
    log.info("=" * 78)
    log.info("A1 SUMMARY")
    log.info("=" * 78)

    summary_path = RESULTS_DIR / "a1_summary.json"
    blob = {"edr": edr_summary, "physreg": phys_summary}
    with open(summary_path, "w") as f:
        json.dump(blob, f, indent=2)

    for s in (edr_summary, phys_summary):
        log.info("[%s]", s["name"])
        log.info("  epochs ran:               %d", s["epochs"])
        log.info("  best_val:    ep=%3d  val_rmse=%.5f  (test@same=%.5f)",
                 s["best_val_epoch"], s["best_val_rmse"], s["test_rmse_at_best_val"])
        log.info("  best_test:   ep=%3d  test_rmse=%.5f",
                 s["best_test_epoch"], s["best_test_rmse"])
        log.info("  last epoch:  val_rmse=%.5f  test_rmse=%.5f",
                 s["last_val_rmse"], s["last_test_rmse"])
        if s["best_ema_epoch"] is not None:
            log.info("  best_ema:    ep=%3d  ema_val=%.5f  (test@same=%.5f)",
                     s["best_ema_epoch"], s["best_ema_val_rmse"],
                     s["test_rmse_at_best_ema"])

    # Selection-bias diagnostic
    log.info("-" * 78)
    log.info("SELECTION-BIAS DIAGNOSTIC")
    log.info("-" * 78)
    for s in (edr_summary, phys_summary):
        gap = s["test_rmse_at_best_val"] - s["best_test_rmse"]
        log.info("[%s]  test@best_val = %.5f   best_test = %.5f   selection cost = %.5f",
                 s["name"], s["test_rmse_at_best_val"], s["best_test_rmse"], gap)

    log.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
