#!/usr/bin/env python3
"""Scan grid-search trained models and produce a comprehensive performance report.

This is the grid-search counterpart of ``analyze_models.py``.  It scans
``Trained_Models_Grid/`` (instead of ``Trained_Models/``) and adds four
grid-specific figures on top of the nine base figures:

  Fig 10 — Top-K Leaderboard  (ranked table per architecture)
  Fig 11 — HP Importance      (mean test RMSE per HP value, per architecture)
  Fig 12 — HP Pair Heatmaps   (2-D RMSE heatmap for key HP pairs)
  Fig 13 — Pareto Front       (test RMSE vs estimated parameter count)

Base figures (identical logic to analyze_models.py):
  Fig 1  — Training Dynamics (best per type)
  Fig 2  — RMSE Comparison
  Fig 3  — R² and Pearson ρ Comparison
  Fig 4  — Per-Joint Heatmaps
  Fig 5  — Multi-Metric Parallel Coordinates
  Fig 6  — R² vs RMSE Scatter
  Fig 7  — MAE and NRMSE Comparison
  Fig 8  — EDR Physics Correction Magnitudes (skipped if no EDR models)
  Fig 9  — Per-Joint R² and RMSE Breakdown

Metrics are read from the correct held-out splits stored in metadata.yaml:
  * ``test_metrics``  — test split (15 %)          → primary evaluation
  * ``val_metrics``   — validation split (15 %)     → secondary evaluation
  * ``metrics``       — checkpoint eval (combined)  → NOT used as final metric
Train-RMSE is extracted from ``training_history.csv`` at the best-checkpoint
epoch (epoch where ``val_rmse`` in the CSV is minimised).

Usage (from repository root)::

    PYTHONPATH=. python Neural_Networks/analyze_models_grid.py
    PYTHONPATH=. python Neural_Networks/analyze_models_grid.py --models-dir path/to/Trained_Models_Grid
    PYTHONPATH=. python Neural_Networks/analyze_models_grid.py --no-plot
    PYTHONPATH=. python Neural_Networks/analyze_models_grid.py --top-k 5
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

# Interactive windows: let matplotlib pick the best available backend.
# On headless systems set MPLBACKEND=Agg in the environment, or pass --no-plot.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NN_ROOT = Path(__file__).resolve().parent
# *** Grid-search output directory — differs from analyze_models.py ***
DEFAULT_MODELS_DIR = str(_NN_ROOT / "Trained_Models_Grid")

JOINT_NAMES = ["J1 (yaw)", "J2 (shoulder)", "J3 (elbow)", "J4 (wrist)", "J5 (wrist roll)"]
JOINT_NAMES_SHORT = ["J1", "J2", "J3", "J4", "J5"]
N_JOINTS = 5

_TYPE_ABBREV: dict[str, str] = {
    "BlackBoxFNN": "FNN",
    "PhysicsRegularizedFNN": "PhysReg",
    "ResidualCorrectionFNN": "ResCorr",
    "EDR": "EDR",
}

# HP keys to render in grid-specific figures (ordered for display)
_GRID_HP_KEYS_FNN:      list[str] = ["hidden_layers", "dropout", "learning_rate", "weight_decay", "batch_size", "activation"]
_GRID_HP_KEYS_PHYSREG:  list[str] = ["hidden_layers", "dropout", "learning_rate", "batch_size", "physics_weight", "physics_warmup_fraction", "phi_lr_ratio"]
_GRID_HP_KEYS_RESIDUAL: list[str] = ["hidden_layers", "dropout", "learning_rate", "weight_decay", "batch_size", "alpha_reg_weight"]
_ARCH_HP_KEYS: dict[str, list[str]] = {
    "BlackBoxFNN":           _GRID_HP_KEYS_FNN,
    "PhysicsRegularizedFNN": _GRID_HP_KEYS_PHYSREG,
    "ResidualCorrectionFNN": _GRID_HP_KEYS_RESIDUAL,
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------

def _setup_plot_style() -> None:
    """Apply a clean, screen-friendly style to all subsequent figures."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "axes.facecolor": "#f7f7f7",
        "axes.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 10,
        "axes.labelpad": 7,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.40,
        "grid.color": "#c8c8c8",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.framealpha": 0.93,
        "legend.edgecolor": "#aaaaaa",
        "legend.fontsize": 9,
        "figure.constrained_layout.use": False,
    })


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_trained_models(models_dir: str) -> list[dict[str, Any]]:
    """Walk *models_dir* and collect every run that has a ``metadata.yaml``."""
    root = Path(models_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Trained-models directory not found: {models_dir}")

    records: list[dict[str, Any]] = []
    for meta_path in sorted(root.rglob("metadata.yaml")):
        try:
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
        except Exception as exc:
            logger.warning("Could not read %s: %s", meta_path, exc)
            continue
        if not isinstance(meta, dict):
            logger.warning("Unexpected format in %s, skipping.", meta_path)
            continue

        record: dict[str, Any] = dict(meta)
        record["_meta_path"] = str(meta_path)
        record["_run_dir"] = str(meta_path.parent)

        hist_path = meta_path.parent / "training_history.csv"
        record["_history_path"] = str(hist_path) if hist_path.is_file() else None
        records.append(record)

    if not records:
        logger.warning("No metadata.yaml files found under %s.", models_dir)
    return records


def group_by_model_type(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group records by ``model_type`` field (or parent folder name as fallback)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        mtype = rec.get("model_type") or Path(rec["_run_dir"]).parent.name
        groups[str(mtype)].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# Metric helpers  (read from correct val/test splits)
# ---------------------------------------------------------------------------

def _get_split(rec: dict[str, Any], split: str) -> dict[str, Any]:
    key_map = {
        "val":        "val_metrics",
        "test":       "test_metrics",
        "checkpoint": "metrics",
    }
    return rec.get(key_map.get(split, f"{split}_metrics"), {}) or {}


def _split_scalar(
    rec: dict[str, Any],
    split: str,
    *keys: str,
    default: float = float("nan"),
) -> float:
    d = _get_split(rec, split)
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def _split_joints(
    rec: dict[str, Any],
    split: str,
    key: str,
) -> list[float]:
    d = _get_split(rec, split)
    v = d.get(key)
    if isinstance(v, list) and len(v) == N_JOINTS:
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            pass
    return [float("nan")] * N_JOINTS


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _load_history(path: str | None) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    if not path or not os.path.isfile(path):
        return result
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    try:
                        result.setdefault(key, []).append(float(val))
                    except (ValueError, TypeError):
                        result.setdefault(key, []).append(float("nan"))
    except OSError as exc:
        logger.warning("Could not read history %s: %s", path, exc)
    return result


def _best_epoch_info(
    history: dict[str, list[float]],
) -> tuple[int, float, float]:
    val_rmse = history.get("val_rmse", [])
    if not val_rmse:
        return (-1, float("nan"), float("nan"))
    best_idx = int(np.nanargmin(val_rmse))
    epochs = history.get("epoch", [])
    epoch_num = int(epochs[best_idx]) if best_idx < len(epochs) else best_idx + 1
    tr_list = history.get("train_rmse", [])
    tr = tr_list[best_idx] if best_idx < len(tr_list) else float("nan")
    return (epoch_num, tr, float(val_rmse[best_idx]))


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _short_label(run_id: str) -> str:
    m = re.search(r"ep(\d+)_rmse([0-9.]+)", run_id)
    prefix = re.match(r"^([A-Za-z0-9]+)", run_id)
    if m and prefix:
        raw_type = prefix.group(1)
        abbrev = _TYPE_ABBREV.get(raw_type, raw_type[:8])
        rmse_str = m.group(2)[:7]
        return f"{abbrev} ep{m.group(1)} r{rmse_str}"
    return run_id[:30]


def _type_color_map(model_types: list[str]) -> dict[str, str]:
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {t: palette[i % len(palette)] for i, t in enumerate(sorted(model_types))}


# ---------------------------------------------------------------------------
# Enrich records
# ---------------------------------------------------------------------------

def enrich_records(records: list[dict[str, Any]]) -> None:
    for rec in records:
        hist = _load_history(rec.get("_history_path"))
        rec["_history"] = hist
        ep, tr, vr = _best_epoch_info(hist)
        rec["_best_epoch"] = ep
        rec["_train_rmse_hist"] = tr
        rec["_val_rmse_hist"] = vr


# ---------------------------------------------------------------------------
# Helpers used across plots
# ---------------------------------------------------------------------------

def _sorted_records(
    groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    all_recs = [r for recs in groups.values() for r in recs]
    all_recs.sort(key=lambda r: _split_scalar(r, "test", "rmse_pooled"))
    return all_recs


def _best_per_type(
    groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    bests: list[dict[str, Any]] = []
    for recs in groups.values():
        best = min(recs, key=lambda r: _split_scalar(r, "test", "rmse_pooled"))
        bests.append(best)
    bests.sort(key=lambda r: _split_scalar(r, "test", "rmse_pooled"))
    return bests


def _model_label(rec: dict[str, Any]) -> str:
    return rec.get("model_type", _short_label(rec.get("run_id", "?")))


def _model_sublabel(rec: dict[str, Any]) -> str:
    mtype = rec.get("model_type", "?")
    detail = _short_label(rec.get("run_id", "?"))
    return f"{mtype}\n({detail})"


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    logger.info("Saved: %s", path)
    plt.draw()


def _fmt(v: float, decimals: int = 5) -> str:
    return f"{v:.{decimals}f}" if v == v else "   —   "


def _annotate_bars(
    ax: plt.Axes,
    bars: Any,
    vals: list[float],
    rotation: int = 90,
    fontsize: float = 6.5,
) -> None:
    for bar, v in zip(bars, vals):
        if v == v:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                f"{v:.4f}",
                ha="center", va="bottom", fontsize=fontsize, rotation=rotation,
            )


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_summary_table(groups: dict[str, list[dict[str, Any]]]) -> None:
    rows: list[dict[str, Any]] = []
    for model_type, recs in groups.items():
        for rec in recs:
            training = rec.get("training", {})
            epochs = rec.get("epochs_trained", training.get("epochs_ran", "?"))
            stopped = rec.get("stopped_early", training.get("stopped_early", False))
            device = rec.get("device", rec.get("hardware", {}).get("device", "?"))
            rows.append({
                "model_type": model_type,
                "run_id": rec.get("run_id", "unknown"),
                "epochs": epochs,
                "early": "Y" if stopped else "N",
                "test_rmse":    _split_scalar(rec, "test", "rmse_pooled"),
                "val_rmse":     _split_scalar(rec, "val",  "rmse_pooled"),
                "test_r2":      _split_scalar(rec, "test", "r2_overall"),
                "val_r2":       _split_scalar(rec, "val",  "r2_overall"),
                "test_mae":     _split_scalar(rec, "test", "mae_mean"),
                "val_mae":      _split_scalar(rec, "val",  "mae_mean"),
                "test_pearson": _split_scalar(rec, "test", "pearson_r_mean"),
                "val_pearson":  _split_scalar(rec, "val",  "pearson_r_mean"),
                "train_rmse_hist": rec.get("_train_rmse_hist", float("nan")),
                "val_rmse_hist":   rec.get("_val_rmse_hist",   float("nan")),
                "best_epoch": rec.get("_best_epoch", -1),
                "device": device,
            })

    rows.sort(key=lambda r: r["test_rmse"] if r["test_rmse"] == r["test_rmse"] else 999.0)

    W = 148
    print("\n" + "=" * W)
    print("  GRID SEARCH — TRAINED MODELS — PERFORMANCE REPORT")
    print("  val/test RMSE, R², MAE, Pearson ρ read from proper held-out splits  |  RMSE & MAE in N·m")
    print("  † train_rmse_hist and val_rmse_hist are from training_history.csv at best checkpoint")
    print("=" * W)

    MT = 28
    hdr = (
        f"  {'#':<3} {'Model Type':<{MT}}  "
        f"{'Ep':>5}  {'ES':>3}  "
        f"{'Test RMSE↓':>11}  {'Val RMSE↓':>11}  "
        f"{'Test R²↑':>10}  {'Val R²↑':>10}  "
        f"{'Test MAE↓':>10}  {'Val MAE↓':>10}  "
        f"{'Test ρ↑':>9}  {'Val ρ↑':>9}  "
        f"{'Tr-RMSE†':>10}  {'V-RMSE†':>10}"
    )
    print(hdr)
    print("-" * W)
    for i, row in enumerate(rows, 1):
        print(
            f"  {i:<3} {row['model_type']:<{MT}}  "
            f"{str(row['epochs']):>5}  {row['early']:>3}  "
            f"{_fmt(row['test_rmse']):>11}  {_fmt(row['val_rmse']):>11}  "
            f"{_fmt(row['test_r2'], 4):>10}  {_fmt(row['val_r2'], 4):>10}  "
            f"{_fmt(row['test_mae']):>10}  {_fmt(row['val_mae']):>10}  "
            f"{_fmt(row['test_pearson'], 4):>9}  {_fmt(row['val_pearson'], 4):>9}  "
            f"{_fmt(row['train_rmse_hist']):>10}  {_fmt(row['val_rmse_hist']):>10}"
        )
    print("-" * W)
    print("  ES=Y: early stopped   †: training-history units (see header)\n")

    print("=== Best per model type (ranked by test RMSE, N·m) ===")
    for mtype in sorted(groups.keys()):
        best = min(groups[mtype], key=lambda r: _split_scalar(r, "test", "rmse_pooled"))
        bp  = _split_scalar(best, "test", "rmse_pooled")
        r2  = _split_scalar(best, "test", "r2_overall")
        mae = _split_scalar(best, "test", "mae_mean")
        print(
            f"  {mtype:<35}  test RMSE={bp:.5f} N·m  "
            f"test R²={r2:.4f}  test MAE={mae:.5f} N·m  →  {best.get('run_id', '?')}"
        )
    print()


# ---------------------------------------------------------------------------
# Fig 1 — Training Dynamics
# ---------------------------------------------------------------------------

def plot_training_dynamics(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    n = len(all_recs)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 5.5 * nrows),
        squeeze=False,
        num="Fig 1 — Training Dynamics (best per type)",
    )
    axes_flat = axes.flatten()
    type_colors = _type_color_map(list(groups.keys()))

    for idx, rec in enumerate(all_recs):
        ax = axes_flat[idx]
        hist = rec.get("_history", {})
        mtype = rec.get("model_type", "unknown")
        run_id = rec.get("run_id", "?")
        best_ep = rec.get("_best_epoch", -1)

        tl = hist.get("train_loss", [])
        vl = hist.get("val_loss", [])
        tr = hist.get("train_rmse", [])
        vr = hist.get("val_rmse", [])
        ep = hist.get("epoch", list(range(1, max(len(tl), len(vl), 1) + 1)))

        c = type_colors.get(mtype, "steelblue")

        if tl:
            ax.plot(ep[:len(tl)], tl, color=c, lw=1.8, label="train loss")
        if vl:
            ax.plot(ep[:len(vl)], vl, color=c, lw=1.8, ls="--", alpha=0.8, label="val loss")

        ax2 = ax.twinx()
        ax2.tick_params(axis="y", colors="darkorange", labelsize=6)
        if tr:
            ax2.plot(ep[:len(tr)], tr, color="darkorange", lw=1.1, alpha=0.55, label="train RMSE†")
        if vr:
            ax2.plot(ep[:len(vr)], vr, color="darkorange", lw=1.1, ls=":", label="val RMSE†")
        ax2.set_ylabel("RMSE†", fontsize=6, color="darkorange")

        if best_ep > 0:
            ax.axvline(best_ep, color="red", lw=1.0, ls=":", alpha=0.75, label=f"best ep={best_ep}")

        tr_nm = _split_scalar(rec, "test", "rmse_pooled")
        vl_nm = _split_scalar(rec, "val",  "rmse_pooled")
        r2_t  = _split_scalar(rec, "test", "r2_overall")
        subtitle = (
            f"test RMSE={tr_nm:.4f} N·m  val RMSE={vl_nm:.4f} N·m\ntest R²={r2_t:.4f}"
            if tr_nm == tr_nm else ""
        )
        ax.set_title(f"{mtype}\n{_short_label(run_id)}\n{subtitle}", fontsize=9, fontweight="bold")
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_ylabel("MSE loss", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

        lines1, lbls1 = ax.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=5.5, loc="upper right")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        "Fig 1 — Training Dynamics  (best model per type)\n"
        "solid/dashed = train/val loss · orange = RMSE† (hist units) · red dot = best checkpoint",
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, output_dir / "fig1_training_dynamics.png")


# ---------------------------------------------------------------------------
# Fig 2 — RMSE Comparison
# ---------------------------------------------------------------------------

def plot_rmse_comparison(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    type_colors = _type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "steelblue") for r in all_recs]
    labels = [_model_label(r) for r in all_recs]

    val_rmse  = [_split_scalar(r, "val",  "rmse_pooled") for r in all_recs]
    test_rmse = [_split_scalar(r, "test", "rmse_pooled") for r in all_recs]
    delta     = [t - v if (t == t and v == v) else float("nan")
                 for t, v in zip(test_rmse, val_rmse)]
    val_r2    = [_split_scalar(r, "val",  "r2_overall") for r in all_recs]
    test_r2   = [_split_scalar(r, "test", "r2_overall") for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.22

    fig, ax = plt.subplots(
        figsize=(max(11, n * 2.8), 7),
        num="Fig 2 — RMSE Comparison (best per type)",
    )
    fig.suptitle(
        "Fig 2 — Pooled RMSE Comparison  (best model per type)\n"
        "Val vs Test RMSE in N·m  ·  Δ = Test − Val  ·  right axis: R²",
        fontsize=12, fontweight="bold",
    )

    bv = ax.bar(x - bw, val_rmse,  bw, color=bar_colors, alpha=0.85,
                edgecolor="white", label="Val RMSE")
    bt = ax.bar(x,      test_rmse, bw, color=bar_colors, alpha=0.55,
                edgecolor="white", hatch="///", label="Test RMSE")
    bd = ax.bar(x + bw, delta,     bw, color="#b0b0b0", alpha=0.80,
                edgecolor="#888888", hatch="xxx", label="Δ (Test − Val)")

    _annotate_bars(ax, bv, val_rmse,  fontsize=8, rotation=75)
    _annotate_bars(ax, bt, test_rmse, fontsize=8, rotation=75)

    for bar, v in zip(bd, delta):
        if v == v:
            colour = "#cc2222" if v > 0 else "#228822"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0) + 0.0003,
                f"{v:+.4f}",
                ha="center", va="bottom", fontsize=7.5, rotation=75, color=colour, fontweight="bold",
            )

    ax_r2 = ax.twinx()
    ax_r2.plot(x - bw, val_r2,  color="purple", lw=2.0, ls="--",
               marker="o", markersize=7, alpha=0.85, label="Val R²")
    ax_r2.plot(x,      test_r2, color="purple", lw=2.0, ls=":",
               marker="^", markersize=7, alpha=0.85, label="Test R²")
    ax_r2.set_ylabel("R²  ↑ higher is better", fontsize=10, color="purple")
    ax_r2.tick_params(axis="y", colors="purple", labelsize=9)
    ax_r2.spines["right"].set_visible(True)
    ax_r2.spines["right"].set_color("purple")
    valid_r2 = [v for v in val_r2 + test_r2 if v == v]
    ax_r2.set_ylim(max(0.0, min(valid_r2) - 0.05) if valid_r2 else 0.0, 1.02)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10, fontweight="bold")
    ax.set_xlabel("Model Architecture", fontsize=11)
    ax.set_ylabel("Pooled RMSE (N·m)  ↓ lower is better", fontsize=11)
    ax.set_xlim(-0.6, n - 0.4)

    type_handles = [Patch(color=type_colors[t], label=t) for t in sorted(type_colors)]
    style_handles = [
        Patch(facecolor="gray", alpha=0.85, label="Solid = Val RMSE"),
        Patch(facecolor="gray", alpha=0.55, hatch="///", label="Hatched = Test RMSE"),
        Patch(facecolor="#b0b0b0", hatch="xxx", label="Δ = Test − Val"),
        Line2D([0], [0], color="purple", lw=2, ls="--", marker="o", label="Val R²"),
        Line2D([0], [0], color="purple", lw=2, ls=":",  marker="^", label="Test R²"),
    ]
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.legend(
        handles=type_handles + style_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=9,
    )

    _save_fig(fig, output_dir / "fig2_rmse_comparison.png")


# ---------------------------------------------------------------------------
# Fig 3 — R² and Pearson ρ Comparison
# ---------------------------------------------------------------------------

def plot_r2_comparison(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    type_colors = _type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "steelblue") for r in all_recs]
    labels = [_model_label(r) for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.22

    panels = [
        ("r2_overall",     "R² Overall (pooled)",   "R²"),
        ("r2_mean",        "R² Mean (per-joint)",    "R²"),
        ("pearson_r_mean", "Pearson ρ Mean",         "ρ"),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(max(16, n * 4.0), 8),
        num="Fig 3 — R² and Pearson ρ Comparison (best per type)",
    )
    fig.suptitle(
        "Fig 3 — R² and Pearson ρ  —  Best Model per Type  (↑ higher is better)\n"
        "Solid = Val  ·  Hatched = Test  ·  Δ = Test − Val  (green = improvement, red = degradation)",
        fontsize=12, fontweight="bold",
    )

    for ax, (metric_key, title, ylabel) in zip(axes, panels):
        vv = [_split_scalar(r, "val",  metric_key) for r in all_recs]
        tv = [_split_scalar(r, "test", metric_key) for r in all_recs]
        dv = [t - v if (t == t and v == v) else float("nan") for t, v in zip(tv, vv)]

        bv = ax.bar(x - bw, vv, bw, color=bar_colors, alpha=0.85, edgecolor="white")
        bt = ax.bar(x,      tv, bw, color=bar_colors, alpha=0.55, edgecolor="white", hatch="///")
        bd = ax.bar(x + bw, dv, bw, color="#b0b0b0",  alpha=0.80, edgecolor="#888888", hatch="xxx")

        _annotate_bars(ax, bv, vv, fontsize=7.5, rotation=75)
        _annotate_bars(ax, bt, tv, fontsize=7.5, rotation=75)

        for bar, v in zip(bd, dv):
            if v == v:
                colour = "#228822" if v >= 0 else "#cc2222"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    (bar.get_height() if v >= 0 else 0) + 0.001,
                    f"{v:+.4f}",
                    ha="center", va="bottom", fontsize=7, rotation=75, color=colour, fontweight="bold",
                )

        valid = [v for v in vv + tv if v == v]
        lo = max(0.0, min(valid) - 0.05) if valid else 0.7
        hi = min(1.03, max(valid) + 0.06) if valid else 1.03
        ax.set_ylim(lo, hi)
        ax.axhline(1.0, color="#44aa44", lw=0.9, alpha=0.5, ls="--", label="perfect = 1.0")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10, fontweight="bold")
        ax.set_xlabel("Model Architecture", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.6, n - 0.4)

    type_handles = [Patch(color=type_colors[t], label=t) for t in sorted(type_colors)]
    style_handles = [
        Patch(facecolor="gray", alpha=0.85, label="Solid = Val"),
        Patch(facecolor="gray", alpha=0.55, hatch="///", label="Hatched = Test"),
        Patch(facecolor="#b0b0b0", hatch="xxx", label="Δ = Test − Val"),
        Line2D([0], [0], color="#44aa44", lw=1.5, ls="--", label="Perfect = 1.0"),
    ]
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.legend(
        handles=type_handles + style_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=9,
    )

    _save_fig(fig, output_dir / "fig3_r2_comparison.png")


# ---------------------------------------------------------------------------
# Fig 4 — Per-Joint Heatmaps (2 × 2 grid)
# ---------------------------------------------------------------------------

def plot_per_joint_heatmaps(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    labels = [_model_label(r) for r in all_recs]

    def _mat(split: str, key: str) -> np.ndarray:
        return np.array([_split_joints(r, split, key) for r in all_recs])

    panels = [
        (0, 0, _mat("test", "rmse"),  "Test RMSE (N·m) ↓  darker=worse", "RdYlGn_r"),
        (0, 1, _mat("val",  "rmse"),  "Val  RMSE (N·m) ↓  darker=worse", "RdYlGn_r"),
        (1, 0, _mat("test", "r2"),    "Test R² ↑  darker=worse",          "RdYlGn"),
        (1, 1, _mat("val",  "r2"),    "Val  R² ↑  darker=worse",          "RdYlGn"),
    ]

    nrows_fig = max(5.0, len(all_recs) * 1.4 + 2.5)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(18, nrows_fig),
        num="Fig 4 — Per-Joint Heatmaps (best per type)",
    )
    fig.suptitle(
        "Fig 4 — Per-Joint Metrics Heatmap  (best model per type)\n"
        "rows = model types sorted best→worst · cols = joints",
        fontsize=13, fontweight="bold",
    )

    for ri, ci, mat, title, cmap in panels:
        ax = axes[ri, ci]
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES, fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(len(labels)):
            for j in range(N_JOINTS):
                v = mat[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save_fig(fig, output_dir / "fig4_per_joint_heatmaps.png")


# ---------------------------------------------------------------------------
# Fig 5 — Multi-Metric Parallel Coordinates
# ---------------------------------------------------------------------------

def plot_parallel_coordinates(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    axes_def = [
        ("rmse_pooled",    "test", "Test\nRMSE↓",    False),
        ("rmse_pooled",    "val",  "Val\nRMSE↓",     False),
        ("r2_overall",     "test", "Test\nR²↑",      True),
        ("r2_overall",     "val",  "Val\nR²↑",       True),
        ("mae_mean",       "test", "Test\nMAE↓",     False),
        ("nrmse_mean",     "test", "Test\nNRMSE↓",   False),
        ("pearson_r_mean", "test", "Test\nPearson↑", True),
    ]
    n_axes = len(axes_def)
    axis_labels = [a[2] for a in axes_def]

    model_rows = []
    for rec in all_recs:
        raw = {lbl: _split_scalar(rec, split, key) for key, split, lbl, _ in axes_def}
        model_rows.append({
            "rec": rec,
            "raw": raw,
            "norm": {},
            "model_type": rec.get("model_type", "unknown"),
            "label": _short_label(rec.get("run_id", "?")),
        })

    axis_ranges: dict[str, tuple[float, float]] = {}
    for key, split, lbl, higher_better in axes_def:
        vals = [d["raw"][lbl] for d in model_rows]
        valid = [v for v in vals if v == v]
        if not valid:
            for d in model_rows:
                d["norm"][lbl] = float("nan")
            axis_ranges[lbl] = (0.0, 1.0)
            continue
        mn, mx = min(valid), max(valid)
        span = mx - mn if mx != mn else 1.0
        axis_ranges[lbl] = (mn, mx)
        for d in model_rows:
            v = d["raw"][lbl]
            if v != v:
                d["norm"][lbl] = float("nan")
                continue
            score = (v - mn) / span
            d["norm"][lbl] = score if higher_better else (1.0 - score)

    type_colors = _type_color_map(list(groups.keys()))
    x_pos = list(range(n_axes))

    fig, ax = plt.subplots(
        figsize=(15, 8),
        num="Fig 5 — Parallel Coordinates (best per type)",
    )
    fig.suptitle(
        "Fig 5 — Multi-Metric Parallel Coordinates  (best model per type)\n"
        "y = 1 → best on that metric · y = 0 → worst · lines labelled by model type",
        fontsize=12, fontweight="bold",
    )

    lw = 2.0 if len(model_rows) <= 6 else 1.4
    alpha = 0.80 if len(model_rows) <= 10 else 0.60

    drawn_types: set[str] = set()
    for d in model_rows:
        mtype = d["model_type"]
        c = type_colors.get(mtype, "steelblue")
        y_vals = [d["norm"].get(lbl, float("nan")) for lbl in axis_labels]
        if any(v != v for v in y_vals):
            continue
        ax.plot(x_pos, y_vals, color=c, lw=lw, alpha=alpha)
        ax.scatter(x_pos, y_vals, color=c, s=60, zorder=4)
        ax.text(n_axes - 0.05, y_vals[-1], f"  {mtype}", fontsize=9,
                color=c, va="center", fontweight="bold")
        drawn_types.add(mtype)

    for xi in x_pos:
        ax.axvline(xi, color="gray", lw=0.5, alpha=0.4)

    for i, (_, _, lbl, _) in enumerate(axes_def):
        lo, hi = axis_ranges[lbl]
        ax.text(i,  1.05, f"{hi:.4f}", ha="center", va="bottom", fontsize=6.5, color="#666")
        ax.text(i, -0.07, f"{lo:.4f}", ha="center", va="top",    fontsize=6.5, color="#666")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(axis_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Performance Score  (1 = best,  0 = worst)", fontsize=11)
    ax.set_ylim(-0.12, 1.15)
    ax.grid(axis="y", alpha=0.18)

    legend_handles = [Patch(color=type_colors[t], label=t) for t in sorted(drawn_types)]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=len(drawn_types), fontsize=9)

    _save_fig(fig, output_dir / "fig5_parallel_coordinates.png")


# ---------------------------------------------------------------------------
# Fig 6 — R² vs RMSE Scatter
# ---------------------------------------------------------------------------

def plot_r2_vs_rmse_scatter(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    type_colors = _type_color_map(list(groups.keys()))
    model_types = sorted(groups.keys())

    fig, ax = plt.subplots(
        figsize=(11, 8),
        num="Fig 6 — R² vs RMSE Scatter",
    )
    fig.suptitle(
        "Fig 6 — R² vs RMSE  —  Best Model per Type\n"
        "Circles = Val  ·  Triangles = Test  ·  top-left corner = ideal",
        fontsize=12, fontweight="bold",
    )

    best_map = {r.get("model_type"): r for r in _best_per_type(groups)}
    for mtype in model_types:
        c = type_colors[mtype]
        for rec in ([best_map[mtype]] if mtype in best_map else []):
            lbl = _model_label(rec)
            vr  = _split_scalar(rec, "val",  "rmse_pooled")
            vr2 = _split_scalar(rec, "val",  "r2_overall")
            tr  = _split_scalar(rec, "test", "rmse_pooled")
            tr2 = _split_scalar(rec, "test", "r2_overall")

            if vr == vr and vr2 == vr2:
                ax.scatter(vr, vr2, color=c, s=200, marker="o", zorder=5, alpha=0.88)
                ax.annotate(lbl + "\n(val)", (vr, vr2),
                            textcoords="offset points", xytext=(8, 4),
                            fontsize=10, color=c, fontweight="bold")
            if tr == tr and tr2 == tr2:
                ax.scatter(tr, tr2, color=c, s=200, marker="^", zorder=5, alpha=0.88)
                ax.annotate(lbl + "\n(test)", (tr, tr2),
                            textcoords="offset points", xytext=(8, -14),
                            fontsize=10, color=c, fontweight="bold")

    best_recs_fig6 = _best_per_type(groups)
    all_test_rmse = [_split_scalar(r, "test", "rmse_pooled") for r in best_recs_fig6]
    all_test_r2   = [_split_scalar(r, "test", "r2_overall")  for r in best_recs_fig6]
    valid_r  = [v for v in all_test_rmse if v == v]
    valid_r2 = [v for v in all_test_r2  if v == v]
    if valid_r and valid_r2:
        ax.annotate(
            "← ideal",
            (min(valid_r), max(valid_r2)),
            textcoords="offset points", xytext=(-35, 6),
            fontsize=9, color="green",
            arrowprops=dict(arrowstyle="->", color="green"),
        )

    type_handles   = [Patch(color=type_colors[t], label=t) for t in model_types]
    marker_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=9, label="Validation  ○"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=9, label="Test  ▲"),
    ]
    ax.set_xlabel("Pooled RMSE (N·m)  ← lower is better", fontsize=11)
    ax.set_ylabel("R² overall  ↑ higher is better", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(
        handles=type_handles + marker_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=len(type_handles) + 2, fontsize=9,
    )

    _save_fig(fig, output_dir / "fig6_r2_vs_rmse_scatter.png")


# ---------------------------------------------------------------------------
# Fig 7 — MAE & NRMSE Comparison
# ---------------------------------------------------------------------------

def plot_mae_nrmse_comparison(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    type_colors = _type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "steelblue") for r in all_recs]
    labels = [_model_label(r) for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.22

    fig, (ax_mae, ax_nrmse) = plt.subplots(
        1, 2,
        figsize=(max(13, n * 3.2), 7),
        num="Fig 7 — MAE and NRMSE Comparison (best per type)",
    )
    fig.suptitle(
        "Fig 7 — MAE (N·m) and NRMSE  —  Best Model per Type  (↓ lower is better)\n"
        "Solid = Val  ·  Hatched = Test  ·  Δ = Test − Val  (red = test worse, green = test better)",
        fontsize=12, fontweight="bold",
    )

    for ax, key, ylabel, title in [
        (ax_mae,   "mae_mean",   "Mean Absolute Error (N·m)  ↓ lower is better", "MAE per Model Type"),
        (ax_nrmse, "nrmse_mean", "Normalised RMSE  ↓ lower is better",           "NRMSE per Model Type"),
    ]:
        vv = [_split_scalar(r, "val",  key) for r in all_recs]
        tv = [_split_scalar(r, "test", key) for r in all_recs]
        dv = [t - v if (t == t and v == v) else float("nan") for t, v in zip(tv, vv)]

        bv = ax.bar(x - bw, vv, bw, color=bar_colors, alpha=0.85, edgecolor="white")
        bt = ax.bar(x,      tv, bw, color=bar_colors, alpha=0.55, edgecolor="white", hatch="///")
        bd = ax.bar(x + bw, dv, bw, color="#b0b0b0",  alpha=0.80, edgecolor="#888888", hatch="xxx")

        _annotate_bars(ax, bv, vv, fontsize=8, rotation=75)
        _annotate_bars(ax, bt, tv, fontsize=8, rotation=75)

        for bar, v in zip(bd, dv):
            if v == v:
                colour = "#cc2222" if v > 0 else "#228822"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height(), 0) + 0.00005,
                    f"{v:+.4f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=75, color=colour, fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10, fontweight="bold")
        ax.set_xlabel("Model Architecture", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.6, n - 0.4)

    type_handles = [Patch(color=type_colors[t], label=t) for t in sorted(type_colors)]
    style_handles = [
        Patch(facecolor="gray", alpha=0.85, label="Solid = Val"),
        Patch(facecolor="gray", alpha=0.55, hatch="///", label="Hatched = Test"),
        Patch(facecolor="#b0b0b0", hatch="xxx", label="Δ = Test − Val"),
    ]
    fig.tight_layout(rect=[0, 0.17, 1, 1])
    fig.legend(
        handles=type_handles + style_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=9,
    )

    _save_fig(fig, output_dir / "fig7_mae_nrmse_comparison.png")


# ---------------------------------------------------------------------------
# Fig 8 — EDR Physics Correction Magnitudes (conditional)
# ---------------------------------------------------------------------------

def plot_edr_physics_corrections(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    edr_recs = groups.get("EDR", [])
    if not edr_recs:
        logger.info("No EDR models — skipping Fig 8.")
        return

    edr_with_history = [
        r for r in edr_recs
        if "mean_abs_delta_g" in r.get("_history", {})
    ]
    if not edr_with_history:
        logger.info("No EDR correction history found — skipping Fig 8.")
        return

    best_edr = min(edr_with_history, key=lambda r: _split_scalar(r, "test", "rmse_pooled"))
    edr_with_data = [best_edr]

    corr_cols = [
        ("mean_abs_delta_g",     "δg — gravity correction  (mean |δg|)"),
        ("mean_frob_delta_M",    "δM — inertia correction  (Frobenius norm)"),
        ("mean_abs_delta_C_qd",  "δC·q̇ — Coriolis correction  (mean |δC·q̇|)"),
        ("mean_abs_delta_tau_f", "δτ_f — friction correction  (mean |δτ_f|)"),
    ]

    cmap = matplotlib.colormaps["tab10"]
    model_colors = {
        r.get("run_id", str(i)): cmap(i % 10)
        for i, r in enumerate(edr_with_data)
    }

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 9),
        num="Fig 8 — EDR Physics Correction Magnitudes",
    )
    fig.suptitle(
        "Fig 8 — EDR Physics Correction Magnitudes Over Training  (best EDR model)\n"
        "★ = best checkpoint epoch  ·  larger value = greater learnable correction applied",
        fontsize=12, fontweight="bold",
    )

    for ax, (col, title) in zip(axes.flatten(), corr_cols):
        for rec in edr_with_data:
            hist = rec.get("_history", {})
            vals = hist.get(col, [])
            if not vals:
                continue
            ep = hist.get("epoch", list(range(1, len(vals) + 1)))
            best_ep = rec.get("_best_epoch", -1)
            run_id = rec.get("run_id", "?")
            c = model_colors[run_id]
            lbl = _short_label(run_id)
            ep_arr = ep[:len(vals)]
            ax.plot(ep_arr, vals, color=c, lw=1.6, alpha=0.8, label=lbl)
            if 0 < best_ep <= len(vals):
                bi = best_ep - 1
                ax.scatter([ep_arr[bi]], [vals[bi]], color=c, s=70, zorder=5, marker="*")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Correction Magnitude", fontsize=10)
        ax.grid(True, alpha=0.3)

    h, lbls = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.legend(h, lbls, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(h), fontsize=9)
    else:
        fig.tight_layout()

    _save_fig(fig, output_dir / "fig8_edr_physics_corrections.png")


# ---------------------------------------------------------------------------
# Fig 9 — Per-Joint R² and RMSE Breakdown  (best per type)
# ---------------------------------------------------------------------------

def plot_per_joint_r2_breakdown(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    type_colors = _type_color_map(list(groups.keys()))
    n_models = len(all_recs)
    model_labels = [_model_label(r) for r in all_recs]
    colors = [type_colors.get(r.get("model_type", "?"), "steelblue") for r in all_recs]

    test_r2_mat   = np.array([_split_joints(r, "test", "r2")   for r in all_recs])
    val_r2_mat    = np.array([_split_joints(r, "val",  "r2")   for r in all_recs])
    test_rmse_mat = np.array([_split_joints(r, "test", "rmse") for r in all_recs])
    val_rmse_mat  = np.array([_split_joints(r, "val",  "rmse") for r in all_recs])
    delta_r2_mat   = test_r2_mat   - val_r2_mat
    delta_rmse_mat = test_rmse_mat - val_rmse_mat

    x = np.arange(N_JOINTS)
    bw = 0.80 / (n_models * 2)
    grp_bw = bw * 2 + 0.03
    grp_offsets = np.linspace(
        -(n_models - 1) / 2.0, (n_models - 1) / 2.0, n_models
    ) * grp_bw

    fig, axes = plt.subplots(
        2, 2,
        figsize=(20, 11),
        num="Fig 9 — Per-Joint R² and RMSE Breakdown (best per type)",
    )
    fig.suptitle(
        "Fig 9 — Per-Joint R² and RMSE  —  Best Model per Type\n"
        "Top: Val vs Test side-by-side  ·  Bottom: Δ = Test − Val  (red = test worse, green = test better)",
        fontsize=12, fontweight="bold",
    )

    for ax, val_mat, test_mat, title, ylabel in [
        (axes[0, 0], val_r2_mat,   test_r2_mat,   "R² per Joint (↑ higher is better)", "R²"),
        (axes[0, 1], val_rmse_mat, test_rmse_mat, "RMSE per Joint  (↓ lower is better)", "RMSE (N·m)"),
    ]:
        for mi, (goff, c, lbl) in enumerate(zip(grp_offsets, colors, model_labels)):
            vv = val_mat[mi]
            tv = test_mat[mi]
            bv = ax.bar(x + goff - bw / 2, vv, bw, color=c, alpha=0.85, edgecolor="white",
                        label=f"{lbl} Val")
            bt = ax.bar(x + goff + bw / 2, tv, bw, color=c, alpha=0.50, edgecolor="white",
                        hatch="///", label=f"{lbl} Test")
            _annotate_bars(ax, bv, list(vv), fontsize=6.5, rotation=75)
            _annotate_bars(ax, bt, list(tv), fontsize=6.5, rotation=75)
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, fontsize=11, fontweight="bold")
        ax.set_xlabel("Joint", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

    delta_bw = 0.65 / n_models
    delta_offsets = np.linspace(
        -(n_models - 1) / 2.0, (n_models - 1) / 2.0, n_models
    ) * (delta_bw + 0.04)

    for ax, delta_mat, title, ylabel, higher_better in [
        (axes[1, 0], delta_r2_mat,   "Δ R² per Joint  (Test − Val)", "Δ R²  (Test − Val)", True),
        (axes[1, 1], delta_rmse_mat, "Δ RMSE per Joint  (Test − Val)", "Δ RMSE in N·m  (Test − Val)", False),
    ]:
        ax.axhline(0, color="#555555", lw=1.2, ls="--", zorder=2)
        for mi, (doff, c, lbl) in enumerate(zip(delta_offsets, colors, model_labels)):
            dv = delta_mat[mi]
            bar_colors_d = []
            for v in dv:
                if v != v:
                    bar_colors_d.append("#aaaaaa")
                elif (higher_better and v >= 0) or (not higher_better and v <= 0):
                    bar_colors_d.append("#228822")
                else:
                    bar_colors_d.append("#cc2222")
            bars = ax.bar(
                x + doff, dv, delta_bw * 0.90,
                color=bar_colors_d, alpha=0.80, edgecolor="white", linewidth=0.5,
                label=lbl,
            )
            for bar, v in zip(bars, dv):
                if v == v:
                    ypos = bar.get_height() + 0.00005 if v >= 0 else bar.get_height() - 0.00005
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        ypos,
                        f"{v:+.4f}",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=6.5, rotation=75,
                        color="#cc2222" if (
                            (higher_better and v < 0) or (not higher_better and v > 0)
                        ) else "#228822",
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, fontsize=11, fontweight="bold")
        ax.set_xlabel("Joint", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

    type_handles = [Patch(color=c, label=lbl) for c, lbl in zip(colors, model_labels)]
    style_handles = [
        Patch(facecolor="gray", alpha=0.85, label="Solid = Val"),
        Patch(facecolor="gray", alpha=0.50, hatch="///", label="Hatched = Test"),
        Patch(facecolor="#228822", alpha=0.80, label="Δ green = test better"),
        Patch(facecolor="#cc2222", alpha=0.80, label="Δ red = test worse"),
    ]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(
        handles=type_handles + style_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=max(3, n_models + 2), fontsize=9,
    )

    _save_fig(fig, output_dir / "fig9_per_joint_r2_breakdown.png")


# ===========================================================================
# ── GRID-SPECIFIC FIGURES (Fig 10–13) ────────────────────────────────────────
# ===========================================================================


def _estimate_params(hidden_layers: list[int], n_in: int = 15, n_out: int = 5) -> int:
    """Estimate MLP trainable-parameter count from layer sizes."""
    sizes = [n_in] + list(hidden_layers) + [n_out]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def _hp_val_str(v: Any) -> str:
    """Human-readable string for an HP value (handles lists like hidden_layers)."""
    if isinstance(v, list):
        return "×".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:.0e}" if abs(v) < 0.01 or abs(v) > 999 else f"{v:.4g}"
    return str(v)


# ---------------------------------------------------------------------------
# Fig 10 — Top-K Leaderboard per Architecture
# ---------------------------------------------------------------------------

def plot_topk_leaderboard(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    top_k: int = 10,
) -> None:
    """Ranked table of the top-K models per architecture sorted by test RMSE.

    Renders one sub-figure per architecture as a matplotlib table with key
    metrics and the most informative HP values highlighted.
    """
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN", "EDR"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order]
    if not arch_order:
        return

    n_archs = len(arch_order)
    fig_h = max(6.0, top_k * 0.55 + 2.5) * n_archs
    fig, axes = plt.subplots(
        n_archs, 1,
        figsize=(22, fig_h),
        num=f"Fig 10 — Top-{top_k} Leaderboard per Architecture",
    )
    if n_archs == 1:
        axes = [axes]
    fig.suptitle(
        f"Fig 10 — Top-{top_k} Leaderboard per Architecture  (ranked by Test RMSE ↑ lower is better)\n"
        "All metrics on held-out test split unless noted",
        fontsize=13, fontweight="bold",
    )

    for ax, mtype in zip(axes, arch_order):
        recs = sorted(groups[mtype], key=lambda r: _split_scalar(r, "test", "rmse_pooled"))
        recs = recs[:top_k]
        hp_keys = _ARCH_HP_KEYS.get(mtype, _GRID_HP_KEYS_FNN)

        col_headers = ["Rank", "Test RMSE↓", "Val RMSE↓", "Test R²↑", "Test MAE↓", "Epochs", "ES"] + hp_keys
        table_data = []
        for rank, rec in enumerate(recs, 1):
            hp = rec.get("hyperparams", {})
            row = [
                str(rank),
                f"{_split_scalar(rec, 'test', 'rmse_pooled'):.5f}",
                f"{_split_scalar(rec, 'val',  'rmse_pooled'):.5f}",
                f"{_split_scalar(rec, 'test', 'r2_overall'):.4f}",
                f"{_split_scalar(rec, 'test', 'mae_mean'):.5f}",
                str(rec.get("epochs_trained", "?")),
                "Y" if rec.get("stopped_early") else "N",
            ]
            for k in hp_keys:
                row.append(_hp_val_str(hp.get(k, "—")))
            table_data.append(row)

        ax.axis("off")
        tbl = ax.table(
            cellText=table_data,
            colLabels=col_headers,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.55)

        # Colour header row
        for col_idx in range(len(col_headers)):
            tbl[(0, col_idx)].set_facecolor("#2c4770")
            tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")

        # Gradient colour for Test RMSE column (col index 1): best=green, worst=red
        if len(recs) > 1:
            rmse_vals = [_split_scalar(r, "test", "rmse_pooled") for r in recs]
            lo, hi = min(rmse_vals), max(rmse_vals)
            span = hi - lo if hi != lo else 1.0
            cmap_g = matplotlib.colormaps["RdYlGn_r"]
            for row_idx, rv in enumerate(rmse_vals, 1):
                t = (rv - lo) / span
                tbl[(row_idx, 1)].set_facecolor(cmap_g(t)[:3] + (0.6,))

        # Alternating row backgrounds
        for row_idx in range(1, len(recs) + 1):
            bg = "#f0f4ff" if row_idx % 2 == 0 else "white"
            for col_idx in range(len(col_headers)):
                if col_idx != 1:  # skip RMSE col already coloured
                    tbl[(row_idx, col_idx)].set_facecolor(bg)

        ax.set_title(
            f"{mtype}  —  Top {min(top_k, len(recs))} of {len(groups[mtype])} runs",
            fontsize=11, fontweight="bold", pad=12,
        )

    fig.tight_layout(pad=1.2)
    _save_fig(fig, output_dir / "fig10_topk_leaderboard.png")


# ---------------------------------------------------------------------------
# Fig 11 — HP Importance (mean test RMSE per HP value, per architecture)
# ---------------------------------------------------------------------------

def plot_hp_importance(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """For each architecture, for each HP dimension that varies across the grid,
    compute the mean test RMSE grouped by HP value and display as grouped bars.

    This gives a quick read of which HP values (and which dimensions) have
    the largest effect on test performance.
    """
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order]
    if not arch_order:
        return

    n_archs = len(arch_order)
    fig_rows = n_archs
    fig_cols_max = 7  # max HP dims per arch to show

    fig, axes_grid = plt.subplots(
        fig_rows, fig_cols_max,
        figsize=(fig_cols_max * 3.5, fig_rows * 4.5),
        squeeze=False,
        num="Fig 11 — HP Importance (mean test RMSE per HP value)",
    )
    fig.suptitle(
        "Fig 11 — Hyperparameter Importance  (mean test RMSE per HP value, per architecture)\n"
        "Lower bar = better mean performance for that HP value  ·  error bars = ±1 std",
        fontsize=13, fontweight="bold",
    )

    for arch_idx, mtype in enumerate(arch_order):
        recs = groups[mtype]
        hp_keys = _ARCH_HP_KEYS.get(mtype, _GRID_HP_KEYS_FNN)

        # Collect (hp_val_str → [test_rmse, ...]) for each HP key
        hp_buckets: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in hp_keys}
        for rec in recs:
            tr = _split_scalar(rec, "test", "rmse_pooled")
            if tr != tr:
                continue
            hp = rec.get("hyperparams", {})
            for k in hp_keys:
                if k in hp:
                    hp_buckets[k][_hp_val_str(hp[k])].append(tr)

        shown = 0
        for col_idx, k in enumerate(hp_keys):
            ax = axes_grid[arch_idx, col_idx]
            bucket = hp_buckets[k]
            if len(bucket) <= 1:
                ax.axis("off")
                continue

            # Sort by numeric value when possible, else lexicographic
            def _sort_key(s: str) -> tuple:
                try:
                    return (0, float(s.replace("×", "0")))
                except ValueError:
                    return (1, s)

            sorted_vals = sorted(bucket.keys(), key=_sort_key)
            means = [float(np.mean(bucket[v])) for v in sorted_vals]
            stds  = [float(np.std(bucket[v])) if len(bucket[v]) > 1 else 0.0 for v in sorted_vals]

            x_pos = np.arange(len(sorted_vals))
            bars = ax.bar(x_pos, means, color="steelblue", alpha=0.80, width=0.6)
            ax.errorbar(x_pos, means, yerr=stds, fmt="none", color="black",
                        capsize=4, linewidth=1.2)

            # Highlight the best (lowest) bar in green
            best_idx_bar = int(np.argmin(means))
            bars[best_idx_bar].set_facecolor("#2e8b57")
            bars[best_idx_bar].set_alpha(0.95)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(sorted_vals, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{k}", fontsize=9, fontweight="bold")
            ax.set_ylabel("Mean Test RMSE (N·m)", fontsize=7)
            ax.set_xlabel("HP value", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
            shown += 1

        # Turn off unused columns
        for col_idx in range(len(hp_keys), fig_cols_max):
            axes_grid[arch_idx, col_idx].axis("off")

        # Architecture label on the left
        axes_grid[arch_idx, 0].set_ylabel(
            f"{mtype}\nMean Test RMSE (N·m)", fontsize=8, fontweight="bold", labelpad=8
        )

    fig.tight_layout(pad=1.5)
    _save_fig(fig, output_dir / "fig11_hp_importance.png")


# ---------------------------------------------------------------------------
# Fig 12 — HP Pair Heatmaps (key 2-D interactions)
# ---------------------------------------------------------------------------

def plot_hp_pair_heatmaps(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """2-D heatmap of mean test RMSE for the most influential HP pairs.

    For each architecture the pair (learning_rate × dropout) and
    (hidden_layers × batch_size) are shown.  Cells with no data are masked.
    """
    # Key pairs to show per architecture
    _PAIR_DEFS: dict[str, list[tuple[str, str]]] = {
        "BlackBoxFNN":           [("learning_rate", "dropout"), ("hidden_layers", "batch_size")],
        "PhysicsRegularizedFNN": [("learning_rate", "physics_weight"), ("hidden_layers", "dropout")],
        "ResidualCorrectionFNN": [("learning_rate", "alpha_reg_weight"), ("hidden_layers", "dropout")],
    }
    _PAIR_DEFS_DEFAULT = [("learning_rate", "dropout"), ("hidden_layers", "batch_size")]

    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order]
    if not arch_order:
        return

    n_pairs = 2
    n_archs = len(arch_order)
    fig, axes_grid = plt.subplots(
        n_archs, n_pairs,
        figsize=(n_pairs * 7, n_archs * 6.5),
        squeeze=False,
        num="Fig 12 — HP Pair Heatmaps (2-D RMSE interactions)",
    )
    fig.suptitle(
        "Fig 12 — HP Pair Heatmaps  (mean test RMSE for 2-D HP interactions)\n"
        "Darker cell = lower mean test RMSE = better  ·  'X' = no data for this combination",
        fontsize=13, fontweight="bold",
    )

    cmap_hm = matplotlib.colormaps["RdYlGn_r"]

    for arch_idx, mtype in enumerate(arch_order):
        recs = groups[mtype]
        pair_defs = _PAIR_DEFS.get(mtype, _PAIR_DEFS_DEFAULT)

        for pair_idx, (key_x, key_y) in enumerate(pair_defs[:n_pairs]):
            ax = axes_grid[arch_idx, pair_idx]

            # Collect all unique values and build a 2-D RMSE matrix
            vals_x: list[Any] = []
            vals_y: list[Any] = []
            for rec in recs:
                hp = rec.get("hyperparams", {})
                if key_x in hp and key_y in hp:
                    vx = _hp_val_str(hp[key_x])
                    vy = _hp_val_str(hp[key_y])
                    if vx not in vals_x:
                        vals_x.append(vx)
                    if vy not in vals_y:
                        vals_y.append(vy)

            def _sort_key_hp(s: str) -> tuple:
                try:
                    return (0, float(s.replace("×", "0")))
                except ValueError:
                    return (1, s)

            vals_x = sorted(vals_x, key=_sort_key_hp)
            vals_y = sorted(vals_y, key=_sort_key_hp)

            if not vals_x or not vals_y:
                ax.axis("off")
                ax.set_title(f"{key_x} × {key_y}\n(no data)", fontsize=9)
                continue

            # Accumulate test RMSE per cell
            cell_data: dict[tuple[str, str], list[float]] = defaultdict(list)
            for rec in recs:
                tr = _split_scalar(rec, "test", "rmse_pooled")
                if tr != tr:
                    continue
                hp = rec.get("hyperparams", {})
                if key_x in hp and key_y in hp:
                    cell_data[(_hp_val_str(hp[key_x]), _hp_val_str(hp[key_y]))].append(tr)

            mat = np.full((len(vals_y), len(vals_x)), np.nan)
            for (vx, vy), rmse_list in cell_data.items():
                if vx in vals_x and vy in vals_y:
                    xi = vals_x.index(vx)
                    yi = vals_y.index(vy)
                    mat[yi, xi] = float(np.mean(rmse_list))

            # Normalise for colour mapping
            valid_vals = mat[~np.isnan(mat)]
            if len(valid_vals) == 0:
                ax.axis("off")
                continue
            vmin, vmax = valid_vals.min(), valid_vals.max()

            masked = np.ma.masked_invalid(mat)
            im = ax.imshow(masked, cmap=cmap_hm, aspect="auto",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="Mean Test RMSE (N·m)")

            ax.set_xticks(range(len(vals_x)))
            ax.set_xticklabels(vals_x, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(vals_y)))
            ax.set_yticklabels(vals_y, fontsize=8)
            ax.set_xlabel(key_x, fontsize=9, fontweight="bold")
            ax.set_ylabel(key_y, fontsize=9, fontweight="bold")
            ax.set_title(
                f"{mtype}\n{key_x} × {key_y}",
                fontsize=10, fontweight="bold",
            )

            for yi in range(len(vals_y)):
                for xi in range(len(vals_x)):
                    v = mat[yi, xi]
                    if np.isnan(v):
                        ax.text(xi, yi, "X", ha="center", va="center",
                                fontsize=9, color="#aaaaaa")
                    else:
                        brightness = (v - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                        txt_color = "white" if brightness > 0.6 else "black"
                        ax.text(xi, yi, f"{v:.4f}", ha="center", va="center",
                                fontsize=7.5, color=txt_color, fontweight="bold")

    fig.tight_layout(pad=1.5)
    _save_fig(fig, output_dir / "fig12_hp_pair_heatmaps.png")


# ---------------------------------------------------------------------------
# Fig 13 — Pareto Front (test RMSE vs estimated parameter count)
# ---------------------------------------------------------------------------

def plot_pareto_front(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Scatter of test RMSE vs estimated MLP parameter count for every run.

    Points on the Pareto frontier (lowest RMSE for a given param budget)
    are connected with a line to highlight the efficiency frontier.
    """
    type_colors = _type_color_map(list(groups.keys()))
    all_recs = [r for recs in groups.values() for r in recs]

    fig, ax = plt.subplots(
        figsize=(14, 8),
        num="Fig 13 — Pareto Front (RMSE vs param count)",
    )
    fig.suptitle(
        "Fig 13 — Pareto Front  —  Test RMSE vs Estimated Parameter Count\n"
        "Each point = one trained run  ·  Pareto frontier per architecture highlighted",
        fontsize=12, fontweight="bold",
    )

    pareto_by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for rec in all_recs:
        tr = _split_scalar(rec, "test", "rmse_pooled")
        if tr != tr:
            continue
        hp = rec.get("hyperparams", {})
        hl = hp.get("hidden_layers", [256, 512, 256])
        n_params = _estimate_params(hl if isinstance(hl, list) else [256, 512, 256])
        mtype = rec.get("model_type", "unknown")
        c = type_colors.get(mtype, "gray")
        ax.scatter(n_params, tr, color=c, s=35, alpha=0.35, zorder=3)
        pareto_by_type[mtype].append((n_params, tr))

    # Draw Pareto frontier for each architecture
    for mtype, pts in pareto_by_type.items():
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0])
        # Keep only Pareto-optimal points (lower RMSE for same or fewer params)
        pareto: list[tuple[float, float]] = []
        best_rmse = float("inf")
        for p, r in pts_sorted:
            if r < best_rmse:
                pareto.append((p, r))
                best_rmse = r
        if len(pareto) >= 2:
            px, py = zip(*pareto)
            c = type_colors.get(mtype, "gray")
            ax.plot(px, py, color=c, lw=2.0, ls="-", marker="D",
                    markersize=8, zorder=5, alpha=0.9, label=f"{mtype} frontier")
            # Annotate the best point on the frontier
            best_pt = min(pareto, key=lambda p: p[1])
            ax.annotate(
                f"best\n{best_pt[1]:.4f} N·m",
                xy=best_pt, xytext=(10, -18),
                textcoords="offset points", fontsize=8,
                color=c, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=c, lw=1.0),
            )

    # Legend: one dot per type (all runs) + frontier line
    type_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=type_colors.get(t, "gray"),
               markersize=9, alpha=0.6, label=f"{t} (all runs)")
        for t in sorted(type_colors)
    ]
    frontier_handles = [
        Line2D([0], [0], color=type_colors.get(t, "gray"), lw=2.0, ls="-",
               marker="D", markersize=8, label=f"{t} Pareto frontier")
        for t in sorted(type_colors) if t in pareto_by_type
    ]

    ax.set_xlabel("Estimated MLP Parameter Count  (lower = smaller model)", fontsize=11)
    ax.set_ylabel("Test RMSE (N·m)  ↓ lower is better", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"
    ))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.legend(
        handles=type_handles + frontier_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=min(6, len(type_handles) + len(frontier_handles)), fontsize=9,
    )

    _save_fig(fig, output_dir / "fig13_pareto_front.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan grid-search trained models and open interactive performance report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help="Root directory containing trained model subdirectories.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip all plots; print summary table only.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top models to show in the leaderboard (Fig 10).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    models_dir = args.models_dir
    output_dir = Path(models_dir) / "analysis"

    logger.info("Scanning: %s", models_dir)
    try:
        records = scan_trained_models(models_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not records:
        logger.error("No trained models found.  Nothing to report.")
        return 1

    enrich_records(records)
    groups = group_by_model_type(records)
    logger.info(
        "Found %d model(s) in %d type(s): %s",
        len(records), len(groups), sorted(groups.keys()),
    )

    print_summary_table(groups)

    if args.no_plot:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving plots to: %s", output_dir)
    _setup_plot_style()

    # ── Base figures (identical to analyze_models.py) ────────────────────────
    plot_training_dynamics(groups, output_dir)
    plot_rmse_comparison(groups, output_dir)
    plot_r2_comparison(groups, output_dir)
    plot_per_joint_heatmaps(groups, output_dir)
    plot_parallel_coordinates(groups, output_dir)
    plot_r2_vs_rmse_scatter(groups, output_dir)
    plot_mae_nrmse_comparison(groups, output_dir)
    plot_per_joint_r2_breakdown(groups, output_dir)
    plot_edr_physics_corrections(groups, output_dir)

    # ── Grid-specific figures ─────────────────────────────────────────────────
    plot_topk_leaderboard(groups, output_dir, top_k=args.top_k)
    plot_hp_importance(groups, output_dir)
    plot_hp_pair_heatmaps(groups, output_dir)
    plot_pareto_front(groups, output_dir)

    n_base = 9 if groups.get("EDR") else 8
    n_figs  = n_base + 4
    print(f"\nPlots saved to: {output_dir}")
    print(f"{n_figs} figure window(s) open — close all windows to exit.")
    plt.show(block=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
