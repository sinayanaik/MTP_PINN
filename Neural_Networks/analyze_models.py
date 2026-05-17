#!/usr/bin/env python3
"""Scan trained models, dynamically group by model_type, and produce an interactive
performance report with live plot windows.

Metrics are read from the **correct** held-out splits stored in metadata.yaml:
  * ``test_metrics``  — test split (15 %)          → primary evaluation
  * ``val_metrics``   — validation split (15 %)     → secondary evaluation
  * ``metrics``       — checkpoint eval (combined)  → NOT used as final metric
Train-RMSE is extracted from ``training_history.csv`` at the best-checkpoint epoch
(the epoch where ``val_rmse`` in the CSV is minimised).

Usage (from repository root)::

    PYTHONPATH=. python Neural_Networks/analyze_models.py
    PYTHONPATH=. python Neural_Networks/analyze_models.py --models-dir path/to/Trained_Models
    PYTHONPATH=. python Neural_Networks/analyze_models.py --no-plot
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
DEFAULT_MODELS_DIR = str(_NN_ROOT / "Trained_Models")

JOINT_NAMES = ["J1 (yaw)", "J2 (shoulder)", "J3 (elbow)", "J4 (wrist)", "J5 (wrist roll)"]
JOINT_NAMES_SHORT = ["J1", "J2", "J3", "J4", "J5"]
N_JOINTS = 5

_TYPE_ABBREV: dict[str, str] = {
    "BlackBoxFNN": "FNN",
    "PhysicsRegularizedFNN": "PhysReg",
    "ResidualCorrectionFNN": "ResCorr",
    "EDR": "EDR",
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
        "figure.constrained_layout.use": False,  # set per-figure below
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
    """Return metric dict for split: 'val'→val_metrics, 'test'→test_metrics."""
    key_map = {
        "val": "val_metrics",
        "test": "test_metrics",
        "checkpoint": "metrics",   # combined eval used for run-id naming only
    }
    return rec.get(key_map.get(split, f"{split}_metrics"), {}) or {}


def _split_scalar(
    rec: dict[str, Any],
    split: str,
    *keys: str,
    default: float = float("nan"),
) -> float:
    """Get a scalar metric from the named split's metric dict."""
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
    """Get per-joint array (len=N_JOINTS) from the named split's metric dict."""
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
    """Load training_history.csv → dict of column_name → list[float]."""
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
    """Return (epoch_1based, train_rmse, val_rmse) at the best checkpoint epoch.

    Best epoch = epoch where val_rmse in the training CSV is minimised —
    the same criterion used by early stopping to save the checkpoint.
    Both values are in training-history units (not directly comparable to N·m
    metrics computed post-training on the full held-out splits).
    """
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
    """Compact label: TypeAbbrev ep<N> r<RMSE>."""
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
# Enrich records (attach derived fields in-place)
# ---------------------------------------------------------------------------

def enrich_records(records: list[dict[str, Any]]) -> None:
    """Attach training history and best-epoch derived fields to each record."""
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
    """All records sorted by test_rmse_pooled ascending (best first)."""
    all_recs = [r for recs in groups.values() for r in recs]
    all_recs.sort(key=lambda r: _split_scalar(r, "test", "rmse_traj_macro", "rmse_pooled"))
    return all_recs


def _best_per_type(
    groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """One record per model type — the one with the lowest test RMSE (pooled).
    Returned list is sorted best → worst by test RMSE."""
    bests: list[dict[str, Any]] = []
    for recs in groups.values():
        best = min(recs, key=lambda r: _split_scalar(r, "test", "rmse_traj_macro", "rmse_pooled"))
        bests.append(best)
    bests.sort(key=lambda r: _split_scalar(r, "test", "rmse_traj_macro", "rmse_pooled"))
    return bests


def _model_label(rec: dict[str, Any]) -> str:
    """Short axis label used in best-per-type comparison plots."""
    return rec.get("model_type", _short_label(rec.get("run_id", "?")))


def _model_sublabel(rec: dict[str, Any]) -> str:
    """Two-line label: type name + run detail (for subplot titles)."""
    mtype = rec.get("model_type", "?")
    detail = _short_label(rec.get("run_id", "?"))
    return f"{mtype}\n({detail})"


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure to *path* and schedule it for interactive display."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    logger.info("Saved: %s", path)
    plt.draw()


def _fmt(v: float, decimals: int = 5) -> str:
    return f"{v:.{decimals}f}" if v == v else "   —   "  # NaN check via identity


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
    """Print ranked summary table using correct val/test split metrics."""
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
                # ── correct split metrics ──────────────────────────────────
                "test_rmse":    _split_scalar(rec, "test", "rmse_traj_macro", "rmse_pooled"),
                "val_rmse":     _split_scalar(rec, "val",  "rmse_traj_macro", "rmse_pooled"),
                "test_r2":      _split_scalar(rec, "test", "r2_overall"),
                "val_r2":       _split_scalar(rec, "val",  "r2_overall"),
                "test_mae":     _split_scalar(rec, "test", "mae_mean"),
                "val_mae":      _split_scalar(rec, "val",  "mae_mean"),
                "test_pearson": _split_scalar(rec, "test", "pearson_r_mean"),
                "val_pearson":  _split_scalar(rec, "val",  "pearson_r_mean"),
                # ── train RMSE from history at best checkpoint epoch ───────
                "train_rmse_hist": rec.get("_train_rmse_hist", float("nan")),
                "val_rmse_hist":   rec.get("_val_rmse_hist",   float("nan")),
                "best_epoch": rec.get("_best_epoch", -1),
                "device": device,
            })

    rows.sort(key=lambda r: r["test_rmse"] if r["test_rmse"] == r["test_rmse"] else 999.0)

    W = 148
    print("\n" + "=" * W)
    print("  TRAINED MODELS — PERFORMANCE REPORT")
    print("  val/test RMSE, R², MAE, Pearson ρ read from proper held-out splits  |  RMSE & MAE in N·m")
    print("  † train_rmse_hist and val_rmse_hist are from training_history.csv at the best checkpoint")
    print("    epoch (training units — NOT directly comparable to the post-training N·m metrics).")
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
        best = min(groups[mtype], key=lambda r: _split_scalar(r, "test", "rmse_traj_macro", "rmse_pooled"))
        bp  = _split_scalar(best, "test", "rmse_traj_macro", "rmse_pooled")
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
    """One subplot per model: loss curves (primary y) + RMSE curves (secondary y),
    with a vertical dashed line marking the best checkpoint epoch."""
    # Show only the best model per type so the layout stays clean
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

        tr_nm = _split_scalar(rec, "test", "rmse_traj_macro", "rmse_pooled")
        vl_nm = _split_scalar(rec, "val",  "rmse_traj_macro", "rmse_pooled")
        r2_t  = _split_scalar(rec, "test", "r2_overall")
        subtitle = (
            f"test RMSE={tr_nm:.4f} N·m  val RMSE={vl_nm:.4f} N·m\ntest R²={r2_t:.4f}" if tr_nm == tr_nm else ""
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
    """Grouped bars: Val / Test / Δ(Test−Val) RMSE per model type.
    A secondary right y-axis shows R² as a line chart.
    Legend is placed outside the axes so it never overlaps the bars."""
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    type_colors = _type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "steelblue") for r in all_recs]
    labels = [_model_label(r) for r in all_recs]

    val_rmse  = [_split_scalar(r, "val",  "rmse_traj_macro", "rmse_pooled") for r in all_recs]
    test_rmse = [_split_scalar(r, "test", "rmse_traj_macro", "rmse_pooled") for r in all_recs]
    delta     = [t - v if (t == t and v == v) else float("nan")
                 for t, v in zip(test_rmse, val_rmse)]
    val_r2    = [_split_scalar(r, "val",  "r2_overall") for r in all_recs]
    test_r2   = [_split_scalar(r, "test", "r2_overall") for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.22   # three bars per group, comfortable spacing

    fig, ax = plt.subplots(
        figsize=(max(11, n * 2.8), 7),
        num="Fig 2 — RMSE Comparison (best per type)",
    )
    fig.suptitle(
        "Fig 2 — Pooled RMSE Comparison  (best model per type)\n"
        "Val vs Test RMSE in N·m  ·  Δ = Test − Val  ·  right axis: R²",
        fontsize=12, fontweight="bold",
    )

    # Val bars (solid), Test bars (hatched), Δ bars (silver)
    bv = ax.bar(x - bw, val_rmse,  bw, color=bar_colors, alpha=0.85,
                edgecolor="white", label="Val RMSE")
    bt = ax.bar(x,      test_rmse, bw, color=bar_colors, alpha=0.55,
                edgecolor="white", hatch="///", label="Test RMSE")
    bd = ax.bar(x + bw, delta,     bw, color="#b0b0b0", alpha=0.80,
                edgecolor="#888888", hatch="xxx", label="Δ (Test − Val)")

    # Annotate bars with values
    _annotate_bars(ax, bv, val_rmse,  fontsize=8, rotation=75)
    _annotate_bars(ax, bt, test_rmse, fontsize=8, rotation=75)

    # Annotate Δ bars with signed value, coloured red/green
    for bar, v in zip(bd, delta):
        if v == v:
            colour = "#cc2222" if v > 0 else "#228822"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0) + 0.0003,
                f"{v:+.4f}",
                ha="center", va="bottom", fontsize=7.5, rotation=75, color=colour, fontweight="bold",
            )

    # R² overlay on secondary y-axis
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

    # Combined legend placed below the figure — outside all axes
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
    """Three grouped bar charts: R² overall, R² mean, Pearson ρ mean (val vs test).
    Adds a Δ(Test−Val) bar per model type; legend is outside the axes."""
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

        # Δ annotation: for R²/ρ, positive Δ = test BETTER = green
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
    """2×2 heatmap grid: test RMSE | val RMSE | test R² | val R²
    Rows = models sorted by test RMSE; columns = joints."""
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
    """Parallel coordinates: one line per model across 7 normalised metrics.
    Y-axis = performance score: 1 = best on that metric, 0 = worst."""
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    # (metric_key, split, display_label, higher_is_better)
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

    # Normalise each axis to [0,1] performance score (1=best)
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
        # Annotate end of line with model type
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
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(drawn_types), fontsize=9)

    _save_fig(fig, output_dir / "fig5_parallel_coordinates.png")


# ---------------------------------------------------------------------------
# Fig 6 — R² vs RMSE Scatter  (val ○  test ▲)
# ---------------------------------------------------------------------------

def plot_r2_vs_rmse_scatter(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Scatter of R² overall vs pooled RMSE — val (circles) and test (triangles)
    on the same axes, coloured by model type."""
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

            vr  = _split_scalar(rec, "val",  "rmse_traj_macro", "rmse_pooled")
            vr2 = _split_scalar(rec, "val",  "r2_overall")
            tr  = _split_scalar(rec, "test", "rmse_traj_macro", "rmse_pooled")
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
    all_test_rmse = [_split_scalar(r, "test", "rmse_traj_macro", "rmse_pooled") for r in best_recs_fig6]
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
        loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(type_handles) + 2, fontsize=9,
    )

    _save_fig(fig, output_dir / "fig6_r2_vs_rmse_scatter.png")


# ---------------------------------------------------------------------------
# Fig 7 — MAE & NRMSE Comparison
# ---------------------------------------------------------------------------

def plot_mae_nrmse_comparison(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Grouped bars: Val / Test / Δ(Test−Val) for MAE (N·m) and NRMSE.
    Legend placed outside axes to avoid overlap."""
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

        # Δ annotation: positive = test worse = red
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
    """Four-panel training curves of EDR physics-correction magnitudes.
    Rendered only when EDR models with correction history columns exist."""
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
    # Use only the best EDR model for a clean, readable figure
    best_edr = min(edr_with_history, key=lambda r: _split_scalar(r, "test", "rmse_traj_macro", "rmse_pooled"))
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
    """2×2 grid: [0,0] R² Val/Test  [0,1] RMSE Val/Test  [1,0] Δ R²  [1,1] Δ RMSE.
    Bottom row shows difference (Test − Val) per joint — red = test worse, green = test better."""
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
    delta_r2_mat   = test_r2_mat   - val_r2_mat     # negative = test worse
    delta_rmse_mat = test_rmse_mat - val_rmse_mat   # positive = test worse

    x = np.arange(N_JOINTS)
    bw = 0.80 / (n_models * 2)           # narrower — two groups (Val/Test) per model
    grp_bw = bw * 2 + 0.03               # spacing between model groups
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

    # ── Top row: grouped Val / Test bars per joint ──────────────────────────
    for ax, val_mat, test_mat, title, ylabel in [
        (axes[0, 0], val_r2_mat,   test_r2_mat,   "R² per Joint (↑ higher is better)", "R²"),
        (axes[0, 1], val_rmse_mat, test_rmse_mat, "RMSE per Joint  (↓ lower is better)", "RMSE (N·m)"),
    ]:
        for mi, (goff, c, lbl) in enumerate(zip(grp_offsets, colors, model_labels)):
            vv = val_mat[mi]
            tv = test_mat[mi]
            bv = ax.bar(x + goff - bw / 2, vv, bw, color=c, alpha=0.85, edgecolor="white", label=f"{lbl} Val")
            bt = ax.bar(x + goff + bw / 2, tv, bw, color=c, alpha=0.50, edgecolor="white", hatch="///", label=f"{lbl} Test")
            _annotate_bars(ax, bv, vv, fontsize=6.5, rotation=75)
            _annotate_bars(ax, bt, tv, fontsize=6.5, rotation=75)
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, fontsize=11, fontweight="bold")
        ax.set_xlabel("Joint", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

    # ── Bottom row: Δ = Test − Val per joint ────────────────────────────────
    delta_bw = 0.65 / n_models
    delta_offsets = np.linspace(-(n_models - 1) / 2.0, (n_models - 1) / 2.0, n_models) * (delta_bw + 0.04)

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
                    bar_colors_d.append("#228822")   # green = test better
                else:
                    bar_colors_d.append("#cc2222")   # red   = test worse
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
                        color="#cc2222" if v in bar_colors_d else "#228822",
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, fontsize=11, fontweight="bold")
        ax.set_xlabel("Joint", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

    # ── Shared legend ────────────────────────────────────────────────────────
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan trained models and open interactive performance report.",
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
        logger.error("No trained models found. Nothing to report.")
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

    plot_training_dynamics(groups, output_dir)
    plot_rmse_comparison(groups, output_dir)
    plot_r2_comparison(groups, output_dir)
    plot_per_joint_heatmaps(groups, output_dir)
    plot_parallel_coordinates(groups, output_dir)
    plot_r2_vs_rmse_scatter(groups, output_dir)
    plot_mae_nrmse_comparison(groups, output_dir)
    plot_per_joint_r2_breakdown(groups, output_dir)
    plot_edr_physics_corrections(groups, output_dir)

    n_figs = 9 if groups.get("EDR") else 8
    print(f"\nPlots saved to: {output_dir}")
    print(f"{n_figs} figure window(s) open — close all windows to exit.")
    plt.show(block=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
