#!/usr/bin/env python3
"""Scan trained models, dynamically group by model_type, and produce a performance report.

Usage (from repository root)::

    PYTHONPATH=. python Neural_Networks/analyze_models.py
    PYTHONPATH=. python Neural_Networks/analyze_models.py --models-dir path/to/Trained_Models
    PYTHONPATH=. python Neural_Networks/analyze_models.py --output-dir /tmp/analysis --no-plot
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Default paths (relative to this file's package root)
# ---------------------------------------------------------------------------
_NN_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = str(_NN_ROOT / "Trained_Models")

JOINT_NAMES = ["J1 (yaw)", "J2 (shoulder)", "J3 (elbow)", "J4 (wrist)", "J5 (wrist roll)"]
N_JOINTS = 5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_trained_models(models_dir: str) -> list[dict[str, Any]]:
    """Walk *models_dir* and collect every run that has a ``metadata.yaml``.

    Returns a list of dicts, one per trained run, with all metadata merged.
    Raises ``FileNotFoundError`` if *models_dir* does not exist.
    """
    root = Path(models_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Trained-models directory not found: {models_dir}")

    records: list[dict[str, Any]] = []
    for meta_path in sorted(root.rglob("metadata.yaml")):
        try:
            with open(meta_path, "r") as f:
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

        # Attach training_history if present
        hist_path = meta_path.parent / "training_history.csv"
        record["_history_path"] = str(hist_path) if hist_path.is_file() else None

        records.append(record)

    if not records:
        logger.warning("No metadata.yaml files found under %s.", models_dir)
    return records


def group_by_model_type(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group records by ``model_type`` field (read from each metadata.yaml).

    Falls back to the parent folder name if the field is missing.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        model_type = rec.get("model_type") or Path(rec["_run_dir"]).parent.name
        groups[str(model_type)].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(obj: Any, *keys: str, default: float = float("nan")) -> float:
    """Drill into nested dicts using *keys* and return a float."""
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _load_history(history_path: str) -> tuple[list[float], list[float], list[float], list[float]]:
    """Load training_history.csv → (train_loss, val_loss, train_rmse, val_rmse)."""
    train_loss, val_loss, train_rmse, val_rmse = [], [], [], []
    try:
        with open(history_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    train_loss.append(float(row["train_loss"]))
                    val_loss.append(float(row["val_loss"]))
                    train_rmse.append(float(row.get("train_rmse", "nan")))
                    val_rmse.append(float(row.get("val_rmse", "nan")))
                except (KeyError, ValueError):
                    continue
    except OSError as exc:
        logger.warning("Could not read training history %s: %s", history_path, exc)
    return train_loss, val_loss, train_rmse, val_rmse


def _short_label(run_id: str, max_len: int = 40) -> str:
    """Shorten a run_id for axis labels."""
    if len(run_id) <= max_len:
        return run_id
    # Keep model type prefix and timestamp suffix
    parts = run_id.split("_")
    return parts[0] + "_..._" + "_".join(parts[-2:])


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _get_metric(rec: dict[str, Any], *keys: str) -> float:
    """Resolve a metric from metadata.yaml (flat keys) or registry (avg_-prefixed keys)."""
    m = rec.get("metrics", {})
    # Direct key (metadata.yaml format)
    for key in keys:
        v = m.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        # Registry format: prefixed with avg_
        v = m.get(f"avg_{key}")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return float("nan")


def print_summary_table(groups: dict[str, list[dict[str, Any]]]) -> None:
    """Print a ranked summary table to stdout."""
    rows: list[dict[str, Any]] = []
    for model_type, recs in groups.items():
        for rec in recs:
            # Support both metadata.yaml (flat) and registry (nested) schemas
            training = rec.get("training", {})
            epochs = rec.get("epochs_trained", training.get("epochs_ran", "?"))
            stopped = rec.get("stopped_early", training.get("stopped_early", False))
            device = rec.get("device", rec.get("hardware", {}).get("device", "?"))
            train_time = rec.get("training_time_formatted", training.get("time_formatted", "-"))
            rows.append(
                {
                    "model_type": model_type,
                    "run_id": rec.get("run_id", "unknown"),
                    "epochs": epochs,
                    "early": "Y" if stopped else "N",
                    "rmse_pooled": _get_metric(rec, "rmse_pooled"),
                    "rmse_mean": _get_metric(rec, "rmse_mean"),
                    "r2_overall": _get_metric(rec, "r2_overall"),
                    "val_rmse": _safe_float(rec, "val_metrics", "rmse_pooled"),
                    "test_rmse": _safe_float(rec, "test_metrics", "rmse_pooled"),
                    "train_time": train_time,
                    "device": device,
                }
            )

    # Sort best (lowest RMSE) first
    rows.sort(key=lambda r: r["rmse_pooled"] if not np.isnan(r["rmse_pooled"]) else 999)
    col_w = {
        "model_type": 30,
        "epochs": 6,
        "E?": 3,
        "rmse_pooled": 12,
        "rmse_mean": 11,
        "r2_overall": 10,
        "val_rmse": 10,
        "test_rmse": 10,
        "train_time": 10,
    }
    header = (
        f"{'Model Type':<{col_w['model_type']}}  "
        f"{'Ep':>{col_w['epochs']}}  "
        f"{'ES':{col_w['E?']}}  "
        f"{'RMSE Pool':>{col_w['rmse_pooled']}}  "
        f"{'RMSE Mean':>{col_w['rmse_mean']}}  "
        f"{'R² Overall':>{col_w['r2_overall']}}  "
        f"{'Val RMSE':>{col_w['val_rmse']}}  "
        f"{'Test RMSE':>{col_w['test_rmse']}}  "
        f"{'Train Time':>{col_w['train_time']}}"
    )
    sep = "-" * len(header)
    print("\n=== Trained Models — Performance Summary ===")
    print(header)
    print(sep)
    for i, row in enumerate(rows, 1):
        rp = f"{row['rmse_pooled']:.5f}" if not np.isnan(row["rmse_pooled"]) else "  —  "
        rm = f"{row['rmse_mean']:.5f}" if not np.isnan(row["rmse_mean"]) else "  —  "
        r2 = f"{row['r2_overall']:.4f}" if not np.isnan(row["r2_overall"]) else "  —  "
        vr = f"{row['val_rmse']:.5f}" if not np.isnan(row["val_rmse"]) else "  —  "
        tr = f"{row['test_rmse']:.5f}" if not np.isnan(row["test_rmse"]) else "  —  "
        print(
            f"#{i:<2} {row['model_type']:<{col_w['model_type']}}  "
            f"{str(row['epochs']):>{col_w['epochs']}}  "
            f"{row['early']:{col_w['E?']}}  "
            f"{rp:>{col_w['rmse_pooled']}}  "
            f"{rm:>{col_w['rmse_mean']}}  "
            f"{r2:>{col_w['r2_overall']}}  "
            f"{vr:>{col_w['val_rmse']}}  "
            f"{tr:>{col_w['test_rmse']}}  "
            f"{row['train_time']:>{col_w['train_time']}}"
        )
    print(sep)
    print(f"  ES = stopped early   |   RMSE in N·m\n")

    # Per-type bests
    print("=== Best per model type ===")
    for model_type, recs in sorted(groups.items()):
        best = min(
            recs,
            key=lambda r: _get_metric(r, "rmse_pooled"),
        )
        bp = _get_metric(best, "rmse_pooled")
        print(f"  {model_type:<35}  RMSE={bp:.5f} N·m  →  {best.get('run_id', '?')}")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _type_color_map(model_types: list[str]) -> dict[str, str]:
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {t: palette[i % len(palette)] for i, t in enumerate(sorted(model_types))}


def plot_training_histories(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """One subplot per model showing train/val loss curves."""
    all_recs = [rec for recs in groups.values() for rec in recs]
    n = len(all_recs)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    colors = _type_color_map(list(groups.keys()))

    for idx, rec in enumerate(all_recs):
        ax = axes_flat[idx]
        hist = rec.get("_history_path")
        model_type = rec.get("model_type", "unknown")
        run_id = rec.get("run_id", "unknown")

        if hist and os.path.isfile(hist):
            tl, vl, _, _ = _load_history(hist)
            epochs = list(range(1, len(tl) + 1))
            ax.plot(epochs, tl, color=colors.get(model_type, "steelblue"), label="train loss", linewidth=1.5)
            ax.plot(epochs, vl, color="darkorange", label="val loss", linewidth=1.5, linestyle="--")
        else:
            ax.text(0.5, 0.5, "No history", ha="center", va="center", transform=ax.transAxes)

        rmse_label = f"RMSE={_get_metric(rec, 'rmse_pooled'):.5f}"
        ax.set_title(f"{model_type}\n{_short_label(run_id)}\n{rmse_label}", fontsize=8)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("MSE loss", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.suptitle("Training History — All Models", fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = output_dir / "training_histories.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_rmse_comparison(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Bar chart — pooled RMSE for every model, grouped by model_type."""
    model_types = sorted(groups.keys())
    colors = _type_color_map(model_types)

    labels: list[str] = []
    values: list[float] = []
    bar_colors: list[str] = []
    x_ticks: list[float] = []
    group_centers: list[tuple[str, float]] = []

    x = 0.0
    gap = 1.5
    bar_width = 0.7

    for mtype in model_types:
        recs = sorted(
            groups[mtype],
            key=lambda r: _get_metric(r, "rmse_pooled"),
        )
        group_start = x
        for rec in recs:
            rmse = _get_metric(rec, "rmse_pooled")
            labels.append(_short_label(rec.get("run_id", "?"), max_len=30))
            values.append(rmse if not np.isnan(rmse) else 0.0)
            bar_colors.append(colors[mtype])
            x_ticks.append(x)
            x += bar_width + 0.2
        group_centers.append((mtype, (group_start + x - bar_width - 0.2) / 2))
        x += gap

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 6))
    bars = ax.bar(x_ticks, values, width=bar_width, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pooled RMSE (N·m)")
    ax.set_title("Pooled RMSE Comparison — All Trained Models")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=7,
        )

    # Legend by model type
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=colors[t], label=t) for t in model_types]
    ax.legend(handles=legend_handles, fontsize=9)

    plt.tight_layout()
    save_path = output_dir / "rmse_comparison.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_per_joint_rmse_heatmap(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Heatmap of per-joint RMSE — rows=models, cols=joints."""
    all_recs = [rec for recs in groups.values() for rec in recs]
    all_recs.sort(key=lambda r: _get_metric(r, "rmse_pooled"))

    run_labels: list[str] = []
    matrix: list[list[float]] = []

    for rec in all_recs:
        # metadata.yaml stores per-joint as 'rmse' list; registry as 'per_joint_rmse'
        m = rec.get("metrics", {})
        pj = m.get("rmse") or m.get("per_joint_rmse")
        if not pj or len(pj) != N_JOINTS:
            continue
        run_labels.append(_short_label(rec.get("run_id", "?"), max_len=35))
        matrix.append([float(v) for v in pj])

    if not matrix:
        logger.warning("No per-joint RMSE data found, skipping heatmap.")
        return

    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, max(3, len(run_labels) * 0.5 + 2)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(N_JOINTS))
    ax.set_xticklabels(JOINT_NAMES, fontsize=9)
    ax.set_yticks(range(len(run_labels)))
    ax.set_yticklabels(run_labels, fontsize=8)
    ax.set_title("Per-Joint RMSE (N·m) — All Models (sorted by pooled RMSE)")
    plt.colorbar(im, ax=ax, label="RMSE (N·m)")

    for i in range(len(run_labels)):
        for j in range(N_JOINTS):
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    save_path = output_dir / "per_joint_rmse_heatmap.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_r2_vs_rmse_scatter(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Scatter plot — R² overall vs pooled RMSE, coloured by model type."""
    model_types = sorted(groups.keys())
    colors = _type_color_map(model_types)

    fig, ax = plt.subplots(figsize=(9, 6))
    for mtype in model_types:
        for rec in groups[mtype]:
            rmse = _get_metric(rec, "rmse_pooled")
            r2 = _get_metric(rec, "r2_overall")
            if np.isnan(rmse) or np.isnan(r2):
                continue
            label = _short_label(rec.get("run_id", "?"), max_len=25)
            ax.scatter(rmse, r2, color=colors[mtype], s=100, zorder=3)
            ax.annotate(
                label,
                (rmse, r2),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=7,
                color=colors[mtype],
            )

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=colors[t], label=t) for t in model_types]
    ax.legend(handles=legend_handles, fontsize=9)
    ax.set_xlabel("Pooled RMSE (N·m) — lower is better →", fontsize=10)
    ax.set_ylabel("R² overall — higher is better ↑", fontsize=10)
    ax.set_title("R² vs RMSE — All Trained Models")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = output_dir / "r2_vs_rmse_scatter.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan trained models and generate performance report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help="Root directory containing trained model subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for saved plots. Defaults to <models-dir>/analysis/.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip all plot generation; print table only.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    models_dir = args.models_dir
    output_dir = Path(args.output_dir) if args.output_dir else Path(models_dir) / "analysis"

    logger.info("Scanning: %s", models_dir)
    try:
        records = scan_trained_models(models_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not records:
        logger.error("No trained models found. Nothing to report.")
        return 1

    groups = group_by_model_type(records)
    logger.info("Found %d model(s) in %d type(s): %s", len(records), len(groups), sorted(groups.keys()))

    print_summary_table(groups)

    if args.no_plot:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving plots to: %s", output_dir)

    plot_training_histories(groups, output_dir)
    plot_rmse_comparison(groups, output_dir)
    plot_per_joint_rmse_heatmap(groups, output_dir)
    plot_r2_vs_rmse_scatter(groups, output_dir)

    print(f"Plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
