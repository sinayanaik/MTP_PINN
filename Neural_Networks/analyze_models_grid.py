#!/usr/bin/env python3
"""Scan grid-search trained models and produce a comprehensive performance report."""

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
import matplotlib.ticker

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NN_ROOT = Path(__file__).resolve().parent
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

# Colorblind-safe Okabe-Ito palette
_OKABE_ITO_PALETTE = [
    "#E69F00",  # orange        -> BlackBoxFNN
    "#56B4E9",  # sky blue      -> PhysicsRegularizedFNN
    "#009E73",  # green         -> ResidualCorrectionFNN
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

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
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.framealpha": 0.93,
        "legend.edgecolor": "#aaaaaa",
        "legend.fontsize": 10,
        "figure.constrained_layout.use": False,
    })


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_trained_models(models_dir: str) -> list[dict[str, Any]]:
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
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        mtype = rec.get("model_type") or Path(rec["_run_dir"]).parent.name
        groups[str(mtype)].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _get_split(rec: dict[str, Any], split: str) -> dict[str, Any]:
    key_map = {
        "val":        "val_metrics",
        "test":       "test_metrics",
        "train":      "train_metrics",
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
# Train metrics computation
# ---------------------------------------------------------------------------

def _compute_train_metrics(rec: dict[str, Any]) -> dict[str, Any]:
    """Load model.pt, run inference on training split, compute full metrics.

    Results cached to <run_dir>/train_metrics_cache.yaml.
    Returns empty dict if model/data unavailable.
    """
    _CACHE_VERSION = 2

    run_dir = Path(rec.get("_run_dir", ""))
    cache_path = run_dir / "train_metrics_cache.yaml"

    if cache_path.is_file():
        try:
            with open(cache_path) as f:
                cached = yaml.safe_load(f)
            if isinstance(cached, dict) and cached.get("_v") == _CACHE_VERSION:
                return {k: v for k, v in cached.items() if k != "_v"}
        except Exception:
            pass

    model_pt = run_dir / "model.pt"
    if not model_pt.is_file():
        logger.debug("No model.pt in %s — skipping train metrics.", run_dir)
        return {}

    data_run_dir = rec.get("data_run_dir", "")
    if not data_run_dir or not Path(data_run_dir).is_dir():
        logger.debug("data_run_dir missing/invalid for %s — skipping.", run_dir)
        return {}

    try:
        import torch
        from Neural_Networks.loader import RobotDataset
        from Neural_Networks.models.torque_models import (
            BlackBoxFNN, PhysicsRegularizedFNN, ResidualCorrectionFNN,
        )
    except ImportError as exc:
        logger.debug("Cannot import torch/model modules: %s", exc)
        return {}

    try:
        mtype = rec.get("model_type", "BlackBoxFNN")
        hp = rec.get("hyperparams", {})
        hidden_layers = hp.get("hidden_layers", [256, 512, 256])
        dropout       = float(hp.get("dropout", 0.1))
        activation    = str(hp.get("activation", "gelu"))

        _cls_map = {
            "BlackBoxFNN":           BlackBoxFNN,
            "PhysicsRegularizedFNN": PhysicsRegularizedFNN,
            "ResidualCorrectionFNN": ResidualCorrectionFNN,
        }

        # Load checkpoint and extract model state with correct key
        ckpt = torch.load(str(model_pt), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            logger.debug("Checkpoint at %s is not a dict — skipping.", model_pt)
            return {}
        model_state = ckpt.get("model_state")
        if model_state is None:
            logger.debug(
                "Key 'model_state' not found in %s (found keys: %s) — skipping.",
                model_pt, list(ckpt.keys())[:8],
            )
            return {}

        # Prefer hparams/model class stored in checkpoint (always in sync with saved weights)
        ckpt_hp   = ckpt.get("hparams") or hp
        ckpt_hl   = ckpt_hp.get("hidden_layers", hidden_layers)
        ckpt_do   = float(ckpt_hp.get("dropout", dropout))
        ckpt_act  = str(ckpt_hp.get("activation", activation))
        cls_final = _cls_map.get(ckpt.get("model_class") or mtype)
        if cls_final is None:
            cls_final = _cls_map.get(mtype)
        if cls_final is None:
            return {}

        model = cls_final(n_joints=N_JOINTS, hidden_layers=ckpt_hl,
                          dropout=ckpt_do, activation=ckpt_act)
        model.load_state_dict(model_state)  # strict=True to catch any weight mismatch
        model.eval()

        dataset = RobotDataset(data_run_dir, split="train", mode="pointwise", normalise=True)
        # Prefer norm_stats from checkpoint (same source as training); fall back to dataset
        ckpt_norm = ckpt.get("norm_stats", {})
        if ckpt_norm and "mean_tau" in ckpt_norm:
            mean_tau = np.asarray(ckpt_norm["mean_tau"], dtype=np.float32)
            std_tau  = np.asarray(ckpt_norm["std_tau"],  dtype=np.float32).clip(min=1e-8)
        else:
            mean_tau = dataset.mean_tau
            std_tau  = dataset.std_tau

        all_preds: list[np.ndarray] = []
        all_tgts:  list[np.ndarray] = []
        batch_size = 2048
        n = len(dataset)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                feat_list, tgt_list, phy_list = [], [], []
                for i in range(start, end):
                    f, t, p = dataset[i]
                    feat_list.append(f); tgt_list.append(t); phy_list.append(p)
                feat = torch.stack(feat_list)
                tgt  = torch.stack(tgt_list)
                phy  = torch.stack(phy_list)
                pred_norm = model(feat, phy)
                pred_phys = pred_norm.numpy() * std_tau + mean_tau
                tgt_phys  = tgt.numpy()       * std_tau + mean_tau
                all_preds.append(pred_phys)
                all_tgts.append(tgt_phys)

        pred_np = np.concatenate(all_preds, axis=0)
        tgt_np  = np.concatenate(all_tgts,  axis=0)

        rmse_j, r2_j, mae_j, nrmse_j, pearson_j = [], [], [], [], []
        for j in range(N_JOINTS):
            p_j = pred_np[:, j]; t_j = tgt_np[:, j]
            res = t_j - p_j
            ss_res = float(np.sum(res ** 2))
            ss_tot = float(np.sum((t_j - t_j.mean()) ** 2))
            r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse = float(np.sqrt(np.mean(res ** 2)))
            mae  = float(np.mean(np.abs(res)))
            nrmse = rmse / float(t_j.std()) if t_j.std() > 0 else float("nan")
            corr  = float(np.corrcoef(p_j, t_j)[0, 1]) if len(p_j) > 1 else float("nan")
            rmse_j.append(rmse); r2_j.append(r2); mae_j.append(mae)
            nrmse_j.append(nrmse); pearson_j.append(corr)

        pred_flat = pred_np.reshape(-1); tgt_flat = tgt_np.reshape(-1)
        ss_res_pool = float(np.sum((tgt_flat - pred_flat) ** 2))
        ss_tot_pool = float(np.sum((tgt_flat - tgt_flat.mean()) ** 2))
        r2_pool   = 1.0 - ss_res_pool / ss_tot_pool if ss_tot_pool > 0 else float("nan")
        rmse_pool = float(np.sqrt(np.mean((pred_flat - tgt_flat) ** 2)))

        result: dict[str, Any] = {
            "rmse": rmse_j, "r2": r2_j, "mae": mae_j,
            "nrmse": nrmse_j, "pearson_r": pearson_j,
            "rmse_mean": float(np.mean(rmse_j)),
            "rmse_pooled": rmse_pool,
            "r2_overall": r2_pool,
            "r2_mean": float(np.nanmean(r2_j)),
            "mae_mean": float(np.mean(mae_j)),
            "nrmse_mean": float(np.nanmean(nrmse_j)),
            "pearson_r_mean": float(np.nanmean(pearson_j)),
        }

        try:
            with open(cache_path, "w") as f:
                yaml.dump({"_v": _CACHE_VERSION, **result}, f)
        except Exception:
            pass

        return result

    except Exception as exc:
        logger.debug("Train metrics computation failed for %s: %s", run_dir, exc)
        return {}


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


def _arch_short_label(mtype: str) -> str:
    return {
        "BlackBoxFNN":           "Black-Box",
        "PhysicsRegularizedFNN": "Physics-Reg",
        "ResidualCorrectionFNN": "Residual-Corr",
        "EDR":                   "EDR",
    }.get(mtype, mtype[:14])


def _type_color_map(model_types: list[str]) -> dict[str, str]:
    sorted_types = sorted(model_types)
    return {t: _OKABE_ITO_PALETTE[i % len(_OKABE_ITO_PALETTE)]
            for i, t in enumerate(sorted_types)}


def _panel_label(ax: "plt.Axes", letter: str, fontsize: float = 13.0) -> None:
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------------------
# Enrich records
# ---------------------------------------------------------------------------

def enrich_records(records: list[dict[str, Any]], compute_train: bool = True) -> None:
    for rec in records:
        hist = _load_history(rec.get("_history_path"))
        rec["_history"] = hist
        ep, tr, vr = _best_epoch_info(hist)
        rec["_best_epoch"] = ep
        rec["_train_rmse_hist"] = tr
        rec["_val_rmse_hist"] = vr
        if compute_train:
            tm = _compute_train_metrics(rec)
        else:
            tm = {}
        # Do NOT fall back to _train_rmse_hist: that value is in normalised units,
        # not physical N·m, so mixing it with test/val metrics causes 5-6× scale errors.
        rec["train_metrics"] = tm


# ---------------------------------------------------------------------------
# Helpers used across plots
# ---------------------------------------------------------------------------

def _sorted_records(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    all_recs = [r for recs in groups.values() for r in recs]
    all_recs.sort(key=lambda r: _split_scalar(r, "test", "rmse_pooled"))
    return all_recs


def _best_per_type(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
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


def _save_fig(fig: "plt.Figure", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    logger.info("Saved: %s  +  %s", pdf_path, png_path)
    plt.draw()


def _fmt(v: float, decimals: int = 5) -> str:
    return f"{v:.{decimals}f}" if v == v else "   -   "


def _annotate_bars(ax: "plt.Axes", bars: Any, vals: list[float],
                   rotation: int = 90, fontsize: float = 8.0) -> None:
    for bar, v in zip(bars, vals):
        if v == v:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0005,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=fontsize, rotation=rotation)


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
    print("  GRID SEARCH - TRAINED MODELS - PERFORMANCE REPORT")
    print("  val/test RMSE, R2, MAE, Pearson read from proper held-out splits  |  RMSE & MAE in N.m")
    print("  + train_rmse_hist and val_rmse_hist are from training_history.csv at best checkpoint")
    print("=" * W)

    MT = 28
    hdr = (
        f"  {'#':<3} {'Model Type':<{MT}}  "
        f"{'Ep':>5}  {'ES':>3}  "
        f"{'Test RMSE':>11}  {'Val RMSE':>11}  "
        f"{'Test R2':>10}  {'Val R2':>10}  "
        f"{'Test MAE':>10}  {'Val MAE':>10}  "
        f"{'Test P':>9}  {'Val P':>9}  "
        f"{'Tr-RMSE+':>10}  {'V-RMSE+':>10}"
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
    print("  ES=Y: early stopped   +: training-history units\n")

    print("=== Best per model type (ranked by test RMSE, N.m) ===")
    for mtype in sorted(groups.keys()):
        best = min(groups[mtype], key=lambda r: _split_scalar(r, "test", "rmse_pooled"))
        bp  = _split_scalar(best, "test", "rmse_pooled")
        r2  = _split_scalar(best, "test", "r2_overall")
        mae = _split_scalar(best, "test", "mae_mean")
        print(
            f"  {mtype:<35}  test RMSE={bp:.5f} N.m  "
            f"test R2={r2:.4f}  test MAE={mae:.5f} N.m  ->  {best.get('run_id', '?')}"
        )
    print()


# ---------------------------------------------------------------------------
# Fig 1 - Training Dynamics
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
        figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    type_colors = _type_color_map(list(groups.keys()))

    panel_letters = "abcdefgh"
    all_losses: list[float] = []

    for idx, rec in enumerate(all_recs):
        ax = axes_flat[idx]
        hist = rec.get("_history", {})
        mtype = rec.get("model_type", "unknown")
        best_ep = rec.get("_best_epoch", -1)

        tl = hist.get("train_loss", [])
        vl = hist.get("val_loss",   [])
        ep = hist.get("epoch", list(range(1, max(len(tl), len(vl), 1) + 1)))

        c = type_colors.get(mtype, "steelblue")

        if tl:
            ax.plot(ep[:len(tl)], tl, color=c, lw=2.0, label="Train Loss")
            all_losses.extend(tl)
        if vl:
            ax.plot(ep[:len(vl)], vl, color=c, lw=2.0, ls="--", alpha=0.8, label="Val Loss")
            all_losses.extend(vl)
        if best_ep > 0:
            ax.axvline(best_ep, color="#CC0000", lw=1.2, ls=":", alpha=0.85,
                       label=f"Best checkpoint (ep {best_ep})")

        letter = panel_letters[idx] if idx < len(panel_letters) else str(idx)
        ax.set_title(f"({letter}) {_arch_short_label(mtype)}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MSE Loss", fontsize=12)
        ax.grid(True, alpha=0.25)

    # Unified y-axis scale
    finite_losses = [v for v in all_losses if np.isfinite(v)]
    if finite_losses:
        lo = max(0.0, np.percentile(finite_losses, 1) * 0.9)
        hi = np.percentile(finite_losses, 99) * 1.05
        for idx in range(n):
            axes_flat[idx].set_ylim(lo, hi)

    # Single shared legend at bottom
    handles, labels = axes_flat[0].get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()),
                   loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=len(seen), fontsize=11, framealpha=0.95,
                   edgecolor="#aaaaaa", borderpad=0.8)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _save_fig(fig, output_dir / "fig1_training_dynamics.png")


# ---------------------------------------------------------------------------
# Fig 2 - RMSE Comparison (Train vs Test)
# ---------------------------------------------------------------------------

def plot_rmse_comparison(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    type_colors = _type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "#888888") for r in all_recs]
    labels = [_arch_short_label(r.get("model_type", "?")) for r in all_recs]

    train_rmse = [_split_scalar(r, "train", "rmse_pooled") for r in all_recs]
    test_rmse  = [_split_scalar(r, "test",  "rmse_pooled") for r in all_recs]
    test_r2    = [_split_scalar(r, "test",  "r2_overall")  for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.32

    fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(max(10, n * 2.8), 6))

    # Panel (a): Train vs Test RMSE
    b_train = ax_rmse.bar(x - bw / 2, train_rmse, bw, color=bar_colors,
                          alpha=0.90, edgecolor="white", linewidth=0.8)
    b_test  = ax_rmse.bar(x + bw / 2, test_rmse,  bw, color=bar_colors,
                          alpha=0.60, edgecolor="white", linewidth=0.8, hatch="////")

    for b, v in zip(b_train, train_rmse):
        if v == v and np.isfinite(v):
            ax_rmse.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                         f"{v:.4f}", ha="center", va="bottom", fontsize=9, rotation=75)
    for b, v in zip(b_test, test_rmse):
        if v == v and np.isfinite(v):
            ax_rmse.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                         f"{v:.4f}", ha="center", va="bottom", fontsize=9, rotation=75)

    valid_rmse = [v for v in train_rmse + test_rmse if v == v and np.isfinite(v)]
    if valid_rmse:
        ax_rmse.set_ylim(max(0.0, min(valid_rmse) * 0.97), max(valid_rmse) * 1.08)

    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(labels, rotation=0, ha="center", fontsize=11, fontweight="bold")
    ax_rmse.set_xlabel("Architecture", fontsize=12)
    ax_rmse.set_ylabel("Pooled RMSE (N.m)  lower is better", fontsize=12)
    ax_rmse.set_xlim(-0.6, n - 0.4)
    _panel_label(ax_rmse, "a")

    proxy_train = Patch(facecolor="#888888", alpha=0.90, label="Train RMSE")
    proxy_test  = Patch(facecolor="#888888", alpha=0.60, hatch="////", label="Test RMSE")
    ax_rmse.legend(handles=[proxy_train, proxy_test], fontsize=10, loc="upper right")

    # Panel (b): Test R2
    b_r2 = ax_r2.bar(x, test_r2, 0.50, color=bar_colors, alpha=0.82,
                     edgecolor="white", linewidth=0.8)
    for b, v in zip(b_r2, test_r2):
        if v == v and np.isfinite(v):
            ax_r2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                       f"{v:.4f}", ha="center", va="bottom", fontsize=9, rotation=75)

    valid_r2 = [v for v in test_r2 if v == v and np.isfinite(v)]
    if valid_r2:
        ax_r2.set_ylim(max(0.0, min(valid_r2) - 0.02), min(1.03, max(valid_r2) + 0.02))

    ax_r2.axhline(1.0, color="#888888", lw=1.0, ls="--", alpha=0.5)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(labels, rotation=0, ha="center", fontsize=11, fontweight="bold")
    ax_r2.set_xlabel("Architecture", fontsize=12)
    ax_r2.set_ylabel("Test R2  higher is better", fontsize=12)
    ax_r2.set_xlim(-0.6, n - 0.4)
    _panel_label(ax_r2, "b")

    arch_handles = [Patch(facecolor=type_colors[t], label=_arch_short_label(t))
                    for t in sorted(type_colors)]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=arch_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=len(arch_handles), fontsize=10)

    _save_fig(fig, output_dir / "fig2_rmse_comparison.png")


# ---------------------------------------------------------------------------
# Fig 3 - R2 and Pearson Comparison (Train vs Test)
# ---------------------------------------------------------------------------

def plot_r2_comparison(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    type_colors = _type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "#888888") for r in all_recs]
    labels = [_arch_short_label(r.get("model_type", "?")) for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.32

    panels = [
        ("r2_overall",     "(a) R2 Overall",    "R2"),
        ("r2_mean",        "(b) R2 Mean",        "R2"),
        ("pearson_r_mean", "(c) Pearson rho Mean", "rho"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(14, n * 4.0), 6))

    for ax, (metric_key, title, ylabel), letter in zip(axes, panels, "abc"):
        tv  = [_split_scalar(r, "test",  metric_key) for r in all_recs]
        trv = [_split_scalar(r, "train", metric_key) for r in all_recs]

        b_tr = ax.bar(x - bw / 2, trv, bw, color=bar_colors, alpha=0.90,
                      edgecolor="white", linewidth=0.8)
        b_te = ax.bar(x + bw / 2, tv,  bw, color=bar_colors, alpha=0.60,
                      edgecolor="white", linewidth=0.8, hatch="////")

        for b, v in zip(b_tr, trv):
            if v == v and np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, rotation=75)
        for b, v in zip(b_te, tv):
            if v == v and np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, rotation=75)

        valid = [v for v in trv + tv if v == v and np.isfinite(v)]
        lo = max(0.0, min(valid) - 0.03) if valid else 0.7
        hi = min(1.03, max(valid) + 0.03) if valid else 1.03
        ax.set_ylim(lo, hi)
        ax.axhline(1.0, color="#888888", lw=0.9, alpha=0.4, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Architecture", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(-0.6, n - 0.4)
        _panel_label(ax, letter)

    proxy_tr = Patch(facecolor="#888888", alpha=0.90, label="Train")
    proxy_te = Patch(facecolor="#888888", alpha=0.60, hatch="////", label="Test")
    arch_handles = [Patch(facecolor=type_colors[t], label=_arch_short_label(t))
                    for t in sorted(type_colors)]
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.legend(handles=arch_handles + [proxy_tr, proxy_te],
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(arch_handles) + 2, fontsize=10)

    _save_fig(fig, output_dir / "fig3_r2_comparison.png")


# ---------------------------------------------------------------------------
# Fig 4 - Per-Joint Heatmaps (2-panel, shared colorbar, plasma)
# ---------------------------------------------------------------------------

def plot_per_joint_heatmaps(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    labels = [_arch_short_label(r.get("model_type", "?")) for r in all_recs]

    test_rmse_mat = np.array([_split_joints(r, "test", "rmse") for r in all_recs])
    val_rmse_mat  = np.array([_split_joints(r, "val",  "rmse") for r in all_recs])

    all_rmse = np.concatenate([test_rmse_mat.flatten(), val_rmse_mat.flatten()])
    valid_rmse = all_rmse[np.isfinite(all_rmse)]
    vmin = float(valid_rmse.min()) if len(valid_rmse) else 0.0
    vmax = float(valid_rmse.max()) if len(valid_rmse) else 1.0

    nrows_fig = max(4.0, len(all_recs) * 1.4 + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(18, nrows_fig))

    panels = [
        (axes[0], test_rmse_mat, "(a) Test RMSE (N.m)"),
        (axes[1], val_rmse_mat,  "(b) Val RMSE (N.m)"),
    ]

    im_last = None
    for ax, mat, title in panels:
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="auto", cmap="plasma",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        im_last = im
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        for i in range(len(labels)):
            for j in range(N_JOINTS):
                v = mat[i, j]
                if v == v and np.isfinite(v):
                    norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    txt_c = "white" if norm_v > 0.6 else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=13, fontweight="bold", color=txt_c)

    if im_last is not None:
        cbar = fig.colorbar(im_last, ax=axes.tolist(), fraction=0.025, pad=0.03)
        cbar.set_label("RMSE (N.m) - lower is better", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    _save_fig(fig, output_dir / "fig4_per_joint_heatmaps.png")


# ---------------------------------------------------------------------------
# Fig 5 - Multi-Metric Parallel Coordinates (Train vs Test)
# ---------------------------------------------------------------------------

def plot_parallel_coordinates(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    all_recs_full = _sorted_records(groups)

    axes_def = [
        ("rmse_pooled",    "train",  "Train\nRMSE",    False),
        ("rmse_pooled",    "test",   "Test\nRMSE",     False),
        ("r2_overall",     "test",   "Test\nR2",       True),
        ("r2_mean",        "test",   "Test\nR2-mean",  True),
        ("mae_mean",       "test",   "Test\nMAE",      False),
        ("nrmse_mean",     "test",   "Test\nNRMSE",    False),
        ("pearson_r_mean", "test",   "Test\nPearson",  True),
    ]
    n_axes = len(axes_def)
    axis_labels = [a[2] for a in axes_def]

    # Normalize across all runs
    axis_ranges: dict[str, tuple[float, float]] = {}
    for key, split, lbl, _ in axes_def:
        vals = [_split_scalar(r, split, key) for r in all_recs_full]
        valid = [v for v in vals if v == v and np.isfinite(v)]
        axis_ranges[lbl] = (min(valid), max(valid)) if valid else (0.0, 1.0)

    model_rows = []
    for rec in all_recs:
        norm = {}
        raw  = {}
        for key, split, lbl, higher_better in axes_def:
            v = _split_scalar(rec, split, key)
            raw[lbl] = v
            mn, mx = axis_ranges[lbl]
            span = mx - mn if mx != mn else 1.0
            if v == v and np.isfinite(v):
                score = (v - mn) / span
                norm[lbl] = score if higher_better else (1.0 - score)
            else:
                norm[lbl] = float("nan")
        model_rows.append({"rec": rec, "raw": raw, "norm": norm,
                           "model_type": rec.get("model_type", "unknown")})

    type_colors = _type_color_map(list(groups.keys()))
    x_pos = list(range(n_axes))

    fig, ax = plt.subplots(figsize=(15, 8))

    drawn_types: set[str] = set()
    for d in model_rows:
        mtype = d["model_type"]
        c = type_colors.get(mtype, "steelblue")
        y_vals = [d["norm"].get(lbl, float("nan")) for lbl in axis_labels]
        if any(v != v or not np.isfinite(v) for v in y_vals):
            continue
        ax.plot(x_pos, y_vals, color=c, lw=2.5, alpha=0.85, marker="o", markersize=8)
        ax.text(n_axes - 0.05, y_vals[-1], f"  {_arch_short_label(mtype)}",
                fontsize=10, color=c, va="center", fontweight="bold")
        drawn_types.add(mtype)

    for xi in x_pos:
        ax.axvline(xi, color="gray", lw=0.6, alpha=0.4)

    for i, (_, _, lbl, _) in enumerate(axes_def):
        lo, hi = axis_ranges[lbl]
        ax.text(i,  1.07, f"{hi:.4f}", ha="center", va="bottom", fontsize=9, color="#555")
        ax.text(i, -0.09, f"{lo:.4f}", ha="center", va="top",    fontsize=9, color="#555")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(axis_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalised Score  (1=best across all runs, 0=worst)", fontsize=11)
    ax.set_ylim(-0.15, 1.20)
    ax.grid(axis="y", alpha=0.18)

    legend_handles = [Patch(color=type_colors[t], label=_arch_short_label(t))
                      for t in sorted(drawn_types)]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=len(drawn_types), fontsize=10)

    _save_fig(fig, output_dir / "fig5_parallel_coordinates.png")


# ---------------------------------------------------------------------------
# Fig 6 - R2 vs RMSE Scatter (Train->Test gap)
# ---------------------------------------------------------------------------

def plot_r2_vs_rmse_scatter(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    type_colors = _type_color_map(list(groups.keys()))
    model_types = sorted(groups.keys())

    fig, ax = plt.subplots(figsize=(11, 7))

    best_map = {r.get("model_type"): r for r in _best_per_type(groups)}
    for mtype in model_types:
        c = type_colors[mtype]
        rec = best_map.get(mtype)
        if rec is None:
            continue

        tr_rmse = _split_scalar(rec, "train", "rmse_pooled")
        if not (tr_rmse == tr_rmse and np.isfinite(tr_rmse)):
            tr_rmse = rec.get("_train_rmse_hist", float("nan"))
        tr_r2 = _split_scalar(rec, "train", "r2_overall")

        te_rmse = _split_scalar(rec, "test", "rmse_pooled")
        te_r2   = _split_scalar(rec, "test", "r2_overall")

        if tr_rmse == tr_rmse and np.isfinite(tr_rmse):
            y_tr = tr_r2 if (tr_r2 == tr_r2 and np.isfinite(tr_r2)) else te_r2
            ax.scatter(tr_rmse, y_tr, color=c, s=160, marker="o",
                       facecolors="none", edgecolors=c, linewidths=2.0, zorder=5)
            if te_rmse == te_rmse and np.isfinite(te_rmse) and te_r2 == te_r2 and np.isfinite(te_r2):
                ax.scatter(te_rmse, te_r2, color=c, s=160, marker="o", zorder=6)
                ax.annotate("", xy=(te_rmse, te_r2), xytext=(tr_rmse, y_tr),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.5), zorder=4)
        elif te_rmse == te_rmse and np.isfinite(te_rmse) and te_r2 == te_r2 and np.isfinite(te_r2):
            ax.scatter(te_rmse, te_r2, color=c, s=160, marker="o", zorder=6)

    best_recs = _best_per_type(groups)
    all_te_rmse = [_split_scalar(r, "test", "rmse_pooled") for r in best_recs]
    all_te_r2   = [_split_scalar(r, "test", "r2_overall")  for r in best_recs]
    vr  = [v for v in all_te_rmse if v == v and np.isfinite(v)]
    vr2 = [v for v in all_te_r2   if v == v and np.isfinite(v)]
    if vr and vr2:
        ax.annotate("Ideal",
                    (min(vr) * 0.99, max(vr2) * 1.002),
                    textcoords="offset points", xytext=(-50, 6),
                    fontsize=11, color="#228822", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#228822", lw=1.2))

    type_handles = [Patch(color=type_colors[t], label=_arch_short_label(t)) for t in model_types]
    marker_handles = [
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="none",
               markeredgewidth=2, markersize=10, linestyle="None", label="Train  (hollow)"),
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
               markersize=10, linestyle="None", label="Test  (filled)"),
    ]
    ax.set_xlabel("Pooled RMSE (N.m)  lower is better", fontsize=12)
    ax.set_ylabel("R2 overall  higher is better", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=type_handles + marker_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(type_handles) + 2, fontsize=10)

    _save_fig(fig, output_dir / "fig6_r2_vs_rmse_scatter.png")


# ---------------------------------------------------------------------------
# Fig 7 - MAE & NRMSE Comparison (Train vs Test)
# ---------------------------------------------------------------------------

def plot_mae_nrmse_comparison(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    all_recs = _best_per_type(groups)
    if not all_recs:
        return

    type_colors = _type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "#888888") for r in all_recs]
    labels = [_arch_short_label(r.get("model_type", "?")) for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.32

    fig, (ax_mae, ax_nrmse) = plt.subplots(1, 2, figsize=(max(11, n * 3.2), 6))

    for ax, key, ylabel, title, letter in [
        (ax_mae,   "mae_mean",   "Mean Absolute Error (N.m)  lower is better", "(a) MAE",   "a"),
        (ax_nrmse, "nrmse_mean", "Normalised RMSE  lower is better",            "(b) NRMSE", "b"),
    ]:
        tr_v = [_split_scalar(r, "train", key) for r in all_recs]
        te_v = [_split_scalar(r, "test",  key) for r in all_recs]

        b_tr = ax.bar(x - bw / 2, tr_v, bw, color=bar_colors, alpha=0.90,
                      edgecolor="white", linewidth=0.8)
        b_te = ax.bar(x + bw / 2, te_v, bw, color=bar_colors, alpha=0.60,
                      edgecolor="white", linewidth=0.8, hatch="////")

        for b, v in zip(b_tr, tr_v):
            if v == v and np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.00005,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9, rotation=75)
        for b, v in zip(b_te, te_v):
            if v == v and np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.00005,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9, rotation=75)

        valid = [v for v in tr_v + te_v if v == v and np.isfinite(v)]
        if valid:
            ax.set_ylim(max(0.0, min(valid) * 0.97), max(valid) * 1.10)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Architecture", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(-0.6, n - 0.4)
        _panel_label(ax, letter)

    proxy_tr = Patch(facecolor="#888888", alpha=0.90, label="Train")
    proxy_te = Patch(facecolor="#888888", alpha=0.60, hatch="////", label="Test")
    arch_handles = [Patch(facecolor=type_colors[t], label=_arch_short_label(t))
                    for t in sorted(type_colors)]
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.legend(handles=arch_handles + [proxy_tr, proxy_te],
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(arch_handles) + 2, fontsize=10)

    _save_fig(fig, output_dir / "fig7_mae_nrmse_comparison.png")


# ---------------------------------------------------------------------------
# Fig 8 - EDR Physics Correction Magnitudes (unchanged)
# ---------------------------------------------------------------------------

def plot_edr_physics_corrections(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    edr_recs = groups.get("EDR", [])
    if not edr_recs:
        logger.info("No EDR models - skipping Fig 8.")
        return

    edr_with_history = [r for r in edr_recs if "mean_abs_delta_g" in r.get("_history", {})]
    if not edr_with_history:
        logger.info("No EDR correction history found - skipping Fig 8.")
        return

    best_edr = min(edr_with_history, key=lambda r: _split_scalar(r, "test", "rmse_pooled"))
    edr_with_data = [best_edr]

    corr_cols = [
        ("mean_abs_delta_g",     "delta_g gravity correction"),
        ("mean_frob_delta_M",    "delta_M inertia correction (Frobenius)"),
        ("mean_abs_delta_C_qd",  "delta_C*qd Coriolis correction"),
        ("mean_abs_delta_tau_f", "delta_tau_f friction correction"),
    ]

    cmap_edr = matplotlib.colormaps["tab10"]
    model_colors = {r.get("run_id", str(i)): cmap_edr(i % 10)
                    for i, r in enumerate(edr_with_data)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

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

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Correction Magnitude", fontsize=11)
        ax.grid(True, alpha=0.3)

    h, lbls = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.legend(h, lbls, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                   ncol=len(h), fontsize=10)
    else:
        fig.tight_layout()

    _save_fig(fig, output_dir / "fig8_edr_physics_corrections.png")


# ---------------------------------------------------------------------------
# Fig 9 - Per-Joint R2 and RMSE Breakdown (Val vs Test, simplified)
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
    model_labels = [_arch_short_label(r.get("model_type", "?")) for r in all_recs]
    colors = [type_colors.get(r.get("model_type", "?"), "#888888") for r in all_recs]

    test_r2_mat   = np.array([_split_joints(r, "test",  "r2")   for r in all_recs])
    train_r2_mat  = np.array([_split_joints(r, "train", "r2")   for r in all_recs])
    test_rmse_mat = np.array([_split_joints(r, "test",  "rmse") for r in all_recs])
    train_rmse_mat = np.array([_split_joints(r, "train", "rmse") for r in all_recs])

    x = np.arange(N_JOINTS)
    bw = 0.70 / (n_models * 2)
    grp_bw = bw * 2 + 0.04
    grp_offsets = np.linspace(-(n_models - 1) / 2.0, (n_models - 1) / 2.0, n_models) * grp_bw

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, train_mat, test_mat, title, ylabel, higher_better in [
        (axes[0], train_r2_mat,   test_r2_mat,
         "(a) R2 per Joint", "R2", True),
        (axes[1], train_rmse_mat, test_rmse_mat,
         "(b) RMSE per Joint (N.m)", "RMSE (N.m)", False),
    ]:
        for mi, (goff, c, lbl) in enumerate(zip(grp_offsets, colors, model_labels)):
            ax.bar(x + goff - bw / 2, train_mat[mi], bw, color=c, alpha=0.88,
                   edgecolor="white", linewidth=0.5)
            ax.bar(x + goff + bw / 2, test_mat[mi],  bw, color=c, alpha=0.55,
                   edgecolor="white", linewidth=0.5, hatch="////")

        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES_SHORT, fontsize=12, fontweight="bold")
        ax.set_xlabel("Joint", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")

        all_vals = list(train_mat.flatten()) + list(test_mat.flatten())
        valid_vals = [v for v in all_vals if v == v and np.isfinite(v)]
        if valid_vals:
            lo = max(0.0, min(valid_vals) - 0.02) if higher_better else max(0.0, min(valid_vals) * 0.97)
            hi = min(1.02, max(valid_vals) + 0.02) if higher_better else max(valid_vals) * 1.08
            ax.set_ylim(lo, hi)

    arch_handles = [Patch(facecolor=c, label=lbl) for c, lbl in zip(colors, model_labels)]
    style_handles = [
        Patch(facecolor="#888888", alpha=0.88, label="Train"),
        Patch(facecolor="#888888", alpha=0.55, hatch="////", label="Test"),
    ]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=arch_handles + style_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=max(3, n_models + 2), fontsize=10)

    _save_fig(fig, output_dir / "fig9_per_joint_r2_breakdown.png")


# ===========================================================================
# Grid-specific figures (Fig 10-13)
# ===========================================================================

def _estimate_params(hidden_layers: list[int], n_in: int = 15, n_out: int = 5) -> int:
    sizes = [n_in] + list(hidden_layers) + [n_out]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def _hp_val_str(v: Any) -> str:
    if isinstance(v, list):
        return "x".join(str(x) for x in v)
    if isinstance(v, float):
        if abs(v) < 0.01 or abs(v) > 999:
            s = f"{v:.2e}"
            s = re.sub(r"0\.00e\+00", "0", s)
            return s
        return f"{v:.4g}"
    s = str(v)
    s = re.sub(r"^0e\+00$", "0", s)
    return s


# ---------------------------------------------------------------------------
# Fig 10 - Top-K Leaderboard
# ---------------------------------------------------------------------------

def plot_topk_leaderboard(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    top_k: int = 10,
) -> None:
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN", "EDR"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order]
    if not arch_order:
        return

    n_archs = len(arch_order)
    fig_h = max(5.0, top_k * 0.52 + 2.0) * n_archs
    fig, axes = plt.subplots(n_archs, 1, figsize=(22, fig_h))
    if n_archs == 1:
        axes = [axes]

    for ax, mtype in zip(axes, arch_order):
        recs = sorted(groups[mtype], key=lambda r: _split_scalar(r, "test", "rmse_pooled"))
        recs = recs[:top_k]
        hp_keys_all = _ARCH_HP_KEYS.get(mtype, _GRID_HP_KEYS_FNN)

        # Drop HP columns that are constant across all top-k runs
        varying_hp_keys: list[str] = []
        for k in hp_keys_all:
            vals = set(_hp_val_str(rec.get("hyperparams", {}).get(k, "-")) for rec in recs)
            if len(vals) > 1:
                varying_hp_keys.append(k)

        _HP_HEADER = {
            "hidden_layers": "Layers", "dropout": "Dropout",
            "learning_rate": "LR", "weight_decay": "WD",
            "batch_size": "BS", "activation": "Act",
            "physics_weight": "Phys-W", "physics_warmup_fraction": "Phys-WF",
            "phi_lr_ratio": "phi-LR", "alpha_reg_weight": "alpha-Reg",
        }

        col_headers = (
            ["Rank", "Test RMSE", "Val RMSE", "Test R2", "Test MAE", "Epochs", "ES"]
            + [_HP_HEADER.get(k, k) for k in varying_hp_keys]
        )
        table_data = []
        for rank, rec in enumerate(recs, 1):
            hp = rec.get("hyperparams", {})
            row = [
                str(rank),
                f"{_split_scalar(rec, 'test', 'rmse_pooled'):.4f}",
                f"{_split_scalar(rec, 'val',  'rmse_pooled'):.4f}",
                f"{_split_scalar(rec, 'test', 'r2_overall'):.4f}",
                f"{_split_scalar(rec, 'test', 'mae_mean'):.4f}",
                str(rec.get("epochs_trained", "?")),
                "Y" if rec.get("stopped_early") else "N",
            ]
            for k in varying_hp_keys:
                v = hp.get(k, "-")
                if k == "learning_rate" and isinstance(v, float):
                    row.append(f"{v:.1e}")
                else:
                    row.append(_hp_val_str(v))
            table_data.append(row)

        ax.axis("off")
        tbl = ax.table(cellText=table_data, colLabels=col_headers,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.60)

        for col_idx in range(len(col_headers)):
            cell = tbl[(0, col_idx)]
            cell.set_facecolor("#2c4770")
            cell.set_text_props(color="white", fontweight="bold")

        for row_idx in range(1, len(recs) + 1):
            bg = "#f4f7fb" if row_idx % 2 == 0 else "white"
            for col_idx in range(len(col_headers)):
                tbl[(row_idx, col_idx)].set_facecolor(bg)

        tbl[(1, 1)].set_text_props(fontweight="bold")

        ax.set_title(
            f"{_arch_short_label(mtype)}  -  Top {min(top_k, len(recs))} of"
            f" {len(groups[mtype])} runs  (ranked by Test RMSE)",
            fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout(pad=1.0)
    _save_fig(fig, output_dir / "fig10_topk_leaderboard.png")


# ---------------------------------------------------------------------------
# Fig 11 - HP Importance (dynamic grid, truncated y-axis)
# ---------------------------------------------------------------------------

def plot_hp_importance(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order]
    if not arch_order:
        return

    n_archs = len(arch_order)

    varying_by_arch: dict[str, list[str]] = {}
    for mtype in arch_order:
        recs = groups[mtype]
        hp_keys = _ARCH_HP_KEYS.get(mtype, _GRID_HP_KEYS_FNN)
        varying = []
        for k in hp_keys:
            vals = set()
            for rec in recs:
                hp = rec.get("hyperparams", {})
                if k in hp:
                    vals.add(_hp_val_str(hp[k]))
            if len(vals) > 1:
                varying.append(k)
        varying_by_arch[mtype] = varying

    max_cols = max((len(v) for v in varying_by_arch.values()), default=1)
    if max_cols == 0:
        logger.info("No varying HPs found - skipping Fig 11.")
        return

    # Pre-pass: compute global y-axis range across ALL arch × HP combinations
    # so every subplot shares the same scale for honest comparison.
    _all_means_flat: list[float] = []
    _all_stds_flat:  list[float] = []
    for _mtype_pre in arch_order:
        for _k_pre in varying_by_arch[_mtype_pre]:
            _bkt_pre: dict[str, list[float]] = defaultdict(list)
            for _rec_pre in groups[_mtype_pre]:
                _tr_pre = _split_scalar(_rec_pre, "test", "rmse_pooled")
                if not (_tr_pre == _tr_pre and np.isfinite(_tr_pre)):
                    continue
                _hp_pre = _rec_pre.get("hyperparams", {})
                if _k_pre in _hp_pre:
                    _bkt_pre[_hp_val_str(_hp_pre[_k_pre])].append(_tr_pre)
            for _vals_pre in _bkt_pre.values():
                if _vals_pre:
                    _m = float(np.mean(_vals_pre))
                    _s = float(np.std(_vals_pre)) if len(_vals_pre) > 1 else 0.0
                    _all_means_flat.append(_m)
                    _all_stds_flat.append(_s)
    if _all_means_flat:
        _g_lo_raw = min(m - s for m, s in zip(_all_means_flat, _all_stds_flat))
        _g_hi_raw = max(m + s for m, s in zip(_all_means_flat, _all_stds_flat))
        _margin   = (_g_hi_raw - _g_lo_raw) * 0.15
        _g_lo = max(0.0, _g_lo_raw - _margin)
        _g_hi = _g_hi_raw + _margin * 2.0   # extra room for n= count labels
    else:
        _g_lo, _g_hi = 0.0, 1.0

    fig, axes_grid = plt.subplots(n_archs, max_cols,
                                  figsize=(max_cols * 3.5, n_archs * 4.5), squeeze=False)

    for arch_idx, mtype in enumerate(arch_order):
        recs = groups[mtype]
        varying = varying_by_arch[mtype]

        hp_buckets: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in varying}
        for rec in recs:
            tr = _split_scalar(rec, "test", "rmse_pooled")
            if tr != tr or not np.isfinite(tr):
                continue
            hp = rec.get("hyperparams", {})
            for k in varying:
                if k in hp:
                    hp_buckets[k][_hp_val_str(hp[k])].append(tr)

        for col_idx in range(max_cols):
            ax = axes_grid[arch_idx, col_idx]
            if col_idx >= len(varying):
                ax.axis("off")
                continue

            k = varying[col_idx]
            bucket = hp_buckets[k]

            def _sort_key(s: str) -> tuple:
                try:
                    return (0, float(s.replace("x", "0")))
                except ValueError:
                    return (1, s)

            sorted_vals = sorted(bucket.keys(), key=_sort_key)
            means = [float(np.mean(bucket[v])) for v in sorted_vals]
            stds  = [float(np.std(bucket[v]))  if len(bucket[v]) > 1 else 0.0
                     for v in sorted_vals]
            n_runs = [len(bucket[v]) for v in sorted_vals]

            x_pos = np.arange(len(sorted_vals))
            bar_col = [_OKABE_ITO_PALETTE[arch_idx % len(_OKABE_ITO_PALETTE)]] * len(sorted_vals)
            ax.bar(x_pos, means, color=bar_col, alpha=0.80, width=0.6)
            ax.errorbar(x_pos, means, yerr=stds, fmt="none", color="black",
                        capsize=4, linewidth=1.2)

            # Annotate each bar with the run count
            _lbl_y_cap = _g_hi - (_g_hi - _g_lo) * 0.07
            for xi, (m_val, s_val, n_val) in enumerate(zip(means, stds, n_runs)):
                _lbl_y = min(m_val + s_val + (_g_hi - _g_lo) * 0.02, _lbl_y_cap)
                ax.text(xi, _lbl_y, f"n={n_val}",
                        ha="center", va="bottom", fontsize=8, color="#555555")

            ax.set_ylim(_g_lo, _g_hi)

            clean_labels = [re.sub(r"^0e\+00$", "0", v) for v in sorted_vals]
            rotate = max(len(v) for v in clean_labels) > 6
            ax.set_xticks(x_pos)
            ax.set_xticklabels(clean_labels,
                               rotation=30 if rotate else 0,
                               ha="right" if rotate else "center",
                               fontsize=10)
            ax.set_title(k, fontsize=11, fontweight="bold")
            ax.tick_params(labelsize=10)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))

            if col_idx == 0:
                ax.set_ylabel(f"{_arch_short_label(mtype)}\nMean Test RMSE (N.m)",
                              fontsize=10, fontweight="bold", labelpad=8)
            else:
                ax.set_ylabel("Mean Test RMSE (N.m)", fontsize=9)

    fig.tight_layout(pad=1.5)
    _save_fig(fig, output_dir / "fig11_hp_importance.png")


# ---------------------------------------------------------------------------
# Fig 12 - HP Pair Heatmaps (viridis_r, shared colorbar, skip degenerate)
# ---------------------------------------------------------------------------

def plot_hp_pair_heatmaps(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
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

    def _sort_hp(s: str) -> tuple:
        try:
            return (0, float(s.replace("x", "0")))
        except ValueError:
            return (1, s)

    # First pass: collect non-degenerate panels
    panel_info = []
    for arch_idx, mtype in enumerate(arch_order):
        recs = groups[mtype]
        pair_defs = _PAIR_DEFS.get(mtype, _PAIR_DEFS_DEFAULT)

        for pair_idx, (key_x, key_y) in enumerate(pair_defs[:n_pairs]):
            vals_x, vals_y = [], []
            for rec in recs:
                hp = rec.get("hyperparams", {})
                if key_x in hp and key_y in hp:
                    vx = _hp_val_str(hp[key_x]); vy = _hp_val_str(hp[key_y])
                    if vx not in vals_x: vals_x.append(vx)
                    if vy not in vals_y: vals_y.append(vy)

            vals_x = sorted(vals_x, key=_sort_hp)
            vals_y = sorted(vals_y, key=_sort_hp)

            if len(vals_x) <= 1 or len(vals_y) <= 1:
                continue

            cell_data: dict[tuple[str, str], list[float]] = defaultdict(list)
            for rec in recs:
                tr = _split_scalar(rec, "test", "rmse_pooled")
                if tr != tr or not np.isfinite(tr):
                    continue
                hp = rec.get("hyperparams", {})
                if key_x in hp and key_y in hp:
                    cell_data[(_hp_val_str(hp[key_x]), _hp_val_str(hp[key_y]))].append(tr)

            mat = np.full((len(vals_y), len(vals_x)), np.nan)
            for (vx, vy), rmse_list in cell_data.items():
                if vx in vals_x and vy in vals_y:
                    mat[vals_y.index(vy), vals_x.index(vx)] = float(np.mean(rmse_list))

            valid_v = mat[~np.isnan(mat)]
            if len(valid_v) < 2:
                continue

            panel_info.append((arch_idx, pair_idx, mtype, key_x, key_y, mat, vals_x, vals_y))

    if not panel_info:
        logger.info("No non-degenerate HP pairs found - skipping Fig 12.")
        return

    all_cell_vals = np.concatenate([info[5][~np.isnan(info[5])].flatten() for info in panel_info])
    global_vmin = float(all_cell_vals.min())
    global_vmax = float(all_cell_vals.max())

    n_panels = len(panel_info)
    n_cols = min(n_pairs, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes_2d = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6.0), squeeze=False)
    axes_flat = [axes_2d[r][c] for r in range(n_rows) for c in range(n_cols)]

    panel_letters = "abcdefghij"
    im_last = None

    for panel_num, (arch_idx, pair_idx, mtype, key_x, key_y,
                    mat, vals_x, vals_y) in enumerate(panel_info):
        ax = axes_flat[panel_num]
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap="viridis_r", aspect="auto",
                       vmin=global_vmin, vmax=global_vmax, interpolation="nearest")
        im_last = im

        ax.set_xticks(range(len(vals_x)))
        ax.set_xticklabels(vals_x, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(len(vals_y)))
        ax.set_yticklabels(vals_y, fontsize=10)
        ax.set_xlabel(key_x, fontsize=11, fontweight="bold")
        ax.set_ylabel(key_y, fontsize=11, fontweight="bold")
        letter = panel_letters[panel_num] if panel_num < len(panel_letters) else str(panel_num)
        ax.set_title(f"({letter}) {_arch_short_label(mtype)}: {key_x} x {key_y}",
                     fontsize=12, fontweight="bold")

        for yi in range(len(vals_y)):
            for xi in range(len(vals_x)):
                v = mat[yi, xi]
                if not np.isnan(v):
                    norm_v = (v - global_vmin) / (global_vmax - global_vmin) if global_vmax > global_vmin else 0.5
                    txt_c = "white" if norm_v < 0.5 else "black"
                    ax.text(xi, yi, f"{v:.4f}", ha="center", va="center",
                            fontsize=10, color=txt_c, fontweight="bold")
                else:
                    ax.text(xi, yi, "-", ha="center", va="center", fontsize=10, color="#aaaaaa")

    for panel_num in range(len(panel_info), len(axes_flat)):
        axes_flat[panel_num].axis("off")

    if im_last is not None:
        cbar = fig.colorbar(im_last, ax=axes_flat[:len(panel_info)],
                            fraction=0.025, pad=0.03)
        cbar.set_label("Mean Test RMSE (N.m)", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    _save_fig(fig, output_dir / "fig12_hp_pair_heatmaps.png")


# ---------------------------------------------------------------------------
# Fig 13 - Architecture RMSE Distribution (Box + Strip plot)
# ---------------------------------------------------------------------------

def plot_pareto_front(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Box + strip plot of test RMSE distribution per architecture."""
    type_colors = _type_color_map(list(groups.keys()))
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order and t != "EDR"]

    arch_data: dict[str, list[float]] = {}
    for mtype in arch_order:
        vals = [_split_scalar(r, "test", "rmse_pooled") for r in groups[mtype]]
        vals = [v for v in vals if v == v and np.isfinite(v)]
        if vals:
            arch_data[mtype] = vals

    if not arch_data:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(arch_data) * 3.0), 7))

    x_positions = np.arange(len(arch_data))
    rng = np.random.default_rng(42)

    for xi, (mtype, vals) in enumerate(arch_data.items()):
        c = type_colors.get(mtype, "#888888")
        arr = np.array(vals)

        ax.boxplot(arr, positions=[xi], widths=0.40, patch_artist=True,
                   showfliers=False,
                   medianprops=dict(color="black", linewidth=2.0),
                   boxprops=dict(facecolor=c, alpha=0.35, linewidth=1.2),
                   whiskerprops=dict(linewidth=1.2, color="#444444"),
                   capprops=dict(linewidth=1.5, color="#444444"))

        jitter = rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(xi + jitter, arr, color=c, s=45, alpha=0.75, zorder=4,
                   edgecolors="white", linewidths=0.5)

        best_val = float(arr.min())
        ax.annotate(f"Best: {best_val:.4f}",
                    xy=(xi, best_val), xytext=(xi + 0.25, best_val - 0.0008),
                    fontsize=9, color=c, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.8))

    all_vals = [v for vals in arch_data.values() for v in vals]
    if all_vals:
        ax.set_ylim(max(0.0, min(all_vals) * 0.994), max(all_vals) * 1.018)

    short_labels = [_arch_short_label(t) for t in arch_data]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Test RMSE (N.m)  lower is better", fontsize=12)
    ax.set_xlabel("Architecture", fontsize=12)
    ax.grid(True, axis="y", alpha=0.35)

    arch_handles = [Patch(facecolor=type_colors.get(t, "#888888"),
                          label=_arch_short_label(t), alpha=0.80)
                    for t in arch_data]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=arch_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=len(arch_handles), fontsize=10)

    _save_fig(fig, output_dir / "fig13_rmse_distribution.png")


# ---------------------------------------------------------------------------
# Fig 14 - Data Efficiency (data_train_fraction vs RMSE and R2)
# ---------------------------------------------------------------------------

def plot_data_efficiency(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Fig 14: Test RMSE and R2 vs training-data fraction, one line per architecture."""
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order]
    if not arch_order:
        return

    type_colors = _type_color_map(list(groups.keys()))

    frac_rmse: dict[str, dict[float, list[float]]] = {a: defaultdict(list) for a in arch_order}
    frac_r2:   dict[str, dict[float, list[float]]] = {a: defaultdict(list) for a in arch_order}

    for mtype in arch_order:
        for rec in groups[mtype]:
            frac = rec.get("hyperparams", {}).get("data_train_fraction")
            if frac is None:
                continue
            frac = float(frac)
            rmse = _split_scalar(rec, "test", "rmse_pooled")
            r2   = _split_scalar(rec, "test", "r2_overall")
            if rmse == rmse and np.isfinite(rmse):
                frac_rmse[mtype][frac].append(rmse)
            if r2 == r2 and np.isfinite(r2):
                frac_r2[mtype][frac].append(r2)

    all_fracs = sorted({f for a in arch_order for f in frac_rmse[a]})
    if not all_fracs:
        logger.info("No data_train_fraction data found - skipping Fig 14.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rng = np.random.default_rng(99)

    for ax, metric_data, ylabel, title in [
        (axes[0], frac_rmse, "Test RMSE (N\u00b7m)  lower is better",
         "(a) Test RMSE vs Training Data Fraction"),
        (axes[1], frac_r2,   "Test R\u00b2  higher is better",
         "(b) Test R\u00b2 vs Training Data Fraction"),
    ]:
        for mtype in arch_order:
            c = type_colors.get(mtype, "#888888")
            fracs_sorted = sorted(metric_data[mtype].keys())
            if not fracs_sorted:
                continue

            xs    = [f * 100 for f in fracs_sorted]
            means = [float(np.mean(metric_data[mtype][f])) for f in fracs_sorted]
            stds  = [float(np.std(metric_data[mtype][f])) if len(metric_data[mtype][f]) > 1 else 0.0
                     for f in fracs_sorted]

            ax.plot(xs, means, color=c, lw=2.2, marker="o", markersize=7, zorder=4,
                    label=_arch_short_label(mtype))
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=c, alpha=0.15)

            for f in fracs_sorted:
                vals = metric_data[mtype][f]
                jitter = rng.uniform(-0.8, 0.8, size=len(vals))
                ax.scatter([f * 100 + j for j in jitter], vals,
                           color=c, s=22, alpha=0.55, zorder=3, edgecolors="none")

        ax.set_xlabel("Training Data Fraction (%)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks([f * 100 for f in all_fracs])
        ax.set_xticklabels([f"{int(round(f * 100))}%" for f in all_fracs], fontsize=11)
        ax.grid(True, axis="y", alpha=0.35)

    arch_handles = [Patch(facecolor=type_colors.get(t, "#888888"), label=_arch_short_label(t))
                    for t in arch_order]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=arch_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(arch_handles), fontsize=11)

    _save_fig(fig, output_dir / "fig14_data_efficiency.png")


# ---------------------------------------------------------------------------
# Fig 15 - Physics Weight Impact (physics_weight vs RMSE and R2)
# ---------------------------------------------------------------------------

def plot_physics_weight_impact(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Fig 15: Test RMSE and R2 vs physics_weight for PhysicsRegularizedFNN,
    with separate lines per data_train_fraction and a pooled-mean dashed line."""
    recs = groups.get("PhysicsRegularizedFNN", [])
    if not recs:
        logger.info("No PhysicsRegularizedFNN models - skipping Fig 15.")
        return

    pw_frac_rmse: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    pw_frac_r2:   dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for rec in recs:
        hp   = rec.get("hyperparams", {})
        pw   = hp.get("physics_weight")
        frac = hp.get("data_train_fraction")
        if pw is None or frac is None:
            continue
        pw   = float(pw)
        frac = float(frac)
        rmse = _split_scalar(rec, "test", "rmse_pooled")
        r2   = _split_scalar(rec, "test", "r2_overall")
        if rmse == rmse and np.isfinite(rmse):
            pw_frac_rmse[pw][frac].append(rmse)
        if r2 == r2 and np.isfinite(r2):
            pw_frac_r2[pw][frac].append(r2)

    all_pw   = sorted(pw_frac_rmse.keys())
    all_frac = sorted({f for d in pw_frac_rmse.values() for f in d})

    if not all_pw:
        logger.info("No physics_weight data found - skipping Fig 15.")
        return

    _frac_cmap = matplotlib.colormaps["viridis"]
    frac_colors = {
        f: _frac_cmap(0.15 + 0.70 * i / max(len(all_frac) - 1, 1))
        for i, f in enumerate(all_frac)
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rng = np.random.default_rng(77)
    _x_range = max(all_pw) - min(all_pw) if len(all_pw) > 1 else 1.0

    for ax, metric_data, ylabel, title in [
        (axes[0], pw_frac_rmse, "Test RMSE (N\u00b7m)  lower is better",
         "(a) Test RMSE vs Physics Weight (\u03bb)"),
        (axes[1], pw_frac_r2,   "Test R\u00b2  higher is better",
         "(b) Test R\u00b2 vs Physics Weight (\u03bb)"),
    ]:
        for frac in all_frac:
            c = frac_colors[frac]
            xs, means, stds = [], [], []
            for pw in all_pw:
                vals = metric_data[pw].get(frac, [])
                if vals:
                    xs.append(pw)
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            if len(xs) < 2:
                continue

            ax.plot(xs, means, color=c, lw=2.0, marker="D", markersize=7, zorder=4,
                    label=f"{int(round(frac * 100))}% data")
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=c, alpha=0.15)

            for pw in xs:
                vals = metric_data[pw].get(frac, [])
                jitter = rng.uniform(-_x_range * 0.008, _x_range * 0.008, size=len(vals))
                ax.scatter([pw + j for j in jitter], vals,
                           color=c, s=22, alpha=0.55, zorder=3, edgecolors="none")

        pooled_xs, pooled_means = [], []
        for pw in all_pw:
            all_vals = [v for frac_d in metric_data[pw].values() for v in frac_d]
            if all_vals:
                pooled_xs.append(pw)
                pooled_means.append(float(np.mean(all_vals)))
        if len(pooled_xs) >= 2:
            ax.plot(pooled_xs, pooled_means, color="black", lw=2.5, ls="--",
                    marker="s", markersize=8, zorder=5, label="All fractions (pooled)")

        ax.set_xlabel("Physics Weight  (\u03bb)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(all_pw)
        ax.set_xticklabels([str(p) for p in all_pw], fontsize=11)
        ax.grid(True, axis="y", alpha=0.35)

    frac_handles = [
        Line2D([0], [0], color=frac_colors[f], lw=2, marker="D", markersize=7,
               label=f"{int(round(f * 100))}% data")
        for f in all_frac
    ]
    pooled_handle = Line2D([0], [0], color="black", lw=2.5, ls="--", marker="s",
                           markersize=8, label="All fractions (pooled)")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=frac_handles + [pooled_handle],
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(frac_handles) + 1, fontsize=10)

    _save_fig(fig, output_dir / "fig15_physics_weight_impact.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan grid-search trained models and open interactive performance report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                        help="Root directory containing trained model subdirectories.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip all plots; print summary table only.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top models to show in the leaderboard (Fig 10).")
    parser.add_argument("--no-train-metrics", action="store_true",
                        help="Skip model inference for train metrics (faster, uses history RMSE only).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
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

    compute_train = not args.no_train_metrics
    if compute_train:
        logger.info("Computing train metrics (model inference on training split)...")
        logger.info("  Use --no-train-metrics to skip this step.")

    enrich_records(records, compute_train=compute_train)
    groups = group_by_model_type(records)
    logger.info("Found %d model(s) in %d type(s): %s",
                len(records), len(groups), sorted(groups.keys()))

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

    plot_topk_leaderboard(groups, output_dir, top_k=args.top_k)
    plot_hp_importance(groups, output_dir)
    plot_hp_pair_heatmaps(groups, output_dir)
    plot_pareto_front(groups, output_dir)
    plot_data_efficiency(groups, output_dir)
    plot_physics_weight_impact(groups, output_dir)

    n_base = 9 if groups.get("EDR") else 8
    n_figs  = n_base + 6
    print(f"\nPlots saved to: {output_dir}")
    print(f"{n_figs} figure window(s) open - close all windows to exit.")
    plt.show(block=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
