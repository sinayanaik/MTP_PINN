#!/usr/bin/env python3
"""
Model registry visualization — enhanced for new registry schema.

What this version does:
- Reads models_registry.yaml (auto-detected)
- Extracts new metadata: training time, epochs, AMP/compile, device, samples
- Shows richly formatted terminal summary table
- Saves/shows 6 plots: RMSE, family comparison, joint error,
  training time, convergence efficiency, epochs utilisation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

# ─── HARDCODED SETTINGS ───────────────────────────────────────────────────────
# Parent of this file is Neural_Networks/analysis/; parent.parent is Neural_Networks/
_THIS_DIR = Path(__file__).resolve().parent.parent

DEFAULT_YAML_CANDIDATES = [
    _THIS_DIR / "Trained_Models" / "models_registry.yaml",   # canonical location
    _THIS_DIR / "models_registry.yaml",
    Path("models_registry.yaml"),
    Path.home() / "Desktop" / "MTP_PINN" / "Neural_Networks" / "Trained_Models" / "models_registry.yaml",
]

SHOW_FIGURES = True
SAVE_FIGURES = False
OUTPUT_DIR   = Path("model_registry_outputs")

DISPLAY_NAMES = {
    # A — Black Box (Data-Driven)
    "BlackBoxFNN":                 "Black Box FNN (1)",
    # B — Physics-Regularized
    "PhysicsRegularizedFNN":       "Physics-Regularized FNN (2)",
    # C — Residual Correction
    "ResidualCorrectionFNN":       "Residual Correction FNN (3)",
    # D — Lagrangian Structured
    "LagrangianStructuredFNN":     "Lagrangian Structured FNN (4)",
    # E.1 — Equation-Constrained PINN
    "EquationConstrainedPINNFNN":  "Eq-Constrained PINN FNN (5)",
    # E.2 — Decomposed Structured PINN
    "DecomposedStructuredPINNFNN": "Decomposed Structured PINN FNN (6)",
}

# Colour per model
FAMILY_COLORS = {
    "BlackBoxFNN":                 "#9ca3af",   # A — grey
    "PhysicsRegularizedFNN":       "#93c5fd",   # B — blue
    "ResidualCorrectionFNN":       "#fdba74",   # C — orange
    "LagrangianStructuredFNN":     "#d8b4fe",   # D — purple
    "EquationConstrainedPINNFNN":  "#6ee7b7",   # E.1 — green
    "DecomposedStructuredPINNFNN": "#fca5a5",   # E.2 — red
}

MODEL_ORDER = [
    "BlackBoxFNN",
    "PhysicsRegularizedFNN",
    "ResidualCorrectionFNN",
    "LagrangianStructuredFNN",
    "EquationConstrainedPINNFNN",
    "DecomposedStructuredPINNFNN",
]

# Map model → category label (for family comparison plot)
MODEL_CATEGORY = {
    "BlackBoxFNN":                 "A — Black Box",
    "PhysicsRegularizedFNN":       "B — Physics-Regularized",
    "ResidualCorrectionFNN":       "C — Residual Correction",
    "LagrangianStructuredFNN":     "D — Lagrangian Structured",
    "EquationConstrainedPINNFNN":  "E.1 — Eq-Constrained PINN",
    "DecomposedStructuredPINNFNN": "E.2 — Decomposed PINN",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#f8f9fa",
    "axes.grid":          True,
    "grid.alpha":         0.4,
    "grid.linewidth":     0.6,
})


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def find_yaml_file() -> Path:
    for path in DEFAULT_YAML_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find models_registry YAML.\nChecked:\n- "
        + "\n- ".join(str(p) for p in DEFAULT_YAML_CANDIDATES)
    )


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError("YAML must contain a top-level 'models' list.")
    return data


def parse_time(value):
    try:
        return pd.to_datetime(value)
    except Exception:
        return pd.Timestamp.min


def keep_latest_models(models: Sequence[dict]) -> List[dict]:
    latest: Dict[str, dict] = {}
    for model in models:
        name = model.get("model_type")
        if not name:
            continue
        if (name not in latest
                or parse_time(model.get("trained_at"))
                >= parse_time(latest[name].get("trained_at"))):
            latest[name] = model
    return list(latest.values())


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def build_dataframe(registry: dict) -> pd.DataFrame:
    """DataFrame with only the LATEST run per model type."""
    return _build_df_from_models(keep_latest_models(registry.get("models", [])))


def build_all_runs_dataframe(registry: dict) -> pd.DataFrame:
    """DataFrame with ALL runs per model type (may have many rows per type)."""
    return _build_df_from_models(registry.get("models", []))


def _safe_float(val, default=np.nan) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _build_df_from_models(models: Sequence[dict]) -> pd.DataFrame:
    rows = []

    for model in models:
        metrics  = model.get("metrics",  {}) or {}
        training = model.get("training", {}) or {}
        hardware = model.get("hardware", {}) or {}
        data_inf = model.get("data",     {}) or {}

        per_joint = metrics.get("per_joint_rmse", []) or []
        rows.append({
            # identity
            "model":   model.get("model_type"),
            "display": display_name(model.get("model_type", "Unknown")),
            "run_dir": model.get("run_dir", ""),
            # test metrics
            "rmse":          _safe_float(metrics.get("test_rmse_mean")),
            "nrmse":         _safe_float(metrics.get("test_nrmse_mean")),
            "mse":           _safe_float(metrics.get("test_mse_mean")),
            "joint_avg_rmse":_safe_float(np.mean(per_joint) if per_joint else np.nan),
            "joint_max_rmse":_safe_float(np.max(per_joint)  if per_joint else np.nan),
            "per_joint":     per_joint,
            "num_joints":    len(per_joint),
            # training metadata (new in updated registry)
            "time_s":       _safe_float(training.get("time_seconds")),
            "time_fmt":     training.get("time_formatted", "—"),
            "epochs_ran":   int(training.get("epochs_ran", 0) or 0),
            "epochs_max":   int(training.get("epochs_max", 0) or 0),
            "stopped_early":bool(training.get("stopped_early", False)),
            "final_val":    _safe_float(training.get("final_val_loss")),
            # hardware
            "device":       hardware.get("device", "—"),
            "torch_ver":    hardware.get("torch_version", "—"),
            # dataset
            "n_train":      int(data_inf.get("num_train_samples", 0) or 0),
            "n_val":        int(data_inf.get("num_val_samples",   0) or 0),
            # trained_at timestamp
            "trained_at":   model.get("trained_at", ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid model entries found in YAML.")

    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    df["order"] = df["model"].map(lambda x: order_map.get(x, 999))
    df = df.sort_values(["order", "rmse"], na_position="last").reset_index(drop=True)
    return df


def _bar_colors(df: pd.DataFrame) -> list:
    return [FAMILY_COLORS.get(m, "#888888") for m in df["model"]]


# ─── PLOTS ────────────────────────────────────────────────────────────────────
def plot_rmse_bar(df: pd.DataFrame):
    tmp    = df.dropna(subset=["rmse"]).sort_values("rmse").reset_index(drop=True)
    colors = _bar_colors(tmp)
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(tmp["display"], tmp["rmse"], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Test RMSE Comparison — All Models (lower = better)", fontsize=13, fontweight="bold")
    ax.set_ylabel("RMSE (N·m)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=32)
    for bar, val in zip(bars, tmp["rmse"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Highlight best
    best_idx = tmp["rmse"].idxmin()
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)
    ax.text(bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
            tmp.iloc[best_idx]["rmse"] / 2, "★ best", ha="center",
            color="white", fontsize=9, fontweight="bold")
    _add_family_legend(ax)
    plt.tight_layout()


def plot_all_runs_per_model(df_all: pd.DataFrame):
    """
    Grouped bar chart — one group per model type, bars within each group = individual
    training runs (sorted oldest → newest, shaded light → dark).

    Shows training variance across repeated runs for each architecture.
    """
    df_valid = df_all.dropna(subset=["rmse"]).copy()
    if df_valid.empty:
        return

    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    df_valid["order"] = df_valid["model"].map(lambda x: order_map.get(x, 999))
    df_valid = df_valid.sort_values(["order", "trained_at"]).reset_index(drop=True)

    groups        = [m for m in MODEL_ORDER if m in df_valid["model"].values]
    n_groups      = len(groups)
    max_runs      = int(df_valid.groupby("model").size().max())
    group_gap     = 0.3
    bar_w         = (1.0 - group_gap) / max(max_runs, 1)
    x_centers     = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(14, n_groups * 2.5), 7))

    for g_idx, model_name in enumerate(groups):
        sub   = df_valid[df_valid["model"] == model_name].reset_index(drop=True)
        color = FAMILY_COLORS.get(model_name, "#888888")
        n_sub = len(sub)

        # Shade intensity: first run = 60% alpha, last run = 100% alpha
        alphas = np.linspace(0.45, 1.0, max(n_sub, 1))
        for r_idx, (_, row) in enumerate(sub.iterrows()):
            x_pos  = g_idx - (n_sub - 1) * bar_w / 2 + r_idx * bar_w
            bar    = ax.bar(x_pos, row["rmse"], width=bar_w * 0.85,
                            color=color, alpha=float(alphas[r_idx]),
                            edgecolor="white", linewidth=0.6)
            ax.text(x_pos, row["rmse"] + 0.0004,
                    f"{row['rmse']:.4f}", ha="center", va="bottom",
                    fontsize=7, rotation=70)

        # Range bracket if > 1 run
        if n_sub > 1:
            vals   = sub["rmse"].values
            y_bot  = vals.min()
            y_top  = vals.max()
            ax.plot([g_idx, g_idx], [y_bot, y_top],
                    color="black", linewidth=1.2, alpha=0.5, zorder=4)
            ax.plot([g_idx - 0.06, g_idx + 0.06], [y_bot, y_bot],
                    color="black", linewidth=1.2, alpha=0.5, zorder=4)
            ax.plot([g_idx - 0.06, g_idx + 0.06], [y_top, y_top],
                    color="black", linewidth=1.2, alpha=0.5, zorder=4)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in groups],
                       rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Test RMSE (N·m)")
    ax.set_title(
        "All Training Runs per Model — Test RMSE (light bar = earliest run, dark = latest)",
        fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _add_family_legend(ax)
    plt.tight_layout()



def plot_joint_heatmap(df: pd.DataFrame):
    """Heatmap of per-joint RMSE across all models."""
    valid = df[df["num_joints"] == 5].reset_index(drop=True)
    if valid.empty:
        return
    matrix = np.array([r for r in valid["per_joint"]])
    joint_names = ["J1 yaw", "J2 shoulder", "J3 elbow", "J4 wrist", "J5 wrist roll"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(valid) * 0.6 + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=matrix.min() * 0.9, vmax=matrix.max() * 1.1)
    plt.colorbar(im, ax=ax, label="RMSE (N·m)", shrink=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(joint_names, rotation=20, ha="right")
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(valid["display"].tolist(), fontsize=9)
    for i in range(len(valid)):
        for j in range(5):
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("Per-Joint RMSE Heatmap — All Models", fontsize=13, fontweight="bold")
    ax.set_xlabel("Joint")
    plt.tight_layout()


def plot_training_time(df: pd.DataFrame):
    """Horizontal bar: training time per model, coloured by family."""
    tmp = df.dropna(subset=["time_s"]).sort_values("time_s", ascending=True).reset_index(drop=True)
    if tmp.empty:
        return
    colors = _bar_colors(tmp)
    fig, ax = plt.subplots(figsize=(10, max(4, len(tmp) * 0.55 + 1.5)))
    bars = ax.barh(tmp["display"], tmp["time_s"], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Training Time per Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training time (s)")
    for bar, (_, row) in zip(bars, tmp.iterrows()):
        label = row["time_fmt"] if row["time_fmt"] != "—" else f"{row['time_s']:.0f}s"
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)
    _add_family_legend(ax)
    plt.tight_layout()


def plot_convergence_efficiency(df: pd.DataFrame):
    """Scatter: training time vs RMSE — ideal is bottom-left."""
    tmp = df.dropna(subset=["rmse", "time_s"]).reset_index(drop=True)
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = _bar_colors(tmp)
    for _, row in tmp.iterrows():
        c = FAMILY_COLORS.get(row["model"], "#888")
        ax.scatter(row["time_s"], row["rmse"], color=c, s=120, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(row["display"],
                    xy=(row["time_s"], row["rmse"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=c)
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Test RMSE (N·m)")
    ax.set_title("Convergence Efficiency  (bottom-left = fast AND accurate)", fontsize=12, fontweight="bold")
    ax.annotate("← faster", xy=(0.05, 0.03), xycoords="axes fraction",
                fontsize=9, color="grey", style="italic")
    ax.annotate("↑ better accuracy", xy=(0.03, 0.05), xycoords="axes fraction",
                fontsize=9, color="grey", style="italic", rotation=90)
    _add_family_legend(ax)
    plt.tight_layout()


def plot_epochs_utilisation(df: pd.DataFrame):
    """Bar: epochs_ran / epochs_max for each model, coloured by early-stopping."""
    tmp = df[df["epochs_max"] > 0].reset_index(drop=True)
    if tmp.empty:
        return
    ratios = (tmp["epochs_ran"] / tmp["epochs_max"].replace(0, np.nan)).fillna(0)
    colors_ep = ["#e74c3c" if se else "#2ecc71" for se in tmp["stopped_early"]]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(tmp["display"], ratios * 100, color=colors_ep, edgecolor="white", linewidth=0.8)
    ax.axhline(100, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Epochs used (%)")
    ax.set_title("Epoch Utilisation  (red = early stopped, green = ran full budget)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    for bar, (_, row) in zip(bars, tmp.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{row['epochs_ran']}/{row['epochs_max']}", ha="center",
                va="bottom", fontsize=8)
    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="Early stopped"),
        mpatches.Patch(color="#2ecc71", label="Ran full budget"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    plt.tight_layout()


def _add_family_legend(ax):
    families = {
        "A — Black Box":             "#9ca3af",
        "B — Physics-Regularized":   "#93c5fd",
        "C — Residual Correction":   "#fdba74",
        "D — Lagrangian Structured": "#d8b4fe",
        "E.1 — Eq-Constrained PINN": "#6ee7b7",
        "E.2 — Decomposed PINN":     "#fca5a5",
    }
    patches = [mpatches.Patch(color=c, label=l) for l, c in families.items()]
    ax.legend(handles=patches, fontsize=8, loc="upper right", framealpha=0.8, ncol=2)


# ─── NEW PLOTS ────────────────────────────────────────────────────────────────

def _load_sidecar_metrics(run_dir: str) -> dict:
    """
    Load per-joint metrics (r2, pearson_r, rmse, nrmse, mae) from the
    metadata.yaml sidecar written by train.py alongside each model checkpoint.

    If run_dir is an absolute path from another machine, falls back to resolving
    relative to this file's Trained_Models directory.
    """
    def _try(p: Path) -> dict:
        if not p.exists():
            return {}
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return data.get("metrics", {})
        except Exception:
            return {}

    # Primary: use path as stored
    m = _try(Path(run_dir) / "metadata.yaml")
    if m:
        return m

    # Fallback: re-anchor at the local Trained_Models directory.
    # The run_dir looks like /.../Trained_Models/<ModelType>/<run_id>/
    pth = Path(run_dir)
    parts = pth.parts
    try:
        tm_idx = next(i for i, p in enumerate(parts) if p == "Trained_Models")
        rel    = Path(*parts[tm_idx:])        # Trained_Models/<ModelType>/<run_id>
        local  = _THIS_DIR / "Trained_Models" / Path(*rel.parts[1:])  # strip first "Trained_Models"
        m = _try(local / "metadata.yaml")
        if m:
            return m
    except (StopIteration, TypeError, ValueError):
        pass

    return {}


def plot_metrics_matrix(df_latest: pd.DataFrame):
    """
    Two-row heatmap figure for the best (latest) run per model:
      Row 1: model × joint  R²   (RdYlGn, 0 → 1)
      Row 2: model × joint  Pearson r  (RdYlGn, −1 → 1)

    Loads per-joint R² and Pearson r from the metadata.yaml sidecar
    (cheaply — does not load model weights).
    """
    valid = df_latest.dropna(subset=["rmse"]).reset_index(drop=True)
    if valid.empty:
        return

    joint_names = ["J1 yaw", "J2 shoulder", "J3 elbow", "J4 wrist", "J5 roll"]
    model_labels = valid["display"].tolist()
    n_models     = len(model_labels)

    r2_mat  = np.full((n_models, 5), np.nan)
    pr_mat  = np.full((n_models, 5), np.nan)

    any_data = False
    for i, (_, row) in enumerate(valid.iterrows()):
        # run_dir is not stored in df; re-derive from registry via model_type + trained_at
        # We load it from the registry-level YAML path stored inside DEFAULT_YAML_CANDIDATES
        run_dir = row.get("run_dir", "")
        if not run_dir:
            continue
        m = _load_sidecar_metrics(str(run_dir))
        r2  = m.get("r2",        [])
        pr  = m.get("pearson_r", [])
        if len(r2) == 5:
            r2_mat[i] = r2
            any_data = True
        if len(pr) == 5:
            pr_mat[i] = pr
            any_data = True

    if not any_data:
        print("[models_analysis] plot_metrics_matrix: no metadata.yaml sidecars found — skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, max(7, n_models * 1.1 + 3)))
    fig.suptitle("Per-Joint Metrics Matrix — Latest Run per Model",
                 fontsize=13, fontweight="bold")

    for ax, mat, title, vmin, vmax, cmap in [
        (ax1, r2_mat,  "R²  (higher = better)",          0.0, 1.0,  "RdYlGn"),
        (ax2, pr_mat,  "Pearson r  (higher = better)",  -1.0, 1.0,  "RdYlGn"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(5))
        ax.set_xticklabels(joint_names, rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(model_labels, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for i in range(n_models):
            for j in range(5):
                v = mat[i, j]
                txt = f"{v:.3f}" if np.isfinite(v) else "—"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8,
                        color="black" if 0.3 < abs(v) < 0.9 else "white"
                        if np.isfinite(v) else "grey")
    plt.tight_layout()


def plot_intermodel_diff_matrix(df_latest: pd.DataFrame):
    """
    N×N pairwise RMSE-delta heatmap.
    Cell (i, j) = RMSE_i − RMSE_j.
    Positive = model i is WORSE than model j.  Diagonal = 0.
    Diverging colourmap centred at 0 makes wins/losses immediately visible.
    """
    valid = df_latest.dropna(subset=["rmse"]).reset_index(drop=True)
    if valid.empty:
        return
    labels = valid["display"].tolist()
    rmses  = valid["rmse"].values
    n      = len(labels)

    delta  = rmses[:, None] - rmses[None, :]   # (n, n) matrix
    abs_max = np.max(np.abs(delta[~np.isnan(delta)])) if n > 1 else 0.01

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5 + 1), max(7, n * 1.5)))
    im = ax.imshow(delta, aspect="auto", cmap="RdYlGn_r",
                   vmin=-abs_max, vmax=abs_max)
    plt.colorbar(im, ax=ax, label="RMSE_row − RMSE_col  (N·m)", shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(
        "Pairwise RMSE Difference — cell(i,j) = RMSE_i − RMSE_j\n"
        "(green = row model is BETTER than col model)",
        fontsize=11, fontweight="bold")

    for i in range(n):
        for j in range(n):
            v   = delta[i, j]
            sgn = "+" if v > 0 else ""
            ax.text(j, i, f"{sgn}{v:.4f}", ha="center", va="center",
                    fontsize=8 if n <= 6 else 7,
                    fontweight="bold" if i == j else "normal",
                    color="white" if abs(v) > abs_max * 0.65 else "black")
    plt.tight_layout()



def print_simple_summary(df: pd.DataFrame):
    tmp = df.dropna(subset=["rmse"]).sort_values("rmse").reset_index(drop=True)
    if tmp.empty:
        return

    best  = tmp.iloc[0]
    worst = tmp.iloc[-1]

    SEP = "=" * 90
    print("\n" + SEP)
    print("  MODEL REGISTRY SUMMARY")
    print(SEP)

    col_w = [4, 34, 8, 8, 8, 6, 12, 12, 16]
    header = (
        f"{'#':>{col_w[0]}}  "
        f"{'Model':<{col_w[1]}}  "
        f"{'RMSE':>{col_w[2]}}  "
        f"{'NRMSE':>{col_w[3]}}  "
        f"{'Epochs':>{col_w[4]}}  "
        f"{'ES?':>{col_w[5]}}  "
        f"{'Train time':>{col_w[6]}}  "
        f"{'Train samples':>{col_w[7]}}  "
        f"{'Device':<{col_w[8]}}"
    )
    print(header)
    print("-" * 90)

    for i, (_, row) in enumerate(tmp.iterrows(), 1):
        prefix = "★ " if i == 1 else "  "
        epoch_str = (f"{row['epochs_ran']}/{row['epochs_max']}"
                     if row["epochs_max"] > 0 else "—")
        es_str    = "YES" if row["stopped_early"] else " no"
        n_train   = f"{row['n_train']:,}" if row["n_train"] > 0 else "—"
        time_str  = row["time_fmt"] if row["time_fmt"] != "—" else "—"
        nrmse_str = f"{row['nrmse']:.4f}" if not np.isnan(row['nrmse']) else "     —"

        print(
            f"{prefix}{i:>{col_w[0]-2}}  "
            f"{row['display']:<{col_w[1]}}  "
            f"{row['rmse']:>{col_w[2]}.5f}  "
            f"{nrmse_str:>{col_w[3]}}  "
            f"{epoch_str:>{col_w[4]}}  "
            f"{es_str:>{col_w[5]}}  "
            f"{time_str:>{col_w[6]}}  "
            f"{n_train:>{col_w[7]}}  "
            f"{str(row['device']):<{col_w[8]}}"
        )

    print("-" * 90)
    print(f"\n  Best  : {best['display']}   RMSE = {best['rmse']:.5f} N·m")
    print(f"  Worst : {worst['display']}   RMSE = {worst['rmse']:.5f} N·m")
    if worst["rmse"] > 0:
        gain = (worst["rmse"] - best["rmse"]) / worst["rmse"] * 100
        print(f"  Gain  : best is {gain:.1f}% lower RMSE than worst")

    # Per-joint summary for best model
    if best["num_joints"] == 5 and best["per_joint"]:
        joint_names = ["J1 yaw", "J2 shoulder", "J3 elbow", "J4 wrist", "J5 roll"]
        print(f"\n  Per-joint RMSE for '{best['display']}':")
        for jn, rv in zip(joint_names, best["per_joint"]):
            bar_len = int(rv / best["rmse"] * 20)
            bar = "█" * bar_len
            print(f"    {jn:<14}  {rv:.5f} N·m  {bar}")

    print("\n" + SEP + "\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def save_open_figures():
    if not SAVE_FIGURES:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, fig_num in enumerate(plt.get_fignums(), 1):
        plt.figure(fig_num).savefig(OUTPUT_DIR / f"figure_{i}.png", dpi=200, bbox_inches="tight")


def main():
    yaml_path = find_yaml_file()
    registry  = load_yaml(yaml_path)
    df        = build_dataframe(registry)       # latest run per model type
    df_all    = build_all_runs_dataframe(registry)  # all runs

    print(f"\nUsing YAML: {yaml_path}")
    print(f"Total runs in registry : {registry.get('total_models', '?')}   "
          f"Unique model types: {df['model'].nunique()}")
    print(f"Last updated           : {registry.get('last_updated', '—')}\n")

    print_simple_summary(df)

    # ── Plots using latest run per model ────────────────────────────────────
    print("[models_analysis] Generating plots …")
    plot_rmse_bar(df)
    plot_joint_heatmap(df)
    plot_training_time(df)
    plot_convergence_efficiency(df)
    plot_epochs_utilisation(df)
    plot_metrics_matrix(df)
    plot_intermodel_diff_matrix(df)

    # ── Plots using ALL runs (shows training variance) ───────────────────
    plot_all_runs_per_model(df_all)

    save_open_figures()

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
