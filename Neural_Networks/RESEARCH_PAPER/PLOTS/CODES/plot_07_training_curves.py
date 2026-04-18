"""
Plot 07 — Validation RMSE Training Curves (2×3 grid, all runs)
===============================================================
One panel per model. All runs shown (alpha-faded older, bold latest).
Journal style using scienceplots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import scienceplots  # noqa: F401
plt.style.use(["science", "grid"])

SCRIPT_DIR  = Path(__file__).resolve().parent
PLOTS_DIR   = SCRIPT_DIR.parent
TRAINED_DIR = PLOTS_DIR.parent.parent / "Trained_Models"
REGISTRY    = TRAINED_DIR / "models_registry.yaml"
OUT_FILE    = PLOTS_DIR / "07_training_curves.png"

DISPLAY = {
    "BlackBoxFNN":                 "Black-Box FNN (A)",
    "PhysicsRegularizedFNN":       "Physics-Regularized FNN (B)",
    "ResidualCorrectionFNN":       "Residual Correction FNN (C)",
    "LagrangianStructuredFNN":     "Lagrangian Structured FNN (D)",
    "EquationConstrainedPINNFNN":  "Eq.-Constrained PINN (E.1)",
    "DecomposedStructuredPINNFNN": "Decomposed Structured PINN (E.2)",
}
COLORS = {
    "BlackBoxFNN":                 "#4e79a7",
    "PhysicsRegularizedFNN":       "#f28e2b",
    "ResidualCorrectionFNN":       "#59a14f",
    "LagrangianStructuredFNN":     "#b07aa1",
    "EquationConstrainedPINNFNN":  "#e15759",
    "DecomposedStructuredPINNFNN": "#76b7b2",
}
MODEL_ORDER = list(DISPLAY.keys())


def load_registry(registry_path: Path):
    with registry_path.open() as f:
        reg = yaml.safe_load(f)
    groups: dict[str, list[dict]] = {m: [] for m in MODEL_ORDER}
    for m in reg.get("models", []):
        mtype = m.get("model_type")
        if mtype in groups:
            groups[mtype].append(m)
    # Sort by trained_at
    for mtype in groups:
        groups[mtype].sort(key=lambda x: x.get("trained_at", ""))
    return groups


def main() -> None:
    groups = load_registry(REGISTRY)

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), constrained_layout=True)
    axes_flat = axes.flatten()

    for pi, mtype in enumerate(MODEL_ORDER):
        ax = axes_flat[pi]
        runs = groups.get(mtype, [])
        color = COLORS[mtype]
        n_runs = len(runs)

        latest_run = None
        if runs:
            latest_run = runs[-1]

        for ri, run in enumerate(runs):
            run_dir = Path(run.get("run_dir") or "")
            csv_path = run_dir / "training_history.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "val_rmse" not in df.columns:
                continue
            epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))
            val_rmse = df["val_rmse"].values

            is_latest = (run is latest_run)
            alpha = 0.85 if is_latest else max(0.12, 0.55 * (ri + 1) / n_runs)
            lw    = 1.6  if is_latest else 0.6
            zord  = 5    if is_latest else 2

            ax.plot(epochs, val_rmse, color=color, alpha=alpha, linewidth=lw, zorder=zord)

        ax.set_title(DISPLAY[mtype], fontsize=7.5)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Val. RMSE", fontsize=7)
        ax.tick_params(labelsize=6.5)

        if n_runs > 0:
            ax.text(0.97, 0.95, f"{n_runs} runs", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, color=color,
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none",
                              boxstyle="round,pad=0.2"))

    # Global legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=1.6, label="Latest run (bold)"),
        Line2D([0], [0], color="gray", linewidth=0.6, alpha=0.4, label="Earlier runs (faded)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=7.5, framealpha=0.7,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Validation RMSE Training Curves — All Runs per Model", fontsize=10)
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
