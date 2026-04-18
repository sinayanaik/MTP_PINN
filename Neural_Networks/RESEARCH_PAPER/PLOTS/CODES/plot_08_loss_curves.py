"""
Plot 08 — Train vs. Validation Loss Curves (2×3 grid, best run)
================================================================
One panel per model. Train loss (solid) and val loss (dashed) for latest run.
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
OUT_FILE    = PLOTS_DIR / "08_loss_curves.png"

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


def load_best(registry_path: Path) -> dict[str, dict]:
    with registry_path.open() as f:
        reg = yaml.safe_load(f)
    best: dict[str, dict] = {}
    for m in reg.get("models", []):
        mtype = m.get("model_type")
        if mtype not in DISPLAY:
            continue
        t = m.get("trained_at", "")
        if mtype not in best or t > best[mtype].get("trained_at", ""):
            best[mtype] = m
    return best


def main() -> None:
    best = load_best(REGISTRY)

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2), constrained_layout=True)
    axes_flat = axes.flatten()

    for pi, mtype in enumerate(MODEL_ORDER):
        ax = axes_flat[pi]
        m = best.get(mtype)
        if m is None:
            ax.set_visible(False)
            continue

        run_dir  = Path(m.get("run_dir") or "")
        csv_path = run_dir / "training_history.csv"
        color    = COLORS[mtype]

        if not csv_path.exists():
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(DISPLAY[mtype], fontsize=7.5)
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            ax.set_title(DISPLAY[mtype], fontsize=7.5)
            continue

        epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))

        if "train_loss" in df.columns:
            ax.plot(epochs, df["train_loss"].values, color=color,
                    linestyle="-", linewidth=1.2, label="Train loss", alpha=0.9)
        if "val_loss" in df.columns:
            ax.plot(epochs, df["val_loss"].values, color=color,
                    linestyle="--", linewidth=1.2, label="Val. loss", alpha=0.9)

        # Mark minimum val_loss
        if "val_loss" in df.columns:
            min_ep  = epochs[np.nanargmin(df["val_loss"].values)]
            min_val = np.nanmin(df["val_loss"].values)
            ax.axvline(min_ep, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
            ax.scatter([min_ep], [min_val], s=20, color="gold",
                       zorder=5, edgecolors="gray", linewidth=0.5)

        ax.set_title(DISPLAY[mtype], fontsize=7.5)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Loss", fontsize=7)
        ax.tick_params(labelsize=6.5)

    # Global legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", linestyle="-",  linewidth=1.2, label="Train loss"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2, label="Val. loss"),
        Line2D([0], [0], marker="o",  color="w", markerfacecolor="gold",
               markeredgecolor="gray", markersize=6, label="Best val. epoch"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=7.5, framealpha=0.7,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Train and Validation Loss — Latest Run per Model", fontsize=10)
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
