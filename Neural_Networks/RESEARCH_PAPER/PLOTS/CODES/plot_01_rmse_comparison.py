"""
Plot 01 — Test RMSE Comparison (Best Run per Model)
====================================================
Horizontal bar chart sorted best → worst, starred best performer.
Journal style using scienceplots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml

import scienceplots  # noqa: F401
plt.style.use(["science", "grid"])

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PLOTS_DIR   = SCRIPT_DIR.parent
REGISTRY    = PLOTS_DIR.parent.parent / "Trained_Models" / "models_registry.yaml"
OUT_FILE    = PLOTS_DIR / "01_rmse_comparison.png"

# ── Config ────────────────────────────────────────────────────────────────────
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


def load_best_per_model(registry_path: Path) -> dict[str, dict]:
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
    best = load_best_per_model(REGISTRY)

    names, rmse_vals, colors = [], [], []
    for mtype in MODEL_ORDER:
        if mtype not in best:
            continue
        m = best[mtype]
        r = (m.get("metrics") or {}).get("test_rmse_mean") or \
            (m.get("metrics") or {}).get("avg_rmse_mean")
        if r is None:
            continue
        names.append(DISPLAY[mtype])
        rmse_vals.append(float(r))
        colors.append(COLORS[mtype])

    # Sort best (lowest) → worst
    order = np.argsort(rmse_vals)
    names     = [names[i] for i in order]
    rmse_vals = [rmse_vals[i] for i in order]
    colors    = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    bars = ax.barh(names, rmse_vals, color=colors, edgecolor="white",
                   linewidth=0.6, height=0.6)

    # Annotate bars
    for bar, val in zip(bars, rmse_vals):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", ha="left", fontsize=7)

    # Star best
    best_idx = 0  # after sort
    bars[best_idx].set_edgecolor("goldenrod")
    bars[best_idx].set_linewidth(1.5)
    ax.text(rmse_vals[best_idx] + 0.0005,
            bars[best_idx].get_y() + bars[best_idx].get_height() / 2 + 0.3,
            r"$\bigstar$ best", va="bottom", ha="left", fontsize=7,
            color="goldenrod", fontstyle="italic")

    # Baseline reference line (BlackBox)
    bb_rmse = None
    for nm, rv in zip(names, rmse_vals):
        if "Black" in nm:
            bb_rmse = rv
    if bb_rmse:
        ax.axvline(bb_rmse, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
                   label=f"Black-Box baseline ({bb_rmse:.5f})")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.6)

    ax.set_xlabel(r"Test RMSE (N$\cdot$m, normalised)")
    ax.set_title("Test RMSE by Model Architecture (Best Run)", fontsize=9)
    ax.set_xlim(0, max(rmse_vals) * 1.18)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
