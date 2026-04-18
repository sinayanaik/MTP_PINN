"""
Plot 03 — All-Runs Variance (Strip Plot + Box)
==============================================
All 50 training runs grouped by model type.
Each dot = one run. Box shows median ± IQR.
Journal style using scienceplots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import scienceplots  # noqa: F401
plt.style.use(["science", "grid"])

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR  = SCRIPT_DIR.parent
REGISTRY   = PLOTS_DIR.parent.parent / "Trained_Models" / "models_registry.yaml"
OUT_FILE   = PLOTS_DIR / "03_all_runs_variance.png"

DISPLAY = {
    "BlackBoxFNN":                 "Black-Box\n(A)",
    "PhysicsRegularizedFNN":       "Phys.-Reg.\n(B)",
    "ResidualCorrectionFNN":       "Residual\nCorr. (C)",
    "LagrangianStructuredFNN":     "Lagrangian\n(D)",
    "EquationConstrainedPINNFNN":  "EC-PINN\n(E.1)",
    "DecomposedStructuredPINNFNN": "Decomp.\nPINN (E.2)",
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


def main() -> None:
    with REGISTRY.open() as f:
        reg = yaml.safe_load(f)

    groups: dict[str, list[float]] = {m: [] for m in MODEL_ORDER}
    for m in reg.get("models", []):
        mtype = m.get("model_type")
        if mtype not in groups:
            continue
        mt = m.get("metrics") or {}
        r = mt.get("test_rmse_mean") or mt.get("avg_rmse_mean")
        if r is not None:
            groups[mtype].append(float(r))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    for gi, mtype in enumerate(MODEL_ORDER):
        vals = np.array(groups[mtype])
        if len(vals) == 0:
            continue
        color = COLORS[mtype]

        # Strip plot with jitter
        jitter = np.random.default_rng(42 + gi).uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), gi) + jitter, vals,
                   color=color, s=22, alpha=0.75, zorder=3, edgecolors="white",
                   linewidths=0.4)

        # Box: median line + IQR box
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        bw = 0.28
        rect = plt.Rectangle((gi - bw / 2, q1), bw, q3 - q1,
                               facecolor=color, alpha=0.22, edgecolor=color,
                               linewidth=1.0, zorder=2)
        ax.add_patch(rect)
        ax.hlines(med, gi - bw / 2, gi + bw / 2, color=color, linewidth=1.8, zorder=4)

        # Min/max whiskers
        ax.vlines(gi, vals.min(), q1, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.vlines(gi, q3, vals.max(), color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.hlines(vals.min(), gi - 0.08, gi + 0.08, color=color, linewidth=0.8)
        ax.hlines(vals.max(), gi - 0.08, gi + 0.08, color=color, linewidth=0.8)

        # Annotate median
        ax.text(gi + 0.22, med, f"{med:.5f}", va="center", fontsize=6, color=color)

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([DISPLAY[m] for m in MODEL_ORDER], fontsize=7.5)
    ax.set_ylabel(r"Test RMSE (N$\cdot$m, normalised)")
    ax.set_title("Training Run Variance — All 50 Runs Across Model Types", fontsize=9)

    # Legend: each dot = 1 run
    ax.text(0.98, 0.97, "Each dot = 1 training run\nBox = IQR, line = median",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray", boxstyle="round,pad=0.3"))

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
