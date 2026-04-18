"""
Plot 10 — Convergence Scatter: Training Time vs. Test RMSE
===========================================================
All 50 runs. Colored by model family. Pareto frontier (lower-left = better).
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

SCRIPT_DIR  = Path(__file__).resolve().parent
PLOTS_DIR   = SCRIPT_DIR.parent
TRAINED_DIR = PLOTS_DIR.parent.parent / "Trained_Models"
REGISTRY    = TRAINED_DIR / "models_registry.yaml"
OUT_FILE    = PLOTS_DIR / "10_convergence_scatter.png"

DISPLAY = {
    "BlackBoxFNN":                 "Black-Box (A)",
    "PhysicsRegularizedFNN":       "Phys.-Reg. (B)",
    "ResidualCorrectionFNN":       "Residual (C)",
    "LagrangianStructuredFNN":     "Lagrangian (D)",
    "EquationConstrainedPINNFNN":  "EC-PINN (E.1)",
    "DecomposedStructuredPINNFNN": "Decomposed (E.2)",
}
COLORS = {
    "BlackBoxFNN":                 "#4e79a7",
    "PhysicsRegularizedFNN":       "#f28e2b",
    "ResidualCorrectionFNN":       "#59a14f",
    "LagrangianStructuredFNN":     "#b07aa1",
    "EquationConstrainedPINNFNN":  "#e15759",
    "DecomposedStructuredPINNFNN": "#76b7b2",
}
MARKERS = {
    "BlackBoxFNN":                 "o",
    "PhysicsRegularizedFNN":       "s",
    "ResidualCorrectionFNN":       "^",
    "LagrangianStructuredFNN":     "D",
    "EquationConstrainedPINNFNN":  "P",
    "DecomposedStructuredPINNFNN": "X",
}
MODEL_ORDER = list(DISPLAY.keys())


def pareto_frontier(times, rmse_vals):
    """Return indices of points on the lower-left Pareto front."""
    pts = sorted(enumerate(zip(times, rmse_vals)), key=lambda x: x[1][0])
    pareto = []
    min_rmse = float("inf")
    for idx, (t, r) in pts:
        if r < min_rmse:
            pareto.append(idx)
            min_rmse = r
    return pareto


def main() -> None:
    with REGISTRY.open() as f:
        reg = yaml.safe_load(f)

    groups: dict[str, list[tuple[float, float]]] = {m: [] for m in MODEL_ORDER}
    all_times, all_rmse, all_types = [], [], []

    for m in reg.get("models", []):
        mtype = m.get("model_type")
        if mtype not in DISPLAY:
            continue
        mt = m.get("metrics") or {}
        tr = m.get("training") or {}
        rmse = mt.get("test_rmse_mean") or mt.get("avg_rmse_mean")
        time = tr.get("time_seconds")
        if rmse is None or time is None:
            continue
        rmse, time = float(rmse), float(time) / 60.0  # min
        groups[mtype].append((time, rmse))
        all_times.append(time)
        all_rmse.append(rmse)
        all_types.append(mtype)

    fig, ax = plt.subplots(figsize=(6, 4))

    for mtype in MODEL_ORDER:
        pts = groups[mtype]
        if not pts:
            continue
        ts, rs = zip(*pts)
        ax.scatter(ts, rs, color=COLORS[mtype], marker=MARKERS[mtype],
                   s=40, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.4,
                   label=DISPLAY[mtype])

    # Pareto frontier (lower-left better)
    pf_idx = pareto_frontier(all_times, all_rmse)
    pf_pts = sorted([(all_times[i], all_rmse[i]) for i in pf_idx])
    if len(pf_pts) >= 2:
        pf_t, pf_r = zip(*pf_pts)
        ax.step(pf_t, pf_r, where="post", color="black", linewidth=0.9,
                linestyle="--", alpha=0.6, zorder=4, label="Pareto frontier")

    ax.set_xlabel("Training Time (min)")
    ax.set_ylabel(r"Test RMSE (N$\cdot$m, normalised)")
    ax.set_title("Efficiency: Training Time vs. Test RMSE — All 50 Runs", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.7, loc="upper right")

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
