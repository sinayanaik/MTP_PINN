"""
Plot 14 — Friction Model Curves
================================
τ_f(q̇) = c·tanh(q̇/ε) + v·q̇ for each of the 5 joints.
Decomposed into viscous (dashed), Coulomb (dotted), total (solid).
Journal style using scienceplots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import scienceplots  # noqa: F401
plt.style.use(["science", "grid"])

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR  = SCRIPT_DIR.parent
OUT_FILE   = PLOTS_DIR / "14_friction_model.png"

# Calibrated from core/physics.py
COULOMB_NM   = [0.134975, 0.278199, 0.201313, 0.088112, 0.203864]
VISCOUS_NM   = [0.300000, 0.300000, 0.245417, 0.040191, 0.046918]
FRICTION_EPS = 0.040469   # rad/s

JOINT_NAMES  = ["J1 yaw", "J2 shoulder", "J3 elbow", "J4 wrist", "J5 roll"]
JOINT_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759"]

qdot = np.linspace(-3.0, 3.0, 600)


def friction(c, v, eps, qdot_arr):
    coulomb  = c * np.tanh(qdot_arr / eps)
    viscous  = v * qdot_arr
    total    = coulomb + viscous
    return viscous, coulomb, total


def main() -> None:
    fig, axes = plt.subplots(1, 5, figsize=(11, 3.2),
                             sharey=False, constrained_layout=True)

    for ji, (ax, jname, color) in enumerate(zip(axes, JOINT_NAMES, JOINT_COLORS)):
        c   = COULOMB_NM[ji]
        v   = VISCOUS_NM[ji]
        vis, coul, tot = friction(c, v, FRICTION_EPS, qdot)

        ax.plot(qdot, tot,  color=color, linewidth=1.6, linestyle="-",  label="Total $\\tau_f$", zorder=4)
        ax.plot(qdot, vis,  color=color, linewidth=0.9, linestyle="--", label="Viscous $v\\dot{q}$", alpha=0.75, zorder=3)
        ax.plot(qdot, coul, color=color, linewidth=0.9, linestyle=":",  label="Coulomb $c\\tanh$", alpha=0.75, zorder=3)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="-", alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="-", alpha=0.4)

        ax.set_title(jname, fontsize=8.5)
        ax.set_xlabel(r"$\dot{q}$ (rad/s)", fontsize=8)
        if ji == 0:
            ax.set_ylabel(r"$\tau_f$ (N$\cdot$m)", fontsize=8)

        # Annotate constants
        ax.text(0.97, 0.95,
                f"$c$ = {c:.4f}\n$v$ = {v:.4f}\n$\\varepsilon$ = {FRICTION_EPS:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
                bbox=dict(fc="white", alpha=0.6, ec="gray", boxstyle="round,pad=0.25"))

        ax.tick_params(labelsize=7)

    # Common legend on last panel
    axes[-1].legend(fontsize=6.5, loc="upper left", framealpha=0.7)

    fig.suptitle(r"Calibrated Friction Model $\tau_f(\dot{q}) = c\tanh(\dot{q}/\varepsilon) + v\dot{q}$ per Joint",
                 fontsize=9.5)
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
