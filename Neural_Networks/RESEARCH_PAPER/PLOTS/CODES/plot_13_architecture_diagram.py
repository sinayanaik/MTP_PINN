"""
Plot 13 — Architecture Diagrams (2×3 grid)
==========================================
Pure matplotlib block-diagram sketches for each model.
No external images. Shows data-flow and loss terms.
Journal style using scienceplots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

import scienceplots  # noqa: F401
plt.style.use("science")
plt.rcParams["text.usetex"] = False  # architecture diagram uses unicode/special chars

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR  = SCRIPT_DIR.parent
OUT_FILE   = PLOTS_DIR / "13_architecture_diagram.png"

# ── Colour palette ─────────────────────────────────────────────────────────
PANEL_COLORS = {
    "BlackBoxFNN":                 "#4e79a7",
    "PhysicsRegularizedFNN":       "#f28e2b",
    "ResidualCorrectionFNN":       "#59a14f",
    "LagrangianStructuredFNN":     "#b07aa1",
    "EquationConstrainedPINNFNN":  "#e15759",
    "DecomposedStructuredPINNFNN": "#76b7b2",
}
MODEL_ORDER = list(PANEL_COLORS.keys())
DISPLAY = {
    "BlackBoxFNN":                 "(A) Black-Box FNN",
    "PhysicsRegularizedFNN":       "(B) Physics-Regularized FNN",
    "ResidualCorrectionFNN":       "(C) Residual Correction FNN",
    "LagrangianStructuredFNN":     "(D) Lagrangian Structured FNN",
    "EquationConstrainedPINNFNN":  "(E.1) Equation-Constrained PINN",
    "DecomposedStructuredPINNFNN": "(E.2) Decomposed Structured PINN",
}


# ── Drawing helpers ────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, label, fc="#dbeafe", ec="#1e3a5f", fs=6.5, bold=False):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.02", fc=fc, ec=ec, linewidth=0.8, zorder=3)
    ax.add_patch(rect)
    fw = "bold" if bold else "normal"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, fontweight=fw, zorder=4)


def arrow(ax, x0, y0, x1, y1, color="gray", lw=0.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=2)


def txt(ax, x, y, s, fs=6, c="gray", ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=c, zorder=5)


def loss_box(ax, cx, cy, label, fc="#fef9c3", ec="#a16207", fs=6):
    box(ax, cx, cy, 1.5, 0.25, label, fc=fc, ec=ec, fs=fs)


def setup_ax(ax, title, color):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=7.5, pad=3,
                 bbox=dict(fc=color, ec="none", alpha=0.18, boxstyle="round,pad=0.3"))


# ── Per-model diagrams ─────────────────────────────────────────────────────

def draw_blackbox(ax):
    setup_ax(ax, DISPLAY["BlackBoxFNN"], PANEL_COLORS["BlackBoxFNN"])
    # Input
    box(ax, 1.0, 2.0, 1.2, 0.45, "x=[q,dq,ddq] in R^15", fc="#eff6ff")
    arrow(ax, 1.6, 2.0, 2.1, 2.0)
    # MLP
    box(ax, 2.9, 2.0, 1.4, 0.5, "MLP theta\n[256-512-256]", fc="#dbeafe", bold=True)
    arrow(ax, 3.6, 2.0, 4.1, 2.0)
    # Output
    box(ax, 4.5, 2.0, 0.8, 0.45, "tau_hat", fc="#eff6ff")
    # Loss
    txt(ax, 2.9, 0.8, r"$\mathcal{L}$ = MSE($\hat{\tau}$, $\tau_{meas}$)", fs=7, c="#1e40af")
    txt(ax, 2.9, 0.5, "No physics", fs=6.5, c="gray")


def draw_physreg(ax):
    setup_ax(ax, DISPLAY["PhysicsRegularizedFNN"], PANEL_COLORS["PhysicsRegularizedFNN"])
    box(ax, 1.0, 2.5, 1.2, 0.4, "x in R^15", fc="#fff7ed")
    arrow(ax, 1.6, 2.5, 2.1, 2.5)
    box(ax, 2.9, 2.5, 1.4, 0.45, "MLP theta\n[256-512-256]", fc="#fed7aa", bold=True)
    arrow(ax, 3.6, 2.5, 4.1, 2.5)
    box(ax, 4.5, 2.5, 0.8, 0.4, "tau_hat", fc="#fff7ed")
    # Physics side
    box(ax, 1.0, 1.2, 1.2, 0.4, "RNEA(q,dq,ddq)", fc="#fde68a")
    arrow(ax, 1.6, 1.2, 2.1, 1.2)
    box(ax, 2.9, 1.2, 1.4, 0.4, "Calib. phi(tau_nom)", fc="#fde68a")
    # Loss
    txt(ax, 2.9, 0.45, r"$\mathcal{L}=w_d\mathcal{L}_{data}+w_p\mathcal{L}_{phys}$", fs=6.8, c="#92400e")
    txt(ax, 2.9, 0.18, "Curriculum schedule w_p (rising)", fs=6, c="gray")


def draw_residual(ax):
    setup_ax(ax, DISPLAY["ResidualCorrectionFNN"], PANEL_COLORS["ResidualCorrectionFNN"])
    box(ax, 0.9, 3.2, 1.0, 0.38, "x in R^15", fc="#f0fdf4")
    box(ax, 0.9, 2.4, 1.1, 0.38, "tau_phys", fc="#dcfce7")
    # Concat
    box(ax, 2.4, 2.8, 0.8, 0.4, "concat\nin R^20", fc="#bbf7d0")
    arrow(ax, 1.4, 3.2, 1.9, 2.9)
    arrow(ax, 1.45, 2.4, 1.9, 2.7)
    arrow(ax, 2.8, 2.8, 3.1, 2.8)
    # Encoder
    box(ax, 3.5, 2.8, 0.8, 0.45, "Encoder\nMLP", fc="#86efac", bold=True)
    arrow(ax, 3.9, 2.8, 4.3, 2.8)
    box(ax, 4.6, 2.8, 0.7, 0.45, "tau_hat", fc="#f0fdf4")
    # Output detail
    txt(ax, 4.6, 2.2, "=softplus(a)*tau_phys+d", fs=6, c="#166534")
    # Loss
    txt(ax, 2.5, 0.65, r"$\mathcal{L}=\mathcal{L}_{data}+\lambda_\alpha\|\alpha-1\|^2$", fs=6.8, c="#166534")


def draw_lagrangian(ax):
    setup_ax(ax, DISPLAY["LagrangianStructuredFNN"], PANEL_COLORS["LagrangianStructuredFNN"])
    # 4 sub-networks
    boxes = [
        (0.9, 3.2, "M-net\n(q->L)", "#f3e8ff"),
        (0.9, 2.4, "C-net\n(q,dq->tC)", "#e9d5ff"),
        (0.9, 1.6, "g-net\n(q->tg)",    "#d8b4fe"),
        (0.9, 0.8, "f-net\n(dq->tf)",   "#c4b5fd"),
    ]
    for bx, by, lbl, fc in boxes:
        box(ax, bx, by, 1.1, 0.36, lbl, fc=fc, fs=6.2)

    # Inertia arrow
    arrow(ax, 1.45, 3.2, 2.0, 2.9)
    txt(ax, 1.75, 3.1, "LL' -> M", fs=5.5, c="#7c3aed")

    # Summation
    box(ax, 2.8, 2.0, 0.5, 0.4, "Sum", fc="#ede9fe", fs=9, bold=True)
    for _, by, _, _ in boxes:
        arrow(ax, 1.45, by, 2.5, 2.0)

    arrow(ax, 3.05, 2.0, 3.5, 2.0)
    box(ax, 3.9, 2.0, 0.7, 0.4, "tau_hat", fc="#f5f3ff")

    txt(ax, 2.8, 0.35, r"$\mathcal{L}=\mathcal{L}_{data}+\lambda_s\mathcal{L}_{SPD}+\lambda_f\mathcal{L}_{fric}$",
        fs=6, c="#6d28d9")


def draw_ecpinn(ax):
    setup_ax(ax, DISPLAY["EquationConstrainedPINNFNN"], PANEL_COLORS["EquationConstrainedPINNFNN"])
    box(ax, 1.0, 2.6, 1.1, 0.4, "x in R^15", fc="#fff1f2")
    arrow(ax, 1.55, 2.6, 2.1, 2.6)
    box(ax, 2.9, 2.6, 1.4, 0.45, "MLP theta\n[256->512->256]", fc="#fecdd3", bold=True)
    arrow(ax, 3.6, 2.6, 4.1, 2.6)
    box(ax, 4.5, 2.6, 0.8, 0.4, "tau_hat", fc="#fff1f2")

    # Physics components
    box(ax, 1.5, 1.5, 1.6, 0.38, "tM+tC+tg+tf", fc="#fca5a5")
    arrow(ax, 2.3, 1.5, 2.7, 1.5)
    box(ax, 3.2, 1.5, 0.8, 0.38, "phi(t_nom)", fc="#fca5a5")

    # Residual
    txt(ax, 3.0, 0.85, r"$r=\hat{\tau}-\varphi(\tau_{nom})$", fs=6.5, c="#9f1239")
    txt(ax, 2.9, 0.50,
        r"$\mathcal{L}=w_d\mathcal{L}_{data}+w_p\mathcal{L}_{phys}+\lambda_{col}\mathcal{L}_{col}$",
        fs=5.8, c="#9f1239")


def draw_decomposed(ax):
    setup_ax(ax, DISPLAY["DecomposedStructuredPINNFNN"], PANEL_COLORS["DecomposedStructuredPINNFNN"])
    # 4 sub-networks
    boxes = [
        (0.85, 3.3, "M-net SPD", "#ccfbf1"),
        (0.85, 2.55, "dC-net", "#99f6e4"),
        (0.85, 1.8,  "dg-net",  "#5eead4"),
        (0.85, 1.05, "df-net",  "#2dd4bf"),
    ]
    for bx, by, lbl, fc in boxes:
        box(ax, bx, by, 1.1, 0.35, lbl, fc=fc, fs=6.2)

    # Nominal physics
    box(ax, 2.3, 2.0, 0.9, 0.38, "t_nom\n(RNEA)", fc="#fde68a", fs=6)

    # Sum
    box(ax, 3.3, 2.0, 0.5, 0.4, "Sum", fc="#ccfbf1", fs=9, bold=True)
    for bx, by, _, _ in boxes:
        arrow(ax, 1.4, by, 3.1, 2.0)
    arrow(ax, 2.75, 2.0, 3.05, 2.0)

    arrow(ax, 3.55, 2.0, 3.95, 2.0)
    box(ax, 4.25, 2.0, 0.55, 0.4, "tau_hat", fc="#f0fdfa")

    txt(ax, 2.6, 0.4,
        r"$\mathcal{L}=\mathcal{L}_{data}+\lambda_s\mathcal{L}_{SPD}+\lambda_c\mathcal{L}_{corr}+\lambda_n\mathcal{L}_{nom}$",
        fs=5.5, c="#0f766e")


# ── Main ──────────────────────────────────────────────────────────────────

DRAWERS = {
    "BlackBoxFNN":                 draw_blackbox,
    "PhysicsRegularizedFNN":       draw_physreg,
    "ResidualCorrectionFNN":       draw_residual,
    "LagrangianStructuredFNN":     draw_lagrangian,
    "EquationConstrainedPINNFNN":  draw_ecpinn,
    "DecomposedStructuredPINNFNN": draw_decomposed,
}


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    axes_flat = axes.flatten()

    for pi, mtype in enumerate(MODEL_ORDER):
        DRAWERS[mtype](axes_flat[pi])

    fig.suptitle("Neural Network Architecture Diagrams — All 6 Models", fontsize=10)
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
