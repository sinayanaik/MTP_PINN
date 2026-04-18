"""
Plot 09 — Physics Weight (w_p) Schedule
========================================
4-panel subplot: one per physics-aware model.
Shows w_p vs epoch from training_history.csv.
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
OUT_FILE    = PLOTS_DIR / "09_physics_weight_schedule.png"

PHYSICS_MODELS = [
    "PhysicsRegularizedFNN",
    "EquationConstrainedPINNFNN",
    "LagrangianStructuredFNN",
    "DecomposedStructuredPINNFNN",
]
DISPLAY = {
    "PhysicsRegularizedFNN":       "Physics-Regularized FNN (B)",
    "EquationConstrainedPINNFNN":  "Eq.-Constrained PINN (E.1)",
    "LagrangianStructuredFNN":     "Lagrangian Structured FNN (D)",
    "DecomposedStructuredPINNFNN": "Decomposed Structured PINN (E.2)",
}
COLORS = {
    "PhysicsRegularizedFNN":       "#f28e2b",
    "EquationConstrainedPINNFNN":  "#e15759",
    "LagrangianStructuredFNN":     "#b07aa1",
    "DecomposedStructuredPINNFNN": "#76b7b2",
}


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

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), constrained_layout=True)

    for pi, mtype in enumerate(PHYSICS_MODELS):
        ax    = axes[pi]
        m     = best.get(mtype)
        color = COLORS[mtype]

        if m is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(DISPLAY.get(mtype, mtype), fontsize=7)
            continue

        run_dir  = Path(m.get("run_dir") or "")
        csv_path = run_dir / "training_history.csv"

        if not csv_path.exists():
            ax.text(0.5, 0.5, "No CSV", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(DISPLAY[mtype], fontsize=7)
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            ax.set_title(DISPLAY[mtype], fontsize=7)
            continue

        epochs = df["epoch"].values if "epoch" in df.columns else np.arange(len(df))

        if "w_p" in df.columns:
            ax.plot(epochs, df["w_p"].values, color=color, linewidth=1.5, zorder=3)

            # Fill under curve
            ax.fill_between(epochs, df["w_p"].values, alpha=0.18, color=color)

            # Peak annotation
            max_wp  = df["w_p"].max()
            max_ep  = epochs[df["w_p"].argmax()]
            ax.axhline(max_wp, color=color, linewidth=0.6, linestyle="--", alpha=0.5)
            ax.text(max_ep + 2, max_wp + 0.005, f"peak={max_wp:.3f}",
                    fontsize=6.5, color=color, va="bottom")

        if "w_d" in df.columns:
            ax.plot(epochs, df["w_d"].values, color="gray", linewidth=0.8,
                    linestyle=":", alpha=0.7, label=r"$w_d$")

        ax.set_title(DISPLAY[mtype], fontsize=7)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel(r"Weight $w_p$", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_ylim(bottom=0)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=1.5, label=r"Physics weight $w_p$"),
        Line2D([0], [0], color="gray", linewidth=0.8, linestyle=":", alpha=0.7,
               label=r"Data weight $w_d$"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=7, framealpha=0.7, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(r"Physics Weight $w_p$ Schedule During Training — Latest Run per Model",
                 fontsize=9.5)
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
