"""
Plot 11 — Epoch Utilisation (Epochs Used vs. Max)
==================================================
Latest run per model. Red = early stopped, Green = completed full run.
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
OUT_FILE    = PLOTS_DIR / "11_epoch_utilisation.png"

DISPLAY = {
    "BlackBoxFNN":                 "Black-Box FNN (A)",
    "PhysicsRegularizedFNN":       "Physics-Regularized FNN (B)",
    "ResidualCorrectionFNN":       "Residual Correction FNN (C)",
    "LagrangianStructuredFNN":     "Lagrangian Structured FNN (D)",
    "EquationConstrainedPINNFNN":  "Eq.-Constrained PINN (E.1)",
    "DecomposedStructuredPINNFNN": "Decomposed Structured PINN (E.2)",
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
    models = [m for m in MODEL_ORDER if m in best]

    names, epochs_ran, epochs_max, stopped, colors = [], [], [], [], []
    for mtype in models:
        m  = best[mtype]
        tr = m.get("training") or {}
        ep_ran = tr.get("epochs_ran") or tr.get("epochs_trained")
        ep_max = tr.get("epochs_max")
        early  = tr.get("stopped_early", False)

        if ep_ran is None or ep_max is None:
            # Try sidecar
            run_dir = Path(m.get("run_dir") or "")
            sc = run_dir / "metadata.yaml"
            if sc.exists():
                try:
                    with sc.open() as f:
                        sd = yaml.safe_load(f) or {}
                    ep_ran = sd.get("epochs_trained") or ep_ran
                    ep_max = sd.get("hyperparams", {}).get("epochs") or ep_max
                    early  = sd.get("stopped_early", early)
                except Exception:
                    pass

        if ep_ran is None:
            continue
        if ep_max is None:
            ep_max = ep_ran

        names.append(DISPLAY[mtype])
        epochs_ran.append(int(ep_ran))
        epochs_max.append(int(ep_max))
        stopped.append(bool(early))
        colors.append("#ef4444" if early else "#22c55e")

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    # Max bars (light background)
    ax.barh(x, epochs_max, color="lightgray", edgecolor="white",
            linewidth=0.6, height=0.55, zorder=1, label="Max epochs")

    # Actual bars
    bars = ax.barh(x, epochs_ran, color=colors, edgecolor="white",
                   linewidth=0.6, height=0.55, zorder=2, alpha=0.85,
                   label="Epochs trained")

    # Utilisation %
    for bar, ran, mx, s in zip(bars, epochs_ran, epochs_max, stopped):
        pct = 100 * ran / mx if mx > 0 else 0
        label_txt = f"{ran}/{mx}  ({pct:.0f}%)"
        if s:
            label_txt += "  [early stop]"
        ax.text(mx + 5, bar.get_y() + bar.get_height() / 2,
                label_txt, va="center", ha="left", fontsize=7)

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Epochs")
    ax.set_title("Epoch Utilisation — Latest Run per Model", fontsize=9)
    ax.set_xlim(0, max(epochs_max) * 1.35)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="lightgray", edgecolor="gray", label="Max configured epochs"),
        Patch(facecolor="#22c55e",   edgecolor="white", label="Trained (full run)"),
        Patch(facecolor="#ef4444",   edgecolor="white", label="Trained (early stop)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right", framealpha=0.7)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
