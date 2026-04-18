"""
Plot 05 — Per-Joint RMSE Heatmap (Models × Joints)
===================================================
Color: RdYlGn_r (red=bad, green=good). Annotated with values.
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
plt.style.use("science")

SCRIPT_DIR  = Path(__file__).resolve().parent
PLOTS_DIR   = SCRIPT_DIR.parent
TRAINED_DIR = PLOTS_DIR.parent.parent / "Trained_Models"
REGISTRY    = TRAINED_DIR / "models_registry.yaml"
OUT_FILE    = PLOTS_DIR / "05_joint_rmse_heatmap.png"

DISPLAY = {
    "BlackBoxFNN":                 "Black-Box FNN (A)",
    "PhysicsRegularizedFNN":       "Physics-Regularized FNN (B)",
    "ResidualCorrectionFNN":       "Residual Correction FNN (C)",
    "LagrangianStructuredFNN":     "Lagrangian Structured FNN (D)",
    "EquationConstrainedPINNFNN":  "Eq.-Constrained PINN (E.1)",
    "DecomposedStructuredPINNFNN": "Decomposed Structured PINN (E.2)",
}
MODEL_ORDER = list(DISPLAY.keys())
JOINT_NAMES = ["J1 yaw", "J2 shoulder", "J3 elbow", "J4 wrist", "J5 roll"]


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


def get_per_joint(m: dict, key: str) -> list[float]:
    pj = (m.get("metrics") or {}).get("per_joint_rmse") if key == "rmse" else None
    if pj and len(pj) == 5:
        return [float(v) for v in pj]
    run_dir = Path(m.get("run_dir") or "")
    sidecar = run_dir / "metadata.yaml"
    if sidecar.exists():
        try:
            with sidecar.open() as f:
                sc = yaml.safe_load(f) or {}
            lst = (sc.get("metrics") or {}).get(key) or []
            if len(lst) == 5:
                return [float(v) for v in lst]
        except Exception:
            pass
    return [np.nan] * 5


def main() -> None:
    best = load_best(REGISTRY)
    models = [m for m in MODEL_ORDER if m in best]
    matrix = np.array([get_per_joint(best[m], "rmse") for m in models])

    vmin, vmax = np.nanmin(matrix) * 0.95, np.nanmax(matrix) * 1.05

    fig, ax = plt.subplots(figsize=(6, 3.2))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r"RMSE (N$\cdot$m)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xticks(range(5))
    ax.set_xticklabels(JOINT_NAMES, fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY[m] for m in models], fontsize=7.5)

    for i in range(len(models)):
        for j in range(5):
            v = matrix[i, j]
            if not np.isnan(v):
                # Dark text on light cells, white on dark
                norm_v = (v - vmin) / (vmax - vmin + 1e-9)
                txt_color = "black" if norm_v < 0.6 else "white"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                        fontsize=7, color=txt_color, fontweight="bold")

    ax.set_title("Per-Joint Test RMSE Heatmap — Best Run per Model", fontsize=9)
    ax.set_xlabel("Joint", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
