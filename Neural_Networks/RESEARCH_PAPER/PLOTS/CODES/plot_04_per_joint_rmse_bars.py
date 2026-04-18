"""
Plot 04 — Per-Joint RMSE Grouped Bar Chart
==========================================
6 model groups, 5 bars each (J1–J5).
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

SCRIPT_DIR   = Path(__file__).resolve().parent
PLOTS_DIR    = SCRIPT_DIR.parent
TRAINED_DIR  = PLOTS_DIR.parent.parent / "Trained_Models"
REGISTRY     = TRAINED_DIR / "models_registry.yaml"
OUT_FILE     = PLOTS_DIR / "04_per_joint_rmse_bars.png"

DISPLAY = {
    "BlackBoxFNN":                 "BB (A)",
    "PhysicsRegularizedFNN":       "PR (B)",
    "ResidualCorrectionFNN":       "RC (C)",
    "LagrangianStructuredFNN":     "Lagr (D)",
    "EquationConstrainedPINNFNN":  "EC (E.1)",
    "DecomposedStructuredPINNFNN": "Dc (E.2)",
}
MODEL_ORDER  = list(DISPLAY.keys())
JOINT_NAMES  = ["J1 yaw", "J2 shoulder", "J3 elbow", "J4 wrist", "J5 roll"]
JOINT_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759"]


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


def get_per_joint_rmse(m: dict) -> list[float]:
    # Try registry per_joint_rmse first
    pj = (m.get("metrics") or {}).get("per_joint_rmse") or []
    if len(pj) == 5:
        return [float(v) for v in pj]
    # Fallback: sidecar metadata.yaml
    run_dir = Path(m.get("run_dir") or "")
    sidecar = run_dir / "metadata.yaml"
    if sidecar.exists():
        try:
            with sidecar.open() as f:
                sc = yaml.safe_load(f) or {}
            rmse_list = (sc.get("metrics") or {}).get("rmse") or []
            if len(rmse_list) == 5:
                return [float(v) for v in rmse_list]
        except Exception:
            pass
    return [np.nan] * 5


def main() -> None:
    best = load_best(REGISTRY)
    models = [m for m in MODEL_ORDER if m in best]
    per_joint = {mtype: get_per_joint_rmse(best[mtype]) for mtype in models}

    n_models = len(models)
    n_joints = 5
    x = np.arange(n_models)
    total_w = 0.8
    bar_w   = total_w / n_joints
    offsets = np.linspace(-(total_w - bar_w) / 2, (total_w - bar_w) / 2, n_joints)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    for ji in range(n_joints):
        vals = [per_joint[mtype][ji] for mtype in models]
        bars = ax.bar(x + offsets[ji], vals, width=bar_w * 0.9,
                      color=JOINT_COLORS[ji], label=JOINT_NAMES[ji],
                      edgecolor="white", linewidth=0.4, alpha=0.88)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{v:.4f}", ha="center", va="bottom",
                        fontsize=5, rotation=80)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in models], fontsize=8.5)
    ax.set_ylabel(r"Test RMSE (N$\cdot$m, normalised)")
    ax.set_title("Per-Joint RMSE by Model — Best Run", fontsize=9)
    ax.legend(title="Joint", fontsize=7, title_fontsize=7,
              ncol=5, loc="upper right", framealpha=0.7)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
