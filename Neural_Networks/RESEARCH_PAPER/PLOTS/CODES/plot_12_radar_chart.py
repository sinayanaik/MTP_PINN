"""
Plot 12 — Radar / Spider Chart
================================
6 performance axes for all 6 models overlaid.
Axes: RMSE_inv, MAE_inv, NRMSE_inv, R², Pearson r, Speed (1/time_norm).
Journal style using scienceplots.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml

import scienceplots  # noqa: F401
plt.style.use("science")

SCRIPT_DIR  = Path(__file__).resolve().parent
PLOTS_DIR   = SCRIPT_DIR.parent
TRAINED_DIR = PLOTS_DIR.parent.parent / "Trained_Models"
REGISTRY    = TRAINED_DIR / "models_registry.yaml"
OUT_FILE    = PLOTS_DIR / "12_radar_chart.png"

DISPLAY = {
    "BlackBoxFNN":                 "Black-Box FNN (A)",
    "PhysicsRegularizedFNN":       "Physics-Reg. FNN (B)",
    "ResidualCorrectionFNN":       "Residual Corr. FNN (C)",
    "LagrangianStructuredFNN":     "Lagrangian FNN (D)",
    "EquationConstrainedPINNFNN":  "EC-PINN (E.1)",
    "DecomposedStructuredPINNFNN": "Decomposed PINN (E.2)",
}
COLORS = {
    "BlackBoxFNN":                 "#4e79a7",
    "PhysicsRegularizedFNN":       "#f28e2b",
    "ResidualCorrectionFNN":       "#59a14f",
    "LagrangianStructuredFNN":     "#b07aa1",
    "EquationConstrainedPINNFNN":  "#e15759",
    "DecomposedStructuredPINNFNN": "#76b7b2",
}
MODEL_ORDER  = list(DISPLAY.keys())
AXIS_LABELS  = ["RMSE↓ (inv.)", "MAE↓ (inv.)", "NRMSE↓ (inv.)", r"$R^2$↑", "Pearson ↑", "Speed↑"]


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


def get_raw_metrics(m: dict) -> dict:
    mt = m.get("metrics") or {}
    tr = m.get("training") or {}
    rmse  = mt.get("test_rmse_mean") or mt.get("avg_rmse_mean")
    nrmse = mt.get("test_nrmse_mean") or mt.get("avg_nrmse_mean")
    r2    = mt.get("avg_r2_overall")
    pearson = mt.get("avg_pearson_mean")
    time  = tr.get("time_seconds")

    # Sidecar for MAE and finer metrics
    mae = None
    run_dir = Path(m.get("run_dir") or "")
    sidecar = run_dir / "metadata.yaml"
    if sidecar.exists():
        try:
            with sidecar.open() as f:
                sc = yaml.safe_load(f) or {}
            mae_list = (sc.get("metrics") or {}).get("mae")
            if mae_list:
                mae = float(np.mean(mae_list))
            if r2 is None:
                r2 = sc.get("metrics", {}).get("r2_overall") or sc.get("metrics", {}).get("avg_r2")
            if pearson is None:
                pr_list = (sc.get("metrics") or {}).get("pearson_r")
                if pr_list:
                    pearson = float(np.mean(pr_list))
        except Exception:
            pass

    return {
        "rmse":    float(rmse)    if rmse    is not None else np.nan,
        "mae":     float(mae)     if mae     is not None else np.nan,
        "nrmse":   float(nrmse)   if nrmse   is not None else np.nan,
        "r2":      float(r2)      if r2      is not None else np.nan,
        "pearson": float(pearson) if pearson is not None else np.nan,
        "time":    float(time)    if time    is not None else np.nan,
    }


def normalise_radar(all_raw: dict[str, dict]) -> dict[str, np.ndarray]:
    """Normalise all metrics to [0, 1], direction = higher is better."""
    keys = list(all_raw.keys())

    def col(field):
        return np.array([all_raw[k][field] for k in keys], dtype=float)

    rmse    = col("rmse");    r_rmse   = 1.0 - (rmse    - np.nanmin(rmse))   / (np.nanmax(rmse)   - np.nanmin(rmse) + 1e-12)
    mae     = col("mae");     r_mae    = 1.0 - (mae     - np.nanmin(mae))    / (np.nanmax(mae)    - np.nanmin(mae)  + 1e-12)
    nrmse   = col("nrmse");   r_nrmse  = 1.0 - (nrmse   - np.nanmin(nrmse))  / (np.nanmax(nrmse)  - np.nanmin(nrmse)+ 1e-12)
    r2      = col("r2");      r_r2     = (r2      - np.nanmin(r2))     / (np.nanmax(r2)     - np.nanmin(r2)  + 1e-12)
    pearson = col("pearson"); r_pearson= (pearson  - np.nanmin(pearson))/ (np.nanmax(pearson)  - np.nanmin(pearson)+1e-12)
    time    = col("time");    r_speed  = 1.0 - (time    - np.nanmin(time))   / (np.nanmax(time)   - np.nanmin(time) + 1e-12)

    result = {}
    for i, k in enumerate(keys):
        result[k] = np.array([r_rmse[i], r_mae[i], r_nrmse[i],
                               r_r2[i], r_pearson[i], r_speed[i]])
    return result


def radar_plot(ax, values, color, label, alpha=0.18):
    n = len(values)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]
    vals   = list(values) + [values[0]]

    ax.plot(angles, vals, color=color, linewidth=1.3, zorder=3, label=label)
    ax.fill(angles, vals, color=color, alpha=alpha, zorder=2)


def main() -> None:
    best    = load_best(REGISTRY)
    raw     = {mtype: get_raw_metrics(best[mtype]) for mtype in MODEL_ORDER if mtype in best}
    normed  = normalise_radar(raw)
    models  = [m for m in MODEL_ORDER if m in normed]

    n = len(AXIS_LABELS)
    angles = [2 * math.pi * i / n for i in range(n)]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(AXIS_LABELS, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.6)

    for mtype in models:
        radar_plot(ax, normed[mtype], COLORS[mtype], DISPLAY[mtype])

    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18),
              fontsize=7.5, framealpha=0.8)
    ax.set_title("Normalised Multi-Metric Radar Chart\n(outer edge = best)",
                 fontsize=9, pad=20)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
