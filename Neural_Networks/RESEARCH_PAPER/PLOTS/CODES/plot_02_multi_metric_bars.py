"""
Plot 02 — Multi-Metric Bar Chart (Best Run per Model)
======================================================
4 metrics side-by-side: RMSE, MAE, NRMSE, (1 - R²).
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
OUT_FILE   = PLOTS_DIR / "02_multi_metric_bars.png"

DISPLAY = {
    "BlackBoxFNN":                 "Black-Box\nFNN (A)",
    "PhysicsRegularizedFNN":       "Phys.-Reg.\nFNN (B)",
    "ResidualCorrectionFNN":       "Residual\nCorr. (C)",
    "LagrangianStructuredFNN":     "Lagrangian\nFNN (D)",
    "EquationConstrainedPINNFNN":  "EC-PINN\n(E.1)",
    "DecomposedStructuredPINNFNN": "Decomposed\nPINN (E.2)",
}
MODEL_ORDER = list(DISPLAY.keys())
METRIC_LABELS = ["RMSE (N·m)", "MAE (N·m)", "NRMSE", r"$1 - R^2$"]
METRIC_KEYS   = ["rmse", "mae", "nrmse", "1mr2"]
HATCHES       = ["", "///", "...", "xxx"]


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


def get_metrics(m: dict) -> dict[str, float]:
    mt = m.get("metrics") or {}
    rmse  = float(mt.get("test_rmse_mean") or mt.get("avg_rmse_mean") or np.nan)
    nrmse = float(mt.get("test_nrmse_mean") or mt.get("avg_nrmse_mean") or np.nan)
    r2    = float(mt.get("avg_r2_overall") or np.nan)

    # Load sidecar metadata.yaml for MAE
    run_dir = Path(m.get("run_dir") or "")
    mae = np.nan
    sidecar = run_dir / "metadata.yaml"
    if sidecar.exists():
        try:
            with sidecar.open() as f:
                sc = yaml.safe_load(f) or {}
            mae_list = (sc.get("metrics") or {}).get("mae")
            if mae_list:
                mae = float(np.mean(mae_list))
        except Exception:
            pass

    return {
        "rmse": rmse,
        "mae":  mae,
        "nrmse": nrmse,
        "1mr2": 1.0 - r2 if not np.isnan(r2) else np.nan,
    }


def main() -> None:
    best = load_best(REGISTRY)
    models = [m for m in MODEL_ORDER if m in best]
    metric_data = {k: [] for k in METRIC_KEYS}

    for mtype in models:
        ms = get_metrics(best[mtype])
        for k in METRIC_KEYS:
            metric_data[k].append(ms[k])

    n_models  = len(models)
    n_metrics = len(METRIC_KEYS)
    x = np.arange(n_models)
    total_w = 0.75
    bar_w   = total_w / n_metrics
    offsets = np.linspace(-(total_w - bar_w) / 2, (total_w - bar_w) / 2, n_metrics)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 3.8))

    for mi, (key, label, hatch) in enumerate(zip(METRIC_KEYS, METRIC_LABELS, HATCHES)):
        vals = metric_data[key]
        xpos = x + offsets[mi]
        bars = ax.bar(xpos, vals, width=bar_w, label=label, color=cmap(mi),
                      hatch=hatch, edgecolor="white", linewidth=0.5, alpha=0.88)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=5.5, rotation=70)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in models], fontsize=7.5)
    ax.set_ylabel("Metric Value")
    ax.set_title("Multi-Metric Comparison — Best Run per Model", fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc="upper right", framealpha=0.7)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
