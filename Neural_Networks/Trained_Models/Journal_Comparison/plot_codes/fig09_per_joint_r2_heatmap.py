#!/usr/bin/env python3
"""fig09 — Per-joint R² heatmap (architecture × joint).

Green = best (highest R²), red = worst, scaled globally across the whole
matrix. Shows how much per-joint torque variance each champion explains; the
greener a row, the better that model fits across joints. Raw values in
tables/per_joint_metrics.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from shared import palette
from shared.config import default_config, replace
from shared.figio import save_pdf
from shared.plotting import champion_results, heatmap
from shared.style import apply_style

# ============================ TWEAKABLES (edit me) ============================
FIG_W, FIG_H      = 6.4, 2.8       # compact, near-square cells (aspect=equal)
DPI_SAVE          = 300
AXES_LABEL_SIZE   = 16.0
TICK_SIZE         = 14.0
ANNOT_SIZE        = 12.0
X_LABEL           = "Joint"
Y_LABEL           = ""             # arch is obvious from the row labels (FNN/EDR…)
CBAR_LABEL        = "Per-joint R²"
VALUE_FMT         = "{:.3f}"
HEATMAP_BEST      = "#1A9850"      # colour of the best (highest R²) cell
HEATMAP_MID       = "#FFFFBF"
HEATMAP_WORST     = "#D73027"      # colour of the worst (lowest R²) cell
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 axes_label_size=AXES_LABEL_SIZE, tick_label_size=TICK_SIZE,
                 annot_size=ANNOT_SIZE, heatmap_best=HEATMAP_BEST,
                 heatmap_mid=HEATMAP_MID, heatmap_worst=HEATMAP_WORST)


def main(cfg=CONFIG):
    apply_style(cfg)
    res = champion_results(cfg)
    archs = palette.ordered_archs(cfg)
    mat = np.array([res[a]["metrics"]["r2"] for a in archs], float)
    rows = [palette.label(cfg, a) for a in archs]

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    heatmap(ax, mat, rows, list(cfg.joint_names), cfg,
            lower_is_better=False, value_fmt=VALUE_FMT, cbar_label=CBAR_LABEL)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    return save_pdf(fig, "fig09_per_joint_r2_heatmap", cfg)


if __name__ == "__main__":
    print(main())
