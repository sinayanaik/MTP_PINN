#!/usr/bin/env python3
"""fig10 — Per-joint R² heatmap (architecture × joint).

Green = best (highest R²), red = worst, scaled globally across the whole
matrix. EDR's row is the greenest overall. Raw values in
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

CONFIG = replace(default_config(), fig_w=7.6, fig_h=3.6)


def main(cfg=CONFIG):
    apply_style(cfg)
    res = champion_results(cfg)
    archs = palette.ordered_archs(cfg)
    mat = np.array([res[a]["metrics"]["r2"] for a in archs], float)
    rows = [palette.label(cfg, a) for a in archs]

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    heatmap(ax, mat, rows, list(cfg.joint_names), cfg,
            lower_is_better=False, value_fmt="{:.3f}",
            cbar_label="Per-joint R²")
    ax.set_xlabel("Joint")
    ax.set_ylabel("Architecture")
    return save_pdf(fig, "fig10_per_joint_r2_heatmap", cfg)


if __name__ == "__main__":
    print(main())
