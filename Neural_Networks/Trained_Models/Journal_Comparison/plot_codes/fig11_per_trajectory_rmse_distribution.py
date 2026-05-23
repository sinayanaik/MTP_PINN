#!/usr/bin/env python3
"""fig11 — Per-trajectory test-RMSE distribution for the three champions.

One violin per architecture over the trajectory-macro RMSE of every test
trajectory (each point is one held-out motion). Shows *consistency across
motions* — distinct from fig01, which spreads over grid runs of one fixed test
set. The black bar is the median, the star/annotation is the worst (highest)
trajectory. A tighter, lower cloud means the model degrades less on hard
motions. (Distribution plot — no Savitzky–Golay; raw values in
tables/per_trajectory_rmse.csv.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from shared import palette
from shared.config import default_config, replace
from shared.figio import save_pdf
from shared.plotting import arch_proxy_handles, champion_results, per_traj_rmse, top_legend
from shared.style import apply_style

# ============================ TWEAKABLES (edit me) ============================
FIG_W, FIG_H      = 7.0, 4.8
DPI_SAVE          = 300
ARCH_COLORS       = {"fnn": "#4C72B0", "physreg": "#55A868", "edr": "#C44E52"}
AXES_LABEL_SIZE   = 16.0
TICK_SIZE         = 14.0
LEGEND_SIZE       = 14.0
ANNOT_SIZE        = 12.0
LEGEND_ANCHOR_Y   = 1.02
Y_LABEL           = "Per-trajectory RMSE (N·m)"
VIOLIN_WIDTH      = 0.75
VIOLIN_ALPHA      = 0.28
POINT_SIZE        = 5.0           # one marker per test trajectory
POINT_ALPHA       = 0.55
JITTER            = 0.07
MEDIAN_LW         = 2.0
WORST_STAR_SIZE   = 16            # marks the hardest trajectory per arch
ANNOT_FMT         = "{:.3f}"
SHOW_WORST        = True          # star + annotate each arch's worst trajectory
GRID_ON           = True
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 arch_colors=dict(ARCH_COLORS), axes_label_size=AXES_LABEL_SIZE,
                 tick_label_size=TICK_SIZE, legend_size=LEGEND_SIZE,
                 annot_size=ANNOT_SIZE, legend_anchor_y=LEGEND_ANCHOR_Y,
                 violin_width=VIOLIN_WIDTH)


def main(cfg=CONFIG):
    apply_style(cfg)
    res = champion_results(cfg)
    archs = palette.ordered_archs(cfg)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for i, a in enumerate(archs):
        vals = np.asarray(per_traj_rmse(res[a]["pred"], res[a]["target"],
                                        res[a]["traj"]), float)
        c = palette.color(cfg, a)
        vp = ax.violinplot(vals, positions=[i], widths=cfg.violin_width,
                           showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(c)
            body.set_alpha(VIOLIN_ALPHA)
            body.set_edgecolor(c)
        xj = i + rng.uniform(-JITTER, JITTER, size=len(vals))
        ax.plot(xj, vals, "o", ms=POINT_SIZE, mfc=c, mec="white", mew=0.4,
                alpha=POINT_ALPHA, zorder=3)
        med = float(np.median(vals))
        ax.hlines(med, i - 0.22, i + 0.22, color="black", lw=MEDIAN_LW, zorder=4)
        if SHOW_WORST:
            worst = float(vals.max())
            ax.plot([i], [worst], marker="*", ms=WORST_STAR_SIZE, mfc=c,
                    mec="black", mew=0.8, zorder=5)
            ax.annotate(ANNOT_FMT.format(worst), (i, worst),
                        textcoords="offset points", xytext=(12, -2),
                        ha="left", va="center", fontsize=cfg.annot_size)

    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([palette.label(cfg, a) for a in archs])
    ax.set_ylabel(Y_LABEL)
    ax.set_xlim(-0.6, len(archs) - 0.4)
    ax.grid(axis="y", visible=GRID_ON)
    ax.grid(axis="x", visible=False)

    handles = arch_proxy_handles(cfg, archs, kind="patch")
    handles += [
        Line2D([0], [0], color="black", lw=MEDIAN_LW, label="median"),
        Line2D([0], [0], marker="*", ls="none", ms=13, mfc="0.5",
               mec="black", label="worst trajectory"),
    ]
    top_legend(ax, handles, cfg)
    return save_pdf(fig, "fig11_per_trajectory_rmse_distribution", cfg)


if __name__ == "__main__":
    print(main())
