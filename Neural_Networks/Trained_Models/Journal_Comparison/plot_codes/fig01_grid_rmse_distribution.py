#!/usr/bin/env python3
"""fig01 — Test-RMSE distribution across the full architecture/HP grid.

One violin per architecture over each family's best 50 full-data runs (frac =
1.0, ranked by test RMSE): the black bar is the median, the star is the family's
best run (annotated). Trimming to the top 50 keeps the families comparable; EDR's
champion (star, ~0.0902 N·m) reaches Physics-Reg.'s low end using an order of
magnitude fewer parameters, while FNN sits clearly higher. (Distribution plot — no
Savitzky–Golay; the raw grid statistics are in tables/grid_runs.csv and
tables/grid_summary.csv.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from shared import dataio, palette
from shared.config import default_config, replace
from shared.figio import save_pdf
from shared.plotting import arch_proxy_handles, top_legend
from shared.style import apply_style

# ============================ TWEAKABLES (edit me) ============================
FIG_W, FIG_H      = 7.0, 4.8          # figure size, inches
DPI_SAVE          = 300               # output resolution
ARCH_COLORS       = {"fnn": "#4C72B0", "physreg": "#55A868", "edr": "#C44E52"}
AXES_LABEL_SIZE   = 16.0              # axis-title font size
TICK_SIZE         = 14.0             # tick-label font size
LEGEND_SIZE       = 14.0             # legend font size
ANNOT_SIZE        = 12.0            # champion value annotation size
LEGEND_ANCHOR_Y   = 1.02            # legend strip height above axes
Y_LABEL           = "Test RMSE (N·m)"
VIOLIN_WIDTH      = 0.75            # violin body width
VIOLIN_ALPHA      = 0.28            # violin fill opacity
POINT_SIZE        = 6.0            # jittered run markers
POINT_ALPHA       = 0.75
JITTER            = 0.06           # horizontal scatter of points
MEDIAN_LW         = 2.0
STAR_SIZE         = 18             # champion marker
ANNOT_FMT         = "{:.4f}"      # champion value format
TOP_N             = 50           # show only each family's best-N runs (by test RMSE)
GRID_ON           = True
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 arch_colors=dict(ARCH_COLORS), axes_label_size=AXES_LABEL_SIZE,
                 tick_label_size=TICK_SIZE, legend_size=LEGEND_SIZE,
                 annot_size=ANNOT_SIZE, legend_anchor_y=LEGEND_ANCHOR_Y,
                 violin_width=VIOLIN_WIDTH, rmse_label=Y_LABEL)


def main(cfg=CONFIG):
    apply_style(cfg)
    gdf = dataio.grid_df()
    gdf = gdf[gdf["status"] == "ok"]
    # Keep only each family's best TOP_N runs (by test RMSE) so the violins stay
    # legible and the families are compared on a comparable number of runs.
    gdf = gdf.sort_values("test_rmse").groupby("arch", sort=False).head(TOP_N)
    archs = palette.ordered_archs(cfg)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for i, a in enumerate(archs):
        vals = gdf.loc[gdf["arch"] == a, "test_rmse"].to_numpy(dtype=float)
        c = palette.color(cfg, a)
        vp = ax.violinplot(vals, positions=[i], widths=cfg.violin_width,
                           showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(c)
            body.set_alpha(VIOLIN_ALPHA)
            body.set_edgecolor(c)
        xj = i + rng.uniform(-JITTER, JITTER, size=len(vals))
        ax.plot(xj, vals, "o", ms=POINT_SIZE, mfc=c, mec="white", mew=0.6,
                alpha=POINT_ALPHA, zorder=3)
        med = float(np.median(vals))
        ax.hlines(med, i - 0.22, i + 0.22, color="black", lw=MEDIAN_LW, zorder=4)
        best = float(vals.min())
        ax.plot([i], [best], marker="*", ms=STAR_SIZE, mfc=c, mec="black",
                mew=0.8, zorder=5)
        ax.annotate(ANNOT_FMT.format(best), (i, best),
                    textcoords="offset points", xytext=(12, -2), ha="left",
                    va="center", fontsize=cfg.annot_size)

    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([palette.label(cfg, a) for a in archs])
    ax.set_ylabel(cfg.rmse_label)
    ax.set_xlim(-0.6, len(archs) - 0.4)
    ax.grid(axis="y", visible=GRID_ON)
    ax.grid(axis="x", visible=False)

    handles = arch_proxy_handles(cfg, archs, kind="patch")
    handles += [
        Line2D([0], [0], color="black", lw=MEDIAN_LW, label="median"),
        Line2D([0], [0], marker="*", ls="none", ms=14, mfc="0.5",
               mec="black", label="champion"),
    ]
    top_legend(ax, handles, cfg)
    return save_pdf(fig, "fig01_grid_rmse_distribution", cfg)


if __name__ == "__main__":
    print(main())
