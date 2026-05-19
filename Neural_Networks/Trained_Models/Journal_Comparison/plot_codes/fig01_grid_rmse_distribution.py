#!/usr/bin/env python3
"""fig01 — Test-RMSE distribution across the full grid sweep, per architecture.

The champion is the starred minimum; the whole violin shows spread + median.
EDR's cloud sits lowest *and* tightest. (Distribution plot — Savitzky–Golay
does not apply; the raw grid statistics are also in tables/grid_runs.csv.)
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

CONFIG = replace(default_config(), fig_w=7.0, fig_h=4.8)


def main(cfg=CONFIG):
    apply_style(cfg)
    gdf = dataio.grid_df()
    gdf = gdf[gdf["status"] == "ok"]
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
            body.set_alpha(0.28)
            body.set_edgecolor(c)
        xj = i + rng.uniform(-0.06, 0.06, size=len(vals))
        ax.plot(xj, vals, "o", ms=6.0, mfc=c, mec="white", mew=0.6,
                alpha=0.75, zorder=3)
        med = float(np.median(vals))
        ax.hlines(med, i - 0.22, i + 0.22, color="black", lw=2.0, zorder=4)
        best = float(vals.min())
        ax.plot([i], [best], marker="*", ms=18, mfc=c, mec="black",
                mew=0.8, zorder=5)
        ax.annotate(f"{best:.4f}", (i, best), textcoords="offset points",
                    xytext=(12, -2), ha="left", va="center",
                    fontsize=cfg.annot_size)

    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([palette.label(cfg, a) for a in archs])
    ax.set_ylabel(cfg.rmse_label)
    ax.set_xlim(-0.6, len(archs) - 0.4)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    handles = arch_proxy_handles(cfg, archs, kind="patch")
    handles += [
        Line2D([0], [0], color="black", lw=2.0, label="median"),
        Line2D([0], [0], marker="*", ls="none", ms=14, mfc="0.5",
               mec="black", label="champion"),
    ]
    top_legend(ax, handles, cfg)
    return save_pdf(fig, "fig01_grid_rmse_distribution", cfg)


if __name__ == "__main__":
    print(main())
