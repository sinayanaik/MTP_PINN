#!/usr/bin/env python3
"""fig06 — Train (dashed) vs validation (solid) RMSE per epoch, 3 champions.

Curves staying together = healthy training. The FNN drives train far below
val (over-fitting); EDR keeps them close. Curves are Savitzky–Golay smoothed
by default (raw per-epoch values in tables/training_curves.csv).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt

from shared import dataio, palette
from shared.config import default_config, replace
from shared.figio import save_pdf
from shared.plotting import arch_proxy_handles, maybe_smooth, top_legend
from shared.style import apply_style

CONFIG = replace(default_config(), fig_w=7.6, fig_h=4.8)


def main(cfg=CONFIG):
    apply_style(cfg)
    champs = dataio.champions(cfg.champion_metric)
    archs = palette.ordered_archs(cfg)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for a in archs:
        h = dataio.load_history(champs[a]["run_dir"])
        c = palette.color(cfg, a)
        ep = h["epoch"].to_numpy()
        ax.plot(ep, maybe_smooth(h["val_rmse"].to_numpy(), cfg), color=c,
                lw=palette.line_width(cfg, a), ls="-",
                zorder=palette.zorder(cfg, a))
        ax.plot(ep, maybe_smooth(h["train_rmse"].to_numpy(), cfg), color=c,
                lw=1.6, ls="--", zorder=palette.zorder(cfg, a))

    ax.set_xlabel(cfg.epoch_label)
    ax.set_ylabel("RMSE (N·m)")
    ax.grid(True)

    handles = arch_proxy_handles(cfg, archs, kind="line")
    handles += [
        Line2D([0], [0], color="0.35", lw=2.2, ls="-", label="validation"),
        Line2D([0], [0], color="0.35", lw=1.6, ls="--", label="train"),
    ]
    top_legend(ax, handles, cfg, ncol=len(handles))
    return save_pdf(fig, "fig06_train_val_curves", cfg)


if __name__ == "__main__":
    print(main())
