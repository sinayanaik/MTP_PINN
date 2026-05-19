#!/usr/bin/env python3
"""fig07 — Over-fitting gap (val − train RMSE) vs epoch, 3 champions.

A gap that grows with training = over-fitting. EDR's gap stays small and
flat; the FNN's widens. Curves are Savitzky–Golay smoothed by default (raw
per-epoch values in tables/training_curves.csv).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
        gap = (h["val_rmse"] - h["train_rmse"]).to_numpy()
        ax.plot(h["epoch"].to_numpy(), maybe_smooth(gap, cfg),
                color=palette.color(cfg, a), lw=palette.line_width(cfg, a),
                zorder=palette.zorder(cfg, a))

    ax.axhline(0.0, color=cfg.reference_color, ls=":", lw=1.4, zorder=1)
    ax.set_xlabel(cfg.epoch_label)
    ax.set_ylabel("Over-fitting gap: val − train RMSE (N·m)")
    ax.grid(True)
    top_legend(ax, arch_proxy_handles(cfg, archs, kind="line"), cfg)
    return save_pdf(fig, "fig07_overfitting_gap_vs_epoch", cfg)


if __name__ == "__main__":
    print(main())
