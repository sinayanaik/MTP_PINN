#!/usr/bin/env python3
"""fig02 — Data efficiency: test RMSE vs amount of training data.

One curve per architecture. EDR holds ~0.090 N·m even at 10 % of the data,
where the black-box FNN is markedly worse. Plotted line is Savitzky–Golay
smoothed by default (raw values in tables/data_efficiency.csv).
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

CONFIG = replace(default_config(), fig_w=7.4, fig_h=4.8)


def main(cfg=CONFIG):
    apply_style(cfg)
    sw = dataio.sweep_df()
    archs = palette.ordered_archs(cfg)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for a in archs:
        d = sw[sw["arch"] == a].sort_values("data_train_fraction")
        frac = d["data_train_fraction"].to_numpy() * 100.0
        mean = d["test_rmse"].to_numpy()
        std = d.get("test_rmse_std")
        y = maybe_smooth(mean, cfg)
        c = palette.color(cfg, a)
        # ±1σ-over-seeds band: shows PhysReg is flat *within noise* rather
        # than spuriously "increasing" (single-seed artefact, now averaged).
        if std is not None:
            s = std.to_numpy()
            if s.any():
                ax.fill_between(frac, mean - s, mean + s, color=c,
                                alpha=0.15, lw=0, zorder=1)
        ax.plot(frac, y, color=c,
                lw=palette.line_width(cfg, a), marker=palette.marker(cfg, a),
                ms=cfg.marker_size, zorder=palette.zorder(cfg, a))

    ax.set_xlabel("Training data used (%)")
    ax.set_ylabel(cfg.rmse_label)
    ax.grid(True)
    top_legend(ax, arch_proxy_handles(cfg, archs, kind="line"), cfg)
    return save_pdf(fig, "fig02_data_efficiency_rmse", cfg)


if __name__ == "__main__":
    print(main())
