#!/usr/bin/env python3
"""fig03 — Generalization gap (test − val RMSE) vs amount of training data.

Smaller is better: a model whose test error tracks its validation error
generalizes. EDR's gap stays lowest at every data budget; the FNN's gap is
the widest. Plotted line is Savitzky–Golay smoothed by default (raw values in
tables/data_efficiency.csv).
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
        gap = (d["test_rmse"] - d["val_rmse"]).to_numpy()
        y = maybe_smooth(gap, cfg)
        c = palette.color(cfg, a)
        std = d.get("test_rmse_std")
        if std is not None:
            s = std.to_numpy()
            if s.any():
                ax.fill_between(frac, gap - s, gap + s, color=c,
                                alpha=0.15, lw=0, zorder=1)
        ax.plot(frac, y, color=c,
                lw=palette.line_width(cfg, a), marker=palette.marker(cfg, a),
                ms=cfg.marker_size, zorder=palette.zorder(cfg, a))

    ax.axhline(0.0, color=cfg.reference_color, ls=":", lw=1.4, zorder=1)
    ax.set_xlabel("Training data used (%)")
    ax.set_ylabel("Generalization gap: test − val RMSE (N·m)")
    ax.grid(True)
    top_legend(ax, arch_proxy_handles(cfg, archs, kind="line"), cfg)
    return save_pdf(fig, "fig03_generalization_gap_vs_fraction", cfg)


if __name__ == "__main__":
    print(main())
