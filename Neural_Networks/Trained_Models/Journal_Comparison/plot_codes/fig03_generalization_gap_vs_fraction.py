#!/usr/bin/env python3
"""fig03 — Generalization gap (test − val RMSE) vs amount of training data.

Smaller is better: a model whose test error tracks its validation error
generalizes. EDR keeps the smallest gap across data budgets, the clearest
expression of its built-in physics inductive bias; the black-box FNN's gap is
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

# ============================ TWEAKABLES (edit me) ============================
FIG_W, FIG_H      = 7.4, 4.8
DPI_SAVE          = 300
ARCH_COLORS       = {"fnn": "#4C72B0", "physreg": "#55A868", "edr": "#C44E52"}
EMPHASIS_SCALE    = 1.35
LINE_W            = 2.4
MARKER_SIZE       = 8.0
AXES_LABEL_SIZE   = 16.0
TICK_SIZE         = 14.0
LEGEND_SIZE       = 14.0
LEGEND_ANCHOR_Y   = 1.02
X_LABEL           = "Training data used (%)"
Y_LABEL           = "Generalization gap: test − val RMSE (N·m)"
ZERO_LINE_COLOR   = "#888888"        # reference line at gap = 0
ZERO_LINE_LW      = 1.4
BAND_ALPHA        = 0.15
SAVGOL_ENABLED    = True
SAVGOL_WINDOW     = 7
SAVGOL_POLYORDER  = 2
GRID_ON           = True
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 arch_colors=dict(ARCH_COLORS), emphasis_line_scale=EMPHASIS_SCALE,
                 line_w=LINE_W, marker_size=MARKER_SIZE,
                 axes_label_size=AXES_LABEL_SIZE, tick_label_size=TICK_SIZE,
                 legend_size=LEGEND_SIZE, legend_anchor_y=LEGEND_ANCHOR_Y,
                 reference_color=ZERO_LINE_COLOR, savgol_enabled=SAVGOL_ENABLED,
                 savgol_window=SAVGOL_WINDOW, savgol_polyorder=SAVGOL_POLYORDER)


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
                                alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.plot(frac, y, color=c, lw=palette.line_width(cfg, a),
                marker=palette.marker(cfg, a), ms=cfg.marker_size,
                zorder=palette.zorder(cfg, a))

    ax.axhline(0.0, color=cfg.reference_color, ls=":", lw=ZERO_LINE_LW, zorder=1)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(GRID_ON)
    top_legend(ax, arch_proxy_handles(cfg, archs, kind="line"), cfg)
    return save_pdf(fig, "fig03_generalization_gap_vs_fraction", cfg)


if __name__ == "__main__":
    print(main())
