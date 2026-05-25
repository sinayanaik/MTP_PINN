#!/usr/bin/env python3
"""fig02 — Data efficiency: test RMSE vs amount of training data.

One curve per architecture across the data-efficiency sweep. EDR holds a low,
nearly flat test RMSE even at 10 % of the training data, where the black-box
FNN is markedly worse; Physics-Reg. is comparable to EDR at high budgets but
relies on far more parameters. Plotted line is Savitzky–Golay smoothed by
default (raw values in tables/data_efficiency.csv).
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
EMPHASIS_SCALE    = 1.35              # EDR line thickness multiplier
LINE_W            = 2.4
MARKER_SIZE       = 8.0
AXES_LABEL_SIZE   = 16.0
TICK_SIZE         = 14.0
LEGEND_SIZE       = 14.0
LEGEND_ANCHOR_Y   = 1.02
X_LABEL           = "Training data used (%)"
Y_LABEL           = "RMSE (N·m)"
BAND_ALPHA        = 0.15             # ±1σ-over-seeds shaded band
SAVGOL_ENABLED    = True            # smooth the drawn line (table stays raw)
SAVGOL_WINDOW     = 7
SAVGOL_POLYORDER  = 2
GRID_ON           = True
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 arch_colors=dict(ARCH_COLORS), emphasis_line_scale=EMPHASIS_SCALE,
                 line_w=LINE_W, marker_size=MARKER_SIZE,
                 axes_label_size=AXES_LABEL_SIZE, tick_label_size=TICK_SIZE,
                 legend_size=LEGEND_SIZE, legend_anchor_y=LEGEND_ANCHOR_Y,
                 rmse_label=Y_LABEL, savgol_enabled=SAVGOL_ENABLED,
                 savgol_window=SAVGOL_WINDOW, savgol_polyorder=SAVGOL_POLYORDER)


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
        if std is not None:
            s = std.to_numpy()
            if s.any():
                ax.fill_between(frac, mean - s, mean + s, color=c,
                                alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.plot(frac, y, color=c, lw=palette.line_width(cfg, a),
                marker=palette.marker(cfg, a), ms=cfg.marker_size,
                zorder=palette.zorder(cfg, a))

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(cfg.rmse_label)
    ax.grid(GRID_ON)
    top_legend(ax, arch_proxy_handles(cfg, archs, kind="line"), cfg)
    return save_pdf(fig, "fig02_data_efficiency_rmse", cfg)


if __name__ == "__main__":
    print(main())
