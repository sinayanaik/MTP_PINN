#!/usr/bin/env python3
"""fig05 — Train (dashed) vs validation (solid) RMSE per epoch, 3 champions.

Train and validation curves staying together signal healthy training. The FNN
drives train well below validation (over-fitting); EDR keeps the two close,
reflecting its physics-constrained capacity. Curves are Savitzky–Golay smoothed
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

# ============================ TWEAKABLES (edit me) ============================
FIG_W, FIG_H      = 7.6, 4.8
DPI_SAVE          = 300
ARCH_COLORS       = {"fnn": "#4C72B0", "physreg": "#55A868", "edr": "#C44E52"}
EMPHASIS_SCALE    = 1.35             # EDR validation-line thickness multiplier
VAL_LINE_W        = 2.4             # validation (solid) base width
TRAIN_LINE_W      = 1.6            # train (dashed) width
AXES_LABEL_SIZE   = 16.0
TICK_SIZE         = 14.0
LEGEND_SIZE       = 14.0
LEGEND_ANCHOR_Y   = 1.02
X_LABEL           = "Epoch"
Y_LABEL           = "RMSE (N·m)"
SAVGOL_ENABLED    = True
SAVGOL_WINDOW     = 7
SAVGOL_POLYORDER  = 2
GRID_ON           = True
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 arch_colors=dict(ARCH_COLORS), emphasis_line_scale=EMPHASIS_SCALE,
                 line_w=VAL_LINE_W, axes_label_size=AXES_LABEL_SIZE,
                 tick_label_size=TICK_SIZE, legend_size=LEGEND_SIZE,
                 legend_anchor_y=LEGEND_ANCHOR_Y, epoch_label=X_LABEL,
                 savgol_enabled=SAVGOL_ENABLED, savgol_window=SAVGOL_WINDOW,
                 savgol_polyorder=SAVGOL_POLYORDER)


def main(cfg=CONFIG):
    apply_style(cfg)
    champs = dataio.champions(cfg)
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
                lw=TRAIN_LINE_W, ls="--", zorder=palette.zorder(cfg, a))

    ax.set_xlabel(cfg.epoch_label)
    ax.set_ylabel(Y_LABEL)
    ax.grid(GRID_ON)

    handles = arch_proxy_handles(cfg, archs, kind="line")
    handles += [
        Line2D([0], [0], color="0.35", lw=VAL_LINE_W, ls="-", label="validation"),
        Line2D([0], [0], color="0.35", lw=TRAIN_LINE_W, ls="--", label="train"),
    ]
    top_legend(ax, handles, cfg, ncol=len(handles))
    return save_pdf(fig, "fig05_train_val_curves", cfg)


if __name__ == "__main__":
    print(main())
