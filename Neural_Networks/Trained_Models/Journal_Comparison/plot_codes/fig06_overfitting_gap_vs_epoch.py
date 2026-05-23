#!/usr/bin/env python3
"""fig06 — Over-fitting gap (val − train RMSE) vs epoch, 3 champions.

A gap that grows with training signals over-fitting. This isolates the *trend*
that fig05 only shows implicitly: EDR's gap stays small and flat, while the
black-box FNN's widens as it memorises the training set. Curves are
Savitzky–Golay smoothed by default (raw per-epoch values in
tables/training_curves.csv).
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
FIG_W, FIG_H      = 7.6, 4.8
DPI_SAVE          = 300
ARCH_COLORS       = {"fnn": "#4C72B0", "physreg": "#55A868", "edr": "#C44E52"}
EMPHASIS_SCALE    = 1.35
LINE_W            = 2.4
AXES_LABEL_SIZE   = 16.0
TICK_SIZE         = 14.0
LEGEND_SIZE       = 14.0
LEGEND_ANCHOR_Y   = 1.02
X_LABEL           = "Epoch"
Y_LABEL           = "Over-fitting gap: val − train RMSE (N·m)"
ZERO_LINE_COLOR   = "#888888"
ZERO_LINE_LW      = 1.4
SAVGOL_ENABLED    = True
SAVGOL_WINDOW     = 7
SAVGOL_POLYORDER  = 2
GRID_ON           = True
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 arch_colors=dict(ARCH_COLORS), emphasis_line_scale=EMPHASIS_SCALE,
                 line_w=LINE_W, axes_label_size=AXES_LABEL_SIZE,
                 tick_label_size=TICK_SIZE, legend_size=LEGEND_SIZE,
                 legend_anchor_y=LEGEND_ANCHOR_Y, epoch_label=X_LABEL,
                 reference_color=ZERO_LINE_COLOR, savgol_enabled=SAVGOL_ENABLED,
                 savgol_window=SAVGOL_WINDOW, savgol_polyorder=SAVGOL_POLYORDER)


def main(cfg=CONFIG):
    apply_style(cfg)
    champs = dataio.champions(cfg)
    archs = palette.ordered_archs(cfg)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for a in archs:
        h = dataio.load_history(champs[a]["run_dir"])
        gap = (h["val_rmse"] - h["train_rmse"]).to_numpy()
        ax.plot(h["epoch"].to_numpy(), maybe_smooth(gap, cfg),
                color=palette.color(cfg, a), lw=palette.line_width(cfg, a),
                zorder=palette.zorder(cfg, a))

    ax.axhline(0.0, color=cfg.reference_color, ls=":", lw=ZERO_LINE_LW, zorder=1)
    ax.set_xlabel(cfg.epoch_label)
    ax.set_ylabel(Y_LABEL)
    ax.grid(GRID_ON)
    top_legend(ax, arch_proxy_handles(cfg, archs, kind="line"), cfg)
    return save_pdf(fig, "fig06_overfitting_gap_vs_epoch", cfg)


if __name__ == "__main__":
    print(main())
