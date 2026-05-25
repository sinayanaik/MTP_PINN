#!/usr/bin/env python3
"""fig10 — Torque tracking over one test trajectory: all five joints.

Five stacked panels, one per joint; each overlays measured truth + FNN +
Physics-Reg. + EDR. Joint identity is the y-axis label (no titles, no (a)/(b)
letters). Which trajectory: set ``trajectory_select`` (None = auto, int = index,
or a geometry name), e.g.
``run_all.py --only fig10 --config-override trajectory_select=helix``. Plotted
lines are Savitzky–Golay smoothed by default; raw samples are in
tables/trajectory_tracking.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from shared import palette
from shared.config import default_config, replace
from shared.figio import save_pdf
from shared.plotting import champion_results, maybe_smooth, select_trajectory, top_legend
from shared.style import apply_style

# ============================ TWEAKABLES (edit me) ============================
FIG_W, FIG_H      = 8.4, 11.0        # tall: 5 stacked joint panels
DPI_SAVE          = 300
ARCH_COLORS       = {"fnn": "#4C72B0", "physreg": "#55A868", "edr": "#C44E52"}
EMPHASIS_SCALE    = 1.35             # EDR prediction-line thickness multiplier
LINE_W            = 2.4
TRUTH_COLOR       = "#222222"        # measured torque
TRUTH_LINE_W      = 2.0
PRED_ALPHA        = 0.9
AXES_LABEL_SIZE   = 16.0
TICK_SIZE         = 14.0
LEGEND_SIZE       = 16.0            # ← increased from 14
LEGEND_ANCHOR_Y   = 1.02
TRAJECTORY_SELECT = None             # None=auto | int index | geometry name
SAVGOL_ENABLED    = True
SAVGOL_WINDOW     = 31               # ← increased from 7 for much heavier smoothing
SAVGOL_POLYORDER  = 3                # ← increased from 2 for smoother curves
GRID_ON           = True
SAMPLING_HZ       = 300.0            # robot feedback sampling rate (≈305 Hz)
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 arch_colors=dict(ARCH_COLORS), emphasis_line_scale=EMPHASIS_SCALE,
                 line_w=LINE_W, truth_color=TRUTH_COLOR,
                 truth_linewidth=TRUTH_LINE_W, axes_label_size=AXES_LABEL_SIZE,
                 tick_label_size=TICK_SIZE, legend_size=LEGEND_SIZE,
                 legend_anchor_y=LEGEND_ANCHOR_Y, trajectory_select=TRAJECTORY_SELECT,
                 savgol_enabled=SAVGOL_ENABLED, savgol_window=SAVGOL_WINDOW,
                 savgol_polyorder=SAVGOL_POLYORDER)


def main(cfg=CONFIG):
    apply_style(cfg)
    res = champion_results(cfg)
    archs = palette.ordered_archs(cfg)
    s, e, geom = select_trajectory(res["edr"]["traj"], cfg)
    n_samples = e - s
    # Convert sample indices to time in seconds
    t = np.arange(n_samples) / SAMPLING_HZ
    nj = len(cfg.joint_names)

    fig, axes = plt.subplots(nj, 1, figsize=(cfg.fig_w, cfg.fig_h), sharex=True)
    for j in range(nj):
        ax = axes[j]
        ax.plot(t, maybe_smooth(res["edr"]["target"][s:e, j], cfg),
                color=cfg.truth_color, lw=cfg.truth_linewidth, zorder=2)
        for a in archs:
            ax.plot(t, maybe_smooth(res[a]["pred"][s:e, j], cfg),
                    color=palette.color(cfg, a), lw=palette.line_width(cfg, a),
                    alpha=PRED_ALPHA, zorder=palette.zorder(cfg, a))
        ax.set_ylabel(f"{cfg.joint_names[j]} ({cfg.torque_unit})")
        ax.grid(GRID_ON)
    axes[-1].set_xlabel(r"$t\;(\mathrm{s})$" + f"  —  test trajectory: {geom}",
                        fontsize=cfg.axes_label_size)

    handles = [Line2D([0], [0], color=cfg.truth_color,
                      lw=cfg.truth_linewidth, label="Measured")]
    handles += [Line2D([0], [0], color=palette.color(cfg, a), lw=LINE_W,
                       label=palette.label(cfg, a)) for a in archs]
    fig.align_ylabels(axes)
    fig.tight_layout()
    top_legend(fig, handles, cfg, ncol=len(handles))
    return save_pdf(fig, "fig10_trajectory_tracking", cfg)


if __name__ == "__main__":
    print(main())
