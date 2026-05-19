#!/usr/bin/env python3
"""fig11 — Torque tracking over one test trajectory: all five joints.

Five stacked panels, one per joint; each overlays measured truth + FNN +
Physics-Reg. + EDR. Joint identity is the y-axis label (no titles/letters).

Which trajectory: set ``trajectory_select`` (None=auto, int=index, or a
geometry name). E.g. ``run_all.py --only fig11 --config-override
trajectory_select=helix``. Plotted lines are Savitzky–Golay smoothed by
default; raw samples are in tables/trajectory_tracking.csv.
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

CONFIG = replace(default_config(), fig_w=8.4, fig_h=11.0)


def main(cfg=CONFIG):
    apply_style(cfg)
    res = champion_results(cfg)
    archs = palette.ordered_archs(cfg)
    s, e, geom = select_trajectory(res["edr"]["traj"], cfg)
    t = np.arange(e - s)
    nj = len(cfg.joint_names)

    fig, axes = plt.subplots(nj, 1, figsize=(cfg.fig_w, cfg.fig_h),
                             sharex=True)
    for j in range(nj):
        ax = axes[j]
        ax.plot(t, maybe_smooth(res["edr"]["target"][s:e, j], cfg),
                color=cfg.truth_color, lw=cfg.truth_linewidth, zorder=2)
        for a in archs:
            ax.plot(t, maybe_smooth(res[a]["pred"][s:e, j], cfg),
                    color=palette.color(cfg, a),
                    lw=palette.line_width(cfg, a), alpha=0.9,
                    zorder=palette.zorder(cfg, a))
        ax.set_ylabel(f"{cfg.joint_names[j]} ({cfg.torque_unit})")
        ax.grid(True)
    axes[-1].set_xlabel(f"Sample (test trajectory: {geom})")

    handles = [Line2D([0], [0], color=cfg.truth_color,
                      lw=cfg.truth_linewidth, label="measured")]
    handles += [Line2D([0], [0], color=palette.color(cfg, a), lw=2.4,
                       label=palette.label(cfg, a)) for a in archs]
    fig.align_ylabels(axes)
    fig.tight_layout()
    top_legend(fig, handles, cfg, ncol=len(handles))
    return save_pdf(fig, "fig11_trajectory_tracking", cfg)


if __name__ == "__main__":
    print(main())
