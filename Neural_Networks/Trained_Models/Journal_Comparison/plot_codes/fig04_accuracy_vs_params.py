#!/usr/bin/env python3
"""fig04 — Model size per architecture (champion trainable-parameter count).

One horizontal bar per family = its champion's trainable-parameter count on a
log axis, annotated with that champion's test RMSE. EDR reaches ~0.0902 N·m with
~13 k parameters, versus ~70 k (FNN, 0.0972) and ~549 k (Physics-Reg., 0.0896):
roughly 5× and 41× fewer parameters at comparable accuracy. (Bar chart — no
Savitzky–Golay; raw values in tables/cost_accuracy.csv.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from shared import dataio, palette
from shared.config import default_config, replace
from shared.figio import save_pdf
from shared.style import apply_style

# ============================ TWEAKABLES (edit me) ============================
FIG_W, FIG_H      = 7.4, 3.4
DPI_SAVE          = 300
ARCH_COLORS       = {"fnn": "#4C72B0", "physreg": "#55A868", "edr": "#C44E52"}
AXES_LABEL_SIZE   = 16.0
TICK_SIZE         = 14.0
ANNOT_SIZE        = 13.0           # param-count + RMSE label at each bar end
X_LABEL           = "Trainable parameters (log scale)"
BAR_HEIGHT        = 0.55
BAR_ALPHA         = 0.9
BAR_BASE          = None           # bar left edge; None = one decade below smallest
RMSE_FMT          = "RMSE {:.4f}"  # champion accuracy annotation
GRID_ON           = True
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 arch_colors=dict(ARCH_COLORS), axes_label_size=AXES_LABEL_SIZE,
                 tick_label_size=TICK_SIZE, annot_size=ANNOT_SIZE)


def _fmt_params(p: float) -> str:
    if p >= 1e6:
        return f"{p / 1e6:.2f}M"
    if p >= 1e3:
        return f"{p / 1e3:.1f}k"
    return f"{int(p)}"


def main(cfg=CONFIG):
    apply_style(cfg)
    champs = dataio.champions(cfg)
    archs = palette.ordered_archs(cfg)
    params = [dataio.param_count(champs[a]["run_dir"]) for a in archs]
    rmses = [champs[a]["test_rmse"] for a in archs]

    # Log-scale bars need a positive base; default to one decade below the
    # smallest model so every bar length is proportional to its order of magnitude.
    base = BAR_BASE or 10.0 ** (np.floor(np.log10(min(params))) - 1)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for i, (a, p, e) in enumerate(zip(archs, params, rmses)):
        c = palette.color(cfg, a)
        ax.barh(i, p - base, left=base, height=BAR_HEIGHT, color=c,
                alpha=BAR_ALPHA, edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(f"{_fmt_params(p)}   {RMSE_FMT.format(e)}", (p, i),
                    textcoords="offset points", xytext=(8, 0), ha="left",
                    va="center", fontsize=cfg.annot_size, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(base, max(params) * 6.0)         # headroom for the end labels
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([palette.label(cfg, a) for a in archs])
    ax.invert_yaxis()                            # first arch on top, like a table
    ax.set_xlabel(X_LABEL)
    ax.grid(axis="x", visible=GRID_ON)
    ax.grid(axis="y", visible=False)
    return save_pdf(fig, "fig04_accuracy_vs_params", cfg)


if __name__ == "__main__":
    print(main())
