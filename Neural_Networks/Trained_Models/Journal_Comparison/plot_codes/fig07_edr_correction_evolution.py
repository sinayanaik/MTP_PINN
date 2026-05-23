#!/usr/bin/env python3
"""fig07 — What EDR learns: magnitude of each physics correction vs epoch.

EDR adds a learned residual to every analytical term; tracking |δg|, ‖δM‖_F,
|δC·q̇| and |δτ_f| over training shows which part of the rigid-body model it
leans on most — interpretability the black-box baselines cannot offer. Curves
are Savitzky–Golay smoothed by default (raw values in
tables/edr_correction_evolution.csv). EDR-only figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

from shared import dataio
from shared.config import default_config, replace
from shared.figio import save_pdf
from shared.plotting import maybe_smooth, top_legend
from shared.style import apply_style

# ============================ TWEAKABLES (edit me) ============================
FIG_W, FIG_H      = 7.4, 4.8
DPI_SAVE          = 300
AXES_LABEL_SIZE   = 16.0
TICK_SIZE         = 14.0
LEGEND_SIZE       = 14.0
LEGEND_ANCHOR_Y   = 1.02
LEGEND_NCOL       = 4
X_LABEL           = "Epoch"
Y_LABEL           = "Mean correction magnitude (N·m)"
LINE_W            = 2.4
LOG_Y             = True            # corrections span orders of magnitude
SAVGOL_ENABLED    = True
SAVGOL_WINDOW     = 7
SAVGOL_POLYORDER  = 2
GRID_ON           = True
# Per-term (history column, legend label, colour):
TERMS = [
    ("mean_abs_delta_g",     r"$|\delta g|$",          "#C44E52"),
    ("mean_frob_delta_M",    r"$\|\delta M\|_F$",      "#4C72B0"),
    ("mean_abs_delta_C_qd",  r"$|\delta C\,\dot q|$",  "#55A868"),
    ("mean_abs_delta_tau_f", r"$|\delta \tau_f|$",     "#8172B3"),
]
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 axes_label_size=AXES_LABEL_SIZE, tick_label_size=TICK_SIZE,
                 legend_size=LEGEND_SIZE, legend_anchor_y=LEGEND_ANCHOR_Y,
                 epoch_label=X_LABEL, savgol_enabled=SAVGOL_ENABLED,
                 savgol_window=SAVGOL_WINDOW, savgol_polyorder=SAVGOL_POLYORDER)


def main(cfg=CONFIG):
    apply_style(cfg)
    edr = dataio.champions(cfg)["edr"]
    h = dataio.load_history(edr["run_dir"])
    ep = h["epoch"].to_numpy()

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    handles = []
    for col, lbl, c in TERMS:
        (ln,) = ax.plot(ep, maybe_smooth(h[col].to_numpy(), cfg), color=c,
                        lw=LINE_W, label=lbl)
        handles.append(ln)
    if LOG_Y:
        ax.set_yscale("log")
    ax.set_xlabel(cfg.epoch_label)
    ax.set_ylabel(Y_LABEL)
    ax.grid(GRID_ON, which="both")
    top_legend(ax, handles, cfg, ncol=LEGEND_NCOL)
    return save_pdf(fig, "fig07_edr_correction_evolution", cfg)


if __name__ == "__main__":
    print(main())
