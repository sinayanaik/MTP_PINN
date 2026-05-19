#!/usr/bin/env python3
"""fig08 — What EDR learns: magnitude of each physics correction vs epoch.

EDR adds a learned residual to every analytical term; |δg|, ‖δM‖_F, |δC·q̇|
and |δτ_f| over training show which term it leans on — interpretability the
black-box baselines cannot offer. Curves Savitzky–Golay smoothed by default
(raw values in tables/edr_correction_evolution.csv).
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

CONFIG = replace(default_config(), fig_w=7.4, fig_h=4.8)

_TERMS = [
    ("mean_abs_delta_g", r"$|\delta g|$", "#C44E52"),
    ("mean_frob_delta_M", r"$\|\delta M\|_F$", "#4C72B0"),
    ("mean_abs_delta_C_qd", r"$|\delta C\,\dot q|$", "#55A868"),
    ("mean_abs_delta_tau_f", r"$|\delta \tau_f|$", "#8172B3"),
]


def main(cfg=CONFIG):
    apply_style(cfg)
    edr = dataio.champions(cfg.champion_metric)["edr"]
    h = dataio.load_history(edr["run_dir"])
    ep = h["epoch"].to_numpy()

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    handles = []
    for col, lbl, c in _TERMS:
        (ln,) = ax.plot(ep, maybe_smooth(h[col].to_numpy(), cfg), color=c,
                        lw=2.4, label=lbl)
        handles.append(ln)
    ax.set_yscale("log")
    ax.set_xlabel(cfg.epoch_label)
    ax.set_ylabel("Mean correction magnitude (N·m)")
    ax.grid(True, which="both")
    top_legend(ax, handles, cfg, ncol=4)
    return save_pdf(fig, "fig08_edr_correction_evolution", cfg)


if __name__ == "__main__":
    print(main())
