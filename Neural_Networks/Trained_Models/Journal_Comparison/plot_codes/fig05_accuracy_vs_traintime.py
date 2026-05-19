#!/usr/bin/env python3
"""fig05 — Accuracy vs training cost (test RMSE vs training wall-time).

Every registry model is a point; the champion of each family is starred.
EDR reaches the best accuracy in the least training time. (Scatter — no
Savitzky–Golay; raw values in tables/cost_accuracy.csv.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from shared import dataio, palette
from shared.config import default_config, replace
from shared.figio import save_pdf
from shared.plotting import arch_proxy_handles, top_legend
from shared.style import apply_style

CONFIG = replace(default_config(), fig_w=7.4, fig_h=4.8)


def main(cfg=CONFIG):
    apply_style(cfg)
    recs = dataio.registry_records()
    champs = dataio.champions(cfg.champion_metric)
    archs = palette.ordered_archs(cfg)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for a in archs:
        c = palette.color(cfg, a)
        t = np.array([r["time_seconds"] for r in recs if r["arch"] == a],
                     float) / 60.0
        e = np.array([r["test_rmse"] for r in recs if r["arch"] == a], float)
        ax.scatter(t, e, s=46, color=c, alpha=0.45, edgecolor="none",
                   zorder=palette.zorder(cfg, a))
        ch = champs[a]
        ax.plot([ch["time_seconds"] / 60.0], [ch["test_rmse"]], marker="*",
                ms=20, mfc=c, mec="black", mew=0.8, zorder=6)

    ax.set_xlabel("Training time (min)")
    ax.set_ylabel(cfg.rmse_label)
    ax.grid(True)

    handles = arch_proxy_handles(cfg, archs, kind="patch")
    handles.append(Line2D([0], [0], marker="*", ls="none", ms=15,
                          mfc="0.5", mec="black", label="champion"))
    top_legend(ax, handles, cfg)
    return save_pdf(fig, "fig05_accuracy_vs_traintime", cfg)


if __name__ == "__main__":
    print(main())
