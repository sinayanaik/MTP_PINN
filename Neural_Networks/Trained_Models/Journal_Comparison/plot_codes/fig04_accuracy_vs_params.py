#!/usr/bin/env python3
"""fig04 — Accuracy vs model size (test RMSE vs parameter count, log x).

Every registry model is a point; the champion of each family is starred.
EDR occupies the lower-left: more accurate with ~13× fewer parameters than
the black-box baselines. (Scatter — no Savitzky–Golay; raw values in
tables/cost_accuracy.csv.)
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

    rows = []
    for r in recs:
        try:
            p = dataio.param_count(r["run_dir"])
        except Exception:  # noqa: BLE001
            continue
        rows.append((r["arch"], p, r["test_rmse"]))

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for a in archs:
        c = palette.color(cfg, a)
        params = np.array([p for arch, p, _ in rows if arch == a], float)
        err = np.array([e for arch, _, e in rows if arch == a], float)
        ax.scatter(params, err, s=46, color=c, alpha=0.45, edgecolor="none",
                   zorder=palette.zorder(cfg, a))
        ch = champs[a]
        ax.plot([dataio.param_count(ch["run_dir"])], [ch["test_rmse"]],
                marker="*", ms=20, mfc=c, mec="black", mew=0.8, zorder=6)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel(cfg.rmse_label)
    ax.grid(True)

    handles = arch_proxy_handles(cfg, archs, kind="patch")
    handles.append(Line2D([0], [0], marker="*", ls="none", ms=15,
                          mfc="0.5", mec="black", label="champion"))
    top_legend(ax, handles, cfg)
    return save_pdf(fig, "fig04_accuracy_vs_params", cfg)


if __name__ == "__main__":
    print(main())
