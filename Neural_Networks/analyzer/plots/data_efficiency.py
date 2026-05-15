"""Fig 7 — Data Efficiency comparison (All Model Types)."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..io.records import arch_short_label, rmse_scalar
from ..style import panel_label, type_color_map
from ._common import save_fig

logger = logging.getLogger(__name__)


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    active_archs = sorted(groups.keys())
    if not active_archs:
        return

    type_colors = type_color_map(active_archs)
    data: dict[str, dict[float, list[float]]] = {a: defaultdict(list) for a in active_archs}

    for mtype in active_archs:
        for rec in groups[mtype]:
            frac = rec.get("hyperparams", {}).get("data_train_fraction")
            if frac is None: continue
            rmse = rmse_scalar(rec, "test")
            if np.isfinite(rmse):
                data[mtype][float(frac)].append(rmse)

    all_fracs = sorted({f for a in active_archs for f in data[a]})
    if not all_fracs:
        return

    fig, ax = plt.subplots(figsize=(10, 7.5))
    rng = np.random.default_rng(99)

    for mtype in active_archs:
        c = type_colors.get(mtype, "steelblue")
        fracs_sorted = sorted(data[mtype].keys())
        if not fracs_sorted: continue
        
        means = [float(np.mean(data[mtype][f])) for f in fracs_sorted]
        stds  = [float(np.std(data[mtype][f])) if len(data[mtype][f]) > 1 else 0.0 for f in fracs_sorted]
        xs = [f * 100 for f in fracs_sorted]

        ax.plot(xs, means, color=c, lw=3, marker="o", markersize=10, zorder=5,
                label=rf"$\mathrm{{{arch_short_label(mtype)}}}$")
        
        ax.fill_between(xs, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)], color=c, alpha=0.12)

        for f in fracs_sorted:
            vals = data[mtype][f]
            jitter = rng.uniform(-1.5, 1.5, size=len(vals))
            ax.scatter([f * 100 + j for j in jitter], vals,
                       color=c, s=35, alpha=0.4, zorder=3, edgecolors="none")

    ax.set_xlabel(r"$\mathrm{Training\ Data\ Fraction\ (\%)}$", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"$\mathrm{Test\ RMSE\ (N\cdot m)}$", fontsize=15, fontweight="bold")
    ax.set_xticks([f * 100 for f in all_fracs])
    ax.set_xticklabels([rf"${int(round(f * 100))}\%$" for f in all_fracs], fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    
    # panel_label removed

    handles = [Patch(facecolor=type_colors[t], label=rf"$\mathrm{{{arch_short_label(t)}}}$", alpha=0.8) 
               for t in active_archs]
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=len(handles), fontsize=14, framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output_dir / "fig7_data_efficiency.pdf")
