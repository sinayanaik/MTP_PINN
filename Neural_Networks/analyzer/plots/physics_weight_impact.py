"""Fig 8 — Physics Regularization weight impact (PhysReg only)."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..io.records import rmse_scalar
from ..style import panel_label
from ._common import save_fig

logger = logging.getLogger(__name__)


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    recs = groups.get("PhysicsRegularizedFNN", [])
    if not recs:
        return

    # ── Data: Group by Data Fraction, then by Physics Weight ───────────────
    data: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in recs:
        hp = rec.get("hyperparams", {})
        pw = hp.get("physics_weight")
        frac = hp.get("data_train_fraction")
        if pw is None or frac is None: continue
        rmse = rmse_scalar(rec, "test")
        if np.isfinite(rmse):
            data[float(frac)][float(pw)].append(rmse)

    all_fracs = sorted(data.keys())
    if not all_fracs:
        return

    fig, ax = plt.subplots(figsize=(10, 7.5))
    rng = np.random.default_rng(77)
    cmap = plt.get_cmap("plasma")

    for i, frac in enumerate(all_fracs):
        pw_sorted = sorted(data[frac].keys())
        means = [float(np.mean(data[frac][pw])) for pw in pw_sorted]
        stds  = [float(np.std(data[frac][pw])) if len(data[frac][pw]) > 1 else 0.0 for pw in pw_sorted]
        
        c = cmap(i / max(1, len(all_fracs) - 1))
        lbl = rf"$\mathrm{{{int(round(frac * 100))}\%\ Data}}$"
        
        ax.plot(pw_sorted, means, color=c, lw=3, marker="s", markersize=9, zorder=5, label=lbl)
        ax.fill_between(pw_sorted, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=c, alpha=0.15)
        
        for pw in pw_sorted:
            vals = data[frac][pw]
            x_range = max(pw_sorted) - min(pw_sorted) if len(pw_sorted) > 1 else 0.1
            jitter = rng.uniform(-x_range * 0.02, x_range * 0.02, size=len(vals))
            ax.scatter([pw + j for j in jitter], vals, color=c, s=30, alpha=0.4, zorder=3, edgecolors="none")

    ax.set_xlabel(r"$\mathrm{Physics\ Regularization\ Weight\ (\lambda)}$", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"$\mathrm{Test\ RMSE\ (N\cdot m)}$", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    # panel_label removed

    handles = [Line2D([0], [0], color=cmap(i / max(1, len(all_fracs) - 1)), lw=3, marker="s", 
                      label=rf"$\mathrm{{{int(round(f * 100))}\%\ Data}}$") for i, f in enumerate(all_fracs)]
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=len(handles), fontsize=13, framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output_dir / "fig8_physics_weight_impact.pdf")
