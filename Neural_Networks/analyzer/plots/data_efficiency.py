"""Fig 14 — data efficiency: test RMSE & R2 vs training-data fraction."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..io.records import arch_short_label, rmse_scalar, split_scalar
from ..style import type_color_map
from ._common import save_fig

logger = logging.getLogger(__name__)


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order]
    if not arch_order:
        return

    type_colors = type_color_map(list(groups.keys()))

    frac_rmse: dict[str, dict[float, list[float]]] = {a: defaultdict(list) for a in arch_order}
    frac_r2:   dict[str, dict[float, list[float]]] = {a: defaultdict(list) for a in arch_order}

    for mtype in arch_order:
        for rec in groups[mtype]:
            frac = rec.get("hyperparams", {}).get("data_train_fraction")
            if frac is None:
                continue
            frac = float(frac)
            rmse = rmse_scalar(rec, "test")
            r2   = split_scalar(rec, "test", "r2_overall")
            if rmse == rmse and np.isfinite(rmse):
                frac_rmse[mtype][frac].append(rmse)
            if r2 == r2 and np.isfinite(r2):
                frac_r2[mtype][frac].append(r2)

    all_fracs = sorted({f for a in arch_order for f in frac_rmse[a]})
    if not all_fracs:
        logger.info("No data_train_fraction data found - skipping Fig 14.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rng = np.random.default_rng(99)

    for ax, metric_data, ylabel, title in [
        (axes[0], frac_rmse, "Test Average RMSE (N·m)",
         "(a) Test RMSE vs Training Data Fraction"),
        (axes[1], frac_r2,   "Test R²",
         "(b) Test R² vs Training Data Fraction"),
    ]:
        for mtype in arch_order:
            c = type_colors.get(mtype, "#888888")
            fracs_sorted = sorted(metric_data[mtype].keys())
            if not fracs_sorted:
                continue

            xs    = [f * 100 for f in fracs_sorted]
            means = [float(np.mean(metric_data[mtype][f])) for f in fracs_sorted]
            stds  = [float(np.std(metric_data[mtype][f])) if len(metric_data[mtype][f]) > 1 else 0.0
                     for f in fracs_sorted]

            ax.plot(xs, means, color=c, lw=2.2, marker="o", markersize=7, zorder=4,
                    label=arch_short_label(mtype))
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=c, alpha=0.15)

            for f in fracs_sorted:
                vals = metric_data[mtype][f]
                jitter = rng.uniform(-0.8, 0.8, size=len(vals))
                ax.scatter([f * 100 + j for j in jitter], vals,
                           color=c, s=22, alpha=0.55, zorder=3, edgecolors="none")

        ax.set_xlabel("Training Data Fraction (%)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks([f * 100 for f in all_fracs])
        ax.set_xticklabels([f"{int(round(f * 100))}%" for f in all_fracs], fontsize=11)
        ax.grid(True, axis="y", alpha=0.35)

    arch_handles = [Patch(facecolor=type_colors.get(t, "#888888"), label=arch_short_label(t))
                    for t in arch_order]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=arch_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(arch_handles), fontsize=11)

    save_fig(fig, output_dir / "fig14_data_efficiency.png")
