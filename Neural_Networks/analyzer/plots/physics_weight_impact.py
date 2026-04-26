"""Fig 15 — Physics weight impact (PhysicsRegularizedFNN only)."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..io.records import rmse_scalar, split_scalar
from ._common import save_fig

logger = logging.getLogger(__name__)


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    recs = groups.get("PhysicsRegularizedFNN", [])
    if not recs:
        logger.info("No PhysicsRegularizedFNN models - skipping Fig 15.")
        return

    pw_frac_rmse: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    pw_frac_r2:   dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for rec in recs:
        hp   = rec.get("hyperparams", {})
        pw   = hp.get("physics_weight")
        frac = hp.get("data_train_fraction")
        if pw is None or frac is None:
            continue
        pw   = float(pw)
        frac = float(frac)
        rmse = rmse_scalar(rec, "test")
        r2   = split_scalar(rec, "test", "r2_overall")
        if rmse == rmse and np.isfinite(rmse):
            pw_frac_rmse[pw][frac].append(rmse)
        if r2 == r2 and np.isfinite(r2):
            pw_frac_r2[pw][frac].append(r2)

    all_pw   = sorted(pw_frac_rmse.keys())
    all_frac = sorted({f for d in pw_frac_rmse.values() for f in d})

    if not all_pw:
        logger.info("No physics_weight data found - skipping Fig 15.")
        return

    _frac_cmap = plt.get_cmap("viridis")
    frac_colors = {
        f: _frac_cmap(0.15 + 0.70 * i / max(len(all_frac) - 1, 1))
        for i, f in enumerate(all_frac)
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rng = np.random.default_rng(77)
    _x_range = max(all_pw) - min(all_pw) if len(all_pw) > 1 else 1.0

    for ax, metric_data, ylabel, title in [
        (axes[0], pw_frac_rmse, "Test Average RMSE (N·m)",
         "(a) Test RMSE vs Physics Weight"),
        (axes[1], pw_frac_r2,   "Test R²",
         "(b) Test R² vs Physics Weight"),
    ]:
        for frac in all_frac:
            c = frac_colors[frac]
            xs, means, stds = [], [], []
            for pw in all_pw:
                vals = metric_data[pw].get(frac, [])
                if vals:
                    xs.append(pw)
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            if len(xs) < 2:
                continue

            ax.plot(xs, means, color=c, lw=2.0, marker="D", markersize=7, zorder=4,
                    label=f"{int(round(frac * 100))}% data")
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=c, alpha=0.15)

            for pw in xs:
                vals = metric_data[pw].get(frac, [])
                jitter = rng.uniform(-_x_range * 0.008, _x_range * 0.008, size=len(vals))
                ax.scatter([pw + j for j in jitter], vals,
                           color=c, s=22, alpha=0.55, zorder=3, edgecolors="none")

        pooled_xs, pooled_means = [], []
        for pw in all_pw:
            all_vals = [v for frac_d in metric_data[pw].values() for v in frac_d]
            if all_vals:
                pooled_xs.append(pw)
                pooled_means.append(float(np.mean(all_vals)))
        if len(pooled_xs) >= 2:
            ax.plot(pooled_xs, pooled_means, color="black", lw=2.5, ls="--",
                    marker="s", markersize=8, zorder=5, label="All fractions (pooled)")

        ax.set_xlabel("Physics Weight  (λ)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(all_pw)
        ax.set_xticklabels([str(p) for p in all_pw], fontsize=11)
        ax.grid(True, axis="y", alpha=0.35)

    frac_handles = [
        Line2D([0], [0], color=frac_colors[f], lw=2, marker="D", markersize=7,
               label=f"{int(round(f * 100))}% data")
        for f in all_frac
    ]
    pooled_handle = Line2D([0], [0], color="black", lw=2.5, ls="--", marker="s",
                           markersize=8, label="All fractions (pooled)")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=frac_handles + [pooled_handle],
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(frac_handles) + 1, fontsize=10)

    save_fig(fig, output_dir / "fig15_physics_weight_impact.png")
