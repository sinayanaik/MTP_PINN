"""Fig 6 — R2 vs RMSE scatter, with train->test arrow per architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..io.records import arch_short_label, best_per_type, rmse_scalar, split_scalar, train_rmse
from ..style import type_color_map
from ._common import save_fig


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    type_colors = type_color_map(list(groups.keys()))
    model_types = sorted(groups.keys())

    fig, ax = plt.subplots(figsize=(11, 7))

    best_map = {r.get("model_type"): r for r in best_per_type(groups)}
    for mtype in model_types:
        c = type_colors[mtype]
        rec = best_map.get(mtype)
        if rec is None:
            continue

        tr_rmse = train_rmse(rec)
        tr_r2   = split_scalar(rec, "train", "r2_overall")
        te_rmse = rmse_scalar(rec, "test")
        te_r2   = split_scalar(rec, "test", "r2_overall")

        if tr_rmse == tr_rmse and np.isfinite(tr_rmse):
            y_tr = tr_r2 if (tr_r2 == tr_r2 and np.isfinite(tr_r2)) else te_r2
            ax.scatter(tr_rmse, y_tr, color=c, s=160, marker="o",
                       facecolors="none", edgecolors=c, linewidths=2.0, zorder=5)
            if te_rmse == te_rmse and np.isfinite(te_rmse) and te_r2 == te_r2 and np.isfinite(te_r2):
                ax.scatter(te_rmse, te_r2, color=c, s=160, marker="o", zorder=6)
                ax.annotate("", xy=(te_rmse, te_r2), xytext=(tr_rmse, y_tr),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.5), zorder=4)
        elif te_rmse == te_rmse and np.isfinite(te_rmse) and te_r2 == te_r2 and np.isfinite(te_r2):
            ax.scatter(te_rmse, te_r2, color=c, s=160, marker="o", zorder=6)

    best_recs = best_per_type(groups)
    all_te_rmse = [rmse_scalar(r, "test") for r in best_recs]
    all_te_r2   = [split_scalar(r, "test", "r2_overall") for r in best_recs]
    vr  = [v for v in all_te_rmse if v == v and np.isfinite(v)]
    vr2 = [v for v in all_te_r2   if v == v and np.isfinite(v)]
    if vr and vr2:
        ax.annotate("Ideal",
                    (min(vr) * 0.99, max(vr2) * 1.002),
                    textcoords="offset points", xytext=(-50, 6),
                    fontsize=11, color="#228822", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#228822", lw=1.2))

    type_handles = [Patch(color=type_colors[t], label=arch_short_label(t)) for t in model_types]
    marker_handles = [
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="none",
               markeredgewidth=2, markersize=10, linestyle="None", label="Train  (hollow)"),
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
               markersize=10, linestyle="None", label="Test  (filled)"),
    ]
    ax.set_xlabel("Average RMSE (N·m)", fontsize=12)
    ax.set_ylabel("R² overall", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=type_handles + marker_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(type_handles) + 2, fontsize=10)

    save_fig(fig, output_dir / "fig6_r2_vs_rmse_scatter.png")
