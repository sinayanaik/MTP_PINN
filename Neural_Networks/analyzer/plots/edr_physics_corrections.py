"""Fig 8 — EDR physics correction magnitudes over training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..io.records import rmse_scalar, short_label
from ._common import save_fig

logger = logging.getLogger(__name__)


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    edr_recs = groups.get("EDR", [])
    if not edr_recs:
        logger.info("No EDR models - skipping Fig 8.")
        return

    edr_with_history = [r for r in edr_recs if "mean_abs_delta_g" in r.get("_history", {})]
    if not edr_with_history:
        logger.info("No EDR correction history found - skipping Fig 8.")
        return

    best_edr = min(edr_with_history, key=lambda r: rmse_scalar(r, "test"))
    edr_with_data = [best_edr]

    corr_cols = [
        ("mean_abs_delta_g",     "delta_g gravity correction"),
        ("mean_frob_delta_M",    "delta_M inertia correction (Frobenius)"),
        ("mean_abs_delta_C_qd",  "delta_C*qd Coriolis correction"),
        ("mean_abs_delta_tau_f", "delta_tau_f friction correction"),
    ]

    cmap_edr = plt.get_cmap("tab10")
    model_colors = {r.get("run_id", str(i)): cmap_edr(i % 10)
                    for i, r in enumerate(edr_with_data)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    for ax, (col, title) in zip(axes.flatten(), corr_cols):
        for rec in edr_with_data:
            hist = rec.get("_history", {})
            vals = hist.get(col, [])
            if not vals:
                continue
            ep = hist.get("epoch", list(range(1, len(vals) + 1)))
            best_ep = rec.get("_best_epoch", -1)
            run_id = rec.get("run_id", "?")
            c = model_colors[run_id]
            lbl = short_label(run_id)
            ep_arr = ep[:len(vals)]
            ax.plot(ep_arr, vals, color=c, lw=1.6, alpha=0.8, label=lbl)
            if 0 < best_ep <= len(vals):
                bi = best_ep - 1
                ax.scatter([ep_arr[bi]], [vals[bi]], color=c, s=70, zorder=5, marker="*")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Correction Magnitude", fontsize=11)
        ax.grid(True, alpha=0.3)

    h, lbls = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.legend(h, lbls, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                   ncol=len(h), fontsize=10)
    else:
        fig.tight_layout()

    save_fig(fig, output_dir / "fig8_edr_physics_corrections.pdf")
