"""Fig 4 — per-joint RMSE heatmaps (test, val) with shared viridis colorbar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..config import JOINT_NAMES, N_JOINTS
from ..io.records import arch_short_label, best_per_type, split_joints
from ._common import save_fig


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    all_recs = best_per_type(groups)
    if not all_recs:
        return

    labels = [arch_short_label(r.get("model_type", "?")) for r in all_recs]

    test_rmse_mat = np.array([split_joints(r, "test", "rmse") for r in all_recs])
    val_rmse_mat  = np.array([split_joints(r, "val",  "rmse") for r in all_recs])

    all_rmse = np.concatenate([test_rmse_mat.flatten(), val_rmse_mat.flatten()])
    valid_rmse = all_rmse[np.isfinite(all_rmse)]
    vmin = float(valid_rmse.min()) if len(valid_rmse) else 0.0
    vmax = float(valid_rmse.max()) if len(valid_rmse) else 1.0

    nrows_fig = max(4.0, len(all_recs) * 1.4 + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(18, nrows_fig))

    panels = [
        (axes[0], test_rmse_mat, "(a) Test RMSE (N.m)"),
        (axes[1], val_rmse_mat,  "(b) Val RMSE (N.m)"),
    ]

    im_last = None
    for ax, mat, title in panels:
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="auto", cmap="plasma",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        im_last = im
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        for i in range(len(labels)):
            for j in range(N_JOINTS):
                v = mat[i, j]
                if v == v and np.isfinite(v):
                    norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    txt_c = "white" if norm_v < 0.5 else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=13, fontweight="bold", color=txt_c)

    if im_last is not None:
        cbar = fig.colorbar(im_last, ax=axes.tolist(), fraction=0.025, pad=0.03)
        cbar.set_label("RMSE (N·m)", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    save_fig(fig, output_dir / "fig4_per_joint_heatmaps.png")
