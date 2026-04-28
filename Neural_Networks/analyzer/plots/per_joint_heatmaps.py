"""Fig 3 — per-joint RMSE heatmaps (test, val) with per-column colour scaling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from ..config import JOINT_NAMES, N_JOINTS
from ..io.records import arch_short_label, best_per_type, split_joints
from ._common import save_fig


def _col_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize each column independently to [0, 1].

    Within each joint column the architecture with the lowest RMSE maps to 0
    (green) and the highest maps to 1 (red).  NaN cells stay NaN.
    All-equal columns are mapped to 0.5.
    """
    result = np.where(np.isfinite(mat), 0.5, np.nan).astype(float)
    for j in range(mat.shape[1]):
        col = mat[:, j].astype(float)
        finite_mask = np.isfinite(col)
        valid = col[finite_mask]
        if len(valid) < 2:
            continue
        cmin, cmax = float(valid.min()), float(valid.max())
        if cmax > cmin:
            result[finite_mask, j] = np.clip(
                (col[finite_mask] - cmin) / (cmax - cmin), 0.0, 1.0
            )
    return result


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    all_recs = best_per_type(groups)
    if not all_recs:
        return

    labels    = [arch_short_label(r.get("model_type", "?")) for r in all_recs]
    n_archs   = len(labels)

    test_rmse_mat = np.array([split_joints(r, "test", "rmse") for r in all_recs], dtype=float)
    val_rmse_mat  = np.array([split_joints(r, "val",  "rmse") for r in all_recs], dtype=float)

    cmap      = plt.get_cmap("RdYlGn_r")   # 0 → green (best), 1 → red (worst)
    nrows_fig = max(4.0, n_archs * 1.4 + 1.5)

    # Leave room on the right for the colorbar (right=0.87)
    fig, axes = plt.subplots(1, 2, figsize=(18, nrows_fig))
    fig.subplots_adjust(left=0.11, right=0.86, bottom=0.14, top=0.90, wspace=0.35)

    panels = [
        (axes[0], test_rmse_mat, "(a) Test RMSE (N·m)"),
        (axes[1], val_rmse_mat,  "(b) Val RMSE (N·m)"),
    ]

    for ax, mat, title in panels:
        norm_mat = _col_normalize(mat)
        masked   = np.ma.masked_invalid(norm_mat)
        ax.imshow(masked, aspect="auto", cmap=cmap,
                  vmin=0.0, vmax=1.0, interpolation="nearest")

        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES, fontsize=12, fontweight="bold")
        ax.set_yticks(range(n_archs))
        ax.set_yticklabels(labels, fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")

        for i in range(n_archs):
            for j in range(N_JOINTS):
                v  = mat[i, j]
                nv = norm_mat[i, j]
                if np.isfinite(v) and np.isfinite(nv):
                    # luminance-based text contrast
                    bg  = cmap(float(nv))
                    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
                    txt_c = "white" if lum < 0.45 else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=12, fontweight="bold", color=txt_c)

    # Dedicated colorbar axes — placed to the right of both panels, no overlap
    cbar_ax = fig.add_axes([0.88, 0.14, 0.018, 0.76])   # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Best\n(lowest)", "Mid", "Worst\n(highest)"])
    cbar.set_label("Relative RMSE per joint\n(0 = best,  1 = worst)",
                   fontsize=11, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    save_fig(fig, output_dir / "fig3_per_joint_heatmaps.pdf")
