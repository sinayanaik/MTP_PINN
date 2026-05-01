"""Fig 3 — per-joint RMSE heatmaps (test, val) with per-panel colour scaling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from ..config import JOINT_NAMES, N_JOINTS
from ..io.records import arch_short_label, best_per_type, split_joints
from ._common import save_fig


def _panel_norm(mat: np.ndarray) -> mcolors.Normalize:
    """Return a split-local color norm.

    The lowest finite RMSE in the panel maps to green and the highest maps
    to red.  Degenerate panels are padded so equal values render at the
    middle of the scale.
    """
    valid = mat[np.isfinite(mat)]
    if valid.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)

    vmin = float(valid.min())
    vmax = float(valid.max())
    if vmax <= vmin:
        pad = max(abs(vmin) * 0.05, 1e-9)
        vmin -= pad
        vmax += pad
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    all_recs = best_per_type(groups)
    if not all_recs:
        return

    labels    = [arch_short_label(r.get("model_type", "?")) for r in all_recs]
    n_archs   = len(labels)
    joint_labels = [name.replace(" (", "\n(") for name in JOINT_NAMES]

    test_rmse_mat = np.array([split_joints(r, "test", "rmse") for r in all_recs], dtype=float)
    val_rmse_mat  = np.array([split_joints(r, "val",  "rmse") for r in all_recs], dtype=float)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rmse_best_to_worst",
        ["#5f8f6a", "#f6efc5", "#b65f52"],  # muted green -> cream -> terracotta
    )
    cmap.set_bad("#f0f0f0")
    nrows_fig = max(5.2, n_archs * 1.10 + 2.2)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, nrows_fig))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.23, top=0.88, wspace=0.38)

    panels = [
        (axes[0], test_rmse_mat, "(a) Test RMSE (N·m)"),
        (axes[1], val_rmse_mat,  "(b) Val RMSE (N·m)"),
    ]

    for ax, mat, title in panels:
        norm = _panel_norm(mat)
        masked = np.ma.masked_invalid(mat)
        x_edges = np.arange(N_JOINTS + 1) - 0.5
        y_edges = np.arange(n_archs + 1) - 0.5
        im = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, norm=norm,
                           shading="flat", edgecolors="none")
        ax.set_xlim(-0.5, N_JOINTS - 0.5)
        ax.set_ylim(n_archs - 0.5, -0.5)
        ax.set_aspect("auto")

        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(joint_labels, fontsize=13, fontweight="bold")
        ax.set_yticks(range(n_archs))
        ax.set_yticklabels(labels, fontsize=16, fontweight="bold")
        ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
        ax.tick_params(axis="both", which="major", labelsize=14, width=1.4, length=6)

        for i in range(n_archs):
            for j in range(N_JOINTS):
                v = mat[i, j]
                if np.isfinite(v):
                    # luminance-based text contrast
                    bg = cmap(norm(float(v)))
                    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
                    txt_c = "white" if lum < 0.45 else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=15, fontweight="bold", color=txt_c)

        vmin, vmax = float(norm.vmin), float(norm.vmax)
        mid = 0.5 * (vmin + vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_ticks([vmin, mid, vmax])
        cbar.set_ticklabels([f"Best\n{vmin:.3f}", f"Mid\n{mid:.3f}",
                             f"Worst\n{vmax:.3f}"])
        cbar.ax.tick_params(labelsize=11, width=1.2, length=5)

    save_fig(fig, output_dir / "fig3_per_joint_heatmaps.pdf")
