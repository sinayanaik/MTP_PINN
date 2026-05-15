"""Fig 9 — per-joint R2 and RMSE breakdown (train vs test) bars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..config import JOINT_NAMES_SHORT, N_JOINTS
from ..io.records import arch_short_label, best_per_type, split_joints
from ..style import type_color_map
from ._common import save_fig


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    all_recs = best_per_type(groups)
    if not all_recs:
        return

    type_colors = type_color_map(list(groups.keys()))
    n_models = len(all_recs)
    model_labels = [arch_short_label(r.get("model_type", "?")) for r in all_recs]
    colors = [type_colors.get(r.get("model_type", "?"), "#888888") for r in all_recs]

    test_r2_mat   = np.array([split_joints(r, "test",  "r2")   for r in all_recs])
    train_r2_mat  = np.array([split_joints(r, "train", "r2")   for r in all_recs])
    test_rmse_mat = np.array([split_joints(r, "test",  "rmse") for r in all_recs])
    train_rmse_mat = np.array([split_joints(r, "train", "rmse") for r in all_recs])

    x = np.arange(N_JOINTS)
    bw = 0.70 / (n_models * 2)
    grp_bw = bw * 2 + 0.04
    grp_offsets = np.linspace(-(n_models - 1) / 2.0, (n_models - 1) / 2.0, n_models) * grp_bw

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, train_mat, test_mat, title, ylabel, higher_better in [
        (axes[0], train_r2_mat,   test_r2_mat,
         "(a) R2 per Joint", "R2", True),
        (axes[1], train_rmse_mat, test_rmse_mat,
         "(b) RMSE per Joint (N.m)", "RMSE (N.m)", False),
    ]:
        for mi, (goff, c, lbl) in enumerate(zip(grp_offsets, colors, model_labels)):
            ax.bar(x + goff - bw / 2, train_mat[mi], bw, color=c, alpha=0.88,
                   edgecolor="white", linewidth=0.5)
            ax.bar(x + goff + bw / 2, test_mat[mi],  bw, color=c, alpha=0.55,
                   edgecolor="white", linewidth=0.5, hatch="////")

        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES_SHORT, fontsize=12, fontweight="bold")
        ax.set_xlabel("Joint", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")

        all_vals = list(train_mat.flatten()) + list(test_mat.flatten())
        valid_vals = [v for v in all_vals if v == v and np.isfinite(v)]
        if valid_vals:
            lo = max(0.0, min(valid_vals) - 0.02) if higher_better else max(0.0, min(valid_vals) * 0.97)
            hi = min(1.02, max(valid_vals) + 0.02) if higher_better else max(valid_vals) * 1.08
            ax.set_ylim(lo, hi)

    arch_handles = [Patch(facecolor=c, label=lbl) for c, lbl in zip(colors, model_labels)]
    style_handles = [
        Patch(facecolor="#888888", alpha=0.88, label="Train"),
        Patch(facecolor="#888888", alpha=0.55, hatch="////", label="Test"),
    ]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=arch_handles + style_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=max(3, n_models + 2), fontsize=10)

    save_fig(fig, output_dir / "fig9_per_joint_r2_breakdown.pdf")
