"""Fig 12 — HP-pair heatmaps (viridis_r), shared colorbar, skip degenerate."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..io.records import arch_short_label, rmse_scalar
from ._common import hp_val_str, save_fig

logger = logging.getLogger(__name__)


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    _PAIR_DEFS: dict[str, list[tuple[str, str]]] = {
        "BlackBoxFNN":           [("learning_rate", "dropout"), ("hidden_layers", "batch_size")],
        "PhysicsRegularizedFNN": [("learning_rate", "physics_weight"), ("hidden_layers", "dropout")],
        "ResidualCorrectionFNN": [("learning_rate", "alpha_reg_weight"), ("hidden_layers", "dropout")],
    }
    _PAIR_DEFS_DEFAULT = [("learning_rate", "dropout"), ("hidden_layers", "batch_size")]

    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order]
    if not arch_order:
        return

    n_pairs = 2

    def _sort_hp(s: str) -> tuple:
        try:
            return (0, float(s.replace("x", "0")))
        except ValueError:
            return (1, s)

    panel_info = []
    for arch_idx, mtype in enumerate(arch_order):
        recs = groups[mtype]
        pair_defs = _PAIR_DEFS.get(mtype, _PAIR_DEFS_DEFAULT)

        for pair_idx, (key_x, key_y) in enumerate(pair_defs[:n_pairs]):
            vals_x, vals_y = [], []
            for rec in recs:
                hp = rec.get("hyperparams", {})
                if key_x in hp and key_y in hp:
                    vx = hp_val_str(hp[key_x]); vy = hp_val_str(hp[key_y])
                    if vx not in vals_x: vals_x.append(vx)
                    if vy not in vals_y: vals_y.append(vy)

            vals_x = sorted(vals_x, key=_sort_hp)
            vals_y = sorted(vals_y, key=_sort_hp)

            if len(vals_x) <= 1 or len(vals_y) <= 1:
                continue

            cell_data: dict[tuple[str, str], list[float]] = defaultdict(list)
            for rec in recs:
                tr = rmse_scalar(rec, "test")
                if tr != tr or not np.isfinite(tr):
                    continue
                hp = rec.get("hyperparams", {})
                if key_x in hp and key_y in hp:
                    cell_data[(hp_val_str(hp[key_x]), hp_val_str(hp[key_y]))].append(tr)

            mat = np.full((len(vals_y), len(vals_x)), np.nan)
            for (vx, vy), rmse_list in cell_data.items():
                if vx in vals_x and vy in vals_y:
                    mat[vals_y.index(vy), vals_x.index(vx)] = float(np.mean(rmse_list))

            valid_v = mat[~np.isnan(mat)]
            if len(valid_v) < 2:
                continue

            panel_info.append((arch_idx, pair_idx, mtype, key_x, key_y, mat, vals_x, vals_y))

    if not panel_info:
        logger.info("No non-degenerate HP pairs found - skipping Fig 12.")
        return

    all_cell_vals = np.concatenate([info[5][~np.isnan(info[5])].flatten() for info in panel_info])
    global_vmin = float(all_cell_vals.min())
    global_vmax = float(all_cell_vals.max())

    n_panels = len(panel_info)
    n_cols = min(n_pairs, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes_2d = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6.0), squeeze=False)
    axes_flat = [axes_2d[r][c] for r in range(n_rows) for c in range(n_cols)]

    panel_letters = "abcdefghij"
    im_last = None

    for panel_num, (arch_idx, pair_idx, mtype, key_x, key_y,
                    mat, vals_x, vals_y) in enumerate(panel_info):
        ax = axes_flat[panel_num]
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap="viridis_r", aspect="auto",
                       vmin=global_vmin, vmax=global_vmax, interpolation="nearest")
        im_last = im

        ax.set_xticks(range(len(vals_x)))
        ax.set_xticklabels(vals_x, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(len(vals_y)))
        ax.set_yticklabels(vals_y, fontsize=10)
        ax.set_xlabel(key_x, fontsize=11, fontweight="bold")
        ax.set_ylabel(key_y, fontsize=11, fontweight="bold")
        letter = panel_letters[panel_num] if panel_num < len(panel_letters) else str(panel_num)
        ax.set_title(f"({letter}) {arch_short_label(mtype)}: {key_x} x {key_y}",
                     fontsize=12, fontweight="bold")

        for yi in range(len(vals_y)):
            for xi in range(len(vals_x)):
                v = mat[yi, xi]
                if not np.isnan(v):
                    norm_v = (v - global_vmin) / (global_vmax - global_vmin) if global_vmax > global_vmin else 0.5
                    txt_c = "white" if norm_v > 0.5 else "black"
                    ax.text(xi, yi, f"{v:.4f}", ha="center", va="center",
                            fontsize=10, color=txt_c, fontweight="bold")
                else:
                    ax.text(xi, yi, "-", ha="center", va="center", fontsize=10, color="#aaaaaa")

    for panel_num in range(len(panel_info), len(axes_flat)):
        axes_flat[panel_num].axis("off")

    if im_last is not None:
        cbar = fig.colorbar(im_last, ax=axes_flat[:len(panel_info)],
                            fraction=0.025, pad=0.03)
        cbar.set_label("Mean Test RMSE (N.m)", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    save_fig(fig, output_dir / "fig12_hp_pair_heatmaps.png")
