"""Fig 11 — HP importance: mean test RMSE per HP value, dynamic grid."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from ..config import _ARCH_HP_KEYS, _GRID_HP_KEYS_FNN
from ..io.records import arch_short_label, rmse_scalar
from ..style import type_color_map
from ._common import hp_val_str, save_fig

logger = logging.getLogger(__name__)


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order]
    if not arch_order:
        return

    n_archs = len(arch_order)
    type_colors = type_color_map(list(groups.keys()))

    varying_by_arch: dict[str, list[str]] = {}
    for mtype in arch_order:
        recs = groups[mtype]
        hp_keys = _ARCH_HP_KEYS.get(mtype, _GRID_HP_KEYS_FNN)
        varying = []
        for k in hp_keys:
            vals = set()
            for rec in recs:
                hp = rec.get("hyperparams", {})
                if k in hp:
                    vals.add(hp_val_str(hp[k]))
            if len(vals) > 1:
                varying.append(k)
        varying_by_arch[mtype] = varying

    max_cols = max((len(v) for v in varying_by_arch.values()), default=1)
    if max_cols == 0:
        logger.info("No varying HPs found - skipping Fig 11.")
        return

    _all_means_flat: list[float] = []
    _all_stds_flat:  list[float] = []
    for _mtype_pre in arch_order:
        for _k_pre in varying_by_arch[_mtype_pre]:
            _bkt_pre: dict[str, list[float]] = defaultdict(list)
            for _rec_pre in groups[_mtype_pre]:
                _tr_pre = rmse_scalar(_rec_pre, "test")
                if not (_tr_pre == _tr_pre and np.isfinite(_tr_pre)):
                    continue
                _hp_pre = _rec_pre.get("hyperparams", {})
                if _k_pre in _hp_pre:
                    _bkt_pre[hp_val_str(_hp_pre[_k_pre])].append(_tr_pre)
            for _vals_pre in _bkt_pre.values():
                if _vals_pre:
                    _m = float(np.mean(_vals_pre))
                    _s = float(np.std(_vals_pre)) if len(_vals_pre) > 1 else 0.0
                    _all_means_flat.append(_m)
                    _all_stds_flat.append(_s)
    if _all_means_flat:
        _g_lo_raw = min(m - s for m, s in zip(_all_means_flat, _all_stds_flat))
        _g_hi_raw = max(m + s for m, s in zip(_all_means_flat, _all_stds_flat))
        _margin   = (_g_hi_raw - _g_lo_raw) * 0.15
        _g_lo = max(0.0, _g_lo_raw - _margin)
        _g_hi = _g_hi_raw + _margin * 2.0
    else:
        _g_lo, _g_hi = 0.0, 1.0

    fig, axes_grid = plt.subplots(n_archs, max_cols,
                                  figsize=(max_cols * 3.5, n_archs * 4.5), squeeze=False)

    for arch_idx, mtype in enumerate(arch_order):
        recs = groups[mtype]
        varying = varying_by_arch[mtype]
        arch_color = type_colors.get(mtype, "#888888")

        hp_buckets: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in varying}
        for rec in recs:
            tr = rmse_scalar(rec, "test")
            if tr != tr or not np.isfinite(tr):
                continue
            hp = rec.get("hyperparams", {})
            for k in varying:
                if k in hp:
                    hp_buckets[k][hp_val_str(hp[k])].append(tr)

        for col_idx in range(max_cols):
            ax = axes_grid[arch_idx, col_idx]
            if col_idx >= len(varying):
                ax.axis("off")
                continue

            k = varying[col_idx]
            bucket = hp_buckets[k]

            def _sort_key(s: str) -> tuple:
                try:
                    return (0, float(s.replace("x", "0")))
                except ValueError:
                    return (1, s)

            sorted_vals = sorted(bucket.keys(), key=_sort_key)
            means = [float(np.mean(bucket[v])) for v in sorted_vals]
            stds  = [float(np.std(bucket[v]))  if len(bucket[v]) > 1 else 0.0
                     for v in sorted_vals]
            n_runs = [len(bucket[v]) for v in sorted_vals]

            x_pos = np.arange(len(sorted_vals))
            ax.bar(x_pos, means, color=arch_color, alpha=0.80, width=0.6)
            ax.errorbar(x_pos, means, yerr=stds, fmt="none", color="black",
                        capsize=4, linewidth=1.2)

            _lbl_y_cap = _g_hi - (_g_hi - _g_lo) * 0.07
            for xi, (m_val, s_val, n_val) in enumerate(zip(means, stds, n_runs)):
                _lbl_y = min(m_val + s_val + (_g_hi - _g_lo) * 0.02, _lbl_y_cap)
                ax.text(xi, _lbl_y, f"n={n_val}",
                        ha="center", va="bottom", fontsize=8, color="#555555")

            ax.set_ylim(_g_lo, _g_hi)

            clean_labels = [re.sub(r"^0e\+00$", "0", v) for v in sorted_vals]
            rotate = max(len(v) for v in clean_labels) > 6
            ax.set_xticks(x_pos)
            ax.set_xticklabels(clean_labels,
                               rotation=30 if rotate else 0,
                               ha="right" if rotate else "center",
                               fontsize=10)
            ax.set_title(k, fontsize=11, fontweight="bold")
            ax.tick_params(labelsize=10)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))

            if col_idx == 0:
                ax.set_ylabel(f"{arch_short_label(mtype)}\nMean Test RMSE (N.m)",
                              fontsize=10, fontweight="bold", labelpad=8)
            else:
                ax.set_ylabel("Mean Test RMSE (N.m)", fontsize=9)

    fig.tight_layout(pad=1.5)
    save_fig(fig, output_dir / "fig11_hp_importance.png")
