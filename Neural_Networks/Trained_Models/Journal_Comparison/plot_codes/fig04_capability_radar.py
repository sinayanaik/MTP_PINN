#!/usr/bin/env python3
"""fig04 — Capability radar comparing three champions on six metrics.

A spider/radar chart comparing the three champions on six per-architecture
scalars: RMSE, overall R², mean MAE, mean NRMSE, parameter count, and
real wall-clock inference time.  Each axis is oriented so that
*outward = better* and normalised onto a common 0–1 radius.

Raw + normalised values in tables/capability_profile.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from shared import dataio, palette
from shared.config import default_config, replace
from shared.figio import save_pdf
from shared.plotting import arch_proxy_handles
from shared.style import apply_style

# ============================ TWEAKABLES (edit me) ============================
FIG_W, FIG_H      = 9.0, 9.5         # taller to leave room for legend at top
DPI_SAVE          = 300
AXES_LABEL_SIZE   = 23.0             # matched to other plots
TICK_SIZE         = 14.0
ANNOT_SIZE        = 13.5             # "best …" sub-label
LINE_W_RADAR      = 2.8
PARAM_FLOOR       = 0.15            # worst model's radius on log axes
INFTIME_FLOOR     = 0.15
FILL_ALPHA        = 0.12
RADIAL_RINGS      = (0.25, 0.50, 0.75, 1.0)
R_HEADROOM        = 0.10
LABEL_PAD         = 25.0
START_AT_TOP      = True
LEGEND_SIZE       = 23.0
LEGEND_MARKER_SZ  = 10.0
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 axes_label_size=AXES_LABEL_SIZE, tick_label_size=TICK_SIZE,
                 annot_size=ANNOT_SIZE, line_w=LINE_W_RADAR,
                 legend_size=LEGEND_SIZE)


def _fmt_params(p: float) -> str:
    if p >= 1e6:
        return f"{p / 1e6:.2f}M"
    if p >= 1e3:
        return f"{p / 1e3:.1f}k"
    return f"{int(p)}"


def _fmt_time(t: float) -> str:
    """Format inference time in µs or ms."""
    us = t * 1e6
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    return f"{us:.0f} µs"


def _ratio_to_best(raw: np.ndarray, lower_is_better: bool) -> np.ndarray:
    """Honest 0–1 score: best -> 1.0, others their true proportion of best."""
    if lower_is_better:
        return float(raw.min()) / raw
    return raw / float(raw.max())


def _log_score(raw: np.ndarray, floor: float) -> np.ndarray:
    """Log score for an axis spanning orders of magnitude (smaller = better)."""
    logs = np.log10(raw.astype(float))
    span = (logs.max() - logs.min()) or 1.0
    s = 1.0 - (logs - logs.min()) / span
    return floor + (1.0 - floor) * s


def _best_raw(raw: np.ndarray, lower_is_better: bool) -> float:
    return float(raw.min()) if lower_is_better else float(raw.max())


def main(cfg=CONFIG):
    apply_style(cfg)
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "serif",
    })

    from shared.plotting import champion_results
    from shared.inference import benchmark_inference_time
    res = champion_results(cfg)
    champs = dataio.champions(cfg)
    archs = palette.ordered_archs(cfg)

    metrics = {a: res[a]["metrics"] for a in archs}
    params = {a: float(dataio.param_count(champs[a]["run_dir"])) for a in archs}

    traj_rmse = {a: champs[a]["test_rmse"] for a in archs}

    # Benchmark real wall-clock inference time
    print("  Measuring inference time ...")
    inf_time = benchmark_inference_time(cfg)

    def mean(x):
        return float(np.mean(np.asarray(x, dtype=float)))

    # 6 axes: 4 accuracy + 2 efficiency
    axis_specs = [
        (r"RMSE",                 True,  "ratio",
         [traj_rmse[a] for a in archs],              "{:.3f}"),
        (r"$R^2$ Overall",       False, "ratio",
         [metrics[a]["r2_overall"] for a in archs],   "{:.3f}"),
        (r"Mean MAE",            True,  "ratio",
         [mean(metrics[a]["mae"]) for a in archs],    "{:.3f}"),
        (r"Mean NRMSE",          True,  "ratio",
         [mean(metrics[a]["nrmse"]) for a in archs],  "{:.3f}"),
        (r"Model Compactness",   True,  "param_log",
         [params[a] for a in archs], _fmt_params),
        (r"Inference Time",      True,  "inftime_log",
         [inf_time.get(a, float("nan")) for a in archs], _fmt_time),
    ]

    labels, scores, rim = [], [], []
    for label, lower, scale, raw, fmt in axis_specs:
        raw = np.asarray(raw, dtype=float)
        if scale == "param_log":
            s = _log_score(raw, PARAM_FLOOR)
        elif scale == "inftime_log":
            s = _log_score(raw, INFTIME_FLOOR)
        else:
            s = _ratio_to_best(raw, lower)
        scores.append(s)
        best = _best_raw(raw, lower)
        best_txt = fmt(best) if callable(fmt) else fmt.format(best)
        labels.append(f"{label}\n(best: {best_txt})")
        rim.append(best)
    scores = np.asarray(scores)

    n = len(axis_specs)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    closed = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(cfg.fig_w, cfg.fig_h))
    ax = fig.add_axes([0.08, 0.02, 0.84, 0.82], polar=True)
    if START_AT_TOP:
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)

    _markers = {"fnn": "o", "physreg": "s", "edr": "D"}

    for j, a in enumerate(archs):
        vals = np.concatenate([scores[:, j], scores[:1, j]])
        c = palette.color(cfg, a)
        lw = LINE_W_RADAR if a != cfg.emphasis_arch else LINE_W_RADAR * 1.2
        ax.plot(closed, vals, color=c, lw=lw,
                marker=_markers.get(a, "o"), markersize=7,
                zorder=palette.zorder(cfg, a), solid_joinstyle="round")
        ax.fill(closed, vals, color=c, alpha=FILL_ALPHA,
                zorder=palette.zorder(cfg, a) - 1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=cfg.axes_label_size,
                       fontweight="medium")
    ax.set_ylim(0.0, 1.0 + R_HEADROOM)
    ax.set_yticks(list(RADIAL_RINGS))
    ax.set_yticklabels([])
    ax.tick_params(axis="x", pad=LABEL_PAD)
    ax.set_rlabel_position(0.0)
    ax.grid(True, alpha=0.30, linewidth=0.7, color="0.65")

    screen = np.pi / 2.0 - angles if START_AT_TOP else angles
    for lab, sa in zip(ax.get_xticklabels(), screen):
        ca, sn = np.cos(sa), np.sin(sa)
        lab.set_horizontalalignment(
            "left" if ca > 0.15 else "right" if ca < -0.15 else "center")
        lab.set_verticalalignment(
            "bottom" if sn > 0.15 else "top" if sn < -0.15 else "center")

    handles = []
    for a in archs:
        c = palette.color(cfg, a)
        lbl = palette.label(cfg, a)
        handles.append(Line2D(
            [0], [0], color=c, lw=LINE_W_RADAR,
            marker=_markers.get(a, "o"), markersize=LEGEND_MARKER_SZ,
            label=lbl))
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(handles), frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.5, columnspacing=3.0,
        borderaxespad=0.0,
    )
    return save_pdf(fig, "fig04_capability_radar", cfg)


if __name__ == "__main__":
    print(main())
