#!/usr/bin/env python3
"""fig04 — Capability radar over metrics no other figure emphasizes.

A spider/radar chart comparing the three champions on five per-architecture
scalars that appear on **no other figure** in this suite: pooled RMSE, overall
R², mean MAE, mean NRMSE and parameter count. Each axis is oriented so that
*outward = better* and normalised onto a common 0–1 radius:

  * the four accuracy/fit axes use **ratio-to-best** (best/value for lower-is-
    better, value/best for higher) so small, honest gaps stay small;
  * the parameter axis spans ~40× while the accuracy axes vary <11 %, so it uses
    a **log** score with a small inner floor — the dramatic size advantage stays
    legible instead of collapsing one model onto the centre.

Each axis is annotated with its real best value, so nothing on the chart is an
abstract normalised number. Physics-Reg. leads the accuracy axes by a hair but
sits near the centre on size; EDR matches it almost everywhere at a fraction of
the parameters. Raw + normalised values in tables/capability_profile.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from shared import dataio, palette
from shared.config import default_config, replace
from shared.figio import save_pdf
from shared.plotting import arch_proxy_handles
from shared.style import apply_style

# ============================ TWEAKABLES (edit me) ============================
FIG_W, FIG_H      = 6.8, 5.8
DPI_SAVE          = 300
AXES_LABEL_SIZE   = 15.0           # per-axis (metric-name) label size
TICK_SIZE         = 12.0
ANNOT_SIZE        = 11.0           # "best …" sub-label under each axis name
PARAM_FLOOR       = 0.12           # worst model's radius on the log param axis
FILL_ALPHA        = 0.10           # polygon fill transparency
RADIAL_RINGS      = (0.2, 0.4, 0.6, 0.8, 1.0)
R_HEADROOM        = 0.06           # empty ring between best vertex and labels
LABEL_PAD         = 10.0           # extra gap from the outer ring to axis labels
START_AT_TOP      = True           # first axis at 12 o'clock, going clockwise
# =============================================================================

CONFIG = replace(default_config(), fig_w=FIG_W, fig_h=FIG_H, dpi_save=DPI_SAVE,
                 axes_label_size=AXES_LABEL_SIZE, tick_label_size=TICK_SIZE,
                 annot_size=ANNOT_SIZE)


def _fmt_params(p: float) -> str:
    if p >= 1e6:
        return f"{p / 1e6:.2f}M"
    if p >= 1e3:
        return f"{p / 1e3:.1f}k"
    return f"{int(p)}"


def _ratio_to_best(raw: np.ndarray, lower_is_better: bool) -> np.ndarray:
    """Honest 0–1 score: best -> 1.0, others their true proportion of best."""
    if lower_is_better:
        return float(raw.min()) / raw
    return raw / float(raw.max())


def _param_log_score(raw: np.ndarray, floor: float) -> np.ndarray:
    """Log score for an axis spanning orders of magnitude (smaller = better).

    Smallest model -> 1.0, largest -> ``floor`` (kept off the dead centre so the
    worst vertex stays a readable point, not a collapse).
    """
    logs = np.log10(raw)
    span = (logs.max() - logs.min()) or 1.0
    s = 1.0 - (logs - logs.min()) / span        # 1 at min params, 0 at max
    return floor + (1.0 - floor) * s


def _best_raw(raw: np.ndarray, lower_is_better: bool) -> float:
    return float(raw.min()) if lower_is_better else float(raw.max())


def main(cfg=CONFIG):
    apply_style(cfg)
    from shared.plotting import champion_results
    res = champion_results(cfg)
    champs = dataio.champions(cfg)
    archs = palette.ordered_archs(cfg)

    metrics = {a: res[a]["metrics"] for a in archs}
    params = {a: float(dataio.param_count(champs[a]["run_dir"])) for a in archs}

    def mean(x):
        return float(np.mean(np.asarray(x, dtype=float)))

    # (display label, lower_is_better, scale, raw values in arch order, formatter)
    axis_specs = [
        ("Pooled\nRMSE", True,  "ratio",
         [metrics[a]["rmse_pooled"] for a in archs], "{:.3f}"),
        ("R² overall",   False, "ratio",
         [metrics[a]["r2_overall"] for a in archs],  "{:.3f}"),
        ("Mean\nMAE",    True,  "ratio",
         [mean(metrics[a]["mae"]) for a in archs],   "{:.3f}"),
        ("Mean\nNRMSE",  True,  "ratio",
         [mean(metrics[a]["nrmse"]) for a in archs], "{:.3f}"),
        ("Param\nefficiency", True, "param_log",
         [params[a] for a in archs], _fmt_params),
    ]

    labels, scores, rim = [], [], []
    for label, lower, scale, raw, fmt in axis_specs:
        raw = np.asarray(raw, dtype=float)
        if scale == "param_log":
            s = _param_log_score(raw, PARAM_FLOOR)
        else:
            s = _ratio_to_best(raw, lower)
        scores.append(s)
        best = _best_raw(raw, lower)
        best_txt = fmt(best) if callable(fmt) else fmt.format(best)
        labels.append(f"{label}\n(best {best_txt})")
        rim.append(best)
    scores = np.asarray(scores)            # (n_axes, n_arch)

    n = len(axis_specs)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    closed = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(cfg.fig_w, cfg.fig_h))
    ax = fig.add_subplot(111, polar=True)
    if START_AT_TOP:
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)

    for j, a in enumerate(archs):
        vals = np.concatenate([scores[:, j], scores[:1, j]])
        c = palette.color(cfg, a)
        ax.plot(closed, vals, color=c, lw=palette.line_width(cfg, a),
                zorder=palette.zorder(cfg, a), solid_joinstyle="round")
        ax.fill(closed, vals, color=c, alpha=FILL_ALPHA,
                zorder=palette.zorder(cfg, a) - 1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=cfg.axes_label_size)
    ax.set_ylim(0.0, 1.0 + R_HEADROOM)      # gap so vertices never touch labels
    ax.set_yticks(list(RADIAL_RINGS))
    ax.set_yticklabels([])                  # rings only; raw best is in the labels
    ax.tick_params(axis="x", pad=LABEL_PAD)
    ax.set_rlabel_position(0.0)
    ax.grid(True, alpha=cfg.grid_alpha, linewidth=cfg.grid_linewidth)

    # Anchor each axis label so its text grows *outward* from the circle (left
    # side right-aligned, right side left-aligned, top/bottom centred) — keeps
    # the multi-line "(best …)" labels off the polygons.
    screen = np.pi / 2.0 - angles if START_AT_TOP else angles
    for lab, sa in zip(ax.get_xticklabels(), screen):
        ca, sn = np.cos(sa), np.sin(sa)
        lab.set_horizontalalignment(
            "left" if ca > 0.15 else "right" if ca < -0.15 else "center")
        lab.set_verticalalignment(
            "bottom" if sn > 0.15 else "top" if sn < -0.15 else "center")

    # The top spoke occupies 12 o'clock, so the shared "legend above" strip would
    # collide with that axis label; place the frameless horizontal legend in the
    # empty wedge at the bottom centre instead (between the 5- and 7-o'clock axes).
    handles = arch_proxy_handles(cfg, archs, kind="line")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.06),
               ncol=len(handles), frameon=cfg.legend_frameon,
               fontsize=cfg.legend_size, handlelength=1.8, columnspacing=1.4,
               borderaxespad=0.0)
    return save_pdf(fig, "fig04_capability_radar", cfg)


if __name__ == "__main__":
    print(main())
