"""
Global summary plots for the bulk torque analysis.

All functions receive the list of per-run summary dicts (as returned by
bulk_analyze.process_one_file) and a save directory.

Plots produced
──────────────
1.  rnea_ratio_violin.png        — RNEA/Load ratio distribution per joint
2.  nrmse_violin.png             — Normalised residual RMS per joint
3.  residual_rms_boxplot.png     — Residual RMS absolute values per joint
4.  load_vs_rnea_scatter.png     — Load RMS vs RNEA RMS scatter (per joint)
5.  accuracy_by_shape.png        — Median NRMSE heatmap: shape × joint
6.  accuracy_by_traj_type.png    — Median NRMSE bar chart per trajectory type
7.  accuracy_vs_radius.png       — NRMSE vs radius per joint (J2–J3 only)
8.  model_coverage_cdf.png       — CDF: fraction of runs vs NRMSE threshold
9.  error_hist_global.png        — Error histograms (3 models) per joint, all runs
10. error_hist_by_shape.png      — Error histograms per trajectory shape (RNEA+Fric)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from . import config as C

# ── colour palette (consistent across all plots) ─────────────
JOINT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a",
                "#984ea3", "#ff7f00", "#a65628"]
JOINT_LABELS = ["J1 (yaw)", "J2", "J3", "J4", "J5", "J6 (tool)"]


def _savefig(fig, path):
    fig.savefig(path, dpi=C.DPI, bbox_inches="tight")
    plt.close(fig)


def _collect(summaries, joint_idx, key):
    return [s["joints"][joint_idx][key]
            for s in summaries
            if s["joints"][joint_idx].get(key) is not None]


# ─────────────────────────────────────────────────────────────
# 1.  RNEA/Load ratio — violin per joint
# ─────────────────────────────────────────────────────────────

def plot_rnea_ratio_violin(summaries: list, save_dir: str):
    """
    Violin plot of the per-run RNEA/Load RMS ratio for each joint.

    Ideal ratio = 1.0 (model perfectly predicts load).
    Ratio << 1 means RNEA under-predicts (unmodelled loads dominate).
    Ratio >> 1 means RNEA over-predicts (model too aggressive).
    """
    data = [_collect(summaries, j, "rnea_over_load") for j in range(C.DOF)]
    data = [d if d else [0.0] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=C.DPI)

    parts = ax.violinplot(data, positions=range(1, C.DOF + 1),
                          showmedians=True, showextrema=True, widths=0.7)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(JOINT_COLORS[i])
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.axhline(1.0, color="green", lw=1.5, ls="--", label="Ideal (ratio = 1.0)")
    ax.axhline(0.5, color="orange", lw=1.0, ls=":", alpha=0.7, label="±50% bounds")
    ax.axhline(2.0, color="orange", lw=1.0, ls=":", alpha=0.7)

    ax.set_xticks(range(1, C.DOF + 1))
    ax.set_xticklabels(JOINT_LABELS)
    ax.set_ylabel("RNEA RMS / Load RMS")
    ax.set_title("RNEA / Load Ratio per Joint\n"
                 "(green dashed = ideal 1.0 | violin = distribution over all runs)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate median
    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i + 1, med + 0.03, f"{med:.2f}", ha="center", va="bottom",
                fontsize=7, color="black", fontweight="bold")

    _savefig(fig, os.path.join(save_dir, "rnea_ratio_violin.png"))


# ─────────────────────────────────────────────────────────────
# 2.  NRMSE violin — normalised residual RMS per joint
# ─────────────────────────────────────────────────────────────

def plot_nrmse_violin(summaries: list, save_dir: str):
    """
    Violin plot of per-run NRMSE = residual_rms / load_rms per joint.

    NRMSE < 0.2 → model explains >80% of load (excellent).
    NRMSE < 0.5 → model explains >50% of load (acceptable).
    """
    data = [_collect(summaries, j, "nrmse") for j in range(C.DOF)]
    data = [d if d else [1.0] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=C.DPI)

    parts = ax.violinplot(data, positions=range(1, C.DOF + 1),
                          showmedians=True, showextrema=True, widths=0.7)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(JOINT_COLORS[i])
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.axhline(0.2, color="green",  lw=1.5, ls="--", label="Excellent (<20%)")
    ax.axhline(0.5, color="orange", lw=1.5, ls="--", label="Acceptable (<50%)")

    ax.set_xticks(range(1, C.DOF + 1))
    ax.set_xticklabels(JOINT_LABELS)
    ax.set_ylabel("NRMSE  = residual RMS / load RMS")
    ax.set_title("Normalised Residual RMS (NRMSE) per Joint\n"
                 "(lower = model explains more of the load)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i + 1, med + 0.01, f"{med*100:.1f}%", ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    _savefig(fig, os.path.join(save_dir, "nrmse_violin.png"))


# ─────────────────────────────────────────────────────────────
# 3.  Residual RMS box plot
# ─────────────────────────────────────────────────────────────

def plot_residual_rms_boxplot(summaries: list, save_dir: str):
    """
    Box plot of absolute residual RMS [N·m] per joint.
    Shows median, IQR, and outliers.
    """
    data = [_collect(summaries, j, "residual_rms") for j in range(C.DOF)]
    data = [d if d else [0.0] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=C.DPI)

    bp = ax.boxplot(data, positions=range(1, C.DOF + 1),
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2),
                    flierprops=dict(marker="o", ms=3, alpha=0.5))

    for patch, color in zip(bp["boxes"], JOINT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, C.DOF + 1))
    ax.set_xticklabels(JOINT_LABELS)
    ax.set_ylabel("Residual RMS  [N·m]")
    ax.set_title("Residual (τ_load − τ_model) RMS per Joint\n"
                 "Residual = unmodelled dynamics left for the PINN to learn")
    ax.grid(True, axis="y", alpha=0.3)

    _savefig(fig, os.path.join(save_dir, "residual_rms_boxplot.png"))


# ─────────────────────────────────────────────────────────────
# 4.  Load RMS vs RNEA RMS scatter
# ─────────────────────────────────────────────────────────────

def plot_load_vs_rnea_scatter(summaries: list, save_dir: str):
    """
    Scatter: Load RMS (x) vs RNEA RMS (y) for joints 1–5.
    Points on the y = x diagonal indicate perfect model agreement.
    """
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), dpi=C.DPI)

    for j, ax in enumerate(axes):
        load_v = np.array(_collect(summaries, j, "load_rms"))
        rnea_v = np.array(_collect(summaries, j, "rnea_rms"))
        n = min(len(load_v), len(rnea_v))
        if n == 0:
            ax.set_visible(False)
            continue

        load_v, rnea_v = load_v[:n], rnea_v[:n]

        ax.scatter(load_v, rnea_v, s=18, color=JOINT_COLORS[j], alpha=0.6,
                   linewidths=0)
        lim = max(load_v.max(), rnea_v.max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (perfect)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Load RMS [N·m]")
        ax.set_ylabel("RNEA RMS [N·m]")
        ax.set_title(JOINT_LABELS[j], fontsize=9)
        ax.grid(True, alpha=0.3)

        # Pearson r
        if n > 2:
            r = np.corrcoef(load_v, rnea_v)[0, 1]
            ax.text(0.05, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
                    fontsize=8, color="black")

    fig.suptitle("Load RMS vs RNEA RMS per Joint\n"
                 "(points on dashed line → perfect model)", fontsize=10)
    plt.tight_layout()
    _savefig(fig, os.path.join(save_dir, "load_vs_rnea_scatter.png"))


# ─────────────────────────────────────────────────────────────
# 5.  Accuracy heatmap by trajectory shape
# ─────────────────────────────────────────────────────────────

def plot_accuracy_by_shape(summaries: list, save_dir: str):
    """
    Heatmap: rows = trajectory shapes, columns = joints J1–J5.
    Cell value = median NRMSE (%).  Green = low error, Red = high.
    """
    shapes = sorted({s["traj_meta"]["shape"] for s in summaries})
    active = list(range(C.ACTIVE_JOINTS))   # joints 0–4
    col_labels = [JOINT_LABELS[j] for j in active]

    matrix = np.full((len(shapes), len(active)), np.nan)
    for i, shape in enumerate(shapes):
        subset = [s for s in summaries if s["traj_meta"]["shape"] == shape]
        for k, j in enumerate(active):
            vals = _collect(subset, j, "nrmse")
            if vals:
                matrix[i, k] = np.median(vals)

    fig, ax = plt.subplots(figsize=(9, max(4, len(shapes) * 0.55 + 1.5)), dpi=C.DPI)
    cmap = plt.cm.RdYlGn_r       # Green (0) → Yellow → Red (high)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1.0, aspect="auto")

    # Annotate cells
    for i in range(len(shapes)):
        for k in range(len(active)):
            v = matrix[i, k]
            if not np.isnan(v):
                txt = f"{v*100:.0f}%"
                color = "white" if v > 0.6 else "black"
                ax.text(k, i, txt, ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_xticks(range(len(active)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(shapes)))
    ax.set_yticklabels(shapes, fontsize=8)
    ax.set_title("Median NRMSE by Trajectory Shape × Joint\n"
                 "(lower % = model better explains the load)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Median NRMSE")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    plt.tight_layout()
    _savefig(fig, os.path.join(save_dir, "accuracy_by_shape.png"))


# ─────────────────────────────────────────────────────────────
# 6.  Accuracy by trajectory profile type — bar chart
# ─────────────────────────────────────────────────────────────

def plot_accuracy_by_traj_type(summaries: list, save_dir: str):
    """
    Grouped bar chart: trajectory type (x) × joint (colour).
    Bar height = median NRMSE.
    """
    traj_types = sorted({s["traj_meta"]["traj_type"] for s in summaries})
    active     = list(range(C.ACTIVE_JOINTS))
    n_tt       = len(traj_types)
    n_j        = len(active)

    matrix = np.full((n_tt, n_j), np.nan)
    counts = []
    for i, tt in enumerate(traj_types):
        subset = [s for s in summaries if s["traj_meta"]["traj_type"] == tt]
        counts.append(len(subset))
        for k, j in enumerate(active):
            vals = _collect(subset, j, "nrmse")
            if vals:
                matrix[i, k] = np.median(vals)

    width   = 0.14
    x       = np.arange(n_tt)
    offsets = np.linspace(-(n_j - 1) * width / 2, (n_j - 1) * width / 2, n_j)

    fig, ax = plt.subplots(figsize=(max(8, n_tt * 1.6), 5), dpi=C.DPI)
    for k, j in enumerate(active):
        vals = matrix[:, k]
        bars = ax.bar(x + offsets[k], np.where(np.isnan(vals), 0, vals),
                      width, color=JOINT_COLORS[j], alpha=0.8,
                      label=JOINT_LABELS[j])

    ax.axhline(0.2, color="green",  lw=1, ls="--", alpha=0.7, label="20% threshold")
    ax.axhline(0.5, color="orange", lw=1, ls="--", alpha=0.7, label="50% threshold")

    xticks = [f"{tt}\n(n={c})" for tt, c in zip(traj_types, counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=8)
    ax.set_ylabel("Median NRMSE")
    ax.set_title("Model NRMSE by Trajectory Profile Type\n"
                 "(Ruckig, Quintic Poly, Cubic Poly, Trapezoidal)")
    ax.set_ylim(0, max(np.nanmax(matrix) * 1.2, 0.6))
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, os.path.join(save_dir, "accuracy_by_traj_type.png"))


# ─────────────────────────────────────────────────────────────
# 7.  NRMSE vs radius
# ─────────────────────────────────────────────────────────────

def plot_accuracy_vs_radius(summaries: list, save_dir: str):
    """
    Scatter: trajectory radius (mm) on x, NRMSE on y, for joints 1–4.
    Shows whether larger / faster motions degrade model accuracy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=C.DPI)
    axes = axes.flat

    for idx, j in enumerate(range(4)):
        ax = axes[idx]
        radii = []
        nrmse = []
        for s in summaries:
            r = s["traj_meta"].get("radius_mm")
            v = s["joints"][j].get("nrmse")
            if r is not None and v is not None:
                radii.append(r)
                nrmse.append(v)

        if not radii:
            ax.set_visible(False)
            continue

        radii = np.array(radii, dtype=float)
        nrmse = np.array(nrmse, dtype=float)

        ax.scatter(radii, nrmse, s=20, color=JOINT_COLORS[j], alpha=0.6,
                   linewidths=0)
        ax.axhline(0.2, color="green",  lw=1, ls="--", alpha=0.7)
        ax.axhline(0.5, color="orange", lw=1, ls="--", alpha=0.7)

        # Trend line
        if len(radii) > 2:
            z = np.polyfit(radii, nrmse, 1)
            xr = np.linspace(radii.min(), radii.max(), 50)
            ax.plot(xr, np.polyval(z, xr), "k-", lw=1.2, alpha=0.7,
                    label=f"slope={z[0]:.4f}")
            ax.legend(fontsize=7)

        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel("NRMSE")
        ax.set_title(JOINT_LABELS[j], fontsize=9)
        ax.set_ylim(0, min(nrmse.max() * 1.3, 2.0))
        ax.grid(True, alpha=0.3)

    fig.suptitle("NRMSE vs Trajectory Radius\n"
                 "(rising trend → model degrades at larger motions)", fontsize=10)
    plt.tight_layout()
    _savefig(fig, os.path.join(save_dir, "accuracy_vs_radius.png"))


# ─────────────────────────────────────────────────────────────
# 8.  CDF of NRMSE — model coverage
# ─────────────────────────────────────────────────────────────

def plot_model_coverage_cdf(summaries: list, save_dir: str):
    """
    CDF curves: fraction of runs vs NRMSE threshold, per joint.
    A curve shifted left → model performs better on that joint.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=C.DPI)

    thresholds = np.linspace(0, 2.0, 200)

    for j in range(C.ACTIVE_JOINTS):
        vals = np.array(_collect(summaries, j, "nrmse"), dtype=float)
        if len(vals) == 0:
            continue
        cdf = np.array([np.mean(vals <= t) for t in thresholds])
        ax.plot(thresholds, cdf, color=JOINT_COLORS[j],
                lw=1.8, label=JOINT_LABELS[j])

    ax.axvline(0.2, color="green",  lw=1, ls="--", alpha=0.7, label="20%")
    ax.axvline(0.5, color="orange", lw=1, ls="--", alpha=0.7, label="50%")
    ax.set_xlabel("NRMSE threshold")
    ax.set_ylabel("Fraction of runs below threshold")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Coverage CDF\n"
                 "(curves further left → model explains more runs well)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _savefig(fig, os.path.join(save_dir, "model_coverage_cdf.png"))


# ─────────────────────────────────────────────────────────────
# 9.  Error histograms — global (all runs combined)
# ─────────────────────────────────────────────────────────────

# Colour + style for the three model types (consistent across all hist plots)
_HIST_STYLES = {
    "rnea":  {"color": "#2166ac", "label": "RNEA only",          "alpha": 0.55},
    "model": {"color": "#1a9641", "label": "RNEA + Friction",    "alpha": 0.55},
    "fric":  {"color": "#d7191c", "label": "Friction only",      "alpha": 0.55},
}


def _plot_hist_axes(ax, bin_edges, hists_dict, title, log_scale=False):
    """
    Draw three normalised error histograms on ax.

    Parameters
    ----------
    bin_edges   : array of 101 bin edge values
    hists_dict  : {"rnea": counts, "model": counts, "fric": counts}
    """
    bin_edges = np.asarray(bin_edges)
    centers   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width     = bin_edges[1] - bin_edges[0]

    for key in ("rnea", "fric", "model"):   # draw model on top
        counts = np.asarray(hists_dict.get(key, []), dtype=float)
        if counts.sum() == 0:
            continue
        density = counts / (counts.sum() * width)   # probability density
        st = _HIST_STYLES[key]
        ax.bar(centers, density, width=width * 0.95,
               color=st["color"], alpha=st["alpha"], label=st["label"])

    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Error  [N·m]", fontsize=8)
    ax.set_ylabel("Probability density", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, axis="y", alpha=0.25)
    if log_scale:
        ax.set_yscale("log")


def plot_error_histograms_global(error_histograms: dict, save_dir: str):
    """
    2×3 grid (J1–J5 + summary stats panel).  Each subplot overlays the error
    histograms for three model variants:

      RNEA only      — τ_load − τ_RNEA
      RNEA+Friction  — τ_load − (τ_RNEA + τ_fric)   [full model residual]
      Friction only  — τ_load − τ_fric

    Data source: pre-aggregated histogram counts from global_summary["error_histograms"].
    """
    if not error_histograms:
        return

    bin_edges = np.asarray(error_histograms["bin_edges"])
    global_h  = error_histograms.get("global", {})
    n_active  = C.ACTIVE_JOINTS   # 5

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=C.DPI)
    axes_flat = axes.flat

    for j in range(6):
        ax = next(axes_flat)
        if j >= n_active:
            # Last panel: summary table of RMS errors
            ax.axis("off")
            rows = [["Joint", "RNEA err\nRMS", "RNEA+Fric\nRMS", "Fric-only\nRMS"]]
            for jj in range(n_active):
                h = global_h.get(str(jj), {})
                def _rms(key):
                    c = np.asarray(h.get(key, []), dtype=float)
                    if c.sum() == 0:
                        return "N/A"
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    density = c / c.sum()
                    return f"{np.sqrt(np.sum(density * centers**2)):.3f}"
                rows.append([JOINT_LABELS[jj], _rms("rnea"), _rms("model"), _rms("fric")])
            tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                           loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.6)
            ax.set_title("Weighted RMS of error\n(from histogram)", fontsize=9)
            continue

        h = global_h.get(str(j), {})
        if not h:
            ax.set_visible(False)
            continue

        _plot_hist_axes(ax, bin_edges, h,
                        title=f"{JOINT_LABELS[j]}  —  all runs combined")

    # Shared legend at figure level
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_HIST_STYLES[k]["color"],
                       alpha=0.7, label=_HIST_STYLES[k]["label"])
        for k in ("rnea", "fric", "model")
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=9,
               title="Model variant", framealpha=0.9)

    fig.suptitle(
        "Error Histogram per Joint — All Runs Combined\n"
        "Error = τ_load − τ_model  |  Narrower / taller = better model",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    _savefig(fig, os.path.join(save_dir, "error_hist_global.png"))


# ─────────────────────────────────────────────────────────────
# 10. Error histograms — per trajectory shape
# ─────────────────────────────────────────────────────────────

def plot_error_histograms_by_shape(error_histograms: dict, save_dir: str):
    """
    One figure per active joint.  Each figure has one subplot per
    trajectory shape, overlaying the three model error histograms.

    This reveals whether model accuracy depends on the motion shape
    (e.g., circular vs sinusoidal vs spiral).
    """
    if not error_histograms:
        return

    bin_edges = np.asarray(error_histograms["bin_edges"])
    by_shape  = error_histograms.get("by_shape", {})
    if not by_shape:
        return

    shapes    = sorted(by_shape.keys())
    n_shapes  = len(shapes)

    for j in range(C.ACTIVE_JOINTS):
        # Grid: ceil(n_shapes/3) rows × 3 cols
        n_cols = 3
        n_rows = (n_shapes + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 5, n_rows * 3.2),
                                 dpi=C.DPI)
        axes_flat = np.array(axes).flat

        for idx, shape in enumerate(shapes):
            ax  = next(axes_flat)
            h   = by_shape[shape].get(str(j), {})
            cnt = sum(np.asarray(h.get(k, [0]), dtype=float).sum()
                      for k in ("rnea", "model", "fric"))

            if cnt == 0:
                ax.set_visible(False)
                continue

            _plot_hist_axes(ax, bin_edges, h, title=shape)

        # Hide any spare axes
        for ax in axes_flat:
            ax.set_visible(False)

        # Shared legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=_HIST_STYLES[k]["color"],
                           alpha=0.7, label=_HIST_STYLES[k]["label"])
            for k in ("rnea", "fric", "model")
        ]
        fig.legend(handles=handles, loc="lower right", fontsize=9,
                   title="Model variant", framealpha=0.9)

        fig.suptitle(
            f"{JOINT_LABELS[j]}  —  Error Histogram by Trajectory Shape\n"
            "Error = τ_load − τ_model  |  Narrower = better model fit",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()
        _savefig(fig, os.path.join(save_dir, f"error_hist_by_shape_J{j+1}.png"))


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def generate_all_global_plots(summaries: list, save_dir: str,
                              error_histograms: dict = None):
    """Generate all global plots and save to save_dir."""
    os.makedirs(save_dir, exist_ok=True)

    steps = [
        ("RNEA ratio violin",         plot_rnea_ratio_violin),
        ("NRMSE violin",              plot_nrmse_violin),
        ("Residual RMS boxplot",      plot_residual_rms_boxplot),
        ("Load vs RNEA scatter",      plot_load_vs_rnea_scatter),
        ("Accuracy by shape",         plot_accuracy_by_shape),
        ("Accuracy by traj type",     plot_accuracy_by_traj_type),
        ("NRMSE vs radius",           plot_accuracy_vs_radius),
        ("Model coverage CDF",        plot_model_coverage_cdf),
    ]

    for name, fn in steps:
        try:
            fn(summaries, save_dir)
            print(f"  [global] {name} — OK")
        except Exception as e:
            print(f"  [global] {name} — FAILED: {e}")

    # Histogram plots (need pre-aggregated data from build_global_summary)
    if error_histograms:
        for name, fn in [
            ("Error hist (global)",    plot_error_histograms_global),
            ("Error hist (by shape)",  plot_error_histograms_by_shape),
        ]:
            try:
                fn(error_histograms, save_dir)
                print(f"  [global] {name} — OK")
            except Exception as e:
                print(f"  [global] {name} — FAILED: {e}")
