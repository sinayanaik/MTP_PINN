"""Small shared plotting helpers (legends, annotations, trajectory picking).

Kept deliberately thin: figures own their layout; this only removes
boilerplate that would otherwise be copy-pasted 20+ times.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .config import PlotConfig
from . import palette


def arch_proxy_handles(cfg: PlotConfig, archs: Sequence[str], *, kind: str = "line") -> list:
    """Legend handles labelled with arch display names, in canonical order."""
    handles = []
    for a in archs:
        c = palette.color(cfg, a)
        lbl = palette.label(cfg, a)
        if kind == "patch":
            handles.append(Patch(facecolor=c, edgecolor="black", label=lbl))
        else:
            handles.append(Line2D([0], [0], color=c, lw=cfg.line_w,
                                   marker=palette.marker(cfg, a),
                                   markersize=cfg.marker_size, label=lbl))
    return handles


def annotate_bars(ax, bars, values, cfg: PlotConfig, fmt: str = "{:.4f}") -> None:
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=cfg.annot_size)


def top_legend(target, handles, cfg: PlotConfig, *, ncol: int | None = None,
                anchor_y: float | None = None):
    """One horizontal, frameless legend centred *above* ``target``.

    ``target`` may be an Axes (legend sits above that axes) or a Figure
    (legend spans the whole figure — used for multi-panel figures so a single
    shared legend sits above every panel).  Placement is uniform everywhere:
    one row, entries side-by-side, never overlapping the data area.
    """
    y = cfg.legend_anchor_y if anchor_y is None else anchor_y
    n = ncol if ncol is not None else len(handles)
    return target.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=max(1, n),
        frameon=cfg.legend_frameon,
        fontsize=cfg.legend_size,
        handlelength=1.8,
        columnspacing=1.4,
        borderaxespad=0.0,
    )


def maybe_smooth(y, cfg: PlotConfig):
    """Savitzky–Golay smoothing of a 1-D series, controlled by ``cfg``.

    On by default and *replacing* the raw series (the returned array is what
    gets plotted). Window is clamped to be odd and within
    ``polyorder < window <= len(y)``; too-short or disabled series are
    returned unchanged. Tables must use the RAW data, never this.
    """
    y = np.asarray(y, dtype=float)
    if not cfg.savgol_enabled or y.ndim != 1:
        return y
    n = y.size
    po = max(1, int(cfg.savgol_polyorder))
    if n < po + 2:
        return y
    w = int(cfg.savgol_window)
    w = min(w, n if n % 2 == 1 else n - 1)   # <= len, odd
    if w % 2 == 0:
        w -= 1
    if w <= po:
        w = po + 1 + (po % 2 == 0)            # smallest odd > po
        if w > n:
            return y
    from scipy.signal import savgol_filter
    return savgol_filter(y, window_length=w, polyorder=po)


def heatmap(ax, matrix, row_labels, col_labels, cfg: PlotConfig, *,
            lower_is_better: bool, value_fmt: str = "{:.4f}",
            cbar_label: str = ""):
    """Annotated arch×metric heatmap with a custom best→worst colour scale.

    Colours run green (best) → pale → red (worst), normalised **globally**
    across the whole matrix and direction-aware: with ``lower_is_better`` the
    matrix minimum is green, otherwise the maximum is green.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    m = np.asarray(matrix, dtype=float)
    nr, nc = m.shape
    cmap = LinearSegmentedColormap.from_list(
        "best_worst", [cfg.heatmap_best, cfg.heatmap_mid, cfg.heatmap_worst])
    lo, hi = float(np.nanmin(m)), float(np.nanmax(m))
    span = (hi - lo) or 1.0
    # t = 0 at the best cell (-> green), 1 at the worst (-> red).
    t = (m - lo) / span if lower_is_better else (hi - m) / span

    # Drawn as vector Rectangle patches (not imshow): the matplotlib PDF
    # backend renders a tiny imshow incorrectly (pale/blank), whereas filled
    # patches are reliable and stay vector.
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                    facecolor=cmap(float(t[i, j])),
                                    edgecolor="white", linewidth=1.5,
                                    zorder=1))
            ax.annotate(value_fmt.format(m[i, j]), (j, i), ha="center",
                        va="center", fontsize=cfg.annot_size, color="black",
                        zorder=2)
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)            # row 0 at the top, like a table
    ax.set_xticks(range(nc))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(row_labels)
    ax.set_aspect("auto")
    ax.grid(False)
    ax.tick_params(length=0)

    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04,
                              ticks=[0.05, 0.95])
    cbar.ax.set_yticklabels(["best", "worst"])
    if cbar_label:
        cbar.set_label(cbar_label)
    return sm


def select_trajectory(traj: list[tuple[int, int, str]], cfg: PlotConfig
                       ) -> tuple[int, int, str]:
    """Resolve ``cfg.trajectory_select`` to one ``(start, end, geom)`` tuple.

    None -> auto (:func:`pick_trajectory`); int -> index; str -> first
    trajectory whose geometry name matches.
    """
    sel = cfg.trajectory_select
    if sel is None:
        return pick_trajectory(traj)
    if isinstance(sel, int):
        return traj[sel]
    matches = [t for t in traj if t[2] == sel]
    if not matches:
        raise ValueError(
            f"trajectory_select={sel!r}: no trajectory with that geometry; "
            f"available: {sorted({t[2] for t in traj})}")
    return max(matches, key=lambda t: t[1] - t[0])


def pick_trajectory(traj: list[tuple[int, int, str]],
                    prefer: Sequence[str] = ("helix", "lissajous", "spiral", "ellipse")) -> tuple[int, int, str]:
    """A representative, information-rich trajectory for time-series plots.

    Prefer a geometrically rich curve; among matches take the longest; if none
    match, take the median-length trajectory.
    """
    for g in prefer:
        cands = [t for t in traj if t[2] == g]
        if cands:
            return max(cands, key=lambda t: t[1] - t[0])
    by_len = sorted(traj, key=lambda t: t[1] - t[0])
    return by_len[len(by_len) // 2]


def per_traj_rmse(pred: np.ndarray, target: np.ndarray,
                  traj: list[tuple[int, int, str]]) -> list[float]:
    """Trajectory-macro RMSE per trajectory (mean over joints of per-joint RMSE)."""
    out = []
    for s, e, _ in traj:
        d = pred[s:e] - target[s:e]
        out.append(float(np.sqrt((d ** 2).mean(axis=0)).mean()))
    return out


def champion_results(cfg: PlotConfig) -> dict[str, dict[str, Any]]:
    """{arch: predict_split result} for the three champions (cache-warmed)."""
    from .inference import prefetch_all
    return prefetch_all(cfg)
