"""Central plot configuration.

Every size, colour and label used by any figure lives here.  Figures never
hardcode these; they read ``CONFIG.<field>``.  A figure customises only the
deltas it needs via ``dataclasses.replace(default_config(), fig_w=..., ...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace  # noqa: F401  (replace re-exported)
from pathlib import Path

from .bootstrap import CACHE_DIR, FIGURES_DIR, TABLES_DIR

# Canonical architecture keys used everywhere downstream.
ARCH_EDR = "edr"
ARCH_FNN = "fnn"
ARCH_PHYSREG = "physreg"


@dataclass(frozen=True)
class PlotConfig:
    # ---- figure sizing (inches) ----------------------------------------
    fig_w: float = 7.2
    fig_h: float = 4.6
    dpi_screen: int = 110
    dpi_save: int = 300

    # ---- typography (deliberately large for print legibility) ----------
    font_family: str = "serif"
    # Times New Roman is primary; STIX is a Times-metric-compatible fallback so
    # math text and any missing glyphs still render in a Times-like face.
    font_serif: tuple[str, ...] = ("Times New Roman", "STIXGeneral", "DejaVu Serif", "serif")
    mathtext_fontset: str = "stix"
    axes_label_size: float = 16.0
    axes_label_weight: str = "normal"
    tick_label_size: float = 14.0
    legend_size: float = 14.0
    annot_size: float = 12.0

    # ---- lines / markers / bars ----------------------------------------
    line_w: float = 2.4
    marker_size: float = 8.0
    err_capsize: float = 4.0
    err_linewidth: float = 1.6
    bar_width: float = 0.26
    violin_width: float = 0.75
    axes_linewidth: float = 1.7
    grid_alpha: float = 0.30
    grid_linewidth: float = 0.8
    scatter_size: float = 9.0
    scatter_alpha: float = 0.35

    # ---- colour ---------------------------------------------------------
    # Colour-blind-safe, high-contrast. EDR is the warm/strong colour so it
    # reads as the protagonist; baselines are cooler/neutral.
    arch_colors: dict[str, str] = field(default_factory=lambda: {
        ARCH_FNN: "#4C72B0",      # blue
        ARCH_PHYSREG: "#55A868",  # green
        ARCH_EDR: "#C44E52",      # red  (emphasis)
    })
    arch_labels: dict[str, str] = field(default_factory=lambda: {
        ARCH_FNN: "FNN",
        ARCH_PHYSREG: "Physics-Reg.",
        ARCH_EDR: "EDR",
    })
    arch_markers: dict[str, str] = field(default_factory=lambda: {
        ARCH_FNN: "o",
        ARCH_PHYSREG: "s",
        ARCH_EDR: "D",
    })
    # Plot order: baselines first, EDR last so it sits rightmost / on top.
    arch_order: tuple[str, ...] = (ARCH_FNN, ARCH_PHYSREG, ARCH_EDR)
    # EDR gets a slightly heavier line / lower alpha so it pops.
    emphasis_arch: str = ARCH_EDR
    emphasis_line_scale: float = 1.35
    truth_color: str = "#222222"
    truth_linewidth: float = 2.0
    reference_color: str = "#888888"
    parity_cmap: str = "viridis"
    # Heatmap colour stops: explicit best→mid→worst (green → pale → red).
    # The heatmap helper normalises globally and direction-aware so that the
    # numerically best cell is always green regardless of the metric.
    heatmap_best: str = "#1A9850"   # green  (best result)
    heatmap_mid: str = "#FFFFBF"    # pale   (middle)
    heatmap_worst: str = "#D73027"  # red    (worst result)

    # ---- labels ---------------------------------------------------------
    joint_names: tuple[str, ...] = ("J1", "J2", "J3", "J4", "J5")
    torque_unit: str = "N·m"
    rmse_label: str = "Test RMSE (N·m)"
    epoch_label: str = "Epoch"

    # ---- behaviour ------------------------------------------------------
    split: str = "test"
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    champion_metric: str = "rmse_traj_macro"
    # Which run becomes each architecture's "champion" for the inference-backed
    # figures (06-12, heatmaps, trajectory, headline). ``champion_basis``:
    #   "test"   -> lowest test RMSE        (default; matches the headline)
    #   "val"    -> lowest validation RMSE  (no test-set selection leakage)
    #   "train"  -> lowest train RMSE @ best-val epoch
    #   "global" -> lowest test RMSE over ALL data fractions (ignores the
    #               full-data restriction; picks reduced-data winners)
    # The make_tables champion_selection.csv reports all four side by side.
    champion_basis: str = "test"
    champion_full_data_only: bool = True   # restrict champions to frac == 1.0
    n_worst_traj: int = 1
    figures_dir: Path = FIGURES_DIR
    tables_dir: Path = TABLES_DIR
    cache_dir: Path = CACHE_DIR
    enforce_no_title: bool = True
    enforce_no_panel_label: bool = True
    # Uniform legend placement: one horizontal frameless strip centred *above*
    # the axes/figure, outside the data area, so it can never overlap data.
    legend_anchor_y: float = 1.02
    legend_frameon: bool = False
    table_float_fmt: str = "%.6g"

    # ---- Savitzky–Golay smoothing --------------------------------------
    # On by default and *replacing* the raw series in the plot (the drawn
    # line IS the smoothed signal). Tables always keep the RAW data.
    # Disable per-run with ``--config-override savgol_enabled=false``.
    savgol_enabled: bool = True
    savgol_window: int = 7
    savgol_polyorder: int = 2

    # ---- trajectory selection (fig10) ----------------------------------
    # None  -> auto (plotting.pick_trajectory, an information-rich curve)
    # int   -> index into the split's trajectory list
    # str   -> first trajectory whose geometry name matches
    trajectory_select: "int | str | None" = None


def default_config() -> PlotConfig:
    """The canonical configuration instance."""
    return PlotConfig()
