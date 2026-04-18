#!/usr/bin/env python3
"""Interactive visualization of hardware execution log files (JSON).

Features:
  - File browser lists all .json logs in the selected directory tree.
  - Clicking a file opens a new visualization window (non-blocking).
  - Multiple visualization windows can be open simultaneously.
  - Each window shows metadata + two stacked plot panels with selectable data sources.
  - The main file browser remains open and usable.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("tkinter is required", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.ndimage import uniform_filter1d
    HAS_UNIFORM_FILTER = True
except ImportError:
    HAS_UNIFORM_FILTER = False

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except ImportError:
    HAS_SAVGOL = False

DEFAULT_LOG_DIR = str("raw_samples")
DEFAULT_MAX_TORQUE_KGCM = 30.0
KGCM_TO_NM = 0.0980665
ST3215_CURRENT_STEP_A = 0.0065
ST3215_TORQUE_CONSTANT_KGCM_PER_A = 11.0
ST3215_TORQUE_CONSTANT_NM_PER_A = ST3215_TORQUE_CONSTANT_KGCM_PER_A * KGCM_TO_NM

__all__ = ["load_log_json", "visualize"]


# ═══════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════

def load_log_json(path: str) -> dict:
    """Load a JSON execution log and extract arrays for plotting.

    Supports both v4 (raw ticks) and v3 (radians) schemas.
    """
    with open(path) as f:
        data = json.load(f)
    entries = data.get("log", [])
    if not entries:
        return {}
    n = len(entries)
    t = np.array([e["t"] for e in entries])

    def _arr(key, default_len=6):
        return np.array([e.get(key, [0] * default_len) for e in entries])

    schema = data.get("schema_version", "")
    ticks_to_rad = data.get("ticks_to_rad", 2.0 * np.pi / 4095.0)
    joint_mapping = data.get("joint_mapping", [])

    result = {
        "path": path,
        "raw": data,
        "schema_version": schema,
        "run_id": data.get("run_id", Path(path).stem),
        "label": data.get("label", ""),
        "success": data.get("success", False),
        "status": data.get("status", ""),
        "port": data.get("port", ""),
        "dof": data.get("dof", 0),
        "control_frequency_hz": data.get("control_frequency_hz", 0),
        "logging_frequency_hz": data.get("logging_frequency_hz", 0),
        "move_to_start_sec": data.get("move_to_start_sec", 0),
        "actual_duration_sec": data.get("actual_duration_sec", 0),
        "actual_control_hz": data.get("actual_control_hz", 0),
        "actual_feedback_hz": data.get("actual_feedback_hz", 0),
        "rows/duration_hz": data.get("rows/duration_hz", 0),
        "servo_ids": data.get("servo_ids", []),
        "geometry": data.get("geometry", {}),
        "trajectory": data.get("trajectory", {}),
        "tracking_quality": data.get("tracking_quality", {}),
        "num_log_entries": data.get("num_log_entries", n),
        "ticks_to_rad": ticks_to_rad,
        "joint_mapping": joint_mapping,
        "t": t,
        "has_current": any("current" in entry for entry in entries),
        "has_voltage": any("voltage" in entry for entry in entries),
    }

    if "hwrl_execution_log_v4" in schema:
        # v4: all values are raw ticks / register integers
        result.update({
            "cmd_pos": _arr("cmd_pos"),
            "cmd_vel": _arr("cmd_vel"),
            "cmd_acc": _arr("cmd_acc"),
            "act_pos": _arr("act_pos"),
            "act_vel": _arr("act_vel"),
            "load": _arr("load"),
            "current": _arr("current"),
            "voltage": _arr("voltage"),
            "cmd_ee": _arr("cmd_ee", 3),
            "act_ee": _arr("act_ee", 3),
            "ee_err": np.array([e.get("ee_err", 0) for e in entries]),
        })
    else:
        # v3 fallback: convert radian fields to ticks for uniform handling
        des_pos_rad = _arr("des_pos")
        act_pos_rad = _arr("act_pos")
        result.update({
            "cmd_pos": des_pos_rad / ticks_to_rad if ticks_to_rad else des_pos_rad,
            "cmd_vel": _arr("des_vel"),  # no clean tick conversion for v3, keep as-is
            "cmd_acc": _arr("des_acc"),
            "act_pos": act_pos_rad / ticks_to_rad if ticks_to_rad else act_pos_rad,
            "act_vel": _arr("act_spd"),
            "load": _arr("load"),
            "current": _arr("current"),
            "voltage": _arr("voltage"),
            "cmd_ee": _arr("des_ee", 3),
            "act_ee": _arr("act_ee", 3),
            "ee_err": np.array([e.get("ee_err", 0) for e in entries]),
        })

    result["load_register_note"] = data.get("load_register_note", "")
    result["current_note"] = data.get("current_note", data.get("current_register_note", ""))
    result["voltage_note"] = data.get("voltage_note", data.get("voltage_register_note", ""))

    return result


def _kgcm_to_nm(value):
    return float(value) * KGCM_TO_NM


def _infer_load_fraction_scale(load_values, load_register_note):
    max_abs_load = float(np.nanmax(np.abs(load_values))) if np.size(load_values) else 0.0
    load_register_note_lower = str(load_register_note).lower()
    if "0.1" in str(load_register_note) and "percentage" in load_register_note_lower:
        return 0.001, "raw * 0.1% => load_fraction = raw * 0.001"
    if max_abs_load <= 100.0 + 1e-9:
        return 0.01, "direct % load => load_fraction = raw * 0.01"
    return 1.0 / 2048.0, "fallback load fraction => load_fraction = raw / 2048"


def _infer_current_amp_scale(current_values, current_note):
    note_lower = str(current_note).lower()
    finite_values = np.asarray(current_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if "ma" in note_lower:
        return 1.0e-3, "mA", "current logged in mA => I[A] = current * 0.001"
    if "raw" in note_lower or "register" in note_lower:
        return ST3215_CURRENT_STEP_A, "raw", "raw current units => I[A] = current * 0.0065"
    if finite_values.size:
        quantized_ma = finite_values / 6.5
        if np.nanmax(np.abs(quantized_ma - np.round(quantized_ma))) < 1e-6:
            return (
                1.0e-3,
                "mA",
                "6.5 mA steps detected => current treated as mA, I[A] = current * 0.001",
            )
    return ST3215_CURRENT_STEP_A, "raw", "raw current units => I[A] = current * 0.0065"


def _fill_zero_signs(sign_values):
    sign_values = np.asarray(sign_values, dtype=float)
    if sign_values.ndim == 1:
        sign_values = sign_values.reshape(-1, 1)
    filled = np.sign(sign_values).copy()
    for joint_idx in range(filled.shape[1]):
        column = filled[:, joint_idx]
        nonzero_idx = np.flatnonzero(column)
        if nonzero_idx.size == 0:
            column[:] = 1.0
            continue
        first_idx = int(nonzero_idx[0])
        column[:first_idx] = column[first_idx]
        last_sign = column[first_idx]
        for sample_idx in range(first_idx + 1, len(column)):
            if column[sample_idx] == 0.0:
                column[sample_idx] = last_sign
            else:
                last_sign = column[sample_idx]
        filled[:, joint_idx] = column
    return filled


# ═══════════════════════════════════════════════════════════
#  Plot Data Sources
# ═══════════════════════════════════════════════════════════

JOINT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]
JOINT_SCOPE_ALL = "All"
OVERLAY_NONE_LABEL = "None"
DEFAULT_LINE_WIDTH = 1.4
PRIMARY_LINE_ALPHA = 0.95
SECONDARY_LINE_ALPHA = 0.48
DEFAULT_MARKER_SIZE = 3.2
DEFAULT_MARKER_EDGE_WIDTH = 0.9
CALCULATED_SOURCE_LABELS = {
    "Calculated Velocity (from pos)",
    "Calculated Acceleration (from pos)",
}
FILTER_METHODS = ["None", "Finite Difference", "Savitzky-Golay"]
FILTER_METHOD_KEYS = ["none", "finite_diff", "savgol"]
FILTER_METHOD_LABEL_BY_KEY = dict(zip(FILTER_METHOD_KEYS, FILTER_METHODS))
WINDOWED_FILTER_METHOD_KEYS = {"finite_diff", "savgol"}

# Unit modes
UNIT_TICKS = 0
UNIT_RAD = 1
UNIT_DEG = 2
UNIT_LABELS = ["Ticks", "Radians", "Degrees"]


@dataclass(frozen=True)
class FilterConfig:
    method: str = "none"
    window: int = 11
    polyorder: int = 3

    @property
    def label(self) -> str:
        return FILTER_METHOD_LABEL_BY_KEY.get(self.method, self.method)


def _normalized_window(window, sample_count):
    """Clamp window to an odd value that fits the available samples."""
    if sample_count <= 0:
        return 1
    try:
        window = int(window)
    except (TypeError, ValueError):
        window = 11
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    max_window = sample_count if sample_count % 2 == 1 else sample_count - 1
    if max_window < 3:
        return sample_count
    return min(window, max_window)


def _normalized_polyorder(polyorder, window, minimum=0):
    """Keep Savitzky-Golay polyorder valid for the chosen window/derivative."""
    try:
        polyorder = int(polyorder)
    except (TypeError, ValueError):
        polyorder = 3
    polyorder = max(minimum, polyorder)
    return min(polyorder, max(minimum, window - 1))


def _sample_period(t):
    """Estimate the nominal sample period for nearly-uniform logs."""
    if len(t) < 2:
        return 1.0
    dt = np.diff(np.asarray(t, dtype=float))
    dt = dt[dt > 0]
    if dt.size == 0:
        return 1.0
    return float(np.median(dt))


def _supports_calculated_series(filter_cfg: FilterConfig) -> bool:
    if filter_cfg.method == "finite_diff":
        return True
    if filter_cfg.method == "savgol":
        return HAS_SAVGOL
    return False


def _calculated_series_note(filter_cfg: FilterConfig) -> str:
    if filter_cfg.method == "savgol" and not HAS_SAVGOL:
        return "Savitzky-Golay requires scipy.signal"
    return "requires Finite Difference or Savitzky-Golay"


def _source_display_label(source_label, filter_cfg: FilterConfig | None = None):
    if not filter_cfg or filter_cfg.method == "none":
        return source_label
    return f"{source_label} [{filter_cfg.label}]"


def _convert_pos(data, ticks_to_rad, unit):
    """Convert position ticks to requested unit."""
    if unit == UNIT_RAD:
        return data * ticks_to_rad
    elif unit == UNIT_DEG:
        return data * ticks_to_rad * (180.0 / np.pi)
    return data  # ticks


def _convert_vel(data, ticks_to_rad, unit):
    """Convert velocity in ticks/s to requested unit."""
    if unit == UNIT_RAD:
        return data * ticks_to_rad
    elif unit == UNIT_DEG:
        return data * ticks_to_rad * (180.0 / np.pi)
    return data  # ticks/s


def _convert_acc(data, ticks_to_rad, unit):
    """Convert acceleration in ticks/s² to requested unit."""
    if unit == UNIT_RAD:
        return data * ticks_to_rad
    elif unit == UNIT_DEG:
        return data * ticks_to_rad * (180.0 / np.pi)
    return data  # ticks/s²


def _pos_ylabel(unit):
    if unit == UNIT_RAD:
        return "position [rad]"
    elif unit == UNIT_DEG:
        return "position [deg]"
    return "position [ticks]"


def _vel_ylabel(unit):
    if unit == UNIT_RAD:
        return "velocity [rad/s]"
    elif unit == UNIT_DEG:
        return "velocity [deg/s]"
    return "velocity [ticks/s]"


def _acc_ylabel(unit):
    if unit == UNIT_RAD:
        return "acceleration [rad/s²]"
    elif unit == UNIT_DEG:
        return "acceleration [deg/s²]"
    return "acceleration [ticks/s²]"


def _build_data_sources(d: dict, unit: int = UNIT_TICKS,
                        filter_cfg: FilterConfig | None = None,
                        joint_selection: str = JOINT_SCOPE_ALL) -> list:
    """Return list of (label, data_fn) for all plottable quantities.

    unit: UNIT_TICKS, UNIT_RAD, or UNIT_DEG — controls position/velocity/acceleration display.
    filter_cfg: controls smoothing/differentiation per subplot or overlay.
    """
    if filter_cfg is None:
        filter_cfg = FilterConfig()
    t = d["t"]
    nj = d["cmd_pos"].shape[1]
    ttr = d.get("ticks_to_rad", 2.0 * np.pi / 4095.0)
    window = _normalized_window(filter_cfg.window, len(t))
    polyorder = _normalized_polyorder(filter_cfg.polyorder, window)
    joint_labels = [f"J{j+1}" for j in range(nj)]
    joint_indices = list(range(nj))
    if isinstance(joint_selection, str) and joint_selection != JOINT_SCOPE_ALL:
        try:
            selected_idx = int(joint_selection.lstrip("Jj")) - 1
        except ValueError:
            selected_idx = None
        if selected_idx is not None and 0 <= selected_idx < nj:
            joint_indices = [selected_idx]
    joint_directions = np.ones(nj, dtype=float)
    for joint_idx, mapping in enumerate(d.get("joint_mapping", [])):
        if joint_idx >= nj:
            break
        joint_directions[joint_idx] = float(mapping.get("direction", 1.0))
    load_scale, _load_torque_note = _infer_load_fraction_scale(
        d.get("load", np.zeros((len(t), nj), dtype=float)),
        d.get("load_register_note", ""),
    )
    current_amp_scale, current_units, _current_torque_note = _infer_current_amp_scale(
        d.get("current", np.zeros((len(t), nj), dtype=float)),
        d.get("current_note", ""),
    )
    max_torque_nm = _kgcm_to_nm(DEFAULT_MAX_TORQUE_KGCM)

    def _smooth_uniform(arr):
        if window < 3:
            return arr
        arr = arr.astype(float, copy=False)
        if not HAS_UNIFORM_FILTER:
            pad = window // 2
            kernel = np.ones(window, dtype=float) / float(window)

            def _smooth_1d(values):
                padded = np.pad(values, (pad, pad), mode="edge")
                return np.convolve(padded, kernel, mode="valid")

            if arr.ndim == 1:
                return _smooth_1d(arr)
            out = np.zeros_like(arr, dtype=float)
            for col in range(arr.shape[1]):
                out[:, col] = _smooth_1d(arr[:, col])
            return out
        if arr.ndim == 1:
            return uniform_filter1d(arr, size=window, mode='nearest')
        out = np.zeros_like(arr, dtype=float)
        for col in range(arr.shape[1]):
            out[:, col] = uniform_filter1d(arr[:, col], size=window, mode='nearest')
        return out

    def _smooth_savgol(arr):
        if not HAS_SAVGOL or window < 3:
            return arr
        arr = arr.astype(float, copy=False)
        return savgol_filter(
            arr,
            window_length=window,
            polyorder=_normalized_polyorder(polyorder, window),
            deriv=0,
            axis=0,
            mode="interp",
        )

    def _smooth(arr):
        """Apply the selected smoothing method to raw series."""
        if filter_cfg.method == "finite_diff":
            return _smooth_uniform(arr)
        if filter_cfg.method == "savgol":
            return _smooth_savgol(arr)
        return arr

    def _gradient(arr):
        if len(t) < 2:
            return np.zeros_like(arr, dtype=float)
        edge_order = 2 if len(t) >= 3 else 1
        return np.gradient(arr, t, axis=0, edge_order=edge_order)

    def _savgol_derivative(arr, deriv):
        if not HAS_SAVGOL or window < 3:
            return np.full_like(arr, np.nan, dtype=float)
        derivative_polyorder = _normalized_polyorder(polyorder, window, minimum=deriv)
        if derivative_polyorder < deriv:
            return np.full_like(arr, np.nan, dtype=float)
        return savgol_filter(
            arr.astype(float, copy=False),
            window_length=window,
            polyorder=derivative_polyorder,
            deriv=deriv,
            delta=_sample_period(t),
            axis=0,
            mode="interp",
        )

    def _joint_specs(arr, alpha=PRIMARY_LINE_ALPHA):
        return [
            {"x": t, "y": arr[:, j], "color": JOINT_COLORS[j % len(JOINT_COLORS)],
             "label": joint_labels[j], "lw": DEFAULT_LINE_WIDTH, "alpha": alpha}
            for j in joint_indices
        ]

    def _joint_raw(key, ylabel):
        """Plot raw register values (no conversion) — for load, etc."""
        arr = _smooth(d[key])
        return (ylabel, _joint_specs(arr))

    def _joint_current():
        if not d.get("has_current", False):
            return ("current [A]", [])
        arr = _smooth(d["current"])
        if current_units == "raw":
            ylabel = "current [raw register]"
        elif current_units == "mA":
            ylabel = "current [mA]"
        else:
            ylabel = "current [A]"
        return (ylabel, _joint_specs(arr))

    def _joint_voltage():
        if not d.get("has_voltage", False):
            return ("voltage [V]", [])
        arr = _smooth(d["voltage"])
        return ("voltage [V]", _joint_specs(arr))

    def _joint_torque():
        load_fraction = _smooth(d["load"]) * joint_directions * load_scale
        load_torque = load_fraction * max_torque_nm
        specs = []
        if d.get("has_current", False):
            current_direction = _fill_zero_signs(_smooth(d["load"]) * joint_directions)
            current_torque = (
                np.abs(_smooth(d["current"])) * current_amp_scale * ST3215_TORQUE_CONSTANT_NM_PER_A * current_direction
            )
        else:
            current_torque = None
        for j in joint_indices:
            color = JOINT_COLORS[j % len(JOINT_COLORS)]
            if current_torque is not None:
                specs.append({
                    "x": t,
                    "y": current_torque[:, j],
                    "color": color,
                    "label": f"{joint_labels[j]} current-derived torque",
                    "lw": DEFAULT_LINE_WIDTH,
                    "alpha": PRIMARY_LINE_ALPHA,
                    "marker": "o",
                })
            specs.append({
                "x": t,
                "y": load_torque[:, j],
                "color": color,
                "label": f"{joint_labels[j]} load-derived torque",
                "lw": DEFAULT_LINE_WIDTH * 0.92,
                "alpha": SECONDARY_LINE_ALPHA,
                "linestyle": "--",
                "marker": None,
            })
        return (
            "torque [N·m]",
            specs,
        )

    def _joint_pos(key):
        arr = _convert_pos(_smooth(d[key]), ttr, unit)
        ylabel = _pos_ylabel(unit)
        return (ylabel, _joint_specs(arr))

    def _joint_vel(key):
        arr = _convert_vel(_smooth(d[key]), ttr, unit)
        ylabel = _vel_ylabel(unit)
        return (ylabel, _joint_specs(arr))

    def _joint_acc(key):
        arr = _convert_acc(_smooth(d[key]), ttr, unit)
        ylabel = _acc_ylabel(unit)
        return (ylabel, _joint_specs(arr))

    def _calc_velocity():
        if not _supports_calculated_series(filter_cfg):
            return (_vel_ylabel(unit), [])
        pos_rad = d["act_pos"] * ttr
        if filter_cfg.method == "savgol":
            # Differentiate position directly with Savitzky-Golay.
            vel = _savgol_derivative(pos_rad, deriv=1)
        else:
            smoothed = _smooth_uniform(pos_rad)
            vel = _gradient(smoothed)
            trim = max(2, window // 2) if window >= 3 else 2
            vel[:trim] = np.nan
            vel[-trim:] = np.nan
        # vel is in rad/s — convert to display unit
        if unit == UNIT_DEG:
            vel = vel * (180.0 / np.pi)
            ylabel = "velocity [deg/s]"
        elif unit == UNIT_TICKS:
            vel = vel / ttr
            ylabel = "velocity [ticks/s]"
        else:
            ylabel = "velocity [rad/s]"
        return (ylabel, _joint_specs(vel))

    def _calc_acceleration():
        if not _supports_calculated_series(filter_cfg):
            return (_acc_ylabel(unit), [])
        pos_rad = d["act_pos"] * ttr
        if filter_cfg.method == "savgol":
            acc = _savgol_derivative(pos_rad, deriv=2)
        else:
            smoothed = _smooth_uniform(pos_rad)
            vel = _gradient(smoothed)
            acc = _gradient(vel)
            trim = max(2, window // 2) if window >= 3 else 2
            acc[:trim] = np.nan
            acc[-trim:] = np.nan
        # acc is in rad/s² — convert to display unit
        if unit == UNIT_DEG:
            acc = acc * (180.0 / np.pi)
            ylabel = "acceleration [deg/s²]"
        elif unit == UNIT_TICKS:
            acc = acc / ttr
            ylabel = "acceleration [ticks/s²]"
        else:
            ylabel = "acceleration [rad/s²]"
        return (ylabel, _joint_specs(acc))

    def _ee_single(key):
        axis_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
        arr = _smooth(d[key])
        return ("EE Position [m]", [
            {"x": t, "y": arr[:, i], "color": axis_colors[i],
             "label": name, "lw": DEFAULT_LINE_WIDTH, "alpha": PRIMARY_LINE_ALPHA}
            for i, name in enumerate(["X", "Y", "Z"])
        ])

    def _ee_err():
        err = _smooth(d["ee_err"]) * 1000
        return ("EE Error [mm]", [
            {"x": t, "y": err, "color": "#d62728", "label": "Error",
             "lw": DEFAULT_LINE_WIDTH, "alpha": PRIMARY_LINE_ALPHA}
        ])

    def _ee_3d():
        return ("EE 3D Path", "3d_ee")

    def _joint_pos_overlay():
        """Overlay desired (cmd) and actual (act) position per joint."""
        cmd = _convert_pos(_smooth(d["cmd_pos"]), ttr, unit)
        act = _convert_pos(_smooth(d["act_pos"]), ttr, unit)
        ylabel = _pos_ylabel(unit)
        specs = []
        for j in joint_indices:
            c = JOINT_COLORS[j % len(JOINT_COLORS)]
            specs.append({"x": t, "y": cmd[:, j], "color": c,
                          "label": f"{joint_labels[j]} desired",
                          "lw": DEFAULT_LINE_WIDTH, "alpha": PRIMARY_LINE_ALPHA})
            specs.append({"x": t, "y": act[:, j], "color": c,
                          "label": f"{joint_labels[j]} actual",
                          "lw": DEFAULT_LINE_WIDTH, "alpha": SECONDARY_LINE_ALPHA})
        return (ylabel, specs)

    return [
        ("Position (Desired vs Actual)",          _joint_pos_overlay),
        ("Desired Position (Planner)",            lambda: _joint_pos("cmd_pos")),
        ("Desired Velocity (Planner)",            lambda: _joint_vel("cmd_vel")),
        ("Desired Acceleration (Planner)",        lambda: _joint_acc("cmd_acc")),
        ("Measured Position (Hardware)",           lambda: _joint_pos("act_pos")),
        ("Measured Velocity (Hardware)",           lambda: _joint_vel("act_vel")),
        ("Calculated Velocity (from pos)",        _calc_velocity),
        ("Calculated Acceleration (from pos)",    _calc_acceleration),
        ("Torque",                                _joint_torque),
        ("Load (Raw Register)",                   lambda: _joint_raw("load", "load [raw register]")),
        ("Current",                               _joint_current),
        ("Voltage",                               _joint_voltage),
        ("Cmd End-Effector",                      lambda: _ee_single("cmd_ee")),
        ("Act End-Effector",                      lambda: _ee_single("act_ee")),
        ("EE Error",                              _ee_err),
        ("EE 3D Path",                            _ee_3d),
    ]


def _restyle_specs(specs, alpha_scale=1.0, include_legend=True, lw_scale=1.0):
    styled = []
    for spec in specs:
        restyled = dict(spec)
        restyled["lw"] = restyled.get("lw", DEFAULT_LINE_WIDTH) * lw_scale
        restyled["alpha"] = max(0.0, min(1.0, restyled.get("alpha", 1.0) * alpha_scale))
        if not include_legend:
            restyled["label"] = "_nolegend_"
        styled.append(restyled)
    return styled


def _prefix_spec_labels(specs, source_label):
    prefixed = []
    for spec in specs:
        updated = dict(spec)
        label = updated.get("label", "")
        updated["label"] = f"{source_label}: {label}" if label else source_label
        prefixed.append(updated)
    return prefixed


def _legend_items(lines):
    legend_items = []
    for line in lines:
        label = line.get_label()
        if label and not label.startswith("_"):
            legend_items.append((line, label))
    return legend_items


def _apply_line_legend(ax, legend_items, loc="upper right"):
    if not legend_items:
        return None
    handles, labels = zip(*legend_items)
    ncol = max(1, len(labels) // 4)
    return ax.legend(
        handles, labels,
        fontsize=7, ncol=ncol, loc=loc, framealpha=0.9,
    )


def _draw_on_ax(ax, specs, title="", clear=True, show_legend=True,
                show_grid=True, show_xlabel=True):
    """Plot a list of line specs onto an axes."""
    if clear:
        ax.cla()
    if not specs:
        message = "No data — select a differentiation method\nand click Compute"
        if "Calculated" in title:
            message = ("Calculated series are only created from measured position\n"
                       "when Filter = Finite Difference or Savitzky-Golay.")
        ax.text(0.5, 0.5, message,
                ha="center", va="center", fontsize=11, color="#888888",
                transform=ax.transAxes)
        if show_xlabel or clear:
            ax.set_xlabel("Time [s]" if show_xlabel else "", fontsize=8)
        if title:
            ax.set_title(title, fontsize=9, fontweight="bold", loc="left", color="#555555")
        return []
    lines = []
    for s in specs:
        line, = ax.plot(
            s["x"], s["y"],
            color=s.get("color"), label=s.get("label"),
            linewidth=s.get("lw", DEFAULT_LINE_WIDTH),
            linestyle=s.get("linestyle", "-"),
            alpha=s.get("alpha", 1.0),
            marker=s.get("marker", "o"),
            markersize=s.get("ms", DEFAULT_MARKER_SIZE),
            markerfacecolor=s.get("markerfacecolor", "none"),
            markeredgecolor=s.get("color"),
            markeredgewidth=s.get("mew", DEFAULT_MARKER_EDGE_WIDTH),
        )
        lines.append(line)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        legend_items = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
        if legend_items:
            legend_handles, legend_labels = zip(*legend_items)
            ncol = max(1, len(legend_labels) // 4)
            ax.legend(legend_handles, legend_labels, fontsize=7, ncol=ncol, loc="upper right")
    if show_grid or clear:
        ax.grid(show_grid, alpha=0.3 if show_grid else 0.0)
    if show_xlabel or clear:
        ax.set_xlabel("Time [s]" if show_xlabel else "", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left", color="#555555")
    return lines


def _draw_3d_ee(ax, d, title="End-Effector 3D Path"):
    """Draw 3D EE paths on a 3D axes."""
    ax.cla()
    path_color = "#1f77b4"
    ax.plot(d["cmd_ee"][:, 0], d["cmd_ee"][:, 1], d["cmd_ee"][:, 2],
            color=path_color, lw=DEFAULT_LINE_WIDTH, label="Commanded", alpha=PRIMARY_LINE_ALPHA)
    ax.plot(d["act_ee"][:, 0], d["act_ee"][:, 1], d["act_ee"][:, 2],
            color=path_color, lw=DEFAULT_LINE_WIDTH, label="Actual", alpha=SECONDARY_LINE_ALPHA)
    ax.scatter(*d["cmd_ee"][0], color="blue", s=40, zorder=5, label="Start")
    ax.scatter(*d["cmd_ee"][-1], color="black", s=40, zorder=5, marker="x", label="End")
    ax.set_xlabel("X [m]", fontsize=7)
    ax.set_ylabel("Y [m]", fontsize=7)
    ax.set_zlabel("Z [m]", fontsize=7)
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)


# ═══════════════════════════════════════════════════════════
#  Metadata Formatting
# ═══════════════════════════════════════════════════════════

def _fmt_metadata(d: dict) -> str:
    """Format metadata as a compact multi-line string."""
    L = []
    L.append(f"Run: {d.get('run_id', '?')}  |  {'OK' if d.get('success') else 'FAIL'}  |  {d.get('status', '?')}")
    L.append(f"Schema: {d.get('schema_version', '?')}  |  Port: {d.get('port', '?')}")
    L.append(f"DOF: {d.get('dof', '?')}  Servos: {d.get('servo_ids', [])}")
    L.append(f"Entries: {d.get('num_log_entries', '?')}  Duration: {d.get('actual_duration_sec', 0):.2f}s  Move-to-start: {d.get('move_to_start_sec', 0):.1f}s")
    L.append(f"Hz  Ctrl(req): {d.get('control_frequency_hz', 0)}  Fb(req): {d.get('logging_frequency_hz', 0)}  Ctrl(act): {d.get('actual_control_hz', 0):.0f}  Fb(act): {d.get('actual_feedback_hz', 0):.0f}  Rows/Dur: {d.get('rows/duration_hz', 0):.0f}")
    if d.get("load_register_note"):
        L.append(f"Load note: {d.get('load_register_note')}")
    if d.get("current_note"):
        L.append(f"Current note: {d.get('current_note')}")
    if d.get("voltage_note"):
        L.append(f"Voltage note: {d.get('voltage_note')}")
    L.append(
        "Torque view: current-derived torque uses Kt=11 kg.cm/A; "
        "load-derived torque assumes max torque 30 kg.cm."
    )

    traj = d.get("trajectory", {})
    if traj:
        L.append(f"Traj  {traj.get('duration_sec', '?')}s  {traj.get('sample_rate_hz', '?')}Hz  {traj.get('num_samples', '?')} samples")

    geom = d.get("geometry", {})
    if geom:
        center = geom.get("center_m", [])
        c_str = f"[{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}]" if center else "?"
        L.append(f"Geom  {geom.get('type', '?')}  actual_r={geom.get('actual_radius_mm', 0):.1f}mm  center={c_str}")
        norm = geom.get('normal', [])
        n_str = f"[{norm[0]:.3f},{norm[1]:.3f},{norm[2]:.3f}]" if len(norm) == 3 else "?"
        L.append(f"      wpts={geom.get('num_waypoints', '?')}  inst={geom.get('instance', '?')}  planner={geom.get('planner', '?')}  normal={n_str}")

    tq = d.get("tracking_quality", {})
    if tq:
        def _r(k): return tq.get(k, 0)
        L.append(f"Joint err  max={_r('joint_max_err_rad'):.4f}rad({np.degrees(_r('joint_max_err_rad')):.1f}°)"  
                 f"  avg={_r('joint_avg_err_rad'):.4f}({np.degrees(_r('joint_avg_err_rad')):.1f}°)"
                 f"  rms={_r('joint_rms_err_rad'):.4f}({np.degrees(_r('joint_rms_err_rad')):.1f}°)")
        L.append(f"EE err     max={_r('ee_max_err_mm'):.1f}mm  avg={_r('ee_avg_err_mm'):.1f}mm  rms={_r('ee_rms_err_mm'):.1f}mm")

    return "\n".join(L)


# ═══════════════════════════════════════════════════════════
#  Visualization Window (one per log file)
# ═══════════════════════════════════════════════════════════

# List of source labels (for browser dropdowns)
SOURCE_LABELS = [
    "Position (Desired vs Actual)",
    "Desired Position (Planner)",
    "Desired Velocity (Planner)",
    "Desired Acceleration (Planner)",
    "Measured Position (Hardware)",
    "Measured Velocity (Hardware)",
    "Calculated Velocity (from pos)",
    "Calculated Acceleration (from pos)",
    "Torque",
    "Load (Raw Register)",
    "Current",
    "Voltage",
    "Cmd End-Effector",
    "Act End-Effector",
    "EE Error",
    "EE 3D Path",
]
OVERLAY_SOURCE_LABELS = [OVERLAY_NONE_LABEL] + SOURCE_LABELS


class PlotWindow:
    """Persistent plot window that can be updated with new data."""

    def __init__(self, master=None):
        self.win = tk.Toplevel(master) if master else tk.Tk()
        self.win.title("Log Viewer")
        self.win.geometry("1520x820")
        self.win.configure(bg="#f0f0f0")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Row 1: Units + Joint scope + Refresh ──
        toolbar = tk.Frame(self.win, bg="#e8e8e8", padx=6, pady=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(toolbar, text="Units:", bg="#e8e8e8",
                 font=("sans-serif", 9, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.combo_unit = ttk.Combobox(toolbar, values=UNIT_LABELS,
                                       state="readonly", width=10,
                                       font=("sans-serif", 9))
        self.combo_unit.current(UNIT_TICKS)
        self.combo_unit.pack(side=tk.LEFT, padx=(0, 10))
        self.combo_unit.bind("<<ComboboxSelected>>", lambda e: self._refresh())

        tk.Label(toolbar, text="Joints:", bg="#e8e8e8",
                 font=("sans-serif", 9, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.combo_joint = ttk.Combobox(toolbar, values=[JOINT_SCOPE_ALL],
                                        state="readonly", width=8,
                                        font=("sans-serif", 9))
        self.combo_joint.current(0)
        self.combo_joint.pack(side=tk.LEFT, padx=(0, 10))
        self.combo_joint.bind("<<ComboboxSelected>>", lambda e: self._refresh())

        tk.Button(toolbar, text="\u21bb Refresh", font=("sans-serif", 9, "bold"),
                  bg="#4a90d9", fg="white", activebackground="#3a7bc8",
                  relief=tk.FLAT, padx=8, pady=1,
                  command=self._refresh).pack(side=tk.LEFT, padx=(6, 4))

        # ── Row 2: Top subplot — source + overlay + filter + smoothing window ──
        top_bar = tk.Frame(self.win, bg="#d6e6f5", padx=6, pady=3)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top_bar, text="Subplot 1:", bg="#d6e6f5",
                 font=("sans-serif", 9, "bold")).pack(side=tk.LEFT, padx=(4, 4))
        tk.Label(top_bar, text="Primary:", bg="#d6e6f5",
                 font=("sans-serif", 8, "bold")).pack(side=tk.LEFT, padx=(2, 2))
        self.combo_top = ttk.Combobox(top_bar, values=SOURCE_LABELS,
                                      state="readonly", width=24,
                                      font=("sans-serif", 9))
        self.combo_top.current(0)
        self.combo_top.pack(side=tk.LEFT, padx=(0, 10))
        self.combo_top.bind("<<ComboboxSelected>>", lambda e: self._refresh())

        tk.Label(top_bar, text="Overlay:", bg="#d6e6f5",
                 font=("sans-serif", 8, "bold")).pack(side=tk.LEFT, padx=(2, 2))
        self.combo_overlay_top = ttk.Combobox(top_bar, values=OVERLAY_SOURCE_LABELS,
                                              state="readonly", width=22,
                                              font=("sans-serif", 9))
        self.combo_overlay_top.current(0)
        self.combo_overlay_top.pack(side=tk.LEFT, padx=(0, 10))
        self.combo_overlay_top.bind("<<ComboboxSelected>>",
                                    lambda e: self._on_overlay_change("top"))

        tk.Label(top_bar, text="Filter:", bg="#d6e6f5",
                 font=("sans-serif", 8, "bold")).pack(side=tk.LEFT, padx=(6, 2))
        self.combo_filter_top = ttk.Combobox(top_bar, values=FILTER_METHODS,
                                             state="readonly", width=16,
                                             font=("sans-serif", 9))
        self.combo_filter_top.current(0)
        self.combo_filter_top.pack(side=tk.LEFT, padx=(0, 6))
        self.combo_filter_top.bind("<<ComboboxSelected>>",
                                   lambda e: self._on_filter_change("top", "primary"))

        tk.Label(top_bar, text="Win:", bg="#d6e6f5",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(2, 1))
        self.spin_win_top = tk.Spinbox(top_bar, from_=3, to=201, increment=2,
                                       width=5, font=("sans-serif", 9))
        self.spin_win_top.delete(0, tk.END)
        self.spin_win_top.insert(0, "11")
        self.spin_win_top.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(top_bar, text="Poly:", bg="#d6e6f5",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(2, 1))
        self.spin_poly_top = tk.Spinbox(top_bar, from_=1, to=9, increment=1,
                                        width=4, font=("sans-serif", 9),
                                        state=tk.DISABLED)
        self.spin_poly_top.delete(0, tk.END)
        self.spin_poly_top.insert(0, "3")
        self.spin_poly_top.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(top_bar, text="Overlay Filter:", bg="#d6e6f5",
                 font=("sans-serif", 8, "bold")).pack(side=tk.LEFT, padx=(8, 2))
        self.combo_filter_overlay_top = ttk.Combobox(top_bar, values=FILTER_METHODS,
                                                     state="disabled", width=16,
                                                     font=("sans-serif", 9))
        self.combo_filter_overlay_top.current(0)
        self.combo_filter_overlay_top.pack(side=tk.LEFT, padx=(0, 6))
        self.combo_filter_overlay_top.bind("<<ComboboxSelected>>",
                                           lambda e: self._on_filter_change("top", "overlay"))

        tk.Label(top_bar, text="Win:", bg="#d6e6f5",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(2, 1))
        self.spin_win_overlay_top = tk.Spinbox(top_bar, from_=3, to=201, increment=2,
                                               width=5, font=("sans-serif", 9),
                                               state=tk.DISABLED)
        self.spin_win_overlay_top.delete(0, tk.END)
        self.spin_win_overlay_top.insert(0, "11")
        self.spin_win_overlay_top.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(top_bar, text="Poly:", bg="#d6e6f5",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(2, 1))
        self.spin_poly_overlay_top = tk.Spinbox(top_bar, from_=1, to=9, increment=1,
                                                width=4, font=("sans-serif", 9),
                                                state=tk.DISABLED)
        self.spin_poly_overlay_top.delete(0, tk.END)
        self.spin_poly_overlay_top.insert(0, "3")
        self.spin_poly_overlay_top.pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(top_bar, text="Compute", font=("sans-serif", 8, "bold"),
                  bg="#5ba55b", fg="white", activebackground="#4a944a",
                  relief=tk.FLAT, padx=8, pady=1,
                  command=self._refresh).pack(side=tk.LEFT, padx=(6, 4))

        # ── Row 3: Bottom subplot — source + overlay + filter + smoothing window ──
        bot_bar = tk.Frame(self.win, bg="#e6d6e6", padx=6, pady=3)
        bot_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(bot_bar, text="Subplot 2:", bg="#e6d6e6",
                 font=("sans-serif", 9, "bold")).pack(side=tk.LEFT, padx=(4, 4))
        tk.Label(bot_bar, text="Primary:", bg="#e6d6e6",
                 font=("sans-serif", 8, "bold")).pack(side=tk.LEFT, padx=(2, 2))
        self.combo_bot = ttk.Combobox(bot_bar, values=SOURCE_LABELS,
                                      state="readonly", width=24,
                                      font=("sans-serif", 9))
        self.combo_bot.current(5)
        self.combo_bot.pack(side=tk.LEFT, padx=(0, 10))
        self.combo_bot.bind("<<ComboboxSelected>>", lambda e: self._refresh())

        tk.Label(bot_bar, text="Overlay:", bg="#e6d6e6",
                 font=("sans-serif", 8, "bold")).pack(side=tk.LEFT, padx=(2, 2))
        self.combo_overlay_bot = ttk.Combobox(bot_bar, values=OVERLAY_SOURCE_LABELS,
                                              state="readonly", width=22,
                                              font=("sans-serif", 9))
        self.combo_overlay_bot.current(0)
        self.combo_overlay_bot.pack(side=tk.LEFT, padx=(0, 10))
        self.combo_overlay_bot.bind("<<ComboboxSelected>>",
                                    lambda e: self._on_overlay_change("bot"))

        tk.Label(bot_bar, text="Filter:", bg="#e6d6e6",
                 font=("sans-serif", 8, "bold")).pack(side=tk.LEFT, padx=(6, 2))
        self.combo_filter_bot = ttk.Combobox(bot_bar, values=FILTER_METHODS,
                                             state="readonly", width=16,
                                             font=("sans-serif", 9))
        self.combo_filter_bot.current(0)
        self.combo_filter_bot.pack(side=tk.LEFT, padx=(0, 6))
        self.combo_filter_bot.bind("<<ComboboxSelected>>",
                                   lambda e: self._on_filter_change("bot", "primary"))

        tk.Label(bot_bar, text="Win:", bg="#e6d6e6",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(2, 1))
        self.spin_win_bot = tk.Spinbox(bot_bar, from_=3, to=201, increment=2,
                                       width=5, font=("sans-serif", 9))
        self.spin_win_bot.delete(0, tk.END)
        self.spin_win_bot.insert(0, "11")
        self.spin_win_bot.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(bot_bar, text="Poly:", bg="#e6d6e6",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(2, 1))
        self.spin_poly_bot = tk.Spinbox(bot_bar, from_=1, to=9, increment=1,
                                        width=4, font=("sans-serif", 9),
                                        state=tk.DISABLED)
        self.spin_poly_bot.delete(0, tk.END)
        self.spin_poly_bot.insert(0, "3")
        self.spin_poly_bot.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(bot_bar, text="Overlay Filter:", bg="#e6d6e6",
                 font=("sans-serif", 8, "bold")).pack(side=tk.LEFT, padx=(8, 2))
        self.combo_filter_overlay_bot = ttk.Combobox(bot_bar, values=FILTER_METHODS,
                                                     state="disabled", width=16,
                                                     font=("sans-serif", 9))
        self.combo_filter_overlay_bot.current(0)
        self.combo_filter_overlay_bot.pack(side=tk.LEFT, padx=(0, 6))
        self.combo_filter_overlay_bot.bind("<<ComboboxSelected>>",
                                           lambda e: self._on_filter_change("bot", "overlay"))

        tk.Label(bot_bar, text="Win:", bg="#e6d6e6",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(2, 1))
        self.spin_win_overlay_bot = tk.Spinbox(bot_bar, from_=3, to=201, increment=2,
                                               width=5, font=("sans-serif", 9),
                                               state=tk.DISABLED)
        self.spin_win_overlay_bot.delete(0, tk.END)
        self.spin_win_overlay_bot.insert(0, "11")
        self.spin_win_overlay_bot.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(bot_bar, text="Poly:", bg="#e6d6e6",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(2, 1))
        self.spin_poly_overlay_bot = tk.Spinbox(bot_bar, from_=1, to=9, increment=1,
                                                width=4, font=("sans-serif", 9),
                                                state=tk.DISABLED)
        self.spin_poly_overlay_bot.delete(0, tk.END)
        self.spin_poly_overlay_bot.insert(0, "3")
        self.spin_poly_overlay_bot.pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(bot_bar, text="Compute", font=("sans-serif", 8, "bold"),
                  bg="#5ba55b", fg="white", activebackground="#4a944a",
                  relief=tk.FLAT, padx=8, pady=1,
                  command=self._refresh).pack(side=tk.LEFT, padx=(6, 4))

        # ── Matplotlib figure ──
        self.fig = Figure(figsize=(14, 8))
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.96,
                                 bottom=0.07, hspace=0.15)
        self.ax_top = self.fig.add_subplot(2, 1, 1)
        self.ax_bot = self.fig.add_subplot(2, 1, 2, sharex=self.ax_top)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.nav = NavigationToolbar2Tk(self.canvas, self.win)
        self.nav.update()
        self.nav.pack(side=tk.BOTTOM, fill=tk.X)

        self._data = None
        self._sources = None
        self._3d = {"top": None, "bot": None}
        self._overlay_axes = {"top": None, "bot": None}
        self._alive = True
        self._sync_filter_state("top", "primary")
        self._sync_filter_state("top", "overlay")
        self._sync_filter_state("bot", "primary")
        self._sync_filter_state("bot", "overlay")

    def is_alive(self):
        return self._alive

    def mainloop(self):
        self.win.mainloop()

    def _on_close(self):
        self._alive = False
        self.win.destroy()

    def _filter_controls(self, which="top", role="primary"):
        if which == "top":
            if role == "overlay":
                return self.combo_filter_overlay_top, self.spin_win_overlay_top, self.spin_poly_overlay_top
            return self.combo_filter_top, self.spin_win_top, self.spin_poly_top
        if role == "overlay":
            return self.combo_filter_overlay_bot, self.spin_win_overlay_bot, self.spin_poly_overlay_bot
        return self.combo_filter_bot, self.spin_win_bot, self.spin_poly_bot

    def _overlay_controls(self, which="top"):
        if which == "top":
            return (self.combo_overlay_top, self.combo_filter_overlay_top,
                    self.spin_win_overlay_top, self.spin_poly_overlay_top)
        return (self.combo_overlay_bot, self.combo_filter_overlay_bot,
                self.spin_win_overlay_bot, self.spin_poly_overlay_bot)

    def _sync_filter_state(self, which="top", role="primary"):
        combo, spin_win, spin_poly = self._filter_controls(which, role)
        if role == "overlay":
            overlay_combo, _, _, _ = self._overlay_controls(which)
            has_overlay = overlay_combo.get().strip() != OVERLAY_NONE_LABEL
            combo.configure(state="readonly" if has_overlay else "disabled")
            method_key = FILTER_METHOD_KEYS[combo.current()] if has_overlay else "none"
            spin_win.config(
                state=tk.NORMAL if has_overlay and method_key in WINDOWED_FILTER_METHOD_KEYS else tk.DISABLED
            )
            spin_poly.config(
                state=tk.NORMAL if has_overlay and method_key == "savgol" else tk.DISABLED
            )
            return
        method_key = FILTER_METHOD_KEYS[combo.current()]
        spin_win.config(state=tk.NORMAL if method_key in WINDOWED_FILTER_METHOD_KEYS else tk.DISABLED)
        spin_poly.config(state=tk.NORMAL if method_key == "savgol" else tk.DISABLED)

    def _on_filter_change(self, which="top", role="primary"):
        self._sync_filter_state(which, role)

    def _on_overlay_change(self, which="top"):
        self._sync_filter_state(which, "overlay")
        self._refresh()

    def _get_filter_params(self, which="top", role="primary"):
        combo, spin_win, spin_poly = self._filter_controls(which, role)
        method = FILTER_METHOD_KEYS[combo.current()]
        try:
            window = int(spin_win.get())
        except ValueError:
            window = 11
        try:
            polyorder = int(spin_poly.get())
        except ValueError:
            polyorder = 3
        return FilterConfig(method=method, window=window, polyorder=polyorder)

    def _get_joint_selection(self):
        selection = self.combo_joint.get().strip()
        return selection or JOINT_SCOPE_ALL

    def _update_joint_selector(self):
        if not self._data:
            return
        nj = self._data["cmd_pos"].shape[1]
        options = [JOINT_SCOPE_ALL] + [f"J{j+1}" for j in range(nj)]
        current = self.combo_joint.get().strip()
        self.combo_joint.configure(values=options)
        if current in options:
            self.combo_joint.set(current)
        else:
            self.combo_joint.current(0)

    def _clear_dynamic_axes(self):
        for key in ("top", "bot"):
            if self._overlay_axes[key] is not None:
                self._overlay_axes[key].remove()
                self._overlay_axes[key] = None
            if self._3d[key] is not None:
                self._3d[key].remove()
                self._3d[key] = None

    def _capture_axis_view(self, ax, role="primary"):
        if ax is None:
            return None
        if getattr(ax, "name", "") == "3d":
            return {
                "kind": "3d",
                "xlim": ax.get_xlim3d(),
                "ylim": ax.get_ylim3d(),
                "zlim": ax.get_zlim3d(),
                "elev": ax.elev,
                "azim": ax.azim,
                "roll": getattr(ax, "roll", None),
            }
        state = {
            "kind": "2d",
            "ylim": ax.get_ylim(),
        }
        if role == "primary":
            state["xlim"] = ax.get_xlim()
        return state

    def _capture_view_state(self):
        state = {}
        for key, ax in (("top", self.ax_top), ("bot", self.ax_bot)):
            primary_ax = self._3d[key] if self._3d[key] is not None else ax
            state[key] = {
                "primary": self._capture_axis_view(primary_ax, role="primary"),
                "overlay": self._capture_axis_view(self._overlay_axes[key], role="overlay"),
            }
        return state

    def _restore_axis_view(self, ax, state, role="primary"):
        if ax is None or not state:
            return
        if state.get("kind") == "3d":
            if getattr(ax, "name", "") != "3d":
                return
            ax.set_xlim3d(state["xlim"])
            ax.set_ylim3d(state["ylim"])
            ax.set_zlim3d(state["zlim"])
            try:
                if state.get("roll") is None:
                    ax.view_init(elev=state["elev"], azim=state["azim"])
                else:
                    ax.view_init(
                        elev=state["elev"],
                        azim=state["azim"],
                        roll=state["roll"],
                    )
            except TypeError:
                ax.view_init(elev=state["elev"], azim=state["azim"])
            return
        if getattr(ax, "name", "") == "3d":
            return
        if role == "primary" and "xlim" in state:
            ax.set_xlim(state["xlim"])
        if "ylim" in state:
            ax.set_ylim(state["ylim"])

    def _restore_view_state(self, state):
        if not state:
            return
        for key, ax in (("top", self.ax_top), ("bot", self.ax_bot)):
            primary_ax = self._3d[key] if self._3d[key] is not None else ax
            axis_state = state.get(key, {})
            self._restore_axis_view(primary_ax, axis_state.get("primary"), role="primary")
            self._restore_axis_view(self._overlay_axes[key], axis_state.get("overlay"), role="overlay")

    def _render_subplot(self, ax, which, primary_label, overlay_label, sources):
        primary_ctx, overlay_ctx = sources
        primary_filter = primary_ctx["filter"]
        overlay_filter = overlay_ctx["filter"]
        primary_sources = primary_ctx["sources"]
        overlay_sources = overlay_ctx["sources"]
        primary_label = primary_label if primary_label in primary_sources else SOURCE_LABELS[0]
        overlay_label = overlay_label if overlay_label in overlay_sources else None
        primary_ylabel, primary_result = primary_sources[primary_label]()
        overlay_note = None
        primary_display = _source_display_label(primary_label, primary_filter)

        if primary_result == "3d_ee":
            ax.set_visible(False)
            pos = ax.get_position()
            ax_3d = self.fig.add_axes(pos, projection="3d")
            self._3d[which] = ax_3d
            if overlay_label:
                overlay_note = f"{overlay_label} unavailable for 3D"
            title = f"Primary: {primary_display}"
            if overlay_note is not None:
                title = f"{title} | {overlay_note}"
            _draw_3d_ee(ax_3d, self._data, title=title)
            return

        ax.set_visible(True)
        title = f"Primary: {primary_display}"
        overlay_ylabel = None
        overlay_result = None
        if primary_label in CALCULATED_SOURCE_LABELS and not primary_result:
            title = f"{title} | {_calculated_series_note(primary_filter)}"

        if overlay_label:
            overlay_display = _source_display_label(overlay_label, overlay_filter)
            overlay_ylabel, overlay_result = overlay_sources[overlay_label]()
            if overlay_result == "3d_ee":
                overlay_note = f"{overlay_label} unavailable in 2D overlay"
                overlay_result = None
            elif overlay_label in CALCULATED_SOURCE_LABELS and not overlay_result:
                overlay_note = f"{overlay_display} {_calculated_series_note(overlay_filter)}"
            else:
                title = f"{title} | Overlay: {overlay_display}"

        primary_specs = _prefix_spec_labels(_restyle_specs(primary_result), primary_display)
        primary_lines = _draw_on_ax(ax, primary_specs, title=title, show_legend=False)
        ax.set_ylabel(primary_ylabel, fontsize=8)
        legend_items = _legend_items(primary_lines)

        if overlay_result:
            overlay_specs = _prefix_spec_labels(
                _restyle_specs(
                    overlay_result,
                    alpha_scale=SECONDARY_LINE_ALPHA,
                    include_legend=True,
                    lw_scale=0.92,
                ),
                overlay_display,
            )
            if overlay_ylabel == primary_ylabel:
                overlay_lines = _draw_on_ax(
                    ax, overlay_specs,
                    clear=False, show_legend=False,
                    show_grid=False, show_xlabel=False,
                )
                legend_items.extend(_legend_items(overlay_lines))
            else:
                overlay_ax = ax.twinx()
                self._overlay_axes[which] = overlay_ax
                overlay_lines = _draw_on_ax(
                    overlay_ax, overlay_specs,
                    clear=True, show_legend=False,
                    show_grid=False, show_xlabel=False,
                )
                overlay_ax.set_ylabel(overlay_ylabel, fontsize=8, color="#666666")
                overlay_ax.tick_params(axis="y", labelsize=8, colors="#666666")
                overlay_ax.spines["right"].set_color("#999999")
                legend_items.extend(_legend_items(overlay_lines))
        elif overlay_note:
            ax.set_title(f"{title} | {overlay_note}",
                         fontsize=9, fontweight="bold", loc="left", color="#555555")

        _apply_line_legend(ax, legend_items)

    def load(self, d: dict):
        """Load new data and refresh plots."""
        self._data = d
        self._update_joint_selector()
        title = d.get('run_id', Path(d.get('path', '')).stem)
        self.win.title(f"Log: {title}")
        self._refresh(preserve_view=False)
        self.win.lift()
        self.win.focus_force()

    def _refresh(self, preserve_view=True):
        if not self._data:
            return
        unit = self.combo_unit.current()
        joint_selection = self._get_joint_selection()
        view_state = self._capture_view_state() if preserve_view else None

        # Build separate data sources for each subplot (independent filter params)
        top_filter = self._get_filter_params("top", "primary")
        top_overlay_filter = self._get_filter_params("top", "overlay")
        bot_filter = self._get_filter_params("bot", "primary")
        bot_overlay_filter = self._get_filter_params("bot", "overlay")
        sources_top_primary = {
            "filter": top_filter,
            "sources": dict(_build_data_sources(
                self._data, unit, top_filter, joint_selection)),
        }
        sources_top_overlay = {
            "filter": top_overlay_filter,
            "sources": dict(_build_data_sources(
                self._data, unit, top_overlay_filter, joint_selection)),
        }
        sources_bot_primary = {
            "filter": bot_filter,
            "sources": dict(_build_data_sources(
                self._data, unit, bot_filter, joint_selection)),
        }
        sources_bot_overlay = {
            "filter": bot_overlay_filter,
            "sources": dict(_build_data_sources(
                self._data, unit, bot_overlay_filter, joint_selection)),
        }

        self._clear_dynamic_axes()

        for ax, combo, overlay_combo, which, sources in [
            (self.ax_top, self.combo_top, self.combo_overlay_top, "top",
             (sources_top_primary, sources_top_overlay)),
            (self.ax_bot, self.combo_bot, self.combo_overlay_bot, "bot",
             (sources_bot_primary, sources_bot_overlay)),
        ]:
            overlay_label = overlay_combo.get().strip()
            if overlay_label == OVERLAY_NONE_LABEL:
                overlay_label = None
            self._render_subplot(ax, which, combo.get().strip(), overlay_label, sources)

        self._restore_view_state(view_state)
        self.canvas.draw_idle()


def visualize(d: dict, top_idx: int = 0, bot_idx: int = 5, master=None):
    """Create a new PlotWindow and load data into it."""
    pw = PlotWindow(master=master)
    pw.combo_top.current(top_idx)
    pw.combo_bot.current(bot_idx)
    pw.load(d)
    return pw


# ═══════════════════════════════════════════════════════════
#  File Browser GUI (main window)
# ═══════════════════════════════════════════════════════════

class LogBrowser:
    """Tkinter file browser that stays open and spawns plot windows."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self._plot_window = None  # single reusable plot window

        self.root = tk.Tk()
        self.root.title("HWRL Log Browser")
        self.root.geometry("620x820")
        self.root.configure(bg="#2b2b2b")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#2b2b2b")
        style.configure("Dark.TLabel", background="#2b2b2b", foreground="#e0e0e0",
                         font=("Monospace", 10))
        style.configure("Title.TLabel", background="#2b2b2b", foreground="#82aaff",
                         font=("Monospace", 12, "bold"))
        style.configure("Dark.TButton", font=("Monospace", 9))

        # Header
        header = ttk.Frame(self.root, style="Dark.TFrame")
        header.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(header, text="Hardware Execution Logs",
                  style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Button(header, text="Refresh", style="Dark.TButton",
                   command=self._refresh_list).pack(side=tk.RIGHT, padx=5)
        ttk.Button(header, text="Open Folder...", style="Dark.TButton",
                   command=self._change_dir).pack(side=tk.RIGHT, padx=5)

        # Directory label
        self._dir_var = tk.StringVar(value=str(self.log_dir))
        dir_frame = ttk.Frame(self.root, style="Dark.TFrame")
        dir_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        ttk.Label(dir_frame, textvariable=self._dir_var,
                  style="Dark.TLabel", wraplength=490).pack(fill=tk.X)

        # File list with scrollbar
        list_frame = tk.Frame(self.root, bg="#1e1e1e", height=220)
        list_frame.pack(fill=tk.X, padx=10, pady=5)
        list_frame.pack_propagate(False)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            list_frame, bg="#1e1e1e", fg="#c0c0c0", selectbackground="#3a5fcd",
            selectforeground="white", font=("Monospace", 9),
            yscrollcommand=scrollbar.set, activestyle="none",
        )
        self.listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind("<Double-Button-1>", self._on_double_click)
        self.listbox.bind("<Return>", self._on_double_click)

        # Status bar
        self._status_var = tk.StringVar(value="Double-click a file to visualize")
        status_bar = ttk.Frame(self.root, style="Dark.TFrame")
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 4))
        ttk.Label(status_bar, textvariable=self._status_var,
                  style="Dark.TLabel", font=("Monospace", 8)).pack(side=tk.LEFT)

        # Metadata preview (bottom panel — scrollable multi-column grid)
        preview_outer = tk.Frame(self.root, bg="#1e1e1e")
        preview_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._meta_canvas = tk.Canvas(preview_outer, bg="#1e1e1e",
                                      highlightthickness=0, borderwidth=0)
        meta_scroll = tk.Scrollbar(preview_outer, command=self._meta_canvas.yview)
        self._meta_canvas.configure(yscrollcommand=meta_scroll.set)
        meta_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._meta_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._meta_inner = tk.Frame(self._meta_canvas, bg="#1e1e1e")
        self._meta_window = self._meta_canvas.create_window(
            (0, 0), window=self._meta_inner, anchor="nw")

        def _on_meta_configure(e):
            self._meta_canvas.configure(scrollregion=self._meta_canvas.bbox("all"))
        self._meta_inner.bind("<Configure>", _on_meta_configure)

        def _on_canvas_configure(e):
            self._meta_canvas.itemconfig(self._meta_window, width=e.width)
        self._meta_canvas.bind("<Configure>", _on_canvas_configure)

        # Placeholder label
        self._meta_placeholder = tk.Label(
            self._meta_inner, text="  Select a file to view metadata",
            bg="#1e1e1e", fg="#555555", font=("Monospace", 9), anchor="w")
        self._meta_placeholder.grid(row=0, column=0, sticky="w", padx=8, pady=8)

        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        self._refresh_list()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        if not self.log_dir.is_dir():
            self._status_var.set(f"Directory not found: {self.log_dir}")
            return
        files = self._scan_log_files()
        self._files = files
        for f in files:
            self.listbox.insert(tk.END, self._display_name(f))
        self._status_var.set(f"{len(files)} JSON files found")

    def _change_dir(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=str(self.log_dir), title="Select Log Directory")
        if d:
            self.log_dir = Path(d)
            self._dir_var.set(str(self.log_dir))
            self._refresh_list()

    # ── Colors for metadata sections ──
    _BG = "#1e1e1e"
    _CARD_BG = "#252525"
    _HDR_FG = "#82aaff"
    _KEY_FG = "#8a8a8a"
    _VAL_FG = "#e0e0e0"
    _NUM_FG = "#c3a76f"
    _OK_FG = "#4ec860"
    _FAIL_FG = "#e05050"
    _SEP_FG = "#3a3a3a"
    _FONT = ("Monospace", 9)
    _FONT_B = ("Monospace", 9, "bold")
    _HDR_FONT = ("Monospace", 10, "bold")

    def _display_name(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.log_dir))
        except ValueError:
            return path.name

    def _scan_log_files(self):
        files = []
        for path in self.log_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".json":
                continue
            try:
                files.append((path.stat().st_mtime, path))
            except OSError:
                continue
        files.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in files]

    def _make_section(self, parent, title, rows, col, row_start):
        """Create a card-like section frame at grid position (row_start, col).
        rows: list of (key, value, value_fg) tuples.
        """
        card = tk.Frame(parent, bg=self._CARD_BG, padx=8, pady=6,
                        highlightbackground=self._SEP_FG, highlightthickness=1)
        card.grid(row=row_start, column=col, sticky="nsew", padx=4, pady=4)

        hdr = tk.Label(card, text=title, bg=self._CARD_BG, fg=self._HDR_FG,
                       font=self._HDR_FONT, anchor="w")
        hdr.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        sep = tk.Frame(card, bg=self._SEP_FG, height=1)
        sep.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        for i, (k, v, fg) in enumerate(rows, start=2):
            tk.Label(card, text=k, bg=self._CARD_BG, fg=self._KEY_FG,
                     font=self._FONT, anchor="w").grid(
                row=i, column=0, sticky="w", padx=(2, 8))
            font = self._FONT_B if fg in (self._OK_FG, self._FAIL_FG) else self._FONT
            tk.Label(card, text=v, bg=self._CARD_BG, fg=fg,
                     font=font, anchor="w").grid(
                row=i, column=1, sticky="w")

        return card

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self._files[idx]

        try:
            d = load_log_json(str(path))
            if not d:
                return
        except Exception:
            return

        # Clear previous content
        for w in self._meta_inner.winfo_children():
            w.destroy()

        parent = self._meta_inner

        # Configure 2-column grid
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        tag = self._OK_FG if d.get("success") else self._FAIL_FG
        V, N = self._VAL_FG, self._NUM_FG

        # ── Column 0, Row 0: Run Info ──
        self._make_section(parent, "Run Info", [
            ("Run ID",  d.get("run_id", "?"), V),
            ("Label",   d.get("label", "?"), V),
            ("Status",  d.get("status", "?"), tag),
            ("Success", "Yes" if d.get("success") else "No", tag),
            ("Schema",  d.get("schema_version", "?"), V),
            ("Port",    d.get("port", "?"), V),
            ("DOF",     f"{d.get('dof', '?')}  Servos: {d.get('servo_ids', [])}", V),
        ], col=0, row_start=0)

        # ── Column 1, Row 0: Execution + Frequencies ──
        exec_rows = [
            ("Log Entries",   str(d.get("num_log_entries", "?")), N),
            ("Duration",      f"{d.get('actual_duration_sec', 0):.2f} s", N),
            ("Move-to-start", f"{d.get('move_to_start_sec', 0):.1f} s", N),
            ("", "", V),  # spacer
            ("Control Hz",   f"{d.get('control_frequency_hz', 0)}  \u2192  {d.get('actual_control_hz', 0):.0f}", N),
            ("Feedback Hz",  f"{d.get('logging_frequency_hz', 0)}  \u2192  {d.get('actual_feedback_hz', 0):.0f}", N),
            ("Rows/Duration", f"{d.get('rows/duration_hz', 0):.0f} Hz", N),
        ]
        self._make_section(parent, "Execution & Frequencies", exec_rows,
                           col=1, row_start=0)

        # ── Column 0, Row 1: Geometry ──
        geom = d.get("geometry", {})
        if geom:
            center = geom.get("center_m", [])
            c_str = f"[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]" if center else "?"
            norm = geom.get("normal", [])
            n_str = f"[{norm[0]:.3f}, {norm[1]:.3f}, {norm[2]:.3f}]" if len(norm) == 3 else "?"
            uvec = geom.get("u", [])
            u_str = f"[{uvec[0]:.3f}, {uvec[1]:.3f}, {uvec[2]:.3f}]" if len(uvec) == 3 else "?"
            vvec = geom.get("v", [])
            v_str = f"[{vvec[0]:.3f}, {vvec[1]:.3f}, {vvec[2]:.3f}]" if len(vvec) == 3 else "?"
            self._make_section(parent, "Geometry", [
                ("Type",      geom.get("type", "?"), V),
                ("Actual R",  f"{geom.get('actual_radius_mm', 0):.1f} mm", N),
                ("Center",    c_str, N),
                ("Normal",    n_str, N),
                ("U axis",    u_str, N),
                ("V axis",    v_str, N),
                ("Waypoints", str(geom.get("num_waypoints", "?")), N),
                ("Instance",  str(geom.get("instance", "?")), V),
                ("Planner",   geom.get("planner", "?"), V),
            ], col=0, row_start=1)

        # ── Column 1, Row 1: Trajectory + Tracking Quality ──
        traj = d.get("trajectory", {})
        tq = d.get("tracking_quality", {})
        tq_rows = []
        if traj:
            tq_rows += [
                ("Traj Duration",  f"{traj.get('duration_sec', '?')} s", N),
                ("Sample Rate",    f"{traj.get('sample_rate_hz', '?')} Hz", N),
                ("Samples",        str(traj.get("num_samples", "?")), N),
            ]
        if traj and tq:
            tq_rows.append(("", "", V))  # spacer
        if tq:
            def _r(k): return tq.get(k, 0)
            jm, ja, jr = _r('joint_max_err_rad'), _r('joint_avg_err_rad'), _r('joint_rms_err_rad')
            em, ea, er = _r('ee_max_err_mm'), _r('ee_avg_err_mm'), _r('ee_rms_err_mm')
            tq_rows += [
                ("Joint Max",  f"{jm:.4f} rad  ({np.degrees(jm):.1f}\u00b0)", N),
                ("Joint Avg",  f"{ja:.4f} rad  ({np.degrees(ja):.1f}\u00b0)", N),
                ("Joint RMS",  f"{jr:.4f} rad  ({np.degrees(jr):.1f}\u00b0)", N),
                ("EE Max",     f"{em:.2f} mm", N),
                ("EE Avg",     f"{ea:.2f} mm", N),
                ("EE RMS",     f"{er:.2f} mm", N),
            ]
        if tq_rows:
            title = "Trajectory & Tracking" if traj and tq else ("Trajectory" if traj else "Tracking Quality")
            self._make_section(parent, title, tq_rows, col=1, row_start=1)

    def _on_double_click(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        path = str(self._files[idx])
        display_name = self._display_name(self._files[idx])
        self._status_var.set(f"Loading {display_name}...")
        self.root.update_idletasks()
        try:
            d = load_log_json(path)
            if not d:
                self._status_var.set(f"Empty or invalid log: {display_name}")
                return
            # Reuse existing plot window or create a new one
            if self._plot_window is None or not self._plot_window.is_alive():
                self._plot_window = PlotWindow(master=self.root)
            self._plot_window.load(d)
            self._status_var.set(f"Showing: {display_name}")
        except Exception as e:
            self._status_var.set(f"Error: {e}")

    def _on_close(self):
        if self._plot_window and self._plot_window.is_alive():
            self._plot_window.win.destroy()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════

def main():
    if len(sys.argv) > 1:
        # Direct file mode: open one file and show
        path = sys.argv[1]
        d = load_log_json(path)
        if not d:
            print(f"Failed to load or empty log: {path}", file=sys.stderr)
            sys.exit(1)
        win = visualize(d)
        win.mainloop()
    else:
        # Browser mode
        log_dir = DEFAULT_LOG_DIR
        browser = LogBrowser(log_dir)
        browser.run()


if __name__ == "__main__":
    main()
