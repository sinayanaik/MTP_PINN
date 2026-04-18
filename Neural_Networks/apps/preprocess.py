#!/usr/bin/env python3
"""
preprocess_data.py -- Interactive Data Preprocessing & Dataset Builder

Usage:  cd /home/san/Desktop/MTP_PINN && python -m Neural_Networks.preprocess_data

GUI Layout
----------
  Left  : file browser (raw JSON trajectory files)
  Right : scrollable stack of 5 plot panels
            q           -- joint positions       (SG smooth: none/savgol)
            qd          -- joint velocities      (SG differentiation from filtered q)
            qdd         -- joint accelerations   (SG differentiation, lock-to-qd option)
            tau_measured -- measured torque        (SG smooth: none/savgol)
            tau_analytical -- RNEA + friction     (computed, optional SG post-filter)
  Bottom: Build Dataset panel (split ratios, output path, build button)

All filtering uses Savitzky-Golay exclusively.  Each panel has its own
(window_length, polyorder) pair.  Build materialises 10 CSV files + t.csv
per split, plus metadata.json with full provenance.  Default output folder names
encode SG parameters, split, trim, and RNEA (see Neural_Networks/dataset_labels.py).
"""

from __future__ import annotations

import json as _json
import os
import sys
import collections
import csv
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.fft import rfft, rfftfreq

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ImportError:
    print("tkinter is required", file=sys.stderr)
    sys.exit(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent   # apps/ -> Neural_Networks/ -> MTP_PINN/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Neural_Networks.data.loader import (
    load_raw_sample, resolve_front_back_trim,
    CSV_T, CSV_RAW_Q, CSV_RAW_QD, CSV_RAW_QDD,
    CSV_RAW_TAU_MEASURED, CSV_RAW_TAU_DECOMPOSED,
    CSV_FILTERED_Q, CSV_FILTERED_QD, CSV_FILTERED_QDD,
    CSV_FILTERED_TAU_MEASURED, CSV_FILTERED_TAU_DECOMPOSED,
    METADATA_FILE, NORMALISATION_KEYS, JOINT_NAMES, JOINT_COLORS,
)
from Neural_Networks.core.physics import (
    ACTIVE_JOINTS, SMOOTH_WINDOW, SAVGOL_POLYORDER,
    VEL_CLIP, ACC_CLIP,
    fix_timestamps, TimestampReport,
    ticks_to_radians, torque_from_load, torque_friction,
    savgol_smooth, sg_differentiate, raw_derivatives,
    validated_sg_window,
    build_pinocchio_model, compute_rnea_decomposition,
)
from Neural_Networks.data.labels import default_run_dir

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

RAW_DIR    = _ROOT / "raw_samples"
TRAIN_DIR  = _HERE.parent / "train_data"   # Neural_Networks/train_data/
XACRO_PATH = _ROOT / "robot_description" / "urdf" / "kikobot.xacro"

BG_MAIN  = "#f0f0f0"
BG_FILE  = "#e8e8e8"
BG_Q     = "#dce8f5"
BG_QD    = "#daf0da"
BG_QDD   = "#f5f0da"
BG_TAU   = "#f0daf5"
BG_ANA   = "#f5e8da"
BG_BUILD = "#e6edd6"
FG_BTN   = "#ffffff"
BG_BTN   = "#4a90d9"
BG_GO    = "#5ba55b"

SG_MODES    = ["interp", "nearest", "mirror", "wrap", "constant"]
PANEL_FIG_H = 3.6  # inches

# Axes position inside the figure (fractions). Kept in sync on resize and redraw.
_PLOT_SUBPLOT_ADJUST = dict(left=0.07, right=0.995, top=0.96, bottom=0.16)


def filtered_qd_qdd_from_params(
    filt_q: np.ndarray,
    dt: float,
    qd_w: int,
    qd_p: int,
    qd_mode: str,
    qdd_locked: bool,
    qdd_w: int,
    qdd_p: int,
    qdd_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filtered joint velocity and acceleration from filtered_q.

    When qdd_locked, qd and qdd are from one SG fit (qd params). When unlocked,
    qd uses the qd fit and qdd uses a separate SG fit (qdd params) for deriv=2.
    """
    if qdd_locked:
        return sg_differentiate(filt_q, dt, qd_w, qd_p, qd_mode)
    filt_qd, _ = sg_differentiate(filt_q, dt, qd_w, qd_p, qd_mode)
    _, filt_qdd = sg_differentiate(filt_q, dt, qdd_w, qdd_p, qdd_mode)
    return filt_qd, filt_qdd


def filtered_qd_qdd_from_panels(
    filt_q: np.ndarray,
    dt: float,
    qd_panel: object,
    qdd_panel: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as filtered_qd_qdd_from_params using GUI panel state."""
    _, qd_w, qd_p = qd_panel.get_sg_params()
    qd_mode = qd_panel.get_mode()
    locked = getattr(qdd_panel, "_lock_var", None) is not None and bool(
        qdd_panel._lock_var.get())
    if locked:
        return filtered_qd_qdd_from_params(
            filt_q, dt, qd_w, qd_p, qd_mode, True, qd_w, qd_p, qd_mode)
    _, qdd_w, qdd_p = qdd_panel.get_sg_params()
    qdd_mode = qdd_panel.get_mode()
    return filtered_qd_qdd_from_params(
        filt_q, dt, qd_w, qd_p, qd_mode, False, qdd_w, qdd_p, qdd_mode)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class _Tooltip:
    _DELAY_MS = 400
    def __init__(self, widget, text):
        self._widget = widget
        self.text = text
        self._tw = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
    def _schedule(self, _=None):
        self._after_id = self._widget.after(self._DELAY_MS, self._show)
    def _show(self):
        if self._tw or not self.text:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tw = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify=tk.LEFT,
                 bg="#ffffdd", fg="#333", relief=tk.SOLID, bd=1,
                 font=("sans-serif", 8), wraplength=400).pack()
    def _hide(self, _=None):
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tw:
            self._tw.destroy()
            self._tw = None


def _tip(widget, text):
    _Tooltip(widget, text)
    return widget


def _vsep(parent, bg):
    tk.Frame(parent, bg="#aaaacc", width=1, height=22).pack(side=tk.LEFT, padx=9)


class ScrollFrame(tk.Frame):
    def __init__(self, parent, bg=BG_MAIN, **kw):
        super().__init__(parent, bg=bg, **kw)
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self._scroll = tk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self.inner = tk.Frame(self._canvas, bg=bg)
        self.inner.bind("<Configure>", lambda _: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._win_id = self._canvas.create_window((0, 0), window=self.inner, anchor=tk.NW)
        self._canvas.configure(yscrollcommand=self._scroll.set)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self.bind_scroll(self._canvas)
        self.bind_scroll(self.inner)

    def _on_canvas_resize(self, event):
        """Stretch the inner frame to match the canvas width."""
        self._canvas.itemconfigure(self._win_id, width=event.width)

    def bind_scroll(self, widget):
        widget.bind("<MouseWheel>", lambda e: self._canvas.yview_scroll(-e.delta // 120, "units"))
        widget.bind("<Button-4>", lambda _: self._canvas.yview_scroll(-3, "units"))
        widget.bind("<Button-5>", lambda _: self._canvas.yview_scroll(3, "units"))


# ═══════════════════════════════════════════════════════════════════════════════
#  SubplotPanel -- one panel per quantity
# ═══════════════════════════════════════════════════════════════════════════════

class SubplotPanel:
    """A single panel: plot + SG controls for one quantity."""

    def __init__(self, parent, title, key, panel_type, bg, app):
        self.key = key
        self.panel_type = panel_type  # "smooth", "derive", "analytical"
        self.app = app
        self._bg = bg
        self._data = None
        self._t = None

        frame = tk.LabelFrame(parent, text=f"  {title}  ", bg=bg,
                               font=("sans-serif", 10, "bold"), padx=4, pady=2)
        frame.pack(fill=tk.X, padx=4, pady=2)
        self._frame = frame

        # Controls row
        ctrl = tk.Frame(frame, bg=bg)
        ctrl.pack(fill=tk.X, pady=(0, 1))

        if panel_type == "smooth":
            self._build_smooth_ctrl(ctrl, bg)
        elif panel_type == "derive":
            self._build_derive_ctrl(ctrl, bg)
        elif panel_type == "analytical":
            self._build_analytical_ctrl(ctrl, bg)

        # Matplotlib figure
        self._fig = Figure(figsize=(10, PANEL_FIG_H), dpi=90, facecolor=bg)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("white")

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=frame)
        canvas_tk = self._canvas_widget.get_tk_widget()
        canvas_tk.configure(bd=0, highlightthickness=0, relief=tk.FLAT, bg=bg)
        canvas_tk.pack(fill=tk.BOTH, expand=True)
        self._resize_after_id = None
        # add="+" keeps FigureCanvasTkAgg's internal <Configure> handler so the
        # figure tracks widget size; our debounced handler applies margins.
        canvas_tk.bind("<Configure>", self._on_canvas_configure, add="+")

        # Toolbar
        tb_frame = tk.Frame(frame, bg=bg)
        tb_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self._canvas_widget, tb_frame)
        toolbar.configure(bg=bg)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        # Info label
        self._info_var = tk.StringVar(value="")
        tk.Label(frame, textvariable=self._info_var, bg=bg,
                 font=("sans-serif", 7), fg="#555").pack(anchor=tk.W)

    def _on_canvas_configure(self, event):
        """Debounced handler: resize figure to match the Tk widget."""
        if self._resize_after_id is not None:
            self._frame.after_cancel(self._resize_after_id)
        self._resize_after_id = self._frame.after(
            80, lambda w=event.width, h=event.height: self._apply_resize(w, h))

    def _apply_plot_margins(self) -> None:
        self._fig.subplots_adjust(**_PLOT_SUBPLOT_ADJUST)

    def _apply_resize(self, w_px, h_px):
        self._resize_after_id = None
        dpi = self._fig.get_dpi()
        w_in = max(w_px / dpi, 1.0)
        h_in = max(h_px / dpi, 0.5)
        self._fig.set_size_inches(w_in, h_in, forward=True)
        self._apply_plot_margins()
        self._canvas_widget.draw_idle()

    # ── smooth controls (q, tau_measured) ─────────────────────────────────

    def _build_smooth_ctrl(self, ctrl, bg):
        self._enabled_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="SG smooth", variable=self._enabled_var,
                       bg=bg, command=self.app._redraw_all).pack(side=tk.LEFT)
        _tip(ctrl.winfo_children()[-1],
             "Enable Savitzky-Golay smoothing (deriv=0) on this signal.")
        tk.Label(ctrl, text="win:", bg=bg, font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(8, 1))
        self._win_sp = tk.Spinbox(ctrl, from_=3, to=201, increment=2, width=5,
                                  font=("sans-serif", 8))
        self._win_sp.delete(0, tk.END)
        self._win_sp.insert(0, "15")
        self._win_sp.pack(side=tk.LEFT)
        tk.Label(ctrl, text="poly:", bg=bg, font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(6, 1))
        self._poly_sp = tk.Spinbox(ctrl, from_=1, to=8, increment=1, width=4,
                                   font=("sans-serif", 8))
        self._poly_sp.delete(0, tk.END)
        self._poly_sp.insert(0, "3")
        self._poly_sp.pack(side=tk.LEFT)
        tk.Button(ctrl, text="Redraw", command=self.app._redraw_all,
                  font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(10, 0))
        self._show_raw_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Show raw", variable=self._show_raw_var,
                       bg=bg, command=self.app._redraw_all).pack(side=tk.LEFT, padx=(10, 0))

    # ── derive controls (qd, qdd) ────────────────────────────────────────

    def _build_derive_ctrl(self, ctrl, bg):
        tk.Label(ctrl, text="SG deriv from filtered q", bg=bg,
                 font=("sans-serif", 8, "bold")).pack(side=tk.LEFT)
        tk.Label(ctrl, text="win:", bg=bg, font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(8, 1))
        self._win_sp = tk.Spinbox(ctrl, from_=5, to=201, increment=2, width=5,
                                  font=("sans-serif", 8))
        self._win_sp.delete(0, tk.END)
        self._win_sp.insert(0, str(SMOOTH_WINDOW))
        self._win_sp.pack(side=tk.LEFT)
        tk.Label(ctrl, text="poly:", bg=bg, font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(6, 1))
        self._poly_sp = tk.Spinbox(ctrl, from_=2, to=8, increment=1, width=4,
                                   font=("sans-serif", 8))
        self._poly_sp.delete(0, tk.END)
        self._poly_sp.insert(0, str(SAVGOL_POLYORDER))
        self._poly_sp.pack(side=tk.LEFT)
        tk.Label(ctrl, text="mode:", bg=bg, font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(6, 1))
        self._mode_var = tk.StringVar(value="interp")
        ttk.Combobox(ctrl, textvariable=self._mode_var, values=SG_MODES,
                     state="readonly", width=8, font=("sans-serif", 8)).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Redraw", command=self.app._redraw_all,
                  font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(10, 0))

        if self.key == "qdd":
            self._lock_var = tk.BooleanVar(value=True)
            tk.Checkbutton(ctrl, text="Lock to qd", variable=self._lock_var,
                           bg=bg, command=self._on_lock_change).pack(side=tk.LEFT, padx=(10, 0))
            _tip(ctrl.winfo_children()[-1],
                 "When locked, qdd uses the same SG params as qd (same polynomial fit).")

    def _sync_spinboxes_from_qd(self) -> None:
        """When qdd is locked to qd, mirror qd's SG parameters in this panel."""
        if not hasattr(self, "_lock_var") or not self._lock_var.get():
            return
        qd = self.app._panels[1]
        self._win_sp.delete(0, tk.END)
        self._win_sp.insert(0, qd._win_sp.get())
        self._poly_sp.delete(0, tk.END)
        self._poly_sp.insert(0, qd._poly_sp.get())
        self._mode_var.set(qd._mode_var.get())

    def _on_lock_change(self):
        if hasattr(self, "_lock_var") and self._lock_var.get():
            self._sync_spinboxes_from_qd()
        self.app._redraw_all()

    # ── analytical controls (tau_analytical) ──────────────────────────────

    def _build_analytical_ctrl(self, ctrl, bg):
        tk.Label(ctrl, text="RNEA + friction from filtered q/qd/qdd", bg=bg,
                 font=("sans-serif", 8, "bold")).pack(side=tk.LEFT)
        _vsep(ctrl, bg)
        self._postfilter_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Post-filter (SG)", variable=self._postfilter_var,
                       bg=bg, command=self.app._redraw_all).pack(side=tk.LEFT)
        tk.Label(ctrl, text="win:", bg=bg, font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(6, 1))
        self._win_sp = tk.Spinbox(ctrl, from_=3, to=201, increment=2, width=5,
                                  font=("sans-serif", 8))
        self._win_sp.delete(0, tk.END)
        self._win_sp.insert(0, "15")
        self._win_sp.pack(side=tk.LEFT)
        tk.Label(ctrl, text="poly:", bg=bg, font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(6, 1))
        self._poly_sp = tk.Spinbox(ctrl, from_=1, to=8, increment=1, width=4,
                                   font=("sans-serif", 8))
        self._poly_sp.delete(0, tk.END)
        self._poly_sp.insert(0, "3")
        self._poly_sp.pack(side=tk.LEFT)
        tk.Button(ctrl, text="Redraw", command=self.app._redraw_all,
                  font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(10, 0))
        self._show_measured_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Show measured tau", variable=self._show_measured_var,
                       bg=bg, command=self.app._redraw_all).pack(side=tk.LEFT, padx=(10, 0))

    # ── common ────────────────────────────────────────────────────────────

    def get_sg_params(self) -> tuple[bool, int, int]:
        """Return (enabled, window, polyorder) for this panel."""
        if self.panel_type == "smooth":
            enabled = self._enabled_var.get()
        elif self.panel_type == "analytical":
            enabled = self._postfilter_var.get()
        else:
            enabled = True
        try:
            win = int(self._win_sp.get())
        except (ValueError, tk.TclError):
            win = 15
        try:
            poly = int(self._poly_sp.get())
        except (ValueError, tk.TclError):
            poly = 3
        return enabled, win, poly

    def get_mode(self) -> str:
        if hasattr(self, '_mode_var'):
            return self._mode_var.get()
        return "interp"

    def load(self, proc: dict) -> None:
        self._data = proc
        self._t = proc["t"]

    def clear(self):
        self._data = None
        self._ax.clear()
        self._ax.set_facecolor("white")
        self._canvas_widget.draw_idle()

    def _redraw(self, *_):
        if self._data is None:
            return
        self._ax.clear()
        self._ax.set_facecolor("white")
        t = self._t
        active = self.app.active_joints
        if not active:
            active = list(range(ACTIVE_JOINTS))

        if self.panel_type == "smooth":
            self._draw_smooth(t, active)
        elif self.panel_type == "derive":
            self._draw_derive(t, active)
        elif self.panel_type == "analytical":
            self._draw_analytical(t, active)

        self._ax.set_xlabel("Time [s]", fontsize=8)
        self._ax.legend(fontsize=7, ncol=min(len(active), 5), loc="upper right")
        self._ax.tick_params(labelsize=7)
        self._ax.grid(True, alpha=0.2, linewidth=0.5)
        self._apply_plot_margins()
        self._canvas_widget.draw_idle()

    def _draw_smooth(self, t, active):
        raw_key = "raw_q" if self.key == "q" else "raw_tau_measured"
        raw = self._data[raw_key]
        enabled, win, poly = self.get_sg_params()
        if enabled:
            filtered = savgol_smooth(raw, win, poly, axis=0)
        else:
            filtered = raw

        for j in active:
            c = JOINT_COLORS[j]
            self._ax.plot(t, filtered[:, j], color=c, linewidth=1.0,
                          label=JOINT_NAMES[j])
            if hasattr(self, '_show_raw_var') and self._show_raw_var.get() and enabled:
                self._ax.plot(t, raw[:, j], color=c, linewidth=0.4, alpha=0.4, linestyle="--")

        unit = "rad" if self.key == "q" else "Nm"
        self._ax.set_ylabel(f"{self.key} [{unit}]", fontsize=8)
        info = f"SG smooth: win={win}, poly={poly}" if enabled else "No filter"
        self._info_var.set(info)

    def _draw_derive(self, t, active):
        q_panel = self.app._panels[0]
        qd_panel = self.app._panels[1]
        qdd_panel = self.app._panels[2]
        raw_q = self._data["raw_q"]
        q_enabled, q_win, q_poly = q_panel.get_sg_params()
        if q_enabled:
            filt_q = savgol_smooth(raw_q, q_win, q_poly, axis=0)
        else:
            filt_q = raw_q

        dt = self._data.get("median_dt", float(np.median(np.diff(t))))
        if dt <= 0:
            dt = 1.0 / 300.0

        filt_qd, filt_qdd = filtered_qd_qdd_from_panels(filt_q, dt, qd_panel, qdd_panel)
        signal = filt_qd if self.key == "qd" else filt_qdd

        for j in active:
            self._ax.plot(t, signal[:, j], color=JOINT_COLORS[j], linewidth=1.0,
                          label=JOINT_NAMES[j])

        unit = "rad/s" if self.key == "qd" else "rad/s^2"
        self._ax.set_ylabel(f"{self.key} [{unit}]", fontsize=8)

        if self.key == "qd":
            _, qd_w, qd_p = qd_panel.get_sg_params()
            d_mode = qd_panel.get_mode()
            rw = validated_sg_window(len(t), qd_w, max(qd_p, 2))
            self._info_var.set(
                f"SG deriv: win={rw}, poly={max(qd_p, 2)}, mode={d_mode}, "
                f"dt={dt*1000:.2f}ms")
        else:
            locked = qdd_panel._lock_var.get()
            if locked:
                _, qd_w, qd_p = qd_panel.get_sg_params()
                d_mode = qd_panel.get_mode()
                rw = validated_sg_window(len(t), qd_w, max(qd_p, 2))
                self._info_var.set(
                    f"SG deriv (locked to qd): win={rw}, poly={max(qd_p, 2)}, "
                    f"mode={d_mode}, dt={dt*1000:.2f}ms")
            else:
                _, qdd_w, qdd_p = qdd_panel.get_sg_params()
                qdd_mode = qdd_panel.get_mode()
                rw = validated_sg_window(len(t), qdd_w, max(qdd_p, 2))
                self._info_var.set(
                    f"SG deriv: win={rw}, poly={max(qdd_p, 2)}, mode={qdd_mode}, "
                    f"dt={dt*1000:.2f}ms")

    def _draw_analytical(self, t, active):
        pin_model = self.app._get_pinocchio()
        if pin_model is None:
            self._ax.text(0.5, 0.5, "Pinocchio model not available",
                          transform=self._ax.transAxes, ha="center", fontsize=10)
            self._info_var.set("RNEA unavailable -- install pinocchio + xacro")
            return

        model, data = pin_model
        q_panel = self.app._panels[0]
        raw_q = self._data["raw_q"]
        q_enabled, q_win, q_poly = q_panel.get_sg_params()
        filt_q = savgol_smooth(raw_q, q_win, q_poly, axis=0) if q_enabled else raw_q

        qd_panel = self.app._panels[1]
        qdd_panel = self.app._panels[2]
        dt = self._data.get("median_dt", float(np.median(np.diff(t))))
        if dt <= 0:
            dt = 1.0 / 300.0

        filt_qd, filt_qdd = filtered_qd_qdd_from_panels(filt_q, dt, qd_panel, qdd_panel)

        tau_rnea, _, _, _ = compute_rnea_decomposition(
            model, data, filt_q, filt_qd, filt_qdd, n_active=ACTIVE_JOINTS)
        tau_fric = torque_friction(filt_qd[:, :ACTIVE_JOINTS])
        tau_ana = tau_rnea + tau_fric

        postfilter_on, pf_win, pf_poly = self.get_sg_params()
        if postfilter_on:
            tau_ana = savgol_smooth(tau_ana, pf_win, pf_poly, axis=0)

        for j in active:
            c = JOINT_COLORS[j]
            self._ax.plot(t, tau_ana[:, j], color=c, linewidth=1.0,
                          label=f"{JOINT_NAMES[j]} analytical")
            if self._show_measured_var.get():
                raw_tau = self._data["raw_tau_measured"]
                self._ax.plot(t, raw_tau[:, j], color=c, linewidth=0.4,
                              alpha=0.4, linestyle="--")

        self._ax.set_ylabel("tau_analytical [Nm]", fontsize=8)
        info = "RNEA + friction"
        if postfilter_on:
            info += f" + SG post-filter (win={pf_win}, poly={pf_poly})"
        self._info_var.set(info)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocessApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Preprocess Data & Build Dataset")
        self.root.geometry("1600x950")
        self.root.minsize(1000, 600)
        self.root.configure(bg=BG_MAIN)

        self._raw_proc = None
        self._loaded_name = ""
        self._ts_report: TimestampReport | None = None
        self._pin_cache = None

        # Geometry composition state (populated by _geom_scan)
        self._geom_rows: dict = {}          # geom_type → {include, count, assign, avail}
        self._total_budget_var = tk.IntVar(value=0)
        self._geom_comp_inner: tk.Frame | None = None  # rebuilt on each scan

        # Joint checkboxes
        top = tk.Frame(self.root, bg="#2e3040")
        top.pack(fill=tk.X)
        self._joint_vars = []
        for j in range(ACTIVE_JOINTS):
            v = tk.BooleanVar(value=True)
            tk.Checkbutton(top, text=JOINT_NAMES[j], variable=v, bg="#2e3040",
                           fg="white", selectcolor="#2e3040",
                           command=self._redraw_all).pack(side=tk.LEFT, padx=4)
            self._joint_vars.append(v)

        _vsep(top, "#2e3040")
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(top, textvariable=self._status_var, bg="#2e3040", fg="#aaa",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=10)

        # Trim controls
        _vsep(top, "#2e3040")
        tk.Label(top, text="Trim front %:", bg="#2e3040", fg="white",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(4, 1))
        self._trim_front = tk.Spinbox(top, from_=0, to=49, width=4, font=("sans-serif", 8))
        self._trim_front.delete(0, tk.END)
        self._trim_front.insert(0, "1")
        self._trim_front.pack(side=tk.LEFT)
        tk.Label(top, text="back %:", bg="#2e3040", fg="white",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(6, 1))
        self._trim_back = tk.Spinbox(top, from_=0, to=49, width=4, font=("sans-serif", 8))
        self._trim_back.delete(0, tk.END)
        self._trim_back.insert(0, "1")
        self._trim_back.pack(side=tk.LEFT)
        tk.Button(top, text="Apply Trim", command=self._reload_with_trim,
                  font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(6, 0))

        self._n_label = tk.StringVar(value="N=--")
        tk.Label(top, textvariable=self._n_label, bg="#2e3040", fg="#ccc",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=10)

        # Timestamp info
        self._ts_info = tk.StringVar(value="")
        tk.Label(top, textvariable=self._ts_info, bg="#2e3040", fg="#ffa",
                 font=("sans-serif", 7)).pack(side=tk.RIGHT, padx=10)

        # Paned window: file browser | panels
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=BG_MAIN, sashwidth=6)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: file browser
        left = tk.Frame(paned, bg=BG_FILE, width=280)
        paned.add(left, minsize=200)
        self._build_file_browser(left)

        # Right: panels + build
        right = tk.Frame(paned, bg=BG_MAIN)
        paned.add(right, minsize=600)

        scroll = ScrollFrame(right, bg=BG_MAIN)
        scroll.pack(fill=tk.BOTH, expand=True)

        self._panels: list[SubplotPanel] = [
            SubplotPanel(scroll.inner, "q  (joint positions)",    "q",              "smooth",     BG_Q,   self),
            SubplotPanel(scroll.inner, "qd (joint velocities)",   "qd",             "derive",     BG_QD,  self),
            SubplotPanel(scroll.inner, "qdd (joint accelerations)","qdd",           "derive",     BG_QDD, self),
            SubplotPanel(scroll.inner, "tau_measured (measured torque)", "tau_measured","smooth",   BG_TAU, self),
            SubplotPanel(scroll.inner, "tau_analytical (RNEA + friction)","tau_analytical","analytical",BG_ANA,self),
        ]
        for p in self._panels:
            p._frame.bind("<MouseWheel>",
                          lambda e: scroll._canvas.yview_scroll(-e.delta // 120, "units"))

        # Build panel
        self._build_dataset_panel(scroll.inner)

    def _suggested_output_path(self) -> str:
        """Default output folder: run_MMDD_HHMM_<param_tag>_<id> under train_data/."""
        try:
            q_enabled, q_win, q_poly = self._panels[0].get_sg_params()
            _, d_win, d_poly = self._panels[1].get_sg_params()
            d_mode = self._panels[1].get_mode()
            qdd_locked = self._panels[2]._lock_var.get()
            _, qdd_w, qdd_p = self._panels[2].get_sg_params()
            qdd_mode = self._panels[2].get_mode()
            tau_enabled, tau_win, tau_poly = self._panels[3].get_sg_params()
            pf_enabled, pf_win, pf_poly = self._panels[4].get_sg_params()
            use_rnea = self._rnea_var.get()
            tr = float(self._split_train.get())
            vl = float(self._split_val.get())
            te = float(self._split_test.get())
            trim_fp = float(self._trim_front.get())
            trim_bp = float(self._trim_back.get())
        except (ValueError, tk.TclError, AttributeError):
            return str(TRAIN_DIR / f"run_{datetime.now().strftime('%m%d_%H%M')}_{uuid.uuid4().hex[:6]}")
        return default_run_dir(
            TRAIN_DIR,
            q_smooth=q_enabled,
            q_win=q_win,
            q_poly=q_poly,
            deriv_win=d_win,
            deriv_poly=d_poly,
            deriv_mode=d_mode,
            qdd_locked=qdd_locked,
            qdd_win=qdd_w,
            qdd_poly=qdd_p,
            qdd_mode=qdd_mode,
            tau_smooth=tau_enabled,
            tau_win=tau_win,
            tau_poly=tau_poly,
            tau_ana_pf=pf_enabled,
            tau_ana_win=pf_win,
            tau_ana_poly=pf_poly,
            use_rnea=use_rnea,
            train_ratio=tr,
            val_ratio=vl,
            test_ratio=te,
            trim_front_pct=trim_fp,
            trim_back_pct=trim_bp,
            unique_suffix=uuid.uuid4().hex[:6],
        )

    @property
    def active_joints(self) -> list[int]:
        return [j for j, v in enumerate(self._joint_vars) if v.get()]

    def _get_pinocchio(self):
        if self._pin_cache is not None:
            return self._pin_cache
        try:
            model, data, _ = build_pinocchio_model(str(XACRO_PATH))
            self._pin_cache = (model, data)
            return self._pin_cache
        except Exception as e:
            self._status(f"Pinocchio error: {e}")
            return None

    # ── File browser ──────────────────────────────────────────────────────

    def _build_file_browser(self, parent):
        tk.Label(parent, text="Raw Samples", bg=BG_FILE,
                 font=("sans-serif", 10, "bold")).pack(anchor=tk.W, padx=6, pady=(6, 2))

        row = tk.Frame(parent, bg=BG_FILE)
        row.pack(fill=tk.X, padx=6)
        self._dir_var = tk.StringVar(value=str(RAW_DIR))
        tk.Entry(row, textvariable=self._dir_var, font=("sans-serif", 8)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text="...", command=self._browse_dir, width=3).pack(side=tk.LEFT, padx=2)
        tk.Button(row, text="Refresh", command=self._refresh_files,
                  font=("sans-serif", 8)).pack(side=tk.LEFT)

        self._file_count = tk.StringVar(value="0 files")
        tk.Label(parent, textvariable=self._file_count, bg=BG_FILE,
                 font=("sans-serif", 8), fg="#666").pack(anchor=tk.W, padx=6)

        self._file_list = tk.Listbox(parent, font=("sans-serif", 8), selectmode=tk.SINGLE)
        self._file_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self._file_list.bind("<<ListboxSelect>>", self._on_file_select)
        self._refresh_files()

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self._dir_var.get())
        if d:
            self._dir_var.set(d)
            self._refresh_files()

    def _refresh_files(self):
        self._file_list.delete(0, tk.END)
        d = self._dir_var.get()
        if not os.path.isdir(d):
            self._file_count.set("Invalid directory")
            return
        files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
        for f in files:
            self._file_list.insert(tk.END, Path(f).stem)
        self._file_count.set(f"{len(files)} files")
        # Refresh geometry composition panel whenever the file list updates
        if hasattr(self, "_geom_comp_inner") and self._geom_comp_inner is not None:
            self._geom_scan()

    def _on_file_select(self, _=None):
        sel = self._file_list.curselection()
        if not sel:
            return
        stem = self._file_list.get(sel[0])
        path = os.path.join(self._dir_var.get(), stem + ".json")
        self._load_file(path)

    def _load_file(self, path):
        self._status(f"Loading {Path(path).name} ...")
        try:
            L, M, N = load_raw_sample(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        # Convert to radians & extract torque
        q_full = ticks_to_radians(L["act_pos"], M["joint_map"], M["ticks_to_rad"], M["dof"])
        ts_report = fix_timestamps(L["t"])
        t = ts_report.t_fixed
        tau_full = torque_from_load(L["load"], L["voltage"], M["joint_map"])

        # Slice to active joints
        nj = ACTIVE_JOINTS
        raw_q = q_full[:, :nj].astype(np.float32)
        raw_tau = tau_full[:, :nj].astype(np.float32)
        t = t.astype(np.float32)

        # Apply trim
        try:
            fp = float(self._trim_front.get())
            bp = float(self._trim_back.get())
            front_n, back_n = resolve_front_back_trim(len(t), fp, bp)
        except (ValueError, tk.TclError):
            front_n, back_n = 0, 0

        end = len(t) - back_n if back_n > 0 else len(t)
        sl = slice(front_n, end)

        self._raw_proc = {
            "raw_q": raw_q[sl],
            "raw_tau_measured": raw_tau[sl],
            "t": t[sl],
            "median_dt": ts_report.median_dt,
            "source_file": M.get("source_file", Path(path).name),
            "meta": M,
        }
        self._ts_report = ts_report
        self._loaded_name = Path(path).name
        n_after = len(t[sl])

        self._n_label.set(f"N={n_after}")
        ts_info = (f"dt={ts_report.median_dt*1000:.2f}ms  "
                   f"mono_fix={ts_report.n_nonmonotonic}  "
                   f"dt_outlier={ts_report.n_outlier_dt}  "
                   f"CV={ts_report.dt_cv_after:.4f}"
                   + ("  RESAMPLED" if ts_report.was_resampled else ""))
        self._ts_info.set(ts_info)
        self._status(f"Loaded {self._loaded_name}  N={n_after}")

        for panel in self._panels:
            panel.load(self._raw_proc)
        self._redraw_all()

    def _reload_with_trim(self):
        if self._raw_proc is None:
            messagebox.showwarning("No data", "Load a file first.")
            return
        sel = self._file_list.curselection()
        if sel:
            stem = self._file_list.get(sel[0])
            path = os.path.join(self._dir_var.get(), stem + ".json")
            self._load_file(path)

    def _redraw_all(self):
        self._panels[2]._sync_spinboxes_from_qd()
        for p in self._panels:
            p._redraw()

    def _status(self, msg):
        self._status_var.set(msg)

    def _log(self, msg):
        if hasattr(self, '_log_text'):
            self._log_text.insert(tk.END, msg + "\n")
            self._log_text.see(tk.END)

    # ── Build panel ───────────────────────────────────────────────────────

    def _build_dataset_panel(self, parent):
        bf = tk.LabelFrame(parent, text="  Build Dataset  ", bg=BG_BUILD,
                           font=("sans-serif", 10, "bold"), padx=6, pady=6)
        bf.pack(fill=tk.X, padx=6, pady=6)

        # Output dir
        row1 = tk.Frame(bf, bg=BG_BUILD)
        row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="Output:", bg=BG_BUILD, font=("sans-serif", 9)).pack(side=tk.LEFT)
        self._run_var = tk.StringVar(value="")
        tk.Entry(row1, textvariable=self._run_var, font=("sans-serif", 8), width=60).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        tk.Button(row1, text="...", command=self._browse_out, width=3).pack(side=tk.LEFT)

        # ── Dataset Composition ──────────────────────────────────────────────
        comp_frame = tk.LabelFrame(bf, text="  Dataset Composition  ", bg=BG_BUILD,
                                   font=("sans-serif", 9, "bold"), padx=4, pady=4)
        comp_frame.pack(fill=tk.X, pady=(4, 2))

        # Budget row
        budget_row = tk.Frame(comp_frame, bg=BG_BUILD)
        budget_row.pack(fill=tk.X, pady=(0, 2))
        tk.Label(budget_row, text="Total trajectories:", bg=BG_BUILD,
                 font=("sans-serif", 9)).pack(side=tk.LEFT)
        self._budget_spin = tk.Spinbox(
            budget_row, from_=1, to=9999, width=5,
            textvariable=self._total_budget_var,
            font=("sans-serif", 9),
            command=self._auto_distribute)
        self._budget_spin.pack(side=tk.LEFT, padx=4)
        self._budget_avail_label = tk.Label(budget_row, text="of 0 available",
                                            bg=BG_BUILD, font=("sans-serif", 8), fg="#555")
        self._budget_avail_label.pack(side=tk.LEFT, padx=4)
        tk.Button(budget_row, text="Auto-Distribute",
                  command=self._auto_distribute,
                  font=("sans-serif", 8)).pack(side=tk.LEFT, padx=6)
        tk.Button(budget_row, text="Refresh Scan",
                  command=self._geom_scan,
                  font=("sans-serif", 8)).pack(side=tk.LEFT)
        _tip(self._budget_spin,
             "Total number of trajectories to include in the dataset.\n"
             "Click Auto-Distribute to spread this budget proportionally\nacross all "
             "included geometry types.")

        # Column headers
        hdr = tk.Frame(comp_frame, bg="#d0d8c0")
        hdr.pack(fill=tk.X)
        for (txt, w) in [("✓", 3), ("Geometry Type", 18), ("Available", 8),
                         ("Use Count", 9), ("Assignment", 22)]:
            tk.Label(hdr, text=txt, bg="#d0d8c0", font=("sans-serif", 8, "bold"),
                     width=w, anchor=tk.W).pack(side=tk.LEFT, padx=2)

        # Scrollable container for per-type rows
        geom_canvas = tk.Canvas(comp_frame, bg=BG_BUILD, height=160, highlightthickness=0)
        geom_scroll = tk.Scrollbar(comp_frame, orient=tk.VERTICAL,
                                   command=geom_canvas.yview)
        geom_canvas.configure(yscrollcommand=geom_scroll.set)
        geom_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        geom_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._geom_canvas = geom_canvas
        self._geom_comp_inner = tk.Frame(geom_canvas, bg=BG_BUILD)
        self._geom_canvas_win = geom_canvas.create_window(
            (0, 0), window=self._geom_comp_inner, anchor="nw")
        self._geom_comp_inner.bind(
            "<Configure>",
            lambda e: geom_canvas.configure(scrollregion=geom_canvas.bbox("all")))
        geom_canvas.bind(
            "<Configure>",
            lambda e: geom_canvas.itemconfig(self._geom_canvas_win, width=e.width))

        # Populate from raw_dir on first build
        self.root.after(200, self._geom_scan)

        # Splits
        row2 = tk.Frame(bf, bg=BG_BUILD)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="Split:", bg=BG_BUILD, font=("sans-serif", 9)).pack(side=tk.LEFT)
        self._split_train = tk.Entry(row2, width=5, font=("sans-serif", 8))
        self._split_train.insert(0, "0.70")
        self._split_train.pack(side=tk.LEFT, padx=2)
        tk.Label(row2, text="/", bg=BG_BUILD).pack(side=tk.LEFT)
        self._split_val = tk.Entry(row2, width=5, font=("sans-serif", 8))
        self._split_val.insert(0, "0.15")
        self._split_val.pack(side=tk.LEFT, padx=2)
        tk.Label(row2, text="/", bg=BG_BUILD).pack(side=tk.LEFT)
        self._split_test = tk.Entry(row2, width=5, font=("sans-serif", 8))
        self._split_test.insert(0, "0.15")
        self._split_test.pack(side=tk.LEFT, padx=2)
        tk.Label(row2, text="(train/val/test)", bg=BG_BUILD,
                 font=("sans-serif", 8), fg="#666").pack(side=tk.LEFT, padx=6)

        self._rnea_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row2, text="Compute RNEA (tau_analytical)",
                       variable=self._rnea_var, bg=BG_BUILD).pack(side=tk.LEFT, padx=10)

        # Split mode
        row2b = tk.Frame(bf, bg=BG_BUILD)
        row2b.pack(fill=tk.X, pady=2)
        tk.Label(row2b, text="Split mode:", bg=BG_BUILD, font=("sans-serif", 9)).pack(side=tk.LEFT)
        self._split_mode_var = tk.StringVar(value="stratified (smart)")
        _split_mode_cb = ttk.Combobox(
            row2b, textvariable=self._split_mode_var,
            values=["stratified (smart)", "random", "temporal"],
            state="readonly", width=18, font=("sans-serif", 8))
        _split_mode_cb.pack(side=tk.LEFT, padx=4)
        _tip(_split_mode_cb,
             "stratified (smart): each geometry type is proportionally represented in every "
             "split.\nrandom: random shuffle of all trajectories (seed=42).\n"
             "temporal: sorted by filename order.")

        # Build button + log
        row3 = tk.Frame(bf, bg=BG_BUILD)
        row3.pack(fill=tk.X, pady=4)
        self._build_btn = tk.Button(row3, text="Build Dataset", bg=BG_GO, fg=FG_BTN,
                                    font=("sans-serif", 10, "bold"),
                                    command=self._build_dataset)
        self._build_btn.pack(side=tk.LEFT, padx=4)

        self._build_summary = tk.Label(row3, text="", bg=BG_BUILD,
                                       font=("sans-serif", 7), fg="#555", anchor=tk.W)
        self._build_summary.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self._log_text = scrolledtext.ScrolledText(bf, height=6, font=("monospace", 8),
                                                   state=tk.NORMAL, bg="#fff")
        self._log_text.pack(fill=tk.X, pady=2)

        self._run_var.set(self._suggested_output_path())

    def _browse_out(self):
        d = filedialog.askdirectory(initialdir=str(TRAIN_DIR))
        if d:
            self._run_var.set(d)

    # ── Geometry composition helpers ──────────────────────────────────────

    @staticmethod
    def _geom_from_stem(stem: str) -> str:
        """Extract geometry type from a raw-sample filename stem.
        e.g. 'circle_r100mm_yz_...' → 'circle'
             'regular_polygon_r70mm_...' → 'regular_polygon'
        """
        import re as _re
        m = _re.match(r'^([a-z_]+?)_r\d', stem)
        return m.group(1) if m else "unknown"

    def _geom_scan(self):
        """Scan raw_dir, build per-geometry-type rows in the Composition panel."""
        d = self._dir_var.get() if hasattr(self, "_dir_var") else str(RAW_DIR)
        if not os.path.isdir(d):
            return

        # Group files by geometry type
        import collections as _c
        grps: dict[str, list[str]] = _c.defaultdict(list)
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                grps[self._geom_from_stem(Path(f).stem)].append(f)

        total_avail = sum(len(v) for v in grps.values())

        # Preserve existing user settings where possible
        prev = {gt: {"include": info["include"].get(),
                     "count":   info["count"].get(),
                     "assign":  info["assign"].get()}
                for gt, info in self._geom_rows.items()}

        # Rebuild inner frame
        for w in self._geom_comp_inner.winfo_children():
            w.destroy()
        self._geom_rows.clear()

        _ASSIGN_OPTS = [
            "all splits (stratified)",
            "train only",
            "val only",
            "test only",
            "exclude",
        ]
        _ASSIGN_COLORS = {
            "all splits (stratified)": BG_BUILD,
            "train only":  "#cce8cc",
            "val only":    "#cce0f5",
            "test only":   "#f5e8cc",
            "exclude":     "#f0c8c8",
        }

        for row_idx, (gt, files) in enumerate(sorted(grps.items())):
            avail = len(files)
            rg = "#f5f5ee" if row_idx % 2 == 0 else BG_BUILD

            include_v = tk.BooleanVar(value=prev.get(gt, {}).get("include", True))
            count_v   = tk.IntVar(value=min(prev.get(gt, {}).get("count", avail), avail))
            assign_v  = tk.StringVar(value=prev.get(gt, {}).get("assign", "all splits (stratified)"))

            row_f = tk.Frame(self._geom_comp_inner, bg=rg)
            row_f.pack(fill=tk.X)

            def _make_toggle(av=assign_v, rf=row_f, iv=include_v):
                def _toggle(*_):
                    _col = _ASSIGN_COLORS.get(av.get(), BG_BUILD)
                    if not iv.get():
                        _col = "#e0e0e0"
                    rf.configure(bg=_col)
                    for w in rf.winfo_children():
                        try:
                            w.configure(bg=_col)
                        except tk.TclError:
                            pass
                return _toggle

            _toggle_fn = _make_toggle()
            include_v.trace_add("write", _toggle_fn)
            assign_v.trace_add("write", _toggle_fn)

            tk.Checkbutton(row_f, variable=include_v, bg=rg,
                           command=_toggle_fn).pack(side=tk.LEFT, padx=2)
            tk.Label(row_f, text=gt, bg=rg, font=("sans-serif", 8),
                     width=18, anchor=tk.W).pack(side=tk.LEFT, padx=2)
            tk.Label(row_f, text=str(avail), bg=rg, font=("sans-serif", 8),
                     width=7, anchor=tk.CENTER).pack(side=tk.LEFT, padx=2)
            count_sp = tk.Spinbox(row_f, from_=1, to=avail, width=5,
                                  textvariable=count_v, font=("sans-serif", 8))
            count_sp.pack(side=tk.LEFT, padx=2)
            assign_cb = ttk.Combobox(row_f, textvariable=assign_v,
                                     values=_ASSIGN_OPTS, state="readonly",
                                     width=20, font=("sans-serif", 8))
            assign_cb.pack(side=tk.LEFT, padx=4)

            self._geom_rows[gt] = {
                "include": include_v,
                "count":   count_v,
                "assign":  assign_v,
                "avail":   avail,
            }

        # Update budget Spinbox range and label
        cur_budget = self._total_budget_var.get()
        if cur_budget == 0 or cur_budget > total_avail:
            self._total_budget_var.set(total_avail)
        self._budget_spin.config(to=total_avail)
        self._budget_avail_label.config(text=f"of {total_avail} available")
        self._geom_canvas.configure(scrollregion=self._geom_canvas.bbox("all"))

    def _auto_distribute(self, *_):
        """Distribute total_budget proportionally across included geometry types."""
        included = [(gt, info) for gt, info in self._geom_rows.items()
                    if info["include"].get()
                    and info.get("assign", tk.StringVar()).get() != "exclude"]
        if not included:
            return
        try:
            total = int(self._total_budget_var.get())
        except (ValueError, tk.TclError):
            return

        avails  = [info["avail"] for _, info in included]
        total_avail = sum(avails)
        if total_avail == 0:
            return
        total = max(1, min(total, total_avail))

        # Proportional floor allocation
        fracs  = [a / total_avail for a in avails]
        allocs = [max(1, int(total * f)) for f in fracs]

        # Distribute remainder to types with the largest fractional shortfall
        remainder = total - sum(allocs)
        if remainder != 0:
            residuals = [(total * fracs[i] - allocs[i], i) for i in range(len(allocs))]
            residuals.sort(reverse=(remainder > 0))
            for k in range(abs(remainder)):
                idx = residuals[k % len(residuals)][1]
                allocs[idx] += 1 if remainder > 0 else -1
                allocs[idx] = max(1, min(allocs[idx], avails[idx]))

        for (_, info), alloc in zip(included, allocs):
            info["count"].set(min(alloc, info["avail"]))



    def _build_dataset(self):
        raw_dir = self._dir_var.get()
        if not os.path.isdir(raw_dir):
            messagebox.showerror("Error", f"Directory not found: {raw_dir}")
            return
        run_dir = self._run_var.get().strip()
        if not run_dir:
            messagebox.showerror("Error", "Output directory is empty.")
            return
        try:
            tr = float(self._split_train.get())
            vl = float(self._split_val.get())
            te = float(self._split_test.get())
        except ValueError:
            messagebox.showerror("Error", "Split ratios must be numbers.")
            return
        if abs(tr + vl + te - 1.0) > 0.01:
            messagebox.showerror("Error", f"Splits must sum to 1.0 (got {tr+vl+te:.3f})")
            return

        # Read all SG params from panels
        q_enabled, q_win, q_poly         = self._panels[0].get_sg_params()
        _, d_win, d_poly                  = self._panels[1].get_sg_params()
        d_mode                            = self._panels[1].get_mode()
        qdd_locked                        = self._panels[2]._lock_var.get()
        tau_enabled, tau_win, tau_poly    = self._panels[3].get_sg_params()
        pf_enabled, pf_win, pf_poly      = self._panels[4].get_sg_params()
        use_rnea = self._rnea_var.get()

        try:
            trim_fp = float(self._trim_front.get())
            trim_bp = float(self._trim_back.get())
        except (ValueError, tk.TclError):
            trim_fp, trim_bp = 0.0, 0.0

        _split_mode_map = {
            "stratified (smart)": "stratified",
            "random": "random",
            "temporal": "temporal",
        }
        split_mode_display = self._split_mode_var.get()
        split_mode = _split_mode_map.get(split_mode_display, "stratified")

        # Confirmation dialog
        msg = (
            f"Build dataset\n\n"
            f"  Source:     {raw_dir}\n"
            f"  Output:     {run_dir}\n"
            f"  Split:      {tr}/{vl}/{te}  ({split_mode_display})\n"
            f"  Trim:       {trim_fp}% front, {trim_bp}% back\n\n"
            f"  q smooth:       {'savgol(win=' + str(q_win) + ',poly=' + str(q_poly) + ')' if q_enabled else 'none'}\n"
            f"  qd deriv:       savgol(win={d_win}, poly={d_poly}, mode={d_mode}, deriv=1)\n"
            + (
                f"  qdd deriv:      locked to qd (deriv=2, same fit)\n"
                if qdd_locked
                else (
                    f"  qdd deriv:      savgol(win={self._panels[2]._win_sp.get()}, "
                    f"poly={self._panels[2]._poly_sp.get()}, "
                    f"mode={self._panels[2]._mode_var.get()}, deriv=2)\n"
                )
            )
            + f"  tau_m smooth:   {'savgol(win=' + str(tau_win) + ',poly=' + str(tau_poly) + ')' if tau_enabled else 'none'}\n"
            + f"  tau_a RNEA:     {'yes' if use_rnea else 'no'}\n"
            + f"  tau_a post-filt:{'savgol(win=' + str(pf_win) + ',poly=' + str(pf_poly) + ')' if pf_enabled else 'none'}\n\n"
            + f"All 10 CSVs + t.csv saved per split.\nProceed?"
        )
        if not messagebox.askyesno("Confirm Build", msg):
            return

        self._build_btn.config(state=tk.DISABLED, text="Building ...")
        self._log("Build started ...")

        qdd_panel = self._panels[2]
        params = {
            "raw_dir": raw_dir, "run_dir": run_dir,
            "split_mode": split_mode,
            "train_ratio": tr, "val_ratio": vl, "test_ratio": te,
            "trim_front_pct": trim_fp, "trim_back_pct": trim_bp,
            "q_smooth_enabled": q_enabled, "q_window": q_win, "q_polyorder": q_poly,
            "deriv_window": d_win, "deriv_polyorder": d_poly, "deriv_mode": d_mode,
            "qdd_locked": qdd_locked,
            "tau_smooth_enabled": tau_enabled, "tau_window": tau_win, "tau_polyorder": tau_poly,
            "use_rnea": use_rnea,
            "tau_ana_postfilter_enabled": pf_enabled,
            "tau_ana_window": pf_win, "tau_ana_polyorder": pf_poly,
        }
        if not qdd_locked:
            _, qdd_w, qdd_p_ord = qdd_panel.get_sg_params()
            params["qdd_window"] = qdd_w
            params["qdd_polyorder"] = qdd_p_ord
            params["qdd_mode"] = qdd_panel.get_mode()

        # Collect geometry composition config from the Composition panel
        geom_config: dict = {}
        for gt, info in self._geom_rows.items():
            assign = info["assign"].get()
            included = info["include"].get() and assign != "exclude"
            geom_config[gt] = {
                "include":    included,
                "count":      info["count"].get() if included else 0,
                "assignment": assign,
                "avail":      info["avail"],
            }
        params["geom_config"] = geom_config

        # Validate: at least 3 trajectories selected (1 per split minimum)
        selected_total = sum(v["count"] for v in geom_config.values() if v["include"])
        if selected_total < 3:
            messagebox.showerror(
                "Error",
                f"Only {selected_total} trajectories selected. Need at least 3 "
                f"(1 per split). Adjust counts or uncheck Exclude."
            )
            self._build_btn.config(state=tk.NORMAL, text="Build Dataset")
            return

        # Warn if any split destination will be empty given current pins
        pin_counts = {"train only": 0, "val only": 0, "test only": 0, "all splits (stratified)": 0}
        for v in geom_config.values():
            if v["include"] and v["count"] > 0:
                a = v["assignment"]
                if a in pin_counts:
                    pin_counts[a] += v["count"]
        strat_total = pin_counts["all splits (stratified)"]
        if strat_total == 0 and (pin_counts["val only"] == 0 or pin_counts["test only"] == 0):
            messagebox.showwarning(
                "Dataset Warning",
                "Some splits may have zero trajectories.\n"
                "Consider setting at least some types to 'all splits (stratified)'."
            )

        def _worker():
            try:
                result = _do_build(params)
                self.root.after(0, self._build_done, result, None)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.root.after(0, self._build_done, None, exc)

        threading.Thread(target=_worker, daemon=True).start()

    def _build_done(self, result, error):
        self._build_btn.config(state=tk.NORMAL, text="Build Dataset")
        if error:
            self._log(f"ERROR: {error}")
            messagebox.showerror("Build failed", str(error))
            return
        for line in result.get("log_lines", []):
            self._log(line)
        meta_path = result.get("meta_path", "?")
        self._log(f"Done -> {meta_path}")
        self._status(f"Dataset saved -> {meta_path}")
        messagebox.showinfo("Done", f"Dataset built!\n{meta_path}")

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
#  Build logic (runs in background thread)
# ═══════════════════════════════════════════════════════════════════════════════

def _do_build(params: dict) -> dict:
    """Process all trajectory files and save 10 CSVs + t.csv per split."""
    raw_dir   = params["raw_dir"]
    run_dir   = params["run_dir"]
    tr_ratio  = params["train_ratio"]
    vl_ratio  = params["val_ratio"]
    te_ratio  = params["test_ratio"]
    trim_fp   = params["trim_front_pct"]
    trim_bp   = params["trim_back_pct"]

    q_smooth  = params["q_smooth_enabled"]
    q_win     = params["q_window"]
    q_poly    = params["q_polyorder"]
    d_win     = params["deriv_window"]
    d_poly    = params["deriv_polyorder"]
    d_mode    = params["deriv_mode"]
    tau_smooth = params["tau_smooth_enabled"]
    tau_win   = params["tau_window"]
    tau_poly  = params["tau_polyorder"]
    use_rnea  = params["use_rnea"]
    pf_on     = params["tau_ana_postfilter_enabled"]
    pf_win    = params["tau_ana_window"]
    pf_poly   = params["tau_ana_polyorder"]
    qdd_locked = params.get("qdd_locked", True)
    qdd_w = params.get("qdd_window", d_win)
    qdd_p_ord = params.get("qdd_polyorder", d_poly)
    qdd_mode = params.get("qdd_mode", d_mode)

    log_lines = []
    def _log(msg):
        log_lines.append(msg)

    # Collect JSON files
    json_files = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {raw_dir}")
    _log(f"Found {len(json_files)} trajectory files")

    # ── Geometry composition filtering ──────────────────────────────────────
    # When geom_config is provided, select only the requested count per type
    # and respect the include/exclude flags.
    import re as _re

    def _geom_from_stem(stem: str) -> str:
        m = _re.match(r'^([a-z_]+?)_r\d', stem)
        return m.group(1) if m else "unknown"

    geom_config: dict = params.get("geom_config", {})
    if geom_config:
        from collections import defaultdict as _dd
        grps_by_type: dict = _dd(list)
        for jf in json_files:
            grps_by_type[_geom_from_stem(Path(jf).stem)].append(jf)

        rng_sel = np.random.default_rng(42)
        filtered_files: list = []
        _log("Geometry selection:")
        _log(f"  {'Type':<22} {'Avail':>6}  {'Used':>5}  Assignment")
        for gt in sorted(grps_by_type):
            files_for_gt = grps_by_type[gt]
            cfg = geom_config.get(gt, {})
            if not cfg.get("include", True):
                _log(f"  {gt:<22} {len(files_for_gt):>6}  {'--':>5}  EXCLUDED")
                continue
            assign = cfg.get("assignment", "all splits (stratified)")
            if assign == "exclude":
                _log(f"  {gt:<22} {len(files_for_gt):>6}  {'--':>5}  EXCLUDED")
                continue
            n_sel = min(cfg.get("count", len(files_for_gt)), len(files_for_gt))
            n_sel = max(1, n_sel)
            shuffled = list(files_for_gt)
            rng_sel.shuffle(shuffled)
            selected = shuffled[:n_sel]
            filtered_files.extend(selected)
            _log(f"  {gt:<22} {len(files_for_gt):>6}  {n_sel:>5}  {assign}")
        json_files = sorted(filtered_files)
        if not json_files:
            raise RuntimeError("No trajectories remain after geometry filtering. "
                               "Check your composition settings.")
        _log(f"Selected {len(json_files)} files after composition filtering")
    else:
        _log("No geometry composition config — using all files")


    # Build pinocchio model
    pin_model = pin_data = None
    if use_rnea:
        xacro = str(XACRO_PATH) if XACRO_PATH.exists() else None
        try:
            pin_model, pin_data, _ = build_pinocchio_model(xacro)
            _log("Pinocchio model loaded")
        except Exception as e:
            _log(f"Pinocchio error: {e} -- RNEA disabled")
            use_rnea = False

    nj = ACTIVE_JOINTS

    # Process each trajectory
    all_trajectories = []
    for i, jpath in enumerate(json_files):
        fname = os.path.basename(jpath)
        try:
            L, M, _ = load_raw_sample(jpath)
        except Exception as e:
            _log(f"  SKIP {fname}: {e}")
            continue

        # Ticks -> rad, fix timestamps, torque from load
        q_full = ticks_to_radians(L["act_pos"], M["joint_map"], M["ticks_to_rad"], M["dof"])
        ts_report = fix_timestamps(L["t"])
        t = ts_report.t_fixed.astype(np.float32)
        tau_full = torque_from_load(L["load"], L["voltage"], M["joint_map"])

        # Slice to active joints
        raw_q = q_full[:, :nj].astype(np.float32)
        raw_tau_m = tau_full[:, :nj].astype(np.float32)

        # Trim
        n_full = len(t)
        try:
            fn, bn = resolve_front_back_trim(n_full, trim_fp, trim_bp)
        except ValueError:
            fn, bn = 0, 0
        end = n_full - bn if bn > 0 else n_full
        sl = slice(fn, end)
        raw_q     = raw_q[sl]
        raw_tau_m = raw_tau_m[sl]
        t         = t[sl]

        dt = ts_report.median_dt
        if dt <= 0:
            dt = 1.0 / 300.0

        # Raw derivatives (np.gradient)
        raw_qd, raw_qdd = raw_derivatives(raw_q, t)

        # Raw tau_analytical
        if use_rnea and pin_model is not None:
            r_rnea, r_tau_g, r_tau_M, r_tau_C = compute_rnea_decomposition(
                pin_model, pin_data, raw_q, raw_qd, raw_qdd, n_active=nj)
            r_fric = torque_friction(raw_qd[:, :nj])
            raw_tau_decomp = np.concatenate([r_tau_g, r_tau_M, r_tau_C, r_fric], axis=1)  # (N, 20)
        else:
            raw_tau_decomp = np.zeros((raw_q.shape[0], 4 * nj), dtype=np.float64)

        # Filtered q
        filt_q = savgol_smooth(raw_q, q_win, q_poly) if q_smooth else raw_q.copy()

        # Filtered qd, qdd (same rules as GUI: lock uses one SG fit; unlock uses qd params for qd, qdd for qdd)
        filt_qd, filt_qdd = filtered_qd_qdd_from_params(
            filt_q, dt, d_win, d_poly, d_mode, qdd_locked, qdd_w, qdd_p_ord, qdd_mode)

        # Filtered tau_measured
        filt_tau_m = savgol_smooth(raw_tau_m, tau_win, tau_poly) if tau_smooth else raw_tau_m.copy()

        # Filtered tau_analytical (RNEA from filtered kinematics + friction)
        if use_rnea and pin_model is not None:
            f_rnea, f_tau_g, f_tau_M, f_tau_C = compute_rnea_decomposition(
                pin_model, pin_data, filt_q, filt_qd, filt_qdd, n_active=nj)
            f_fric = torque_friction(filt_qd[:, :nj])
            if pf_on:
                f_tau_g = savgol_smooth(f_tau_g, pf_win, pf_poly)
                f_tau_M = savgol_smooth(f_tau_M, pf_win, pf_poly)
                f_tau_C = savgol_smooth(f_tau_C, pf_win, pf_poly)
                f_fric  = savgol_smooth(f_fric,  pf_win, pf_poly)
            filt_tau_decomp = np.concatenate([f_tau_g, f_tau_M, f_tau_C, f_fric], axis=1)  # (N, 20)
            filt_tau_a = f_rnea + torque_friction(filt_qd[:, :nj])  # 5-dim total for reference only
        else:
            filt_tau_decomp = np.zeros((filt_q.shape[0], 4 * nj), dtype=np.float64)
            filt_tau_a = np.zeros_like(filt_q)

        geom = M.get("geometry", {})
        if isinstance(geom, dict):
            geom_type = geom.get("type", "unknown")
            geom_full = {
                "type":          geom.get("type", "unknown"),
                "radius_mm":     geom.get("actual_radius_mm", geom.get("radius_mm", None)),
                "center_m":      geom.get("center_m", None),
                "normal":        geom.get("normal", None),
                "planner":       geom.get("planner", None),
                "num_waypoints": geom.get("num_waypoints", None),
                "instance":      geom.get("instance", None),
            }
        else:
            geom_type = str(geom)
            geom_full = {"type": geom_type}

        tracking = M.get("tracking", {})
        if not isinstance(tracking, dict):
            tracking = {}

        all_trajectories.append({
            "t": t,
            "raw_q": raw_q, "raw_qd": raw_qd, "raw_qdd": raw_qdd,
            "raw_tau_measured": raw_tau_m, "raw_tau_decomposed": raw_tau_decomp,
            "filtered_q": filt_q, "filtered_qd": filt_qd, "filtered_qdd": filt_qdd,
            "filtered_tau_measured": filt_tau_m, "filtered_tau_decomposed": filt_tau_decomp,
            "source_file": M.get("source_file", fname),
            "geometry_type": geom_type,
            "geometry_full": geom_full,
            "tracking":      tracking,
            "ctrl_hz":       M.get("ctrl_hz", None),
            "fb_hz":         M.get("fb_hz", None),
            "duration":      float(M.get("duration", 0)),
            "original_samples": n_full,
            "trim_front": fn, "trim_back": bn,
            "ts_report": ts_report.to_dict(),
        })
        if (i + 1) % 20 == 0:
            _log(f"  Processed {i+1}/{len(json_files)}")

    if not all_trajectories:
        raise RuntimeError("No trajectories could be processed.")
    _log(f"Processed {len(all_trajectories)} trajectories")

    # Split: stratified (default), random, or temporal
    split_mode = str(params.get("split_mode", "stratified")).strip().lower()
    if split_mode == "temporal":
        indices = np.arange(len(all_trajectories), dtype=np.int64)
        n_train = int(len(indices) * tr_ratio)
        n_val   = int(len(indices) * vl_ratio)
        splits = {
            "train": list(indices[:n_train]),
            "val":   list(indices[n_train:n_train + n_val]),
            "test":  list(indices[n_train + n_val:]),
        }
        _log("Split mode: temporal (trajectories in sorted filename order)")
    elif split_mode == "random":
        rng_r = np.random.default_rng(42)
        indices = rng_r.permutation(len(all_trajectories))
        n_train = int(len(indices) * tr_ratio)
        n_val   = int(len(indices) * vl_ratio)
        splits = {
            "train": list(indices[:n_train]),
            "val":   list(indices[n_train:n_train + n_val]),
            "test":  list(indices[n_train + n_val:]),
        }
        _log("Split mode: random (seed=42)")
    else:  # "stratified" (default) or assignment-aware
        # Separate pinned trajectories from the stratified pool
        _geom_cfg = params.get("geom_config", {})
        pinned_train: list = []
        pinned_val:   list = []
        pinned_test:  list = []
        stratified_indices: list = []

        groups: dict = collections.defaultdict(list)
        for idx, traj in enumerate(all_trajectories):
            groups[traj["geometry_type"]].append(idx)

        rng_s = np.random.default_rng(42)
        _log("Split mode: stratified by geometry type (with assignment pins, seed=42)")
        _log(f"  {'Geometry':<26} {'Total':>5}  {'Pin':<10}  {'Train':>5}  {'Val':>5}  {'Test':>5}")

        for gtype in sorted(groups):
            grp = list(groups[gtype])
            rng_s.shuffle(grp)
            n = len(grp)
            assign = _geom_cfg.get(gtype, {}).get("assignment", "all splits (stratified)")

            if assign == "train only":
                pinned_train.extend(grp)
                _log(f"  {gtype:<26} {n:>5}  {'train':10}  {n:>5}  {'0':>5}  {'0':>5}")
            elif assign == "val only":
                pinned_val.extend(grp)
                _log(f"  {gtype:<26} {n:>5}  {'val':10}  {'0':>5}  {n:>5}  {'0':>5}")
            elif assign == "test only":
                pinned_test.extend(grp)
                _log(f"  {gtype:<26} {n:>5}  {'test':10}  {'0':>5}  {'0':>5}  {n:>5}")
            else:  # "all splits (stratified)" — proportional
                stratified_indices.extend(grp)
                # Per-geometry proportional allocation logged below

        # Run proportional allocation on the stratified pool
        train_idx: list = []
        val_idx:   list = []
        test_idx:  list = []

        if stratified_indices:
            strat_groups: dict = collections.defaultdict(list)
            for idx in stratified_indices:
                strat_groups[all_trajectories[idx]["geometry_type"]].append(idx)

            rng_s2 = np.random.default_rng(42)
            for gtype in sorted(strat_groups):
                grp = strat_groups[gtype]
                rng_s2.shuffle(grp)
                n = len(grp)
                n_tr = max(1, round(n * tr_ratio))
                remaining = n - n_tr
                if remaining > 0 and (vl_ratio + te_ratio) > 0:
                    n_vl = round(remaining * vl_ratio / (vl_ratio + te_ratio))
                    n_vl = min(max(n_vl, 0), remaining)
                else:
                    n_vl = 0
                if n >= 2 and (n_vl + (n - n_tr - n_vl)) == 0:
                    n_tr = n - 1
                    n_vl = 1
                n_te = n - n_tr - n_vl
                train_idx.extend(grp[:n_tr])
                val_idx.extend(grp[n_tr:n_tr + n_vl])
                test_idx.extend(grp[n_tr + n_vl:])
                _log(f"  {gtype:<26} {n:>5}  {'stratified':10}  {n_tr:>5}  {n_vl:>5}  {n_te:>5}")

        # Merge pinned + stratified
        final_train = pinned_train + train_idx
        final_val   = pinned_val   + val_idx
        final_test  = pinned_test  + test_idx

        rng_s3 = np.random.default_rng(42)
        rng_s3.shuffle(final_train)
        rng_s3.shuffle(final_val)
        rng_s3.shuffle(final_test)

        if not final_train:
            _log("  WARNING: train split is empty — check assignment pins")
        if not final_val:
            _log("  WARNING: val split is empty — check assignment pins")
        if not final_test:
            _log("  WARNING: test split is empty — check assignment pins")

        splits = {"train": final_train, "val": final_val, "test": final_test}


    header = ",".join(JOINT_NAMES)
    header_decomp = ",".join(
        [f"{j}_g" for j in JOINT_NAMES] +
        [f"{j}_M" for j in JOINT_NAMES] +
        [f"{j}_C" for j in JOINT_NAMES] +
        [f"{j}_f" for j in JOINT_NAMES]
    )
    os.makedirs(run_dir, exist_ok=True)

    split_stats = {}
    all_norm_data = {"q": [], "qd": [], "qdd": [], "tau": [], "tau_a": []}

    for split_name, idx_arr in splits.items():
        split_dir = os.path.join(run_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Concatenate
        arrays = {k: [] for k in [
            "t", "raw_q", "raw_qd", "raw_qdd", "raw_tau_measured", "raw_tau_decomposed",
            "filtered_q", "filtered_qd", "filtered_qdd",
            "filtered_tau_measured", "filtered_tau_decomposed"]}
        traj_entries = []
        offset = 0

        for idx in idx_arr:
            traj = all_trajectories[idx]
            n_s = len(traj["t"])
            for k in arrays:
                arrays[k].append(traj[k])
            traj_entries.append({
                "source_file":       traj["source_file"],
                "geometry_type":     traj["geometry_type"],
                "geometry":          traj["geometry_full"],
                "tracking":          traj["tracking"],
                "ctrl_hz":           traj["ctrl_hz"],
                "fb_hz":             traj["fb_hz"],
                "duration_sec":      traj["duration"],
                "original_samples":  traj["original_samples"],
                "trim_front":        traj["trim_front"],
                "trim_back":         traj["trim_back"],
                "n_samples":         n_s,
                "start_idx":         offset,
                "end_idx_exclusive": offset + n_s,
                "timestamp_repair":  traj["ts_report"],
            })
            offset += n_s

        for k in arrays:
            arrays[k] = np.concatenate(arrays[k], axis=0)

        # Save CSVs with headers
        def _save(fname, arr, hdr=None):
            path = os.path.join(split_dir, fname)
            if arr.ndim == 1:
                np.savetxt(path, arr.reshape(-1, 1), delimiter=",", fmt="%.8f",
                           header="t", comments="")
            else:
                np.savetxt(path, arr, delimiter=",", fmt="%.8f",
                           header=hdr or header, comments="")

        _save(CSV_T, arrays["t"])
        _save(CSV_RAW_Q, arrays["raw_q"])
        _save(CSV_RAW_QD, arrays["raw_qd"])
        _save(CSV_RAW_QDD, arrays["raw_qdd"])
        _save(CSV_RAW_TAU_MEASURED, arrays["raw_tau_measured"])
        _save(CSV_RAW_TAU_DECOMPOSED, arrays["raw_tau_decomposed"], header_decomp)
        _save(CSV_FILTERED_Q, arrays["filtered_q"])
        _save(CSV_FILTERED_QD, arrays["filtered_qd"])
        _save(CSV_FILTERED_QDD, arrays["filtered_qdd"])
        _save(CSV_FILTERED_TAU_MEASURED, arrays["filtered_tau_measured"])
        _save(CSV_FILTERED_TAU_DECOMPOSED, arrays["filtered_tau_decomposed"], header_decomp)

        n_total = arrays["filtered_q"].shape[0]
        split_stats[split_name] = {
            "n_samples": n_total,
            "n_trajectories": len(idx_arr),
            "trajectories": traj_entries,
        }

        if split_name == "train":
            all_norm_data["q"].append(arrays["filtered_q"])
            all_norm_data["qd"].append(arrays["filtered_qd"])
            all_norm_data["qdd"].append(arrays["filtered_qdd"])
            all_norm_data["tau"].append(arrays["filtered_tau_measured"])
            # Compute tau_a norm from RNEA total (sum of all 4 component blocks) — same joint space.
            decomp = arrays["filtered_tau_decomposed"]  # (N, 20)
            nj = ACTIVE_JOINTS
            tau_a_total = (decomp[:, :nj] + decomp[:, nj:2*nj]
                           + decomp[:, 2*nj:3*nj] + decomp[:, 3*nj:])  # (N, 5)
            all_norm_data["tau_a"].append(tau_a_total)

        _log(f"  {split_name.upper()}: {len(idx_arr)} traj, {n_total:,} samples")

        # Write trajectories_catalog.csv for this split
        catalog_fields = [
            "source_file", "geometry_type", "radius_mm", "planner",
            "ctrl_hz", "fb_hz", "duration_sec", "n_samples",
            "ee_rms_err_mm", "start_idx", "end_idx_exclusive",
        ]
        catalog_path = os.path.join(split_dir, "trajectories_catalog.csv")
        with open(catalog_path, "w", newline="") as _cf:
            writer = csv.DictWriter(_cf, fieldnames=catalog_fields, extrasaction="ignore")
            writer.writeheader()
            for te in traj_entries:
                geom_d  = te["geometry"] if isinstance(te["geometry"], dict) else {}
                track_d = te["tracking"] if isinstance(te["tracking"], dict) else {}
                writer.writerow({
                    "source_file":       te["source_file"],
                    "geometry_type":     te["geometry_type"],
                    "radius_mm":         geom_d.get("radius_mm", ""),
                    "planner":           geom_d.get("planner", ""),
                    "ctrl_hz":           te.get("ctrl_hz", ""),
                    "fb_hz":             te.get("fb_hz", ""),
                    "duration_sec":      te.get("duration_sec", ""),
                    "n_samples":         te["n_samples"],
                    "ee_rms_err_mm":     track_d.get("ee_rms_err_mm", ""),
                    "start_idx":         te["start_idx"],
                    "end_idx_exclusive": te["end_idx_exclusive"],
                })

    # Normalisation from training split
    q_all    = np.concatenate(all_norm_data["q"])
    qd_all   = np.concatenate(all_norm_data["qd"])
    qdd_all  = np.concatenate(all_norm_data["qdd"])
    tau_all  = np.concatenate(all_norm_data["tau"])
    tau_a_all = np.concatenate(all_norm_data["tau_a"])

    normalisation = {
        "mean_q":     q_all.mean(axis=0).tolist(),
        "std_q":      q_all.std(axis=0).clip(min=1e-8).tolist(),
        "mean_qd":    qd_all.mean(axis=0).tolist(),
        "std_qd":     qd_all.std(axis=0).clip(min=1e-8).tolist(),
        "mean_qdd":   qdd_all.mean(axis=0).tolist(),
        "std_qdd":    qdd_all.std(axis=0).clip(min=1e-8).tolist(),
        "mean_tau":   tau_all.mean(axis=0).tolist(),
        "std_tau":    tau_all.std(axis=0).clip(min=1e-8).tolist(),
        "mean_tau_a": tau_a_all.mean(axis=0).tolist(),
        "std_tau_a":  tau_a_all.std(axis=0).clip(min=1e-8).tolist(),
    }

    # Geometry distribution across splits
    geom_dist: dict = collections.defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for _sn, _idx_arr in splits.items():
        for _idx in _idx_arr:
            geom_dist[all_trajectories[_idx]["geometry_type"]][_sn] += 1
    geom_dist = {k: dict(v) for k, v in sorted(geom_dist.items())}

    # metadata.json
    meta_doc = {
        "format": "v4_stratified_metadata",
        "created_at": datetime.now().isoformat(),
        "raw_dir": raw_dir,
        "run_dir": run_dir,
        "n_trajectories_total": len(json_files),
        "n_trajectories_processed": len(all_trajectories),

        "preprocessing": {
            "trim": {"front_percent": trim_fp, "back_percent": trim_bp},
            "q_smooth": {
                "enabled": q_smooth,
                "method": "savgol" if q_smooth else "none",
                "window_length": q_win,
                "polyorder": q_poly,
            },
            "differentiation": {
                "method": "savgol",
                "qd": {
                    "window_length": d_win,
                    "polyorder": d_poly,
                    "mode": d_mode,
                    "note": "deriv=1 of Savitzky-Golay on filtered_q",
                },
                "qdd": {
                    "locked_to_qd": qdd_locked,
                    "window_length": d_win if qdd_locked else qdd_w,
                    "polyorder": d_poly if qdd_locked else qdd_p_ord,
                    "mode": d_mode if qdd_locked else qdd_mode,
                    "note": (
                        "deriv=2 from same SG fit as qd when locked; "
                        "separate SG fit for deriv=2 when unlocked"
                    ),
                },
            },
            "tau_measured_smooth": {
                "enabled": tau_smooth,
                "method": "savgol" if tau_smooth else "none",
                "window_length": tau_win,
                "polyorder": tau_poly,
            },
            "tau_analytical": {
                "rnea_enabled": use_rnea,
                "xacro_path": str(XACRO_PATH) if use_rnea else None,
                "friction_model": "coulomb_viscous",
                "source": "RNEA(filtered_q, filtered_qd, filtered_qdd) + friction(filtered_qd)",
                "output_format": "decomposed_20dim",
                "layout": "tau_g(5), tau_M(5), tau_C(5), tau_f(5)",
            },
            "tau_analytical_postfilter": {
                "enabled": pf_on,
                "method": "savgol" if pf_on else "none",
                "window_length": pf_win,
                "polyorder": pf_poly,
            },
        },

        "split": {
            "mode": split_mode,
            "strategy": (
                "stratified_by_geometry_type" if split_mode == "stratified"
                else split_mode
            ),
            "geometry_distribution": geom_dist,
            "geometry_config": geom_config if geom_config else None,
            "ratios": {"train": tr_ratio, "val": vl_ratio, "test": te_ratio},
            "stats": {k: {
                "n_samples":      v["n_samples"],
                "n_trajectories": v["n_trajectories"],
                "trajectories":   v["trajectories"],
            } for k, v in split_stats.items()},
        },

        "normalisation": normalisation,

        "files_per_split": [
            CSV_T,
            CSV_RAW_Q, CSV_RAW_QD, CSV_RAW_QDD,
            CSV_RAW_TAU_MEASURED, CSV_RAW_TAU_DECOMPOSED,
            CSV_FILTERED_Q, CSV_FILTERED_QD, CSV_FILTERED_QDD,
            CSV_FILTERED_TAU_MEASURED, CSV_FILTERED_TAU_DECOMPOSED,
            "trajectories_catalog.csv",
        ],
    }

    meta_path = os.path.join(run_dir, METADATA_FILE)
    with open(meta_path, "w") as f:
        _json.dump(meta_doc, f, indent=2, default=str)

    _log(f"Metadata saved -> {meta_path}")

    return {"meta_path": meta_path, "log_lines": log_lines}


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = PreprocessApp()
    app.run()


if __name__ == "__main__":
    main()
