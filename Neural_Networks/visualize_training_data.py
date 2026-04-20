#!/usr/bin/env python3
"""
visualize_training_data.py -- Interactive Training Data Explorer

Browse pre-built training runs, select splits and individual trajectories,
compare raw vs filtered signals, and inspect the RNEA physics decomposition.

Usage::

    cd /home/san/Desktop/MTP_PINN && python -m Neural_Networks.visualize_training_data
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

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
    print("tkinter is required (install python3-tk)", file=sys.stderr)
    sys.exit(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

from Neural_Networks.loader import (
    ACTIVE_JOINTS,
    JOINT_NAMES,
    JOINT_COLORS,
    CSV_T,
    CSV_RAW_Q, CSV_RAW_QD, CSV_RAW_QDD,
    CSV_RAW_TAU_MEASURED, CSV_RAW_TAU_DECOMPOSED,
    CSV_FILTERED_Q, CSV_FILTERED_QD, CSV_FILTERED_QDD,
    CSV_FILTERED_TAU_MEASURED, CSV_FILTERED_TAU_DECOMPOSED,
    METADATA_FILE,
)

TRAIN_DIR = _HERE / "train_data"

# ── Colour palette ─────────────────────────────────────────────────────────
BG_MAIN  = "#f0f0f0"
BG_LEFT  = "#e8e8e8"
BG_Q     = "#dce8f5"
BG_QD    = "#daf0da"
BG_QDD   = "#f5f0da"
BG_TAU   = "#f0daf5"
BG_ANA   = "#f5e8da"
FG_BTN   = "#ffffff"
BG_BTN   = "#4a90d9"

PANEL_FIG_H = 3.6
_PLOT_SUBPLOT_ADJUST = dict(left=0.07, right=0.995, top=0.96, bottom=0.16)

# Physics decomposition: filtered_tau_decomposed layout is tau_g(5)|tau_M(5)|tau_C(5)|tau_f(5)
_COMP_OPTS  = ["total (sum)", "tau_g  (gravity)", "tau_M  (inertia)",
               "tau_C  (Coriolis)", "tau_f  (friction)"]
_COMP_BLOCK = {
    "tau_g  (gravity)":  0,
    "tau_M  (inertia)":  1,
    "tau_C  (Coriolis)": 2,
    "tau_f  (friction)": 3,
}

# =============================================================================
#  Data loading helpers
# =============================================================================

def _load_csvs(split_dir: str) -> dict:
    """Load all 11 signal CSVs from a split directory into numpy arrays."""
    def _r(fname: str) -> np.ndarray:
        return np.loadtxt(os.path.join(split_dir, fname), delimiter=",", skiprows=1)

    return {
        "t":                       _r(CSV_T).ravel(),
        "raw_q":                   _r(CSV_RAW_Q),
        "raw_qd":                  _r(CSV_RAW_QD),
        "raw_qdd":                 _r(CSV_RAW_QDD),
        "raw_tau_measured":        _r(CSV_RAW_TAU_MEASURED),
        "raw_tau_decomposed":      _r(CSV_RAW_TAU_DECOMPOSED),
        "filtered_q":              _r(CSV_FILTERED_Q),
        "filtered_qd":             _r(CSV_FILTERED_QD),
        "filtered_qdd":            _r(CSV_FILTERED_QDD),
        "filtered_tau_measured":   _r(CSV_FILTERED_TAU_MEASURED),
        "filtered_tau_decomposed": _r(CSV_FILTERED_TAU_DECOMPOSED),
    }


def _load_catalog(split_dir: str) -> list[dict]:
    """Read trajectories_catalog.csv, return list of trajectory dicts."""
    path = os.path.join(split_dir, "trajectories_catalog.csv")
    if not os.path.isfile(path):
        return []
    rows: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "source_file":       row.get("source_file", ""),
                "geometry_type":     row.get("geometry_type", ""),
                "n_samples":         int(row.get("n_samples", 0) or 0),
                "start_idx":         int(row.get("start_idx", 0) or 0),
                "end_idx_exclusive": int(row.get("end_idx_exclusive", 0) or 0),
            })
    return rows


# =============================================================================
#  GUI helpers  (same style as preprocess_data.py)
# =============================================================================

class _Tooltip:
    _DELAY_MS = 400

    def __init__(self, widget, text: str):
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
        tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            bg="#ffffdd", fg="#333", relief=tk.SOLID, bd=1,
            font=("sans-serif", 8), wraplength=400,
        ).pack()

    def _hide(self, _=None):
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tw:
            self._tw.destroy()
            self._tw = None


def _tip(widget, text: str):
    _Tooltip(widget, text)
    return widget


def _vsep(parent, bg: str):
    tk.Frame(parent, bg="#aaaacc", width=1, height=22).pack(side=tk.LEFT, padx=9)


class ScrollFrame(tk.Frame):
    """Vertically scrollable frame with mouse-wheel support."""

    def __init__(self, parent, bg=BG_MAIN, **kw):
        super().__init__(parent, bg=bg, **kw)
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self._scroll = tk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self.inner = tk.Frame(self._canvas, bg=bg)
        self.inner.bind(
            "<Configure>",
            lambda _: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._win_id = self._canvas.create_window((0, 0), window=self.inner, anchor=tk.NW)
        self._canvas.configure(yscrollcommand=self._scroll.set)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self.bind_scroll(self._canvas)
        self.bind_scroll(self.inner)

    def _on_canvas_resize(self, event):
        self._canvas.itemconfigure(self._win_id, width=event.width)

    def bind_scroll(self, widget):
        widget.bind(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(-e.delta // 120, "units"),
        )
        widget.bind("<Button-4>", lambda _: self._canvas.yview_scroll(-3, "units"))
        widget.bind("<Button-5>", lambda _: self._canvas.yview_scroll(3, "units"))


# =============================================================================
#  DataPanel -- one signal panel
# =============================================================================

class DataPanel:
    """One panel: matplotlib figure + controls for one signal or decomposition."""

    def __init__(self, parent, title: str, key: str, panel_type: str, bg: str, app):
        self.key = key
        self.panel_type = panel_type   # "signal" | "decomposition"
        self.app = app
        self._bg = bg
        self._data: dict | None = None  # thinned/sliced data dict

        frame = tk.LabelFrame(
            parent, text=f"  {title}  ", bg=bg,
            font=("sans-serif", 10, "bold"), padx=4, pady=2,
        )
        frame.pack(fill=tk.X, padx=4, pady=2)
        self._frame = frame

        ctrl = tk.Frame(frame, bg=bg)
        ctrl.pack(fill=tk.X, pady=(0, 1))

        if panel_type == "signal":
            self._build_signal_ctrl(ctrl, bg)
        elif panel_type == "decomposition":
            self._build_decomp_ctrl(ctrl, bg)

        # Matplotlib figure
        self._fig = Figure(figsize=(10, PANEL_FIG_H), dpi=90, facecolor=bg)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("white")

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=frame)
        canvas_tk = self._canvas_widget.get_tk_widget()
        canvas_tk.configure(bd=0, highlightthickness=0, relief=tk.FLAT, bg=bg)
        canvas_tk.pack(fill=tk.BOTH, expand=True)
        self._resize_after_id = None
        canvas_tk.bind("<Configure>", self._on_canvas_configure, add="+")

        tb_frame = tk.Frame(frame, bg=bg)
        tb_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self._canvas_widget, tb_frame)
        toolbar.configure(bg=bg)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        self._info_var = tk.StringVar(value="")
        tk.Label(
            frame, textvariable=self._info_var, bg=bg,
            font=("sans-serif", 7), fg="#555",
        ).pack(anchor=tk.W)

    # ── resize ─────────────────────────────────────────────────────────────

    def _on_canvas_configure(self, event):
        if self._resize_after_id is not None:
            self._frame.after_cancel(self._resize_after_id)
        self._resize_after_id = self._frame.after(
            80,
            lambda w=event.width, h=event.height: self._apply_resize(w, h),
        )

    def _apply_plot_margins(self):
        self._fig.subplots_adjust(**_PLOT_SUBPLOT_ADJUST)

    def _apply_resize(self, w_px, h_px):
        self._resize_after_id = None
        dpi = self._fig.get_dpi()
        self._fig.set_size_inches(max(w_px / dpi, 1.0), max(h_px / dpi, 0.5), forward=True)
        self._apply_plot_margins()
        self._canvas_widget.draw_idle()

    # ── control builders ───────────────────────────────────────────────────

    def _build_signal_ctrl(self, ctrl, bg):
        self._show_filtered_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            ctrl, text="Filtered", variable=self._show_filtered_var,
            bg=bg, command=self._redraw,
        ).pack(side=tk.LEFT)

        self._show_raw_var = tk.BooleanVar(value=False)
        cb = tk.Checkbutton(
            ctrl, text="Raw overlay", variable=self._show_raw_var,
            bg=bg, command=self._redraw,
        )
        cb.pack(side=tk.LEFT, padx=(8, 0))
        _tip(cb, "Overlay the unfiltered / gradient-based raw signal as a dashed line.")

    def _build_decomp_ctrl(self, ctrl, bg):
        tk.Label(ctrl, text="Component:", bg=bg,
                 font=("sans-serif", 8)).pack(side=tk.LEFT)
        self._comp_var = tk.StringVar(value="total (sum)")
        comp_cb = ttk.Combobox(
            ctrl, textvariable=self._comp_var, values=_COMP_OPTS,
            state="readonly", width=24, font=("sans-serif", 8),
        )
        comp_cb.pack(side=tk.LEFT, padx=4)
        comp_cb.bind("<<ComboboxSelected>>", lambda _: self._redraw())

        _vsep(ctrl, bg)
        self._show_measured_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            ctrl, text="Show tau_measured", variable=self._show_measured_var,
            bg=bg, command=self._redraw,
        ).pack(side=tk.LEFT)

        _vsep(ctrl, bg)
        self._use_raw_decomp_var = tk.BooleanVar(value=False)
        cb_raw = tk.Checkbutton(
            ctrl, text="Use raw decomp", variable=self._use_raw_decomp_var,
            bg=bg, command=self._redraw,
        )
        cb_raw.pack(side=tk.LEFT)
        _tip(cb_raw, "Switch from filtered_tau_decomposed to raw_tau_decomposed (gradient-based kinematics).")

    # ── public API ─────────────────────────────────────────────────────────

    def load(self, data: dict):
        """Receive a pre-sliced/thinned data dict and redraw."""
        self._data = data
        self._redraw()

    def clear(self):
        self._data = None
        self._ax.clear()
        self._ax.set_facecolor("white")
        self._canvas_widget.draw_idle()

    # ── drawing ────────────────────────────────────────────────────────────

    def _redraw(self, *_):
        if self._data is None:
            return
        self._ax.clear()
        self._ax.set_facecolor("white")

        active = self.app.active_joints or list(range(ACTIVE_JOINTS))
        t = self._data["t"]

        if self.panel_type == "signal":
            self._draw_signal(t, active)
        elif self.panel_type == "decomposition":
            self._draw_decomposition(t, active)

        self._ax.set_xlabel("Time [s]", fontsize=8)
        self._ax.legend(fontsize=7, ncol=min(len(active) + 2, 8), loc="upper right")
        self._ax.tick_params(labelsize=7)
        self._ax.grid(True, alpha=0.2, linewidth=0.5)
        self._apply_plot_margins()
        self._canvas_widget.draw_idle()

    def _draw_signal(self, t, active):
        key_filt = f"filtered_{self.key}"
        key_raw  = f"raw_{self.key}"
        show_filt = self._show_filtered_var.get()
        show_raw  = self._show_raw_var.get()

        unit_map = {"q": "rad", "qd": "rad/s", "qdd": "rad/s\u00b2", "tau_measured": "Nm"}
        self._ax.set_ylabel(f"{self.key} [{unit_map.get(self.key, '')}]", fontsize=8)

        for j in active:
            c = JOINT_COLORS[j]
            lbl = JOINT_NAMES[j]
            if show_filt:
                self._ax.plot(t, self._data[key_filt][:, j],
                              color=c, linewidth=1.0, label=lbl)
            if show_raw:
                self._ax.plot(t, self._data[key_raw][:, j],
                              color=c, linewidth=0.5, alpha=0.45, linestyle="--",
                              label=(lbl + " raw") if not show_filt else None)

        parts = (["filtered"] if show_filt else []) + (["raw (dashed)"] if show_raw else [])
        self._info_var.set("  ".join(parts) + f"  |  N={len(t):,}")

    def _draw_decomposition(self, t, active):
        nj = ACTIVE_JOINTS
        use_raw = self._use_raw_decomp_var.get()
        decomp_key = "raw_tau_decomposed" if use_raw else "filtered_tau_decomposed"
        decomp = self._data[decomp_key]   # (N, 20)

        self._ax.set_ylabel("torque [Nm]", fontsize=8)
        comp = self._comp_var.get()

        if comp == "total (sum)":
            total = (
                decomp[:, :nj] + decomp[:, nj:2*nj]
                + decomp[:, 2*nj:3*nj] + decomp[:, 3*nj:]
            )
            for j in active:
                self._ax.plot(t, total[:, j], color=JOINT_COLORS[j],
                              linewidth=1.0, label=f"{JOINT_NAMES[j]} RNEA")
        else:
            block = _COMP_BLOCK[comp]
            cols = decomp[:, block*nj:(block+1)*nj]
            for j in active:
                self._ax.plot(t, cols[:, j], color=JOINT_COLORS[j],
                              linewidth=1.0, label=JOINT_NAMES[j])

        if self._show_measured_var.get():
            tau_m_key = "raw_tau_measured" if use_raw else "filtered_tau_measured"
            tau_m = self._data[tau_m_key]
            for j in active:
                self._ax.plot(t, tau_m[:, j], color=JOINT_COLORS[j],
                              linewidth=0.5, alpha=0.5, linestyle="--",
                              label=f"{JOINT_NAMES[j]} measured")

        src = "raw" if use_raw else "filtered"
        self._info_var.set(f"{comp}  |  source: {src}  |  N={len(t):,}")


# =============================================================================
#  Main Application
# =============================================================================

class VisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Training Data Explorer")
        self.root.geometry("1600x950")
        self.root.minsize(1000, 600)
        self.root.configure(bg=BG_MAIN)

        self._run_dir: str | None = None
        self._full_data: dict | None = None   # all loaded CSV arrays for current split
        self._catalog: list[dict] = []
        self._current_sl: slice = slice(None)

        # ── Top bar ───────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg="#2e3040")
        top.pack(fill=tk.X)

        self._joint_vars: list[tk.BooleanVar] = []
        for j in range(ACTIVE_JOINTS):
            v = tk.BooleanVar(value=True)
            tk.Checkbutton(
                top, text=JOINT_NAMES[j], variable=v, bg="#2e3040",
                fg="white", selectcolor="#2e3040", command=self._redraw_all,
            ).pack(side=tk.LEFT, padx=4)
            self._joint_vars.append(v)

        _vsep(top, "#2e3040")
        self._status_var = tk.StringVar(value="Select a run from the left panel")
        tk.Label(
            top, textvariable=self._status_var, bg="#2e3040", fg="#aaa",
            font=("sans-serif", 8),
        ).pack(side=tk.LEFT, padx=10)

        # Max-samples spinbox
        _vsep(top, "#2e3040")
        tk.Label(top, text="Max plot pts:", bg="#2e3040", fg="white",
                 font=("sans-serif", 8)).pack(side=tk.LEFT, padx=(4, 2))
        self._max_samples_var = tk.IntVar(value=10000)
        max_sp = tk.Spinbox(
            top, from_=100, to=500000, increment=5000, width=7,
            textvariable=self._max_samples_var, font=("sans-serif", 8),
            command=self._apply_selection,
        )
        max_sp.pack(side=tk.LEFT)
        _tip(max_sp,
             "Limit plotted points for performance.\n"
             "Data is always loaded in full; this only thins the drawn lines.")

        self._sel_info_var = tk.StringVar(value="")
        tk.Label(
            top, textvariable=self._sel_info_var, bg="#2e3040", fg="#ffa",
            font=("sans-serif", 7),
        ).pack(side=tk.RIGHT, padx=10)

        # ── Paned layout ──────────────────────────────────────────────────
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=BG_MAIN, sashwidth=6)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = tk.Frame(paned, bg=BG_LEFT, width=300)
        paned.add(left, minsize=220)
        self._build_left_panel(left)

        right = tk.Frame(paned, bg=BG_MAIN)
        paned.add(right, minsize=600)
        scroll = ScrollFrame(right, bg=BG_MAIN)
        scroll.pack(fill=tk.BOTH, expand=True)

        self._panels: list[DataPanel] = [
            DataPanel(scroll.inner, "q  (joint positions)",           "q",             "signal",        BG_Q,   self),
            DataPanel(scroll.inner, "qd (joint velocities)",          "qd",            "signal",        BG_QD,  self),
            DataPanel(scroll.inner, "qdd (joint accelerations)",      "qdd",           "signal",        BG_QDD, self),
            DataPanel(scroll.inner, "tau_measured (torque sensor)",   "tau_measured",  "signal",        BG_TAU, self),
            DataPanel(scroll.inner, "tau_analytical (RNEA+friction)", "tau_analytical","decomposition", BG_ANA, self),
        ]
        for p in self._panels:
            p._frame.bind(
                "<MouseWheel>",
                lambda e: scroll._canvas.yview_scroll(-e.delta // 120, "units"),
            )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def active_joints(self) -> list[int]:
        return [j for j, v in enumerate(self._joint_vars) if v.get()]

    # ── Left panel ─────────────────────────────────────────────────────────

    def _build_left_panel(self, parent):
        tk.Label(parent, text="Training Data Explorer", bg=BG_LEFT,
                 font=("sans-serif", 10, "bold")).pack(anchor=tk.W, padx=6, pady=(6, 2))

        # Base directory row
        dir_row = tk.Frame(parent, bg=BG_LEFT)
        dir_row.pack(fill=tk.X, padx=6)
        self._base_dir_var = tk.StringVar(value=str(TRAIN_DIR))
        tk.Entry(dir_row, textvariable=self._base_dir_var,
                 font=("sans-serif", 8)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(dir_row, text="...", command=self._browse_base_dir,
                  width=3).pack(side=tk.LEFT, padx=2)
        tk.Button(dir_row, text="Scan", command=self._scan_runs,
                  font=("sans-serif", 8)).pack(side=tk.LEFT)

        # Run list
        tk.Label(parent, text="Runs:", bg=BG_LEFT,
                 font=("sans-serif", 8, "bold")).pack(anchor=tk.W, padx=6, pady=(6, 0))
        run_frame = tk.Frame(parent, bg=BG_LEFT)
        run_frame.pack(fill=tk.X, padx=6)
        run_scroll = tk.Scrollbar(run_frame, orient=tk.VERTICAL)
        self._run_list = tk.Listbox(
            run_frame, font=("monospace", 8), selectmode=tk.SINGLE,
            height=5, yscrollcommand=run_scroll.set,
        )
        run_scroll.config(command=self._run_list.yview)
        run_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._run_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._run_list.bind("<<ListboxSelect>>", self._on_run_select)

        # Split radio buttons
        split_row = tk.Frame(parent, bg=BG_LEFT)
        split_row.pack(fill=tk.X, padx=6, pady=(6, 2))
        tk.Label(split_row, text="Split:", bg=BG_LEFT,
                 font=("sans-serif", 9)).pack(side=tk.LEFT)
        self._split_var = tk.StringVar(value="train")
        for s in ("train", "val", "test"):
            tk.Radiobutton(
                split_row, text=s, variable=self._split_var, value=s,
                bg=BG_LEFT, command=self._on_split_change,
            ).pack(side=tk.LEFT, padx=4)

        # Load split button
        self._load_btn = tk.Button(
            parent, text="Load Split", bg=BG_BTN, fg=FG_BTN,
            font=("sans-serif", 9, "bold"), command=self._load_split,
        )
        self._load_btn.pack(fill=tk.X, padx=6, pady=2)

        # Trajectory list
        tk.Label(parent, text="Trajectories:", bg=BG_LEFT,
                 font=("sans-serif", 8, "bold")).pack(anchor=tk.W, padx=6, pady=(4, 0))
        traj_frame = tk.Frame(parent, bg=BG_LEFT)
        traj_frame.pack(fill=tk.BOTH, expand=True, padx=6)
        traj_scroll = tk.Scrollbar(traj_frame, orient=tk.VERTICAL)
        self._traj_list = tk.Listbox(
            traj_frame, font=("monospace", 8), selectmode=tk.SINGLE,
            yscrollcommand=traj_scroll.set,
        )
        traj_scroll.config(command=self._traj_list.yview)
        traj_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._traj_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._traj_list.bind("<<ListboxSelect>>", self._on_traj_select)

        # Metadata summary
        self._meta_text = scrolledtext.ScrolledText(
            parent, height=9, font=("monospace", 7), state=tk.DISABLED, bg="#f8f8f8",
        )
        self._meta_text.pack(fill=tk.X, padx=6, pady=(4, 4))

        self.root.after(100, self._scan_runs)

    def _browse_base_dir(self):
        d = filedialog.askdirectory(initialdir=self._base_dir_var.get())
        if d:
            self._base_dir_var.set(d)
            self._scan_runs()

    def _scan_runs(self):
        """Find all subdirectories containing a metadata.json."""
        base = self._base_dir_var.get()
        self._run_list.delete(0, tk.END)
        if not os.path.isdir(base):
            return
        runs = sorted(
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
            and os.path.isfile(os.path.join(base, d, METADATA_FILE))
        )
        for r in runs:
            self._run_list.insert(tk.END, r)
        msg = (f"{len(runs)} run(s) found \u2014 select one." if runs
               else "No runs found \u2014 build a dataset first.")
        self._status(msg)

    def _on_run_select(self, _=None):
        sel = self._run_list.curselection()
        if not sel:
            return
        run_name = self._run_list.get(sel[0])
        self._run_dir = os.path.join(self._base_dir_var.get(), run_name)
        self._full_data = None
        self._catalog = []
        self._traj_list.delete(0, tk.END)
        self._status(f"{run_name}  \u2192  press Load Split")

        meta_path = os.path.join(self._run_dir, METADATA_FILE)
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    self._show_meta(json.load(f))
            except Exception:
                pass

    def _show_meta(self, meta: dict):
        lines: list[str] = []
        lines.append(f"Created:  {meta.get('created_at', '?')[:19]}")
        n_proc = meta.get("n_trajectories_processed", "?")
        lines.append(f"Processed: {n_proc} trajectories")
        stats = meta.get("split", {}).get("stats", {})
        for sn in ("train", "val", "test"):
            if sn in stats:
                s = stats[sn]
                n_s = s.get("n_samples", "?")
                n_t = s.get("n_trajectories", "?")
                lines.append(f"  {sn}: {n_t} traj, {n_s:,} samples"
                             if isinstance(n_s, int) else f"  {sn}: {n_t} traj, {n_s} samples")
        pre = meta.get("preprocessing", {})
        q = pre.get("q_smooth", {})
        lines.append(f"q_smooth: {'savgol win='+str(q.get('window_length')) if q.get('enabled') else 'none'}")
        diff = pre.get("differentiation", {})
        qd  = diff.get("qd", {})
        qdd = diff.get("qdd", {})
        lines.append(f"qd:  savgol win={qd.get('window_length')} poly={qd.get('polyorder')} mode={qd.get('mode')}")
        lines.append(f"qdd: {'locked to qd' if qdd.get('locked_to_qd') else 'separate SG win='+str(qdd.get('window_length'))}")
        tau_a = pre.get("tau_analytical", {})
        lines.append(f"RNEA: {'enabled' if tau_a.get('rnea_enabled') else 'disabled'}")
        pf = pre.get("tau_analytical_postfilter", {})
        if pf.get("enabled"):
            lines.append(f"RNEA post-filter: savgol win={pf.get('window_length')}")
        trim = pre.get("trim", {})
        lines.append(f"Trim: {trim.get('front_percent', 0)}% front, {trim.get('back_percent', 0)}% back")

        self._meta_text.config(state=tk.NORMAL)
        self._meta_text.delete("1.0", tk.END)
        self._meta_text.insert(tk.END, "\n".join(lines))
        self._meta_text.config(state=tk.DISABLED)

    def _on_split_change(self):
        self._full_data = None
        self._traj_list.delete(0, tk.END)
        self._catalog = []
        for p in self._panels:
            p.clear()
        if self._run_dir:
            self._status(f"Split \u2192 {self._split_var.get()}  (press Load Split)")

    def _load_split(self):
        if not self._run_dir:
            messagebox.showwarning("No run selected", "Select a run from the list first.")
            return
        split = self._split_var.get()
        split_dir = os.path.join(self._run_dir, split)
        if not os.path.isdir(split_dir):
            messagebox.showerror("Not found", f"Split directory not found:\n{split_dir}")
            return

        self._status(f"Loading {split} CSVs \u2026")
        self.root.update_idletasks()

        try:
            self._full_data = _load_csvs(split_dir)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self._status("Load failed.")
            return

        self._catalog = _load_catalog(split_dir)
        n_total = len(self._full_data["t"])

        # Populate trajectory listbox
        self._traj_list.delete(0, tk.END)
        self._traj_list.insert(tk.END,
            f"[ All {len(self._catalog)} trajectories  N={n_total:,} ]")
        for i, traj in enumerate(self._catalog):
            self._traj_list.insert(tk.END,
                f"{i:3d}  {traj['geometry_type']:<12}  N={traj['n_samples']:>7,}")
        self._traj_list.selection_set(0)

        self._status(f"Loaded {split}: {len(self._catalog)} traj, {n_total:,} samples")
        self._current_sl = slice(None)
        self._apply_selection()

    def _on_traj_select(self, _=None):
        sel = self._traj_list.curselection()
        if not sel or self._full_data is None:
            return
        idx = sel[0]
        if idx == 0:
            self._current_sl = slice(None)
            self._sel_info_var.set("All trajectories")
        else:
            traj = self._catalog[idx - 1]
            self._current_sl = slice(traj["start_idx"], traj["end_idx_exclusive"])
            self._sel_info_var.set(
                f"Traj {idx-1}: {traj['geometry_type']}  N={traj['n_samples']:,}"
            )
        self._apply_selection()

    def _apply_selection(self):
        """Thin the current slice to max_plot_samples and push to all panels."""
        if self._full_data is None:
            return
        sl = self._current_sl
        n = len(self._full_data["t"][sl])
        max_s = max(1, self._max_samples_var.get())
        step = max(1, n // max_s)

        plot_data = {k: v[sl][::step] for k, v in self._full_data.items()}
        n_plot = len(plot_data["t"])

        old = self._sel_info_var.get().split("  |")[0]
        self._sel_info_var.set(f"{old}  |  {n_plot:,} / {n:,} pts plotted")

        for panel in self._panels:
            panel.load(plot_data)

    def _redraw_all(self):
        for p in self._panels:
            p._redraw()

    def _status(self, msg: str):
        self._status_var.set(msg)

    def run(self):
        self.root.mainloop()


# =============================================================================
#  Entry point
# =============================================================================

def main():
    app = VisApp()
    app.run()


if __name__ == "__main__":
    main()
