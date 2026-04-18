"""
Enhanced GUI Visualizer and Model Comparison Tool.
visualizer.py

Usage
-----
    cd /home/san/Desktop/MTP_PINN
    python -m Neural_Networks.visualizer

Features
--------
• Registry Browser — load any trained model directly from models_registry.yaml
• Load N models dynamically; compare predictions vs ground truth
• Per-model line style / colour / width customisation
• Info panel: per-joint RMSE bars, training time, epochs for selected model
• Comparison window tabs: Metrics Table, RMSE Bars, NRMSE per Joint,
  Training History (loss curves from CSV), Hyperparameters
• Denormalized predictions — all metrics in physical N·m
• Sample range zoom + joint filter
• Export current plot to PNG
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import os
import re
import sys
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import torch
from scipy.signal import savgol_filter as _savgol

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Neural_Networks.models import (
    MODEL_REGISTRY, LAGRANGIAN_MODELS, PHYSICS_INPUT_MODELS, DECOMPOSED_MODELS,
    EQUATION_CONSTRAINED_MODELS,
)
from Neural_Networks.data.loader import RobotDataset, _load_run_metadata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOINT_NAMES   = ["J1 (yaw)", "J2 (shoulder)", "J3 (elbow)", "J4 (wrist)", "J5 (wrist roll)"]
LINE_STYLES   = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
MARKER_OPTS   = ["none", ".", "o", "^", "s", "x"]
DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

_NN_DIR          = os.path.dirname(os.path.abspath(__file__))
REGISTRY_FILE    = os.path.join(_NN_DIR, "Trained_Models", "models_registry.yaml")
_TRAIN_DATA_DIR  = os.path.join(_NN_DIR, "train_data")

# Human-readable short names for display
_SHORT_NAMES = {
    "BlackBoxFNN":                "Black Box FNN (1)",
    "PhysicsRegularizedFNN":      "Physics-Regularized FNN (2)",
    "ResidualCorrectionFNN":      "Residual Correction FNN (3)",
    "LagrangianStructuredFNN":    "Lagrangian Structured FNN (4)",
    "EquationConstrainedPINNFNN": "Eq-Constrained PINN FNN (5)",
    "DecomposedStructuredPINNFNN": "Decomposed PINN FNN (6)",
}


# =============================================================================
# REGISTRY UTILITIES
# =============================================================================

def load_registry() -> list[dict]:
    """Read models_registry.yaml and return list of model entries."""
    if not _HAS_YAML:
        print("[Visualizer] WARNING: PyYAML not installed — registry browser unavailable.")
        return []
    if not os.path.exists(REGISTRY_FILE):
        print(f"[Visualizer] WARNING: Registry not found at {REGISTRY_FILE}")
        return []
    try:
        with open(REGISTRY_FILE, "r") as f:
            data = yaml.safe_load(f) or {}
        entries = data.get("models", [])
        def _reg_rmse(e: dict) -> float:
            m = e.get("metrics") or {}
            rp = m.get("test_rmse_pooled")
            if isinstance(rp, (int, float)):
                return float(rp)
            rm = m.get("test_rmse_mean")
            return float(rm) if isinstance(rm, (int, float)) else 9999.0

        entries.sort(key=_reg_rmse)
        print(f"[Visualizer] Registry loaded: {len(entries)} entries  ({REGISTRY_FILE})")
        for i, e in enumerate(entries, 1):
            _r = _reg_rmse(e)
            rmse = _r if _r < 9998.0 else None
            rmse_str = f"{rmse:.5f}" if isinstance(rmse, float) else "?"
            print(f"[Visualizer]   {i:2d}. {e.get('model_type','?'):<35}  RMSE_p={rmse_str}")
        return entries
    except Exception as ex:
        print(f"[Visualizer] ERROR reading registry: {ex}")
        return []


def registry_display_name(entry: dict) -> str:
    """Short display name: '<ShortType>  RMSE=X.XXXXX'."""
    cls   = entry.get("model_type", "Unknown")
    short = _SHORT_NAMES.get(cls, cls)
    m = entry.get("metrics") or {}
    rmse = m.get("test_rmse_pooled", m.get("test_rmse_mean"))
    if rmse is not None:
        return f"{short}  (RMSE_p={float(rmse):.5f})"
    return short


# =============================================================================
# MODEL LOADING / INFERENCE
# =============================================================================

def _detect_stale(class_name: str, hparams: dict, state: dict) -> str | None:
    """
    Return a human-readable reason string if the checkpoint state_dict is
    incompatible with the current model class definition, else None.

    Checks every known architecture change between checkpoint and current code.
    """
    n = hparams.get("n_joints", 5)

    if class_name == "ResidualCorrectionFNN":
        # First layer weight key
        key = next((k for k in state if k.endswith(".0.weight") or k == "net.0.weight"), None)
        if key and state[key].shape[1] != n * 4:
            old_in = state[key].shape[1]
            return (
                f"ResidualCorrectionFNN input_size changed: "
                f"checkpoint={old_in}, current={n * 4}."
            )

    if class_name == "EquationConstrainedPINNFNN":
        # TauEquationCalibration submodule was added after initial training runs.
        # Checkpoints missing tau_calib.* keys are incompatible with current code.
        if not any(k.startswith("tau_calib.") for k in state):
            return (
                "EquationConstrainedPINNFNN is missing the TauEquationCalibration "
                "submodule (tau_calib.*). The architecture was updated after this "
                "checkpoint was saved. Please retrain EquationConstrainedPINNFNN."
            )

    return None  # looks compatible


def _filter_hparams(cls: type, hparams: dict) -> dict:
    """
    Return only the hparam keys accepted by cls.__init__.
    Drops stale keys saved by older architectures so ModelClass(**hparams) never
    raises an unexpected-keyword-argument error.
    If the constructor uses **kwargs, the full dict is returned unchanged.
    """
    sig = inspect.signature(cls.__init__)
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return hparams  # **kwargs — pass everything through
    valid   = set(sig.parameters.keys()) - {"self"}
    dropped = [k for k in hparams if k not in valid]
    if dropped:
        print(f"[Visualizer]   INFO: dropping stale hparam key(s) for "
              f"{cls.__name__}: {dropped}")
    return {k: v for k, v in hparams.items() if k in valid}


def load_model_from_dir(model_dir: str) -> tuple[torch.nn.Module | None, dict]:
    """
    Load a trained model from its save directory.
    Returns (model, metadata) or (None, {error}) on failure.

    Stale checkpoints (architecture changed since training):
      - Detected by comparing state_dict shapes against current model definition.
      - Marked with meta["stale_reason"] so the UI can warn the user.
      - Returned as (model=None) so inference is cleanly blocked rather than
        silently producing garbage predictions.
    """
    model_path = os.path.join(model_dir, "model.pt")
    meta_path  = os.path.join(model_dir, "metadata.json")

    # Re-anchor cross-machine paths (e.g. checkpoints saved on /home/sinayan_iitp/...)
    if not os.path.isdir(model_dir):
        tail = os.path.join(*model_dir.replace("\\", "/").rstrip("/").split("/")[-2:])
        candidate = os.path.join(_NN_DIR, "Trained_Models", tail)
        if os.path.isdir(candidate):
            print(f"[Visualizer]   Re-anchored path: {model_dir}")
            print(f"[Visualizer]                 -> {candidate}")
            model_dir  = candidate
            model_path = os.path.join(model_dir, "model.pt")
            meta_path  = os.path.join(model_dir, "metadata.json")

    print(f"[Visualizer] Loading checkpoint: {model_dir}")

    if not os.path.exists(model_path):
        err = f"model.pt not found in {model_dir}"
        print(f"[Visualizer]   ERROR: {err}")
        return None, {"error": err}

    try:
        ckpt       = torch.load(model_path, map_location="cpu", weights_only=False)
        class_name = ckpt.get("model_class", "")
        hparams    = ckpt.get("hparams", {})
        metrics    = ckpt.get("metrics", {})

        print(f"[Visualizer]   class={class_name}  "
              f"RMSE={metrics.get('rmse_mean', '?'):.5f}" if isinstance(metrics.get('rmse_mean'), float)
              else f"[Visualizer]   class={class_name}")

        if class_name not in MODEL_REGISTRY:
            err = (f"Unknown model class '{class_name}'. "
                   f"Known: {list(MODEL_REGISTRY.keys())}")
            print(f"[Visualizer]   ERROR: {err}")
            return None, {"error": err}

        # torch.compile() prefixes all parameter names with "_orig_mod."
        state = ckpt["model_state"]
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

        # --- Detect stale checkpoint BEFORE constructing the model ---
        stale_reason = _detect_stale(class_name, hparams, state)
        if stale_reason:
            print(f"[Visualizer]   STALE CHECKPOINT: {stale_reason}")
            print(f"[Visualizer]   Blocked — retrain {class_name} to get a fresh checkpoint.")
            meta = {
                "model_class":  class_name,
                "hparams":      hparams,
                "metrics":      metrics,
                "model_dir":    model_dir,
                "norm_stats":   ckpt.get("norm_stats", {}),
                "stale_reason": stale_reason,
            }
            _yaml_meta_path = os.path.join(model_dir, "metadata.yaml")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    saved = json.load(f)
                saved.update(meta)   # meta values win
                meta = saved
            elif _HAS_YAML and os.path.exists(_yaml_meta_path):
                with open(_yaml_meta_path, "r") as f:
                    saved = yaml.safe_load(f) or {}
                saved.update(meta)   # meta values win
                meta = saved
            # Return a stub model so the entry appears in the list (but inference blocked)
            ModelClass = MODEL_REGISTRY[class_name]
            stub_model = ModelClass(**_filter_hparams(ModelClass, hparams))
            stub_model.eval()
            return stub_model, meta

        # --- Build and load normally ---
        ModelClass = MODEL_REGISTRY[class_name]
        model      = ModelClass(**_filter_hparams(ModelClass, hparams))
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"[Visualizer]   Loaded OK — {sum(p.numel() for p in model.parameters()):,} params")

        norm_stats = ckpt.get("norm_stats", {})
        meta = {"model_class": class_name, "hparams": hparams,
                "metrics": metrics, "model_dir": model_dir,
                "norm_stats": norm_stats}
        _yaml_meta_path = os.path.join(model_dir, "metadata.yaml")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                saved = json.load(f)
            meta.update(saved)
            if not meta.get("norm_stats"):
                meta["norm_stats"] = norm_stats
        elif _HAS_YAML and os.path.exists(_yaml_meta_path):
            with open(_yaml_meta_path, "r") as f:
                saved = yaml.safe_load(f) or {}
            meta.update(saved)
            if not meta.get("norm_stats"):
                meta["norm_stats"] = norm_stats

        return model, meta

    except Exception as e:
        print(f"[Visualizer]   EXCEPTION during load: {e}")
        return None, {"error": str(e)}


def run_inference(model: torch.nn.Module, data_path: str,
                  seq_len: int = 50,
                  norm_stats: dict | None = None,
                  sample_slice: tuple[int, int] | None = None) -> dict:
    """
    Run pointwise FNN inference on a processed split directory.

    Parameters
    ----------
    model        : trained torch model in eval mode
    data_path    : run_dir  OR  run_dir/split  (split auto-detected from basename)
    seq_len      : legacy parameter — kept for call-site compatibility, not used
    norm_stats   : tau normalisation stats from model.pt (mean_tau, std_tau)
    sample_slice : (start_idx, end_idx) into the flat split CSV.
                   None = use all samples in the split.

    Returns
    -------
    dict with keys:
        pred   (N, 5) float32 — denormalised predictions  [N·m]
        target (N, 5) float32 — denormalised ground truth [N·m]
        t      (N,)   float64 | None  — elapsed timestamps from t.csv  [s]
    """
    from torch.utils.data import DataLoader

    cls_name = model.__class__.__name__

    # Resolve run_dir and split name from data_path
    _dp = os.path.abspath(data_path)
    _split_name = os.path.basename(_dp)
    _run_dir = os.path.dirname(_dp)
    if _split_name not in ("train", "val", "test"):
        _run_dir = _dp
        _split_name = "test"

    ds = RobotDataset(_run_dir, split=_split_name, mode="pointwise")

    # Apply trajectory slice: mutate the underlying numpy arrays before DataLoader
    _slice_obj: slice | None = None
    if sample_slice is not None:
        n_total   = ds.q.shape[0]
        _s_start  = max(0, int(sample_slice[0]))
        _s_end    = min(n_total, int(sample_slice[1]))
        _slice_obj = slice(_s_start, _s_end)
        ds.q              = ds.q[_slice_obj]
        ds.qd             = ds.qd[_slice_obj]
        ds.qdd            = ds.qdd[_slice_obj]
        ds.tau_measured   = ds.tau_measured[_slice_obj]
        ds.tau_analytical = ds.tau_analytical[_slice_obj]

    # Load timestamps for time-axis plotting (gracefully absent if t.csv missing)
    t_arr: np.ndarray | None = None
    t_csv = os.path.join(_run_dir, _split_name, "t.csv")
    if os.path.exists(t_csv):
        try:
            raw_t = np.loadtxt(t_csv, delimiter=",", skiprows=1, dtype=np.float64)
            if _slice_obj is not None:
                raw_t = raw_t[_slice_obj]
            # Zero-based elapsed time so all trajectories start at 0 s
            t_arr = raw_t - raw_t[0]
        except Exception:
            pass

    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    # Forward-pass dispatch:
    #   DECOMPOSED_MODELS  / PHYSICS_INPUT_MODELS  → forward(features, physics)
    #   All others (BlackBox, PhysicsReg, Lagrangian, EC) → forward(features)
    _needs_physics = cls_name in DECOMPOSED_MODELS or cls_name in PHYSICS_INPUT_MODELS

    all_pred, all_target = [], []
    with torch.no_grad():
        for features, target, physics in dl:
            if _needs_physics:
                out = model(features, physics)
            else:
                out = model(features)
            if isinstance(out, tuple):
                out = out[0]   # tau_hat from (tau_hat, components)
            all_pred.append(out.numpy())
            all_target.append(target.numpy())

    pred_arr   = np.concatenate(all_pred,   axis=0)
    target_arr = np.concatenate(all_target, axis=0)

    # --- Denormalization ---
    # Predictions: use model's OWN training stats (saved in model.pt).
    # This is correct even when the loaded split differs from the training run.
    test_std_tau  = getattr(ds, "std_tau",  np.ones(5,  dtype=np.float32))
    test_mean_tau = getattr(ds, "mean_tau", np.zeros(5, dtype=np.float32))

    if norm_stats and "std_tau" in norm_stats and "mean_tau" in norm_stats:
        model_std_tau  = np.array(norm_stats["std_tau"],  dtype=np.float32).clip(min=1e-8)
        model_mean_tau = np.array(norm_stats["mean_tau"], dtype=np.float32)
    else:
        model_std_tau  = test_std_tau.clip(min=1e-8)
        model_mean_tau = test_mean_tau

    # Denormalize both arrays to physical N·m using each dataset's own stats.
    # pred_arr was produced by a model trained under model_std/mean;
    # target_arr was normalized by the test dataset's std/mean.
    # Both are converted to physical units → metrics are on a common scale.
    pred_arr   = pred_arr   * model_std_tau  + model_mean_tau
    target_arr = target_arr * test_std_tau.clip(min=1e-8) + test_mean_tau

    return {"pred": pred_arr, "target": target_arr, "t": t_arr}


def _pearson_r_safe_viz(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute per-joint and summary metrics. Inputs in physical N·m (aligned with train.py)."""
    diff  = pred - target

    mse   = (diff ** 2).mean(axis=0)
    rmse  = np.sqrt(mse)
    mae   = np.abs(diff).mean(axis=0)
    max_e = np.abs(diff).max(axis=0)

    rng   = target.max(axis=0) - target.min(axis=0)
    nrmse = rmse / (rng + 1e-8)

    ss_res = (diff ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2     = 1.0 - ss_res / (ss_tot + 1e-10)

    pearson_r = np.array([
        _pearson_r_safe_viz(pred[:, j], target[:, j])
        for j in range(pred.shape[1])
    ])

    exp_var = 1.0 - np.var(diff, axis=0) / (np.var(target, axis=0) + 1e-10)

    mse_pooled  = float(np.mean(diff.astype(np.float64, copy=False) ** 2))
    rmse_pooled = float(np.sqrt(mse_pooled))
    pv = pred.reshape(-1).astype(np.float64, copy=False)
    tv = target.reshape(-1).astype(np.float64, copy=False)
    ss_res_all = float(np.sum((pv - tv) ** 2))
    ss_tot_all = float(np.sum((tv - tv.mean()) ** 2))
    r2_overall = float(1.0 - ss_res_all / (ss_tot_all + 1e-10))

    return {
        "mse":               mse.tolist(),
        "rmse":              rmse.tolist(),
        "nrmse":             nrmse.tolist(),
        "mae":               mae.tolist(),
        "max_error":         max_e.tolist(),
        "r2":                r2.tolist(),
        "pearson_r":         pearson_r.tolist(),
        "explained_variance": exp_var.tolist(),
        "mse_mean":          float(mse.mean()),
        "rmse_mean":         float(rmse.mean()),
        "rmse_macro_mean":   float(rmse.mean()),
        "mse_pooled":        mse_pooled,
        "rmse_pooled":       rmse_pooled,
        "r2_overall":        r2_overall,
        "nrmse_mean":        float(nrmse.mean()),
        "mae_mean":          float(mae.mean()),
        "r2_mean":           float(r2.mean()),
        "pearson_r_mean":    float(np.mean(pearson_r)),
    }


def load_training_history(model_dir: str) -> dict[str, list] | None:
    """Read training_history.csv → {epoch, train_loss, val_loss}."""
    csv_path = os.path.join(model_dir, "training_history.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        epochs, train_loss, val_loss = [], [], []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
        return {"epoch": epochs, "train_loss": train_loss, "val_loss": val_loss}
    except Exception:
        return None


def detect_trajectory_boundaries(split_dir: str) -> list[tuple[int, int, float, float]]:
    """
    Detect individual trajectory segments in a flat-concatenated split CSV.

    Boundaries are placed where `t.csv` shows a clock reset (t[i+1] < t[i])
    or a large inter-sample gap (> 25 × median_dt), indicating the start of a
    new recording.

    Parameters
    ----------
    split_dir : path to a split directory containing t.csv

    Returns
    -------
    list of (start_idx, end_idx_exclusive, t_start, t_end) tuples.
    Returns an empty list if t.csv is absent or unreadable.
    """
    t_csv = os.path.join(split_dir, "t.csv")
    if not os.path.exists(t_csv):
        return []
    try:
        t = np.loadtxt(t_csv, delimiter=",", skiprows=1, dtype=np.float64)
    except Exception:
        return []

    N = len(t)
    if N == 0:
        return []
    if N == 1:
        return [(0, 1, float(t[0]), float(t[0]))]

    dt = np.diff(t)
    pos_dt = dt[dt > 0]
    med_dt = float(np.median(pos_dt)) if len(pos_dt) > 0 else 1.0
    # Threshold: clock reset OR gap > 25× typical inter-sample interval (≈ 82 ms at 300 Hz)
    thresh = max(med_dt * 25.0, 0.05)

    breaks = [0]
    for i, d in enumerate(dt):
        if t[i + 1] < t[i] or d > thresh:
            breaks.append(i + 1)
    breaks.append(N)

    segments: list[tuple[int, int, float, float]] = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b > a:
            segments.append((int(a), int(b), float(t[a]), float(t[b - 1])))
    return segments


# =============================================================================
# TRAJECTORY BROWSER DIALOG
# =============================================================================

class TrajectoryBrowserDialog(tk.Toplevel):
    """
    Browse preprocessed run directories and select a single trajectory segment
    (or all samples from a split) for inference and visualisation.

    After closing:
        self.result  is None if cancelled, or a dict:
          {
            "run_dir":   str,
            "split":     str,       # "train" | "val" | "test"
            "start_idx": int,
            "end_idx":   int,       # exclusive
            "t_start":   float,     # seconds
            "t_end":     float,     # seconds
            "label":     str,       # human-readable description
          }
        When the "ALL" row is selected, start_idx=0 and end_idx=n_total.
    """

    COLUMNS = ("#", "Source", "N samples", "t start (s)", "t end (s)", "duration (s)")

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Browse Data — Select Run / Split / Trajectory")
        self.geometry("1200x580")
        self.resizable(True, True)
        self.grab_set()

        self.result: dict | None = None
        self._segments: list[tuple[int, int, float, float]] = []
        self._n_total: int = 0

        # Discover available preprocessed run directories
        self._run_dirs: list[str] = []
        if os.path.isdir(_TRAIN_DATA_DIR):
            self._run_dirs = sorted([
                os.path.join(_TRAIN_DATA_DIR, d)
                for d in os.listdir(_TRAIN_DATA_DIR)
                if os.path.isdir(os.path.join(_TRAIN_DATA_DIR, d))
                and not d.startswith(".")
            ])

        self._build_ui()

        if self._run_dirs:
            self._run_var.set(os.path.basename(self._run_dirs[0]))
            self._refresh()

    def _build_ui(self):
        # ── Top controls ────────────────────────────────────────────────────
        ctrl = tk.Frame(self)
        ctrl.pack(fill="x", padx=10, pady=8)

        tk.Label(ctrl, text="Run directory:", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(0, 6))
        self._run_var   = tk.StringVar()
        run_names       = [os.path.basename(d) for d in self._run_dirs]
        self._run_combo = ttk.Combobox(ctrl, textvariable=self._run_var,
                                       values=run_names, state="readonly", width=42)
        self._run_combo.grid(row=0, column=1, sticky="w")
        self._run_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh())

        tk.Button(ctrl, text="Browse…",
                  command=self._browse_rundir).grid(row=0, column=2, padx=6)

        tk.Label(ctrl, text="Split:", anchor="w").grid(
            row=1, column=0, sticky="w", pady=4)
        self._split_var = tk.StringVar(value="test")
        self._split_combo = ttk.Combobox(
            ctrl, textvariable=self._split_var,
            values=["train", "val", "test"], state="readonly", width=10)
        self._split_combo.grid(row=1, column=1, sticky="w")
        self._split_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh())

        self._info_var = tk.StringVar(value="Select a run directory to list trajectories.")
        tk.Label(ctrl, textvariable=self._info_var, anchor="w",
                 fg="#444", font=("TkDefaultFont", 9)).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # ── Treeview ────────────────────────────────────────────────────────
        tf = tk.Frame(self)
        tf.pack(fill="both", expand=True, padx=10)

        self._tree = ttk.Treeview(tf, columns=self.COLUMNS,
                                   show="headings", selectmode="browse", height=16)
        widths  = [38, 230, 90, 105, 105, 90]
        anchors = ["center", "w", "center", "center", "center", "center"]
        for col, w, anc in zip(self.COLUMNS, widths, anchors):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor=anc, minwidth=w)

        vsb = ttk.Scrollbar(tf, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(tf, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tf.rowconfigure(0, weight=1)
        tf.columnconfigure(0, weight=1)

        self._tree.tag_configure("all_row", background="#ddeeff")
        self._tree.tag_configure("even",    background="#f5f5f5")
        self._tree.bind("<Double-1>", lambda _: self._confirm())

        # ── Buttons ─────────────────────────────────────────────────────────
        bf = tk.Frame(self)
        bf.pack(fill="x", padx=10, pady=8)
        tk.Button(bf, text="Load Selected", command=self._confirm,
                  bg="#336699", fg="white",
                  font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=4)
        tk.Button(bf, text="Cancel",
                  command=self.destroy).pack(side="right", padx=4)

    @staticmethod
    def _fmt_source(entry: dict) -> str:
        """Return a compact label from a trajectory metadata entry."""
        src  = entry.get("source_file", "")
        stem = os.path.splitext(os.path.basename(src))[0]
        # Keep parts before coordinate segment (starts with 'cx')
        parts = stem.split("_")
        label_parts: list[str] = []
        for p in parts:
            if re.match(r"^cx", p):
                break
            label_parts.append(p)
        return "_".join(label_parts) or stem[:30]

    def _browse_rundir(self):
        d = filedialog.askdirectory(
            title="Select preprocessed run directory (contains train/ val/ test/)",
            initialdir=_TRAIN_DATA_DIR if os.path.isdir(_TRAIN_DATA_DIR) else _NN_DIR,
        )
        if not d:
            return
        if d not in self._run_dirs:
            self._run_dirs.append(d)
            names = [os.path.basename(x) for x in self._run_dirs]
            self._run_combo["values"] = names
        self._run_var.set(os.path.basename(d))
        self._refresh()

    def _current_run_dir(self) -> str | None:
        name = self._run_var.get()
        for d in self._run_dirs:
            if os.path.basename(d) == name:
                return d
        return None

    def _refresh(self):
        run_dir = self._current_run_dir()
        split   = self._split_var.get()
        self._tree.delete(*self._tree.get_children())
        self._segments  = []
        self._n_total   = 0

        if run_dir is None:
            self._info_var.set("No run directory selected.")
            return

        split_dir = os.path.join(run_dir, split)
        if not os.path.isdir(split_dir):
            self._info_var.set(
                f"Split '{split}' not found in {os.path.basename(run_dir)}")
            return

        # Count rows in filtered_q.csv (cheap: just count newlines)
        q_csv = os.path.join(split_dir, "filtered_q.csv")
        if os.path.exists(q_csv):
            try:
                with open(q_csv) as fh:
                    self._n_total = sum(1 for _ in fh) - 1  # subtract header row
            except Exception:
                pass

        segs = detect_trajectory_boundaries(split_dir)
        self._segments = segs

        # Per-trajectory source labels from metadata.json (if available)
        src_lookup: dict[int, str] = {}
        meta_path = os.path.join(run_dir, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                import json as _json
                _meta = _json.load(open(meta_path))
                for t_entry in (_meta.get("split", {})
                                .get("stats", {})
                                .get(split, {})
                                .get("trajectories", [])):
                    src_lookup[int(t_entry["start_idx"])] = \
                        self._fmt_source(t_entry)
            except Exception:
                pass

        # "ALL" row — uses the entire split without any slice
        t_start_all = float(segs[0][2]) if segs else 0.0
        t_end_all   = float(segs[-1][3]) if segs else 0.0
        self._tree.insert("", "end", iid="all",
                          values=("ALL", "— all trajectories —",
                                  f"{self._n_total:,}",
                                  f"{t_start_all:.3f}", f"{t_end_all:.3f}",
                                  f"{t_end_all - t_start_all:.3f}"),
                          tags=("all_row",))

        for i, (s, e, ts, te) in enumerate(segs):
            tag = ("even",) if i % 2 == 0 else ()
            src_label = src_lookup.get(s, f"traj {i + 1}")
            self._tree.insert("", "end", iid=str(i),
                              values=(i + 1, src_label, e - s,
                                      f"{ts:.3f}", f"{te:.3f}",
                                      f"{te - ts:.3f}"),
                              tags=tag)

        n_segs = len(segs)
        self._info_var.set(
            f"{os.path.basename(run_dir)} / {split} — "
            f"{self._n_total:,} samples total  |  {n_segs} trajectory segment(s) detected"
        )

        # Auto-select first individual trajectory if available
        if segs:
            self._tree.selection_set("0")
            self._tree.see("0")

    def _confirm(self):
        run_dir = self._current_run_dir()
        split   = self._split_var.get()
        if run_dir is None:
            messagebox.showwarning("Select Data", "Please select a run directory first.")
            return
        sel = self._tree.selection()
        if not sel:
            messagebox.showwarning("Select Data", "Please select a row (or 'ALL').")
            return

        iid = sel[0]
        base = os.path.basename(run_dir)
        if iid == "all":
            ts = self._segments[0][2] if self._segments else 0.0
            te = self._segments[-1][3] if self._segments else 0.0
            self.result = {
                "run_dir":   run_dir,
                "split":     split,
                "start_idx": 0,
                "end_idx":   self._n_total,
                "t_start":   ts,
                "t_end":     te,
                "label":     f"All  —  {base}/{split}",
            }
        else:
            idx      = int(iid)
            s, e, ts, te = self._segments[idx]
            src_label = self._tree.set(iid, "Source")
            self.result = {
                "run_dir":   run_dir,
                "split":     split,
                "start_idx": s,
                "end_idx":   e,
                "t_start":   ts,
                "t_end":     te,
                "label":     f"#{idx + 1} {src_label}  —  {base}/{split}  [{s}:{e}]",
            }
        self.destroy()


# =============================================================================
# REGISTRY BROWSER DIALOG
# =============================================================================

class RegistryBrowserDialog(tk.Toplevel):
    """
    Browse all entries in models_registry.yaml and select one or more to load.
    After closing, self.selected_dirs contains the model_dir paths to load.
    """

    COLUMNS = ("Rank", "Model Type", "RMSE_p", "NRMSE", "R²", "Epochs", "ES?",
               "Train Time", "Trained At", "Samples", "Device")

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Registry Browser — Select Models to Load")
        self.geometry("1380x520")
        self.grab_set()

        self.selected_dirs: list[str] = []
        self._entries = load_registry()

        if not self._entries:
            tk.Label(self, text="No registry entries found.\n"
                     f"Check: {REGISTRY_FILE}", pady=30).pack()
            tk.Button(self, text="Close", command=self.destroy).pack()
            return

        self._build_ui()

    def _build_ui(self):
        # Info bar
        info = tk.Label(self,
                        text=f"  {len(self._entries)} models in registry — sorted by RMSE (best first)."
                             "  Double-click or select + Load.",
                        anchor="w", fg="#444")
        info.pack(fill="x", padx=6, pady=4)

        # Treeview
        frame = tk.Frame(self)
        frame.pack(fill="both", expand=True, padx=6)

        self._tree = ttk.Treeview(frame, columns=self.COLUMNS,
                                   show="headings", selectmode="extended")
        widths = [40, 220, 90, 75, 60, 75, 40, 90, 135, 90, 220]
        for col, w in zip(self.COLUMNS, widths):
            self._tree.heading(col, text=col,
                               command=lambda c=col: self._sort(c))
            self._tree.column(col, width=w, anchor="center")
        self._tree.column("Model Type", anchor="w")
        self._tree.column("Device",     anchor="w")

        vsb = ttk.Scrollbar(frame, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._tree.bind("<Double-1>", self._on_double)

        self._populate()

        # Tag best model
        self._tree.tag_configure("best", background="#d4edda")

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=6, pady=6)
        tk.Button(btn_frame, text="Load Selected",
                  command=self._load_selected, bg="#4a7", fg="white",
                  font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Load All",
                  command=self._load_all).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Cancel",
                  command=self.destroy).pack(side="right", padx=4)

    def _populate(self):
        self._tree.delete(*self._tree.get_children())
        for rank, entry in enumerate(self._entries, 1):
            metrics  = entry.get("metrics", {})
            training = entry.get("training", {})
            hw       = entry.get("hardware", {})
            data     = entry.get("data", {})

            rmse_p = metrics.get("test_rmse_pooled", metrics.get("test_rmse_mean", 0.0))
            nrmse  = metrics.get("test_nrmse_mean", 0.0)
            r2     = metrics.get("test_r2_overall", 0.0)
            epochs = f"{training.get('epochs_ran','?')}/{training.get('epochs_max','?')}"
            es     = "✓" if training.get("stopped_early") else "–"
            t_fmt  = training.get("time_formatted", "–")
            raw_ta = entry.get("trained_at", "")
            try:
                from datetime import datetime as _dt
                trained_at = _dt.fromisoformat(str(raw_ta)).strftime("%d %b %Y  %H:%M")
            except Exception:
                trained_at = str(raw_ta)[:16] if raw_ta else "–"
            n_samp = data.get("num_train_samples", 0)
            device = hw.get("device", "–").replace("cuda:", "GPU:")

            tag = ("best",) if rank == 1 else ()
            self._tree.insert("", "end",
                              iid=str(rank - 1),
                              values=(rank,
                                      entry.get("model_type", "–"),
                                      f"{rmse_p:.5f}",
                                      f"{nrmse:.4f}",
                                      f"{r2:.4f}",
                                      epochs, es, t_fmt,
                                      trained_at,
                                      f"{n_samp:,}",
                                      device),
                              tags=tag)

    def _sort(self, col):
        """Sort table by clicked column."""
        col_idx = self.COLUMNS.index(col)
        data = [(self._tree.set(iid, col), iid)
                for iid in self._tree.get_children("")]
        try:
            data.sort(key=lambda x: float(x[0].replace(",", "").replace("✓","1").replace("–","0")))
        except ValueError:
            data.sort(key=lambda x: x[0])
        for pos, (_, iid) in enumerate(data):
            self._tree.move(iid, "", pos)

    def _on_double(self, _event):
        self._load_selected()

    def _load_selected(self):
        sels = self._tree.selection()
        if not sels:
            messagebox.showinfo("Load", "Select at least one model.")
            return
        self.selected_dirs = [
            self._entries[int(iid)].get("run_dir", "")
            for iid in sels
        ]
        self.destroy()

    def _load_all(self):
        self.selected_dirs = [e.get("run_dir", "") for e in self._entries]
        self.destroy()


# =============================================================================
# STYLE DIALOG
# =============================================================================

class StyleDialog(tk.Toplevel):
    """Dialog to configure line style for a single model."""

    LINE_STYLES_NAMED = {"solid": "-", "dashed": "--", "dashdot": "-.", "dotted": ":"}
    LINE_WIDTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    def __init__(self, parent, title: str, current_style: dict):
        super().__init__(parent)
        self.title(f"Line Style — {title}")
        self.resizable(False, False)
        self.grab_set()
        self.result = dict(current_style)

        tk.Label(self, text="Color:", anchor="w").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self._color_var = tk.StringVar(value=current_style.get("color", "#1f77b4"))
        self._color_btn = tk.Button(self, text="  ", width=4,
                                    bg=self._color_var.get(),
                                    command=self._pick_color)
        self._color_btn.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self, text="Line style:", anchor="w").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self._ls_var = tk.StringVar(value=self._rev_ls(current_style.get("linestyle", "-")))
        ttk.Combobox(self, textvariable=self._ls_var,
                     values=list(self.LINE_STYLES_NAMED.keys()), width=12).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self, text="Line width:", anchor="w").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self._lw_var = tk.DoubleVar(value=current_style.get("linewidth", 1.5))
        ttk.Combobox(self, textvariable=self._lw_var,
                     values=self.LINE_WIDTHS, width=12).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self, text="Opacity (0-1):", anchor="w").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self._alpha_var = tk.DoubleVar(value=current_style.get("alpha", 0.85))
        ttk.Spinbox(self, from_=0.1, to=1.0, increment=0.05,
                    textvariable=self._alpha_var, width=12).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(self, text="Marker:", anchor="w").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self._marker_var = tk.StringVar(value=current_style.get("marker", "none") or "none")
        ttk.Combobox(self, textvariable=self._marker_var,
                     values=MARKER_OPTS, width=12).grid(row=4, column=1, padx=5, pady=5)

        ttk.Separator(self, orient="horizontal").grid(row=5, column=0, columnspan=2,
                                                       sticky="ew", padx=10, pady=4)
        tk.Label(self, text="SG window (0=off):", anchor="w").grid(
            row=6, column=0, padx=10, pady=5, sticky="w")
        self._sg_win_var = tk.IntVar(value=int(current_style.get("sg_window", 0)))
        ttk.Spinbox(self, from_=0, to=201, increment=2,
                    textvariable=self._sg_win_var, width=12).grid(row=6, column=1, padx=5, pady=5)

        tk.Label(self, text="SG poly order:", anchor="w").grid(
            row=7, column=0, padx=10, pady=5, sticky="w")
        self._sg_poly_var = tk.IntVar(value=int(current_style.get("sg_poly", 3)))
        ttk.Spinbox(self, from_=1, to=6, increment=1,
                    textvariable=self._sg_poly_var, width=12).grid(row=7, column=1, padx=5, pady=5)

        btn_frame = tk.Frame(self)
        btn_frame.grid(row=8, column=0, columnspan=2, pady=10)
        tk.Button(btn_frame, text="Apply",  command=self._apply).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="left", padx=5)

    def _rev_ls(self, ls):
        for name, val in self.LINE_STYLES_NAMED.items():
            if val == ls:
                return name
        return "solid"

    def _pick_color(self):
        color = colorchooser.askcolor(color=self._color_var.get(), parent=self)
        if color[1]:
            self._color_var.set(color[1])
            self._color_btn.config(bg=color[1])

    def _apply(self):
        self.result = {
            "color":      self._color_var.get(),
            "linestyle":  self.LINE_STYLES_NAMED[self._ls_var.get()],
            "linewidth":  float(self._lw_var.get()),
            "alpha":      float(self._alpha_var.get()),
            "marker":     "" if self._marker_var.get() == "none" else self._marker_var.get(),
            "markersize": 3,
            "sg_window":  int(self._sg_win_var.get()),
            "sg_poly":    int(self._sg_poly_var.get()),
        }
        self.destroy()


# =============================================================================
# COMPARISON WINDOW
# =============================================================================

class ComparisonWindow(tk.Toplevel):
    """
    Model comparison window — 4 tabs + left sidebar for add/remove.

    Tabs:  Summary | Heatmaps | Training History | Hyperparameters
    Sidebar: lists active models; Remove (from view), Add (load from registry).
    """

    def __init__(self, parent: tk.Tk, model_infos: list[dict]):
        super().__init__(parent)
        self.title("Model Comparison — Detailed Analysis")
        self.geometry("1380x740")
        self.minsize(900, 560)

        self._parent_app = parent   # VisualizerApp reference for Add Model
        # Deep-copy so sidebar changes don't mutate the caller's list
        self._all_infos: list[dict] = list(model_infos)
        # Active = those with metrics; user can toggle via sidebar
        self._active: list[dict] = [i for i in self._all_infos if i.get("metrics")]

        if not self._active:
            tk.Label(self, text="Run inference first — no metrics available.",
                     pady=40).pack()
            return

        self._build_ui()

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Outer horizontal split: sidebar (left) + notebook (right)
        outer = tk.PanedWindow(self, orient="horizontal", sashwidth=5)
        outer.pack(fill="both", expand=True)

        # ── Left sidebar ───────────────────────────────────────────────────
        sb = tk.Frame(outer, width=220, relief="groove", bd=1)
        outer.add(sb, minsize=160)

        tk.Label(sb, text="Models in Comparison",
                 font=("TkDefaultFont", 9, "bold")).pack(pady=(8, 2), padx=6, anchor="w")

        list_frame = tk.Frame(sb)
        list_frame.pack(fill="both", expand=True, padx=6)
        self._sb_list = tk.Listbox(list_frame, selectmode="single",
                                    exportselection=False, font=("TkFixedFont", 8),
                                    height=14)
        sb_vsb = ttk.Scrollbar(list_frame, orient="vertical",
                                command=self._sb_list.yview)
        self._sb_list.configure(yscrollcommand=sb_vsb.set)
        self._sb_list.pack(side="left", fill="both", expand=True)
        sb_vsb.pack(side="right", fill="y")

        btn_f = tk.Frame(sb)
        btn_f.pack(fill="x", padx=6, pady=6)
        tk.Button(btn_f, text="✕ Remove", command=self._sb_remove,
                  fg="#c00").pack(side="left", padx=2)
        tk.Button(btn_f, text="＋ Add",   command=self._sb_add).pack(side="left", padx=2)

        self._sb_refresh()

        # ── Notebook (right) ───────────────────────────────────────────────
        nb_frame = tk.Frame(outer)
        outer.add(nb_frame, minsize=600)

        self._nb = ttk.Notebook(nb_frame)
        self._nb.pack(fill="both", expand=True)

        self._tab_frames: dict[str, ttk.Frame] = {}
        for label in ("Summary", "Heatmaps", "Training History", "Hyperparameters"):
            f = ttk.Frame(self._nb)
            self._nb.add(f, text=label)
            self._tab_frames[label] = f

        self._rebuild_tabs()

    # ── Sidebar helpers ────────────────────────────────────────────────────

    def _sb_refresh(self):
        self._sb_list.delete(0, "end")
        for info in self._active:
            self._sb_list.insert("end", info["name"])

    def _sb_remove(self):
        sel = self._sb_list.curselection()
        if not sel:
            return
        self._active.pop(sel[0])
        if not self._active:
            messagebox.showinfo("Compare", "No models left — add at least one.")
            return
        self._sb_refresh()
        self._rebuild_tabs()

    def _sb_add(self):
        """Open RegistryBrowserDialog; load selected model(s) into parent app + this window."""
        dlg = RegistryBrowserDialog(self)
        self.wait_window(dlg)
        if not dlg.selected_dirs:
            return
        for d in dlg.selected_dirs:
            # Load into parent app
            self._parent_app._load_model_dir(d)
            # The freshly added entry is the last in parent's model list
            if not self._parent_app._models:
                continue
            new_entry = self._parent_app._models[-1]
            # Build an info dict for the comparison window
            m = (new_entry["data"].get("metrics", {})
                 if new_entry["data"] is not None else {})
            new_info = {
                "name":      new_entry["name"],
                "metrics":   m,
                "color":     new_entry["style"]["color"],
                "hparams":   new_entry["meta"].get("hparams", {}),
                "model_dir": new_entry.get("model_dir", ""),
            }
            self._all_infos.append(new_info)
            if m:  # only add to active if metrics exist
                self._active.append(new_info)
        self._sb_refresh()
        self._rebuild_tabs()

    # ── Tab content (rebuilt on sidebar change) ────────────────────────────

    def _rebuild_tabs(self):
        for label, frame in self._tab_frames.items():
            for child in frame.winfo_children():
                child.destroy()
            if label == "Summary":
                self._build_summary(frame)
            elif label == "Heatmaps":
                self._build_heatmaps(frame)
            elif label == "Training History":
                self._build_history_chart(frame)
            elif label == "Hyperparameters":
                self._build_hp_table(frame)

    @property
    def model_infos(self) -> list[dict]:
        """Alias so helper methods work unchanged."""
        return self._active

    # ── Tab 1: Summary ─────────────────────────────────────────────────────

    def _build_summary(self, parent):
        """Compact treeview + two mini-charts below."""
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # ── Compact metrics treeview ───────────────────────────────────────
        tbl_frame = tk.Frame(parent)
        tbl_frame.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))

        METRIC_COLS = ("Model", "RMSE \u2193", "R\u00b2", "Pearson", "NRMSE", "MAE")
        tree = ttk.Treeview(tbl_frame, columns=METRIC_COLS,
                             show="headings", height=min(len(self._active) + 1, 10))
        col_widths = [220, 80, 72, 72, 72, 72]
        for col, w in zip(METRIC_COLS, col_widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center")
        tree.column("Model", anchor="w")

        # Compute per-column best values for green highlighting
        def _safe_mean(info, key):
            v = info["metrics"].get(key)
            if v is None:
                return float("nan")
            return float(np.nanmean(v)) if isinstance(v, list) else float(v)

        metrics_rows = []
        for info in self._active:
            metrics_rows.append({
                "name":    info["name"],
                "rmse":    _safe_mean(info, "rmse"),
                "r2":      _safe_mean(info, "r2"),
                "pearson": _safe_mean(info, "pearson_r"),
                "nrmse":   _safe_mean(info, "nrmse"),
                "mae":     _safe_mean(info, "mae"),
            })

        best_rmse    = min((r["rmse"]    for r in metrics_rows if np.isfinite(r["rmse"])),    default=0)
        best_r2      = max((r["r2"]      for r in metrics_rows if np.isfinite(r["r2"])),      default=0)
        best_pearson = max((r["pearson"] for r in metrics_rows if np.isfinite(r["pearson"])), default=0)
        best_nrmse   = min((r["nrmse"]   for r in metrics_rows if np.isfinite(r["nrmse"])),   default=0)
        best_mae     = min((r["mae"]     for r in metrics_rows if np.isfinite(r["mae"])),     default=0)

        tree.tag_configure("best_rmse",    background="#d4edda")
        tree.tag_configure("best_r2",      background="#cce5ff")
        tree.tag_configure("best_pearson", background="#e2ccff")
        tree.tag_configure("best_nrmse",   background="#fff3cd")
        tree.tag_configure("best_mae",     background="#fde2e4")

        def _fmt(v): return f"{v:.5f}" if np.isfinite(v) else "\u2013"

        for r in metrics_rows:
            tags = []
            if abs(r["rmse"]    - best_rmse)    < 1e-9: tags.append("best_rmse")
            if abs(r["r2"]      - best_r2)      < 1e-9: tags.append("best_r2")
            if abs(r["pearson"] - best_pearson) < 1e-9: tags.append("best_pearson")
            if abs(r["nrmse"]   - best_nrmse)   < 1e-9: tags.append("best_nrmse")
            if abs(r["mae"]     - best_mae)     < 1e-9: tags.append("best_mae")
            tag = tags[0] if tags else ""
            tree.insert("", "end", values=(
                r["name"], _fmt(r["rmse"]), _fmt(r["r2"]),
                _fmt(r["pearson"]), _fmt(r["nrmse"]), _fmt(r["mae"]),
            ), tags=(tag,))

        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=hsb.set)
        tree.pack(fill="x")
        hsb.pack(fill="x")

        tk.Label(parent,
                 text="  Color: green=best RMSE  blue=best R²  purple=best Pearson  "
                      "yellow=best NRMSE  pink=best MAE",
                 anchor="w", fg="#555", font=("TkDefaultFont", 8)).grid(
            row=1, column=0, sticky="w", padx=6)

        # ── Two mini-charts ────────────────────────────────────────────────
        chart_frame = tk.Frame(parent)
        chart_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=2)
        parent.rowconfigure(2, weight=1)

        infos_s = sorted(self._active, key=lambda i: _safe_mean(i, "rmse"))
        names   = [i["name"] for i in infos_s]
        colors  = [i.get("color", DEFAULT_COLORS[k % len(DEFAULT_COLORS)])
                   for k, i in enumerate(infos_s)]

        fig = Figure(figsize=(13, 4), tight_layout=True)

        # Left: mean RMSE bar
        ax1   = fig.add_subplot(121)
        rmses = [_safe_mean(i, "rmse") for i in infos_s]
        x     = np.arange(len(names))
        bars  = ax1.bar(x, rmses, color=colors, edgecolor="black", linewidth=0.5, zorder=3)
        ax1.axhline(rmses[0], color="green", linestyle="--", linewidth=0.8,
                    label=f"Best: {rmses[0]:.5f}", zorder=4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=22, ha="right", fontsize=8)
        ax1.set_ylabel("Mean RMSE (N·m)")
        ax1.set_title("RMSE — best → worst")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.35, zorder=0)
        for bar, val in zip(bars, rmses):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                     f"{val:.5f}", ha="center", va="bottom", fontsize=7)

        # Right: per-joint RMSE grouped bars
        ax2      = fig.add_subplot(122)
        n_m      = len(self._active)
        xj       = np.arange(5)
        bar_w    = 0.75 / max(n_m, 1)
        for k, info in enumerate(self._active):
            rmse_j = [float(v) if np.isfinite(v) else 0.0
                      for v in info["metrics"].get("rmse", [0.0] * 5)]
            ax2.bar(xj + k * bar_w, rmse_j, bar_w,
                    label=info["name"],
                    color=info.get("color", DEFAULT_COLORS[k % len(DEFAULT_COLORS)]),
                    edgecolor="black", linewidth=0.3, alpha=0.88)
        ax2.set_xticks(xj + bar_w * (n_m - 1) / 2)
        ax2.set_xticklabels([f"J{j+1}" for j in range(5)], fontsize=9)
        ax2.set_ylabel("RMSE (N·m)")
        ax2.set_title("Per-Joint RMSE")
        ax2.legend(fontsize=7, ncol=max(1, n_m // 3))
        ax2.grid(axis="y", alpha=0.35)

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Tab 2: Heatmaps ────────────────────────────────────────────────────

    def _build_heatmaps(self, parent):
        """3 stacked heatmaps: R², Pearson r, NRMSE — rows=models, cols=joints."""
        n_m = len(self._active)
        names = [i["name"] for i in self._active]

        def _mat(key, fallback=float("nan")):
            rows = []
            for info in self._active:
                v = info["metrics"].get(key, [fallback] * 5)
                rows.append([float(x) if np.isfinite(float(x)) else float("nan") for x in v])
            return np.array(rows)   # (n_m, 5)

        mat_r2  = _mat("r2")
        mat_pr  = _mat("pearson_r")
        mat_nr  = _mat("nrmse")

        fig_h = max(4, n_m * 0.7) * 3 + 1
        fig = Figure(figsize=(12, min(fig_h, 18)), tight_layout=True)

        def _draw_heatmap(ax, mat, title, vmin, vmax, cmap, fmt=".3f"):
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(5))
            ax.set_xticklabels([f"J{j+1} ({JOINT_NAMES[j].split()[0]})"
                                 for j in range(5)], rotation=15, ha="right", fontsize=8)
            ax.set_yticks(range(n_m))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_title(title, fontsize=10, fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            for i in range(n_m):
                for j in range(5):
                    val = mat[i, j]
                    txt = f"{val:{fmt}}" if np.isfinite(val) else "\u2013"
                    # choose text colour for readability
                    bg = im.cmap(im.norm(val)) if np.isfinite(val) else (0.5, 0.5, 0.5, 1)
                    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                    tc  = "black" if lum > 0.45 else "white"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7, color=tc, fontweight="bold")

        ax1 = fig.add_subplot(311)
        _draw_heatmap(ax1, mat_r2, "R²  (green = perfect fit)", -0.2, 1.0, "RdYlGn")

        ax2 = fig.add_subplot(312)
        _draw_heatmap(ax2, mat_pr, "Pearson r  (green = high correlation)", -1.0, 1.0, "RdYlGn")

        ax3 = fig.add_subplot(313)
        # For NRMSE, lower = better, so invert colormap
        _draw_heatmap(ax3, mat_nr, "NRMSE  (green = low error)", 0.0,
                      float(np.nanmax(mat_nr)) * 1.05 if np.any(np.isfinite(mat_nr)) else 1.0,
                      "RdYlGn_r", fmt=".4f")

        sf = tk.Frame(parent)
        sf.pack(fill="both", expand=True)
        canvas = FigureCanvasTkAgg(fig, master=sf)
        canvas.draw()

        # Scrollable canvas widget for many models
        canvas_w = canvas.get_tk_widget()
        vsb = ttk.Scrollbar(sf, orient="vertical")
        hsb = ttk.Scrollbar(sf, orient="horizontal")
        canvas_w.configure()
        canvas_w.pack(fill="both", expand=True)
        vsb.pack_forget(); hsb.pack_forget()
        canvas_w.pack(fill="both", expand=True)

    # ── Tab 3: Training History ────────────────────────────────────────────

    def _build_history_chart(self, parent):
        fig = Figure(figsize=(13, 5), tight_layout=True)
        ax_train = fig.add_subplot(121)
        ax_val   = fig.add_subplot(122)

        any_data = False
        for i, info in enumerate(self._active):
            model_dir = info.get("model_dir", "")
            hist = load_training_history(model_dir) if model_dir else None
            if hist is None:
                continue
            any_data = True
            color = info.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            label = info["name"]
            ax_train.plot(hist["epoch"], hist["train_loss"],
                          color=color, label=label, linewidth=1.2, alpha=0.85)
            ax_val.plot(hist["epoch"],   hist["val_loss"],
                        color=color, label=label, linewidth=1.2, alpha=0.85)

        for ax, title in [(ax_train, "Training Loss"), (ax_val, "Validation Loss")]:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (normalised MSE)")
            ax.set_title(title)
            h, l = ax.get_legend_handles_labels()
            if h:
                ax.legend(h, l, fontsize=7, loc="upper right")
            ax.grid(alpha=0.35)

        if not any_data:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No training_history.csv found for loaded models.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Tab 4: Hyperparameters ─────────────────────────────────────────────

    def _build_hp_table(self, parent):
        all_keys: list[str] = []
        for info in self._active:
            for k in info.get("hparams", {}).keys():
                if k not in all_keys:
                    all_keys.append(k)

        cols = ["Hyperparameter"] + [i["name"] for i in self._active]
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=30)
        tree.heading("Hyperparameter", text="Hyperparameter")
        tree.column("Hyperparameter", width=200, anchor="w")
        for info in self._active:
            tree.heading(info["name"], text=info["name"])
            tree.column(info["name"], width=160, anchor="center")

        for key in all_keys:
            vals = [str(info.get("hparams", {}).get(key, "\u2013")) for info in self._active]
            tag  = ("differ",) if len(set(vals)) > 1 else ()
            tree.insert("", "end", values=[key] + vals, tags=tag)

        tree.tag_configure("differ", background="#fff3cd")

        note = tk.Label(parent,
                        text="  Yellow rows: hyperparameters that differ between models.",
                        anchor="w", fg="#555", font=("TkDefaultFont", 9))
        note.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=2)

        vsb = ttk.Scrollbar(parent, orient="vertical",   command=tree.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")
        hsb.grid(row=2, column=0, sticky="ew")
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class VisualizerApp(tk.Tk):
    """Main GUI application."""

    def __init__(self):
        super().__init__()
        self.title("Neural Network Torque Prediction — Model Visualizer")
        self.geometry("1700x950")
        self.minsize(1100, 650)

        self._models:      list[dict]  = []
        self._gt_style:    dict        = {"color": "#e55", "linestyle": "-",
                                          "linewidth": 2.0, "alpha": 0.9,
                                          "marker": "", "markersize": 3}
        self._data_path:   str | None  = None
        self._traj_slice:  tuple[int, int] | None = None   # (start, end) for selected traj
        self._traj_label:  str | None  = None              # human-readable trajectory label
        self._joint_mask:  list[bool]  = [True] * 5
        self._sample_range             = [0, 3000]

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ── UI Construction ────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Toolbar ────────────────────────────────────────────────────────
        toolbar = tk.Frame(self, relief="raised", bd=1)
        toolbar.pack(side="top", fill="x")

        tk.Button(toolbar, text="📂 Browse Registry",
                  command=self.browse_registry,
                  bg="#2255aa", fg="white",
                  font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=4, pady=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4, pady=3)
        tk.Button(toolbar, text="Load Model (dir)",
                  command=self.load_model_from_filesystem).pack(side="left", padx=4, pady=4)
        tk.Button(toolbar, text="Remove Model",
                  command=self.remove_model).pack(side="left", padx=4, pady=4)
        tk.Button(toolbar, text="Browse Data",
                  command=self.load_test_data_browser,
                  bg="#556b2f", fg="white").pack(side="left", padx=4, pady=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6, pady=3)
        tk.Button(toolbar, text="▶  Run Inference",
                  command=self.run_all_inference,
                  bg="#4a7", fg="white",
                  font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=4, pady=4)
        tk.Button(toolbar, text="Comparison Window",
                  command=self.open_comparison).pack(side="left", padx=4, pady=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6, pady=3)
        tk.Button(toolbar, text="Export PNG",
                  command=self.export_png).pack(side="left", padx=4, pady=4)
        tk.Button(toolbar, text="Help",
                  command=self.show_help).pack(side="right", padx=8, pady=4)

        # ── Status bar ─────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready — click '📂 Browse Registry' to load models, then 'Browse Data' to select a trajectory.")
        tk.Label(self, textvariable=self._status_var,
                 bd=1, relief="sunken", anchor="w").pack(side="bottom", fill="x")

        # ── Main paned layout ──────────────────────────────────────────────
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        left = ttk.Frame(pane, width=300)
        pane.add(left, weight=0)
        self._build_left_panel(left)

        right = ttk.Frame(pane)
        pane.add(right, weight=1)
        self._build_plot_area(right)

    def _build_left_panel(self, parent):
        parent.columnconfigure(0, weight=1)

        # ── Model list (Treeview) ───────────────────────────────────────────
        tk.Label(parent, text="Loaded Models",
                 font=("TkDefaultFont", 10, "bold")).pack(pady=(10, 2), anchor="w", padx=10)

        ml_frame = tk.Frame(parent)
        ml_frame.pack(fill="x", padx=10, pady=2)
        ml_frame.columnconfigure(0, weight=1)

        _ML_COLS = ("on", "name", "rmse", "r2")
        self._model_tree = ttk.Treeview(
            ml_frame, columns=_ML_COLS, show="headings",
            selectmode="browse", height=8)
        self._model_tree.heading("on",   text="✔")
        self._model_tree.heading("name", text="Name")
        self._model_tree.heading("rmse", text="RMSE")
        self._model_tree.heading("r2",   text="R²")
        self._model_tree.column("on",   width=24,  anchor="center", stretch=False)
        self._model_tree.column("name", width=180, anchor="w")
        self._model_tree.column("rmse", width=64,  anchor="center")
        self._model_tree.column("r2",   width=54,  anchor="center")
        ml_vsb = ttk.Scrollbar(ml_frame, orient="vertical",
                                command=self._model_tree.yview)
        self._model_tree.configure(yscrollcommand=ml_vsb.set)
        self._model_tree.grid(row=0, column=0, sticky="nsew")
        ml_vsb.grid(row=0, column=1, sticky="ns")
        ml_frame.rowconfigure(0, weight=1)
        self._model_tree.bind("<<TreeviewSelect>>", self._on_model_select)
        self._model_tree.tag_configure("hidden",  foreground="#aaa")
        self._model_tree.tag_configure("stale",   foreground="#c80")

        btn_row = tk.Frame(parent)
        btn_row.pack(fill="x", padx=10, pady=2)
        tk.Button(btn_row, text="Edit Style", command=self.edit_selected_style,
                  width=8).pack(side="left", padx=1)
        tk.Button(btn_row, text="Toggle",    command=self.toggle_selected,
                  width=6).pack(side="left", padx=1)
        tk.Button(btn_row, text="▲",         command=self._move_model_up,
                  width=2).pack(side="left", padx=1)
        tk.Button(btn_row, text="▼",         command=self._move_model_down,
                  width=2).pack(side="left", padx=1)
        tk.Button(btn_row, text="✕ Remove",  command=self.remove_model,
                  fg="#c00", width=7).pack(side="left", padx=1)

        # ── Info panel for selected model ──────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=6)
        tk.Label(parent, text="Model Info",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=10)

        self._info_text = tk.Text(parent, height=8, width=32, font=("TkFixedFont", 8),
                                   state="disabled", relief="flat", bg=self.cget("bg"),
                                   wrap="word")
        self._info_text.pack(fill="x", padx=10, pady=2)

        # Mini per-joint RMSE bar (canvas)
        tk.Label(parent, text="Per-Joint RMSE (N·m)",
                 font=("TkDefaultFont", 8)).pack(anchor="w", padx=10)
        self._rmse_canvas = tk.Canvas(parent, height=70, bg="#f8f8f8",
                                       relief="sunken", bd=1)
        self._rmse_canvas.pack(fill="x", padx=10, pady=2)

        # ── Ground Truth style ─────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=6)
        tk.Label(parent, text="Ground Truth Style",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=10)
        tk.Button(parent, text="Edit GT Style",
                  command=self.edit_gt_style).pack(padx=10, pady=3, anchor="w")

        # ── Joint filter ───────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=6)
        tk.Label(parent, text="Joint Filter",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=10)
        self._joint_vars = []
        for j in range(5):
            var = tk.BooleanVar(value=True)
            tk.Checkbutton(parent, text=JOINT_NAMES[j], variable=var,
                           command=self._update_plot).pack(anchor="w", padx=20)
            self._joint_vars.append(var)

        # ── Sample range ───────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=6)
        tk.Label(parent, text="Sample Range",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=10)
        rf = tk.Frame(parent)
        rf.pack(fill="x", padx=10, pady=2)
        tk.Label(rf, text="From:").grid(row=0, column=0, sticky="w")
        self._range_from = tk.Entry(rf, width=8)
        self._range_from.insert(0, "0")
        self._range_from.grid(row=0, column=1, padx=4)
        tk.Label(rf, text="To:").grid(row=1, column=0, sticky="w")
        self._range_to = tk.Entry(rf, width=8)
        self._range_to.insert(0, "3000")
        self._range_to.grid(row=1, column=1, padx=4)
        tk.Button(parent, text="Apply Range",
                  command=self._apply_range).pack(padx=10, pady=3, anchor="w")

    def _build_plot_area(self, parent):
        self._fig    = Figure(figsize=(14, 8), tight_layout=True)
        self._axes: list[plt.Axes] = []
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        nav = NavigationToolbar2Tk(self._canvas, parent)
        nav.update()

    # ── Registry Browser ───────────────────────────────────────────────────

    def browse_registry(self):
        if not _HAS_YAML:
            messagebox.showerror("Error", "PyYAML not installed. Cannot read registry.")
            return
        dlg = RegistryBrowserDialog(self)
        self.wait_window(dlg)
        for d in dlg.selected_dirs:
            if d:
                self._load_model_dir(d)

    # ── Model Management ───────────────────────────────────────────────────

    def load_model_from_filesystem(self):
        models_base = os.path.join(_NN_DIR, "Trained_Models")
        directory   = filedialog.askdirectory(
            title="Select Model Directory (contains model.pt)",
            initialdir=models_base if os.path.isdir(models_base) else _NN_DIR,
        )
        if directory:
            self._load_model_dir(directory)

    def _load_model_dir(self, directory: str):
        self._status(f"Loading {os.path.basename(directory)} ...")
        model, meta = load_model_from_dir(directory)
        if model is None:
            messagebox.showerror("Load Error", meta.get("error", "Unknown error"))
            self._status("Load failed.")
            return

        cls   = meta.get("model_class", "Model")
        short = _SHORT_NAMES.get(cls, cls)
        # Store display name WITHOUT embedded RMSE — the legend shows live per-joint
        # RMSE after inference, so embedding a second "saved" RMSE causes confusion.
        stale  = meta.get("stale_reason")
        suffix = "  ⚠ STALE" if stale else ""
        model_name = f"{short} #{len(self._models)+1}{suffix}"

        if stale:
            messagebox.showwarning(
                "Stale Checkpoint — Inference Disabled",
                f"This checkpoint was trained with an older architecture:\n\n"
                f"{stale}\n\n"
                f"The model appears in the list but CANNOT run inference.\n"
                f"Please retrain {cls} to get a fresh checkpoint.",
            )

        idx    = len(self._models)
        entry  = {
            "name":      model_name,
            "model":     model,
            "meta":      meta,
            "data":      None,
            "enabled":   True,
            "model_dir": directory,
            "style": {
                "color":      DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
                "linestyle":  "-",
                "linewidth":  1.8,
                "alpha":      0.85,
                "marker":     "",
                "markersize": 3,
                "sg_window":  0,
                "sg_poly":    3,
            },
        }
        self._models.append(entry)
        self._refresh_model_list()
        self._status(f"Loaded: {model_name}")

    def remove_model(self):
        idx = self._selected_model_idx()
        if idx is None:
            messagebox.showinfo("Remove", "Select a model to remove.")
            return
        name = self._models[idx]["name"]
        self._models.pop(idx)
        self._refresh_model_list()
        self._update_plot()
        self._clear_info_panel()
        self._status(f"Removed: {name}")

    def _selected_model_idx(self) -> int | None:
        """Return the index into self._models of the currently selected treeview row."""
        sel = self._model_tree.selection()
        if not sel:
            return None
        try:
            return int(sel[0])
        except (ValueError, IndexError):
            return None

    def _refresh_model_list(self):
        # Remember which iid was selected
        sel = self._model_tree.selection()
        prev = sel[0] if sel else None

        self._model_tree.delete(*self._model_tree.get_children())
        for i, entry in enumerate(self._models):
            m    = entry.get("data", {}) or {}
            mets = m.get("metrics", {}) if isinstance(m, dict) else {}
            rmse_v = mets.get("rmse_mean")
            r2_v   = mets.get("r2_mean")
            rmse_s = f"{rmse_v:.4f}" if isinstance(rmse_v, float) else "–"
            r2_s   = f"{r2_v:.3f}"   if isinstance(r2_v,   float) else "–"
            enabled = entry["enabled"]
            stale   = bool(entry["meta"].get("stale_reason"))
            on_s    = "✔" if enabled else "□"
            tag     = "stale" if stale else ("" if enabled else "hidden")
            self._model_tree.insert("", "end", iid=str(i),
                                    values=(on_s, entry["name"], rmse_s, r2_s),
                                    tags=(tag,))
        # Restore selection if still valid
        if prev is not None and self._model_tree.exists(prev):
            self._model_tree.selection_set(prev)
            self._model_tree.see(prev)

    def _on_model_select(self, _event=None):
        idx = self._selected_model_idx()
        if idx is not None and idx < len(self._models):
            self._update_info_panel(self._models[idx])

    def toggle_selected(self):
        idx = self._selected_model_idx()
        if idx is None:
            return
        self._models[idx]["enabled"] = not self._models[idx]["enabled"]
        self._refresh_model_list()
        self._update_plot()

    def _move_model_up(self):
        idx = self._selected_model_idx()
        if idx is None or idx == 0:
            return
        self._models[idx - 1], self._models[idx] = self._models[idx], self._models[idx - 1]
        self._refresh_model_list()
        self._model_tree.selection_set(str(idx - 1))
        self._update_plot()

    def _move_model_down(self):
        idx = self._selected_model_idx()
        if idx is None or idx >= len(self._models) - 1:
            return
        self._models[idx], self._models[idx + 1] = self._models[idx + 1], self._models[idx]
        self._refresh_model_list()
        self._model_tree.selection_set(str(idx + 1))
        self._update_plot()

    def edit_selected_style(self):
        idx = self._selected_model_idx()
        if idx is None:
            messagebox.showinfo("Style", "Select a model first.")
            return
        entry = self._models[idx]
        dlg   = StyleDialog(self, entry["name"], entry["style"])
        self.wait_window(dlg)
        if dlg.result != entry["style"]:
            self._models[idx]["style"] = dlg.result
            self._update_plot()

    def edit_gt_style(self):
        dlg = StyleDialog(self, "Ground Truth", self._gt_style)
        self.wait_window(dlg)
        if dlg.result != self._gt_style:
            self._gt_style = dlg.result
            self._update_plot()

    # ── Info Panel ─────────────────────────────────────────────────────────

    def _update_info_panel(self, entry: dict):
        meta = entry.get("meta", {})
        cls  = meta.get("model_class", "–")

        # Registry data from metadata.json / metadata.yaml
        training = meta.get("training", {})
        hw       = meta.get("hardware", {})
        data_m   = meta.get("data", {})

        t_fmt    = training.get("time_formatted") or meta.get("training_time_formatted", "–")
        epochs_r = training.get("epochs_ran") or meta.get("epochs_trained", "–")
        epochs_m = training.get("epochs_max") or meta.get("hyperparams", {}).get("epochs", "–")
        es       = "Yes" if training.get("stopped_early") else "No"
        device   = (hw.get("device") or meta.get("device", "–")).replace("cuda:", "GPU:")
        n_train  = data_m.get("num_train_samples") or "–"
        n_val    = data_m.get("num_val_samples") or "–"
        f_val    = training.get("final_val_loss")
        f_train  = training.get("final_train_loss")

        # trained-at timestamp
        raw_ta = meta.get("trained_at", "")
        try:
            from datetime import datetime as _dt
            trained_at = _dt.fromisoformat(str(raw_ta)).strftime("%d %b %Y, %H:%M")
        except Exception:
            trained_at = str(raw_ta)[:16] if raw_ta else "–"

        # run ID (truncate if long)
        run_id = meta.get("run_id", "")
        run_id_disp = (run_id[:48] + "…") if len(run_id) > 48 else run_id

        # data source (basename of data_run_dir)
        data_run_dir = meta.get("data_run_dir", "")
        data_src = os.path.basename(data_run_dir.rstrip("/\\")) if data_run_dir else "–"

        lines = [
            f"Type:     {_SHORT_NAMES.get(cls, cls)}",
            f"Trained:  {trained_at}",
        ]
        if run_id_disp:
            lines.append(f"Run ID:   {run_id_disp}")
        lines += [
            f"Epochs:   {epochs_r}/{epochs_m}  (ES: {es})",
            f"Time:     {t_fmt}",
            f"Device:   {device}",
            f"Train N:  {f'{n_train:,}' if isinstance(n_train, int) else n_train}",
            f"Val N:    {f'{n_val:,}' if isinstance(n_val, int) else n_val}",
        ]
        if data_src and data_src != "–":
            lines.append(f"Data:     {data_src}")
        if f_val is not None:
            lines.append(f"Val loss: {f_val:.5f}")
        if f_train is not None:
            lines.append(f"Trn loss: {f_train:.5f}")

        # best-val metrics from metadata.yaml
        bvl = meta.get("best_val_loss")
        bvr = meta.get("best_val_rmse")
        if bvl is not None or bvr is not None:
            parts = []
            if bvl is not None:
                parts.append(f"loss={bvl:.5f}")
            if bvr is not None:
                parts.append(f"RMSE={bvr:.5f}")
            lines.append(f"Best val: {', '.join(parts)}")

        # physics scheduler config
        psc = meta.get("physics_sched_config")
        if psc and isinstance(psc, dict):
            wp   = psc.get("weight", psc.get("w_p", ""))
            ns   = psc.get("nudge_step", "")
            auto = "(auto)" if psc.get("nudge_step_auto") else ""
            mb   = psc.get("max_bad_nudges", "")
            lines.append(f"Physics:  {psc.get('mode','?')}  w_p={wp}  step={ns}{auto}  max_bad={mb}")
        elif meta.get("hyperparams", {}).get("physics_weight") is not None:
            pw = meta["hyperparams"]["physics_weight"]
            lines.append(f"Physics:  w_p(init)={pw}")

        # saved test metrics
        saved_metrics = meta.get("metrics", {})
        if saved_metrics:
            rp   = saved_metrics.get("rmse_pooled",  saved_metrics.get("test_rmse_pooled"))
            r2ov = saved_metrics.get("r2_overall",   saved_metrics.get("test_r2_overall"))
            if any(v is not None for v in [rp, r2ov]):
                lines.append("")
                lines.append("── Saved test metrics ──")
            if rp is not None:
                lines.append(f"RMSE_p:   {rp:.5f} N·m")
            if r2ov is not None:
                lines.append(f"R² ov:    {r2ov:.4f}")

        if meta.get("stale_reason"):
            lines.append("\n⚠ STALE CHECKPOINT — inference disabled")
            lines.append("Architecture changed since training.")
            lines.append("Please retrain this model.")

        # live inference metrics
        if entry.get("data") and entry["data"].get("metrics"):
            m       = entry["data"]["metrics"]
            rmse    = m.get("rmse",      [])
            r2      = m.get("r2",        [])
            pearson = m.get("pearson_r", [])
            lines.append(f"\n── Live inference ──")
            lines.append(f"RMSE:     {m.get('rmse_mean', 0):.5f} N·m  (pooled: {m.get('rmse_pooled', m.get('rmse_mean', 0)):.5f})")
            lines.append(f"R²:       {m.get('r2_mean', 0):.4f}  (overall: {m.get('r2_overall', m.get('r2_mean', 0)):.4f})")
            lines.append(f"Pearson:  {m.get('pearson_r_mean', 0):.4f}")
            lines.append(f"\nPer-joint RMSE / R²:")
            for j in range(len(rmse)):
                r2_j = r2[j] if j < len(r2) else 0.0
                lines.append(f"  J{j+1}: {rmse[j]:.5f}  R²={r2_j:.4f}")

        self._info_text.config(state="normal")
        self._info_text.delete("1.0", "end")
        self._info_text.insert("end", "\n".join(lines))
        self._info_text.config(state="disabled")

        # Mini RMSE bar
        self._draw_rmse_bars(entry)

    def _clear_info_panel(self):
        self._info_text.config(state="normal")
        self._info_text.delete("1.0", "end")
        self._info_text.config(state="disabled")
        self._rmse_canvas.delete("all")

    def _draw_rmse_bars(self, entry: dict):
        """Draw tiny horizontal RMSE bars on the info canvas."""
        self._rmse_canvas.delete("all")

        # Prefer live inference metrics; fall back to registry saved metrics
        rmse = None
        if entry.get("data") and entry["data"].get("metrics"):
            rmse = entry["data"]["metrics"].get("rmse")
        if rmse is None:
            saved = entry.get("meta", {}).get("metrics", {})
            # registry stores per_joint_rmse; metadata.json stores rmse list
            rmse = (saved.get("rmse") or
                    saved.get("per_joint_rmse"))

        if not rmse or len(rmse) < 5:
            return

        cw    = self._rmse_canvas.winfo_width() or 260
        ch    = self._rmse_canvas.winfo_height() or 70
        n     = 5
        row_h = ch / n
        pad_l = 30
        max_v = max(rmse) if max(rmse) > 0 else 1.0

        for j, val in enumerate(rmse):
            y0   = j * row_h + 3
            y1   = (j + 1) * row_h - 3
            bar_w = (val / max_v) * (cw - pad_l - 6)
            fill  = entry["style"]["color"]
            self._rmse_canvas.create_rectangle(pad_l, y0, pad_l + bar_w, y1,
                                                fill=fill, outline="")
            self._rmse_canvas.create_text(pad_l - 2, (y0 + y1) / 2,
                                           text=f"J{j+1}", anchor="e",
                                           font=("TkFixedFont", 7))
            self._rmse_canvas.create_text(pad_l + bar_w + 2, (y0 + y1) / 2,
                                           text=f"{val:.4f}", anchor="w",
                                           font=("TkFixedFont", 7))

    # ── Data Loading ───────────────────────────────────────────────────────

    def load_test_data_browser(self):
        """Open the Trajectory Browser dialog and store selected data path + slice."""
        dlg = TrajectoryBrowserDialog(self)
        self.wait_window(dlg)
        if dlg.result is None:
            return

        r = dlg.result
        # Store split directory so run_inference auto-detects the split name
        self._data_path  = os.path.join(r["run_dir"], r["split"])
        self._traj_label = r["label"]

        # If the user selected ALL, no slice needed (pass None to run_inference)
        if r["label"].startswith("All"):
            self._traj_slice = None
            n_pts = r["end_idx"]
        else:
            self._traj_slice = (r["start_idx"], r["end_idx"])
            n_pts = r["end_idx"] - r["start_idx"]

        # Reset the plot sample-range to show up to 5000 samples of the selection
        disp_to = min(5000, n_pts)
        self._range_from.delete(0, "end"); self._range_from.insert(0, "0")
        self._range_to.delete(0, "end");   self._range_to.insert(0, str(disp_to))
        self._sample_range = [0, disp_to]

        msg = f"Data: {r['label']}  ({n_pts:,} samples)"
        print(f"[Visualizer] {msg}")
        self._status(msg)

    def load_test_data(self):
        """Fallback: pick a run/split directory directly (no trajectory selection)."""
        path = filedialog.askdirectory(
            title="Select dataset run directory (or split sub-directory)",
            initialdir=_TRAIN_DATA_DIR if os.path.isdir(_TRAIN_DATA_DIR) else _NN_DIR,
        )
        if not path:
            return
        self._data_path  = path
        self._traj_slice = None
        self._traj_label = os.path.basename(path)
        try:
            _dp    = os.path.abspath(path)
            _split = os.path.basename(_dp)
            _run   = os.path.dirname(_dp) if _split in ("train", "val", "test") else _dp
            meta   = _load_run_metadata(_run)
            split_key = _split if _split in ("train", "val", "test") else "test"
            n_samp = meta.get("split", {}).get("stats", {}).get(
                split_key, {}).get("n_samples", 0) if meta else 0
            if n_samp:
                disp_to = min(5000, n_samp)
                self._range_to.delete(0, "end")
                self._range_to.insert(0, str(disp_to))
                self._sample_range[1] = disp_to
            msg = f"Dataset: {os.path.basename(path)}"
            self._status(msg)
        except Exception as e:
            self._status(f"Warning: {e}")

    # ── Inference ──────────────────────────────────────────────────────────

    def run_all_inference(self):
        if self._data_path is None:
            messagebox.showinfo("Inference", "Load test data (.json or .npz) first.")
            return
        if not self._models:
            messagebox.showinfo("Inference", "Load at least one model first.")
            return

        n_total = len(self._models)
        print(f"\n[Visualizer] === Running inference on {n_total} model(s) ===")
        self._status(f"Running inference on {n_total} model(s) ...")
        n_ok = n_skip = n_err = 0

        for entry in self._models:
            name  = entry["name"]
            stale = entry["meta"].get("stale_reason")

            if stale:
                print(f"[Visualizer]   SKIP (stale) : {name}")
                print(f"[Visualizer]     Reason: {stale}")
                n_skip += 1
                continue

            print(f"[Visualizer]   Inferring   : {name}")
            try:
                seq_len    = entry["meta"].get("hyperparams", {}).get("seq_len", 50)
                norm_stats = entry["meta"].get("norm_stats", {})
                results    = run_inference(entry["model"], self._data_path,
                                           seq_len=seq_len, norm_stats=norm_stats,
                                           sample_slice=self._traj_slice)
                results["metrics"] = compute_metrics(results["pred"], results["target"])
                entry["data"] = results
                m = results["metrics"]
                print(f"[Visualizer]     RMSE={m['rmse_mean']:.5f}  "
                      f"R²={m['r2_mean']:.4f}  "
                      f"Pearson={m['pearson_r_mean']:.4f}")
                for j, (r, r2) in enumerate(zip(m["rmse"], m["r2"])):
                    print(f"[Visualizer]       J{j+1}: RMSE={r:.5f}  R²={r2:.4f}")
                n_ok += 1
            except Exception as e:
                print(f"[Visualizer]     ERROR: {e}")
                messagebox.showerror("Inference Error", f"'{name}':\n{e}")
                n_err += 1

        print(f"[Visualizer] === Done: {n_ok} OK  {n_skip} skipped (stale)  {n_err} errors ===\n")
        self._refresh_model_list()   # update RMSE/R² columns
        self._update_plot()
        idx = self._selected_model_idx()
        if idx is not None and idx < len(self._models):
            self._update_info_panel(self._models[idx])
        msg = f"Inference: {n_ok}/{n_total} OK"
        if n_skip:
            msg += f"  ({n_skip} stale — retrain to use)"
        self._status(msg)

    # ── Plotting ───────────────────────────────────────────────────────────

    def _apply_range(self):
        try:
            fr = int(self._range_from.get())
            to = int(self._range_to.get())
            if fr < to:
                self._sample_range = [fr, to]
        except ValueError:
            pass
        self._update_plot()

    def _update_plot(self):
        self._joint_mask = [v.get() for v in self._joint_vars]
        active_joints    = [j for j, m in enumerate(self._joint_mask) if m]
        if not active_joints:
            return

        self._fig.clear()
        n_cols = min(len(active_joints), 3)
        n_rows = math.ceil(len(active_joints) / n_cols)
        self._axes = [self._fig.add_subplot(n_rows, n_cols, k + 1)
                      for k in range(len(active_joints))]

        s0, s1 = self._sample_range

        # Determine whether any model exposes a time array (from t.csv)
        _has_time = any(
            e.get("data") is not None and e["data"].get("t") is not None
            for e in self._models
        )
        xlabel = "time (s)" if _has_time else "sample index"
        traj_suffix = f"  [{self._traj_label}]" if self._traj_label else ""

        for ax_idx, j in enumerate(active_joints):
            ax = self._axes[ax_idx]

            # Ground truth (from the first model that has inference results)
            gt_plotted = False
            for entry in self._models:
                if entry["data"] is None or gt_plotted:
                    continue
                target = entry["data"]["target"]
                n      = target.shape[0]
                sl     = slice(min(s0, n - 1), min(s1, n))
                t_e    = entry["data"].get("t")
                x_gt   = (t_e[sl] if (t_e is not None and len(t_e) == n)
                          else np.arange(sl.start, min(sl.stop, n)))
                _gt_kw = {k: v for k, v in self._gt_style.items()
                          if k not in ("sg_window", "sg_poly")}
                ax.plot(x_gt, target[sl, j], label="Ground Truth", **_gt_kw)
                gt_plotted = True
                break

            # Model predictions
            for entry in self._models:
                if not entry["enabled"] or entry["data"] is None:
                    continue
                pred = entry["data"]["pred"]
                n    = pred.shape[0]
                sl   = slice(min(s0, n - 1), min(s1, n))
                t_e  = entry["data"].get("t")
                x_pr = (t_e[sl] if (t_e is not None and len(t_e) == n)
                        else np.arange(sl.start, min(sl.stop, n)))
                m    = entry["data"].get("metrics", {})
                rmse = m.get("rmse", [0.0] * 5)[j] if m else 0.0
                r2   = m.get("r2",   [0.0] * 5)[j] if m else 0.0
                lbl  = f"{entry['name']}  RMSE={rmse:.4f} R²={r2:.3f}"
                y_pr = pred[sl, j]
                sg_w = entry["style"].get("sg_window", 0)
                sg_p = entry["style"].get("sg_poly",   3)
                if sg_w > 1 and len(y_pr) > sg_w:
                    sg_w = sg_w | 1                        # ensure odd
                    sg_p = min(sg_p, sg_w - 1)             # poly < window
                    try:
                        y_pr = _savgol(y_pr, sg_w, sg_p)
                    except Exception:
                        pass
                plot_kw = {k: v for k, v in entry["style"].items()
                           if k not in ("sg_window", "sg_poly")}
                ax.plot(x_pr, y_pr, label=lbl, **plot_kw)

            ax.set_title(f"{JOINT_NAMES[j]}{traj_suffix}", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("torque (N·m)", fontsize=8)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7, loc="best")
            ax.grid(True, alpha=0.28)

        self._fig.tight_layout()
        self._canvas.draw()

    # ── Comparison Window ──────────────────────────────────────────────────

    def open_comparison(self):
        infos = []
        for entry in self._models:
            m = (entry["data"].get("metrics", {})
                 if entry["data"] is not None else {})
            infos.append({
                "name":      entry["name"],
                "metrics":   m,
                "color":     entry["style"]["color"],
                "hparams":   entry["meta"].get("hparams", {}),
                "model_dir": entry.get("model_dir", ""),
            })
        if not any(i["metrics"] for i in infos):
            messagebox.showinfo("Compare", "Run inference first — no metrics available yet.")
            return
        ComparisonWindow(self, infos)

    # ── Export ─────────────────────────────────────────────────────────────

    def export_png(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*")],
            title="Export Plot",
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
            self._status(f"Exported → {path}")

    # ── Help ───────────────────────────────────────────────────────────────

    def show_help(self):
        msg = (
            "NEURAL NETWORK TORQUE PREDICTION — VISUALIZER\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "QUICK START\n"
            "1. Click '📂 Browse Registry' — shows all trained models from\n"
            "   models_registry.yaml sorted by RMSE. Select one or more.\n"
            "2. Click 'Browse Data' — pick a preprocessed run directory,\n"
            "   choose a split (train/val/test), and select a single trajectory\n"
            "   segment OR 'ALL' to use the full split.\n"
            "3. Click '▶ Run Inference' — compute predictions for all models.\n"
            "4. Click 'Comparison Window' for detailed charts and tables.\n\n"
            "TRAJECTORY SELECTION (Browse Data)\n"
            "• Boundaries are auto-detected from t.csv time jumps / resets.\n"
            "• Each row shows its sample count, start/end timestamps, duration.\n"
            "• Selecting a trajectory slices inference to those samples only;\n"
            "  the plot x-axis becomes elapsed time (s) from that trajectory.\n\n"
            "LEFT PANEL\n"
            "• Model Info: epochs, training time, device, per-joint RMSE bars\n"
            "• Edit Style: customise line colour, style, width per model\n"
            "• Toggle:     hide/show a model without removing it\n"
            "• Joint Filter: show only selected joints\n"
            "• Sample Range: zoom into a timestep subset of the inferred data\n\n"
            "COMPARISON WINDOW TABS\n"
            "• Metrics Table:    full per-joint RMSE/NRMSE/MAE/R²/Pearson table\n"
            "• RMSE Comparison:  sorted bar chart with best-model line\n"
            "• R² & Pearson:     mean R² bars + per-joint Pearson r heatmap\n"
            "• NRMSE per Joint:  grouped bar chart per joint\n"
            "• Training History: loss curves from training_history.csv\n"
            "• Hyperparameters:  side-by-side HP table (yellow = differs)\n\n"
            "KEYBOARD (matplotlib toolbar)\n"
            "  Zoom: drag magnifier.  Pan: drag hand.  Home: reset view."
        )
        messagebox.showinfo("Help", msg)

    # ── Utility ────────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self._status_var.set(msg)
        self.update_idletasks()

    def on_close(self):
        self.quit()
        self.destroy()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("=" * 60)
    print("  Neural Network Torque Prediction — Model Visualizer")
    print(f"  PyTorch {torch.__version__}   CUDA: {torch.cuda.is_available()}")
    print(f"  Registry : {REGISTRY_FILE}")
    print("=" * 60)
    app = VisualizerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
