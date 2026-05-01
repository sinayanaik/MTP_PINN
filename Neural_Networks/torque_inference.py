#!/usr/bin/env python3
"""GUI-first torque model inference, comparison, plotting, and export.

Run from the repository root with:

    PYTHONPATH=. python3 Neural_Networks/torque_inference.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mtp-pinn")

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from Neural_Networks.loader import (
    ACTIVE_JOINTS,
    CSV_FILTERED_Q,
    CSV_FILTERED_QD,
    CSV_FILTERED_QDD,
    CSV_FILTERED_TAU_DECOMPOSED,
    CSV_FILTERED_TAU_MEASURED,
    CSV_T,
)
from Neural_Networks.models.shared.metrics_numpy import compute_metrics
from Neural_Networks.models.torque_models import (
    BlackBoxFNN,
    PhysicsRegularizedFNN,
    ResidualCorrectionFNN,
    build_mlp,
    reduce_physics_to_total,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_STANDALONE_MODEL_ROOT = ROOT / "Trained_Models"
DEFAULT_GRID_MODEL_ROOT = ROOT / "Trained_Models_Grid"
DEFAULT_MODEL_ROOTS = [DEFAULT_STANDALONE_MODEL_ROOT, DEFAULT_GRID_MODEL_ROOT]
DEFAULT_DATA_ROOT = ROOT / "train_data"
DEFAULT_RAW_ROOT = ROOT.parent / "raw_samples"
EDR_DIR = ROOT / "models" / "Equivariant-Decomposed-Residual"
CATALOG_FILENAMES = ("trajectories_catalog.csv", "trajectory_catalog.csv")
TRAJECTORY_ALL_LABEL = "[All catalog trajectories]"
JOINT_LABELS = ["J1 yaw", "J2 shoulder", "J3 elbow", "J4 wrist", "J5 wrist roll"]
MODEL_COLORS = {
    "BlackBoxFNN": "#e41a1c",
    "FNN": "#e41a1c",
    "PhysicsRegularizedFNN": "#148414",
    "ResidualCorrectionFNN": "#8a148a",
    "EDR": "#0047ff",
}
MODEL_ALIASES = {
    "BlackBoxFNN": "FNN",
    "PhysicsRegularizedFNN": "PINN_FNN",
    "ResidualCorrectionFNN": "Residual",
    "EDR": "EDR",
}


@dataclass(frozen=True)
class ModelRecord:
    label: str
    run_dir: Path
    model_path: Path
    metadata_path: Path
    model_type: str
    run_id: str
    data_run_dir: Path | None
    rmse: float | None
    r2: float | None
    rmse_by_split: dict[str, float | None]
    r2_by_split: dict[str, float | None]
    epochs: int | None
    data_fraction: float | None
    physics_tag: str
    data_available: bool


@dataclass(frozen=True)
class ModelGroup:
    label: str
    roots: tuple[Path, ...]
    kind: str


@dataclass
class PredictionResult:
    record: ModelRecord
    pred: np.ndarray
    target: np.ndarray
    residual: np.ndarray
    time: np.ndarray
    dataset_sample_indices: np.ndarray
    source_file_per_sample: np.ndarray | None
    trajectory_label: str
    raw_paths: list[Path]
    physics_total: np.ndarray | None
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TrajectoryEntry:
    source_file: str
    geometry_type: str
    radius_mm: float | None
    planner: str
    ctrl_hz: float | None
    fb_hz: float | None
    duration_sec: float | None
    n_samples: int
    ee_rms_err_mm: float | None
    start_idx: int
    end_idx_exclusive: int
    raw_path: Path | None
    raw_exists: bool


class LegacyPhysicsRegularizedFNN(torch.nn.Module):
    """Compatibility wrapper for older PhysReg checkpoints.

    Historical grid checkpoints used the physics signal as a training loss only,
    not as an additive term in ``forward``. Some checkpoints still contain
    calibration parameters; they are kept so state loading is exact, but are not
    used for prediction.
    """

    def __init__(
        self,
        hidden_layers: list[int],
        dropout: float,
        activation: str,
        in_dim: int,
        compat_params: str | None = None,
    ) -> None:
        super().__init__()
        self.n_joints = ACTIVE_JOINTS
        self.in_dim = in_dim
        self.net = build_mlp(in_dim, hidden_layers, ACTIVE_JOINTS, activation, dropout)
        if compat_params == "tau":
            self.tau_scale = torch.nn.Parameter(torch.ones(ACTIVE_JOINTS))
            self.tau_bias = torch.nn.Parameter(torch.zeros(ACTIVE_JOINTS))
        elif compat_params == "cal":
            self.cal_scale = torch.nn.Parameter(torch.ones(ACTIVE_JOINTS))
            self.cal_bias = torch.nn.Parameter(torch.zeros(ACTIVE_JOINTS))

    def forward(self, features: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        if self.in_dim == ACTIVE_JOINTS * 3:
            x = features
        elif self.in_dim == ACTIVE_JOINTS * 4:
            if physics is None:
                raise ValueError("LegacyPhysicsRegularizedFNN requires decomposed physics.")
            tau_phys = reduce_physics_to_total(physics, self.n_joints)
            x = torch.cat([features, tau_phys], dim=-1)
        else:
            raise ValueError(f"unsupported legacy PhysReg input dim: {self.in_dim}")
        return self.net(x)


class LegacyResidualCorrectionFNN(torch.nn.Module):
    """Compatibility wrapper for older residual checkpoints."""

    def __init__(self, hidden_layers: list[int], dropout: float, activation: str, in_dim: int) -> None:
        super().__init__()
        self.n_joints = ACTIVE_JOINTS
        self.in_dim = in_dim
        self.net = build_mlp(in_dim, hidden_layers, ACTIVE_JOINTS, activation, dropout)

    def forward(self, features: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        if physics is None:
            raise ValueError("LegacyResidualCorrectionFNN requires decomposed physics.")
        tau_phys = reduce_physics_to_total(physics, self.n_joints)
        if self.in_dim == ACTIVE_JOINTS * 3:
            x = features
        elif self.in_dim == ACTIVE_JOINTS * 4:
            x = torch.cat([features, tau_phys], dim=-1)
        else:
            raise ValueError(f"unsupported legacy residual input dim: {self.in_dim}")
        return tau_phys + self.net(x)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _safe_torch_load(path: Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_state_prefix(key: str) -> str:
    for prefix in ("_orig_mod.", "module."):
        while key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _checkpoint_model_state(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = ckpt.get("model_state") or {}
    return {_strip_state_prefix(str(k)): v for k, v in state.items()}


def _first_net_input_dim(state: dict[str, torch.Tensor]) -> int | None:
    weight = state.get("net.0.weight")
    if hasattr(weight, "shape") and len(weight.shape) == 2:
        return int(weight.shape[1])
    return None


def _metric_float_from_section(meta: dict[str, Any], section: str, *keys: str) -> float | None:
    d = meta.get(section) or {}
    for key in keys:
        if key in d and d[key] is not None:
            try:
                return float(d[key])
            except (TypeError, ValueError):
                pass
    return None


def _stored_metric(meta: dict[str, Any], split: str, base_key: str) -> float | None:
    section_by_split = {
        "avg": "metrics",
        "val": "val_metrics",
        "test": "test_metrics",
        "train": "train_metrics",
    }
    section = section_by_split[split]
    prefix = "avg" if split == "avg" else split
    value = _metric_float_from_section(meta, section, base_key, f"{prefix}_{base_key}")
    if value is None and split != "avg":
        value = _metric_float_from_section(meta, "metrics", f"{prefix}_{base_key}")
    return value


def _stored_metric_map(meta: dict[str, Any], base_key: str) -> dict[str, float | None]:
    return {split: _stored_metric(meta, split, base_key) for split in ("avg", "val", "test", "train")}


def _fmt_optional_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _model_alias(model_type: str) -> str:
    return MODEL_ALIASES.get(model_type, model_type)


def _model_color(model_type: str, index: int = 0) -> str:
    fallback = ["#0047ff", "#148414", "#8a148a", "#e41a1c", "#ff7f00", "#4d4d4d"]
    return MODEL_COLORS.get(model_type, fallback[index % len(fallback)])


def _physics_tag(hp: dict[str, Any]) -> str:
    if "physics_weight" in hp:
        return f"pw={hp['physics_weight']}"
    if "alpha_reg_weight" in hp:
        return f"alpha={hp['alpha_reg_weight']}"
    if "lambda_correction_reg" in hp:
        return f"creg={hp['lambda_correction_reg']}"
    return "-"


def _valid_data_dir(path: Path) -> bool:
    return (path / "metadata.json").is_file() and all((path / s).is_dir() for s in ("train", "val", "test"))


def _resolve_existing_data_dir(data_run_dir: Path | None) -> Path | None:
    if data_run_dir is None:
        return None
    if data_run_dir.is_dir() and _valid_data_dir(data_run_dir):
        return data_run_dir
    candidate = DEFAULT_DATA_ROOT / data_run_dir.name
    if candidate.is_dir() and _valid_data_dir(candidate):
        return candidate
    return None


def _has_model_metadata(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        next(path.rglob("metadata.yaml"))
        return True
    except StopIteration:
        return False


def discover_model_groups(extra_groups: list[ModelGroup] | None = None) -> list[ModelGroup]:
    groups: list[ModelGroup] = []
    if DEFAULT_GRID_MODEL_ROOT.is_dir():
        for data_group in sorted(DEFAULT_GRID_MODEL_ROOT.iterdir(), key=lambda p: p.name):
            if data_group.is_dir() and _has_model_metadata(data_group):
                groups.append(ModelGroup(f"Grid: {data_group.name}", (data_group,), "grid"))
    if _has_model_metadata(DEFAULT_STANDALONE_MODEL_ROOT):
        groups.append(ModelGroup("Standalone: Trained_Models", (DEFAULT_STANDALONE_MODEL_ROOT,), "standalone"))
    if extra_groups:
        groups.extend(extra_groups)
    return groups


def discover_models(model_roots: list[Path]) -> list[ModelRecord]:
    records: list[ModelRecord] = []
    seen: set[Path] = set()
    for root in model_roots:
        if not root.is_dir():
            continue
        for meta_path in root.rglob("metadata.yaml"):
            run_dir = meta_path.parent
            model_path = run_dir / "model.pt"
            if not model_path.is_file():
                model_path = run_dir / "model_final.pt"
            if not model_path.is_file() or model_path in seen:
                continue
            seen.add(model_path)
            try:
                meta = _read_yaml(meta_path)
            except Exception:
                continue
            model_type = str(meta.get("model_type") or run_dir.parent.name)
            run_id = str(meta.get("run_id") or run_dir.name)
            data_raw = meta.get("data_run_dir") or meta.get("run_dir")
            data_run_dir = Path(data_raw).expanduser() if data_raw else None
            rmse_by_split = _stored_metric_map(meta, "rmse_pooled")
            r2_by_split = _stored_metric_map(meta, "r2_overall")
            rmse = rmse_by_split.get("avg") or rmse_by_split.get("test") or rmse_by_split.get("val")
            r2 = r2_by_split.get("avg") or r2_by_split.get("test") or r2_by_split.get("val")
            epochs = meta.get("epochs_trained")
            try:
                epochs = int(epochs) if epochs is not None else None
            except (TypeError, ValueError):
                epochs = None
            hp = meta.get("hyperparams") or meta.get("exhaustive_hyperparams") or {}
            try:
                data_fraction = float(hp["data_train_fraction"]) if "data_train_fraction" in hp else None
            except (TypeError, ValueError):
                data_fraction = None
            short = run_id
            if len(short) > 86:
                short = short[:83] + "..."
            label = f"{model_type:<24} RMSE={rmse:.5f}  {short}" if rmse else f"{model_type:<24} {short}"
            records.append(
                ModelRecord(
                    label,
                    run_dir,
                    model_path,
                    meta_path,
                    model_type,
                    run_id,
                    data_run_dir,
                    rmse,
                    r2,
                    rmse_by_split,
                    r2_by_split,
                    epochs,
                    data_fraction,
                    _physics_tag(hp),
                    _resolve_existing_data_dir(data_run_dir) is not None,
                )
            )
    records.sort(key=lambda r: (float("inf") if r.rmse is None else r.rmse, r.model_type, r.run_id))
    return records


def discover_datasets(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        return []
    out = []
    for p in sorted(data_root.iterdir()):
        if (p / "metadata.json").is_file() and all((p / s).is_dir() for s in ("train", "val", "test")):
            out.append(p)
    return out


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_field(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _catalog_path(split_dir: Path) -> Path | None:
    for name in CATALOG_FILENAMES:
        path = split_dir / name
        if path.is_file():
            return path
    return None


def _raw_dir_for_data_run(data_run_dir: Path) -> Path:
    meta_path = data_run_dir / "metadata.json"
    if meta_path.is_file():
        try:
            with meta_path.open("r") as f:
                raw = json.load(f).get("raw_dir")
            if raw:
                raw_dir = Path(raw).expanduser()
                if not raw_dir.is_absolute():
                    raw_dir = (data_run_dir / raw_dir).resolve()
                return raw_dir
        except Exception:
            pass
    return DEFAULT_RAW_ROOT


def _resolve_raw_sample_path(raw_dir: Path, source_file: str) -> Path | None:
    if not source_file:
        return None
    source_path = Path(source_file)
    if source_path.is_absolute():
        candidates = [source_path]
    else:
        candidates = [raw_dir / source_file]
        if raw_dir != DEFAULT_RAW_ROOT:
            candidates.append(DEFAULT_RAW_ROOT / source_file)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else None


def load_trajectory_catalog(data_run_dir: Path, split: str) -> list[TrajectoryEntry]:
    split_dir = data_run_dir / split
    path = _catalog_path(split_dir)
    if path is None:
        return []
    raw_dir = _raw_dir_for_data_run(data_run_dir)
    rows: list[TrajectoryEntry] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            source_file = str(row.get("source_file") or "").strip()
            raw_path = _resolve_raw_sample_path(raw_dir, source_file)
            start_idx = _int_field(row.get("start_idx"), 0)
            end_idx = _int_field(row.get("end_idx_exclusive"), start_idx)
            rows.append(
                TrajectoryEntry(
                    source_file=source_file,
                    geometry_type=str(row.get("geometry_type") or "").strip() or "unknown",
                    radius_mm=_optional_float(row.get("radius_mm")),
                    planner=str(row.get("planner") or "").strip(),
                    ctrl_hz=_optional_float(row.get("ctrl_hz")),
                    fb_hz=_optional_float(row.get("fb_hz")),
                    duration_sec=_optional_float(row.get("duration_sec")),
                    n_samples=_int_field(row.get("n_samples"), max(0, end_idx - start_idx)),
                    ee_rms_err_mm=_optional_float(row.get("ee_rms_err_mm")),
                    start_idx=start_idx,
                    end_idx_exclusive=end_idx,
                    raw_path=raw_path,
                    raw_exists=bool(raw_path and raw_path.is_file()),
                )
            )
    return rows


def _trajectory_choice_label(index: int, entry: TrajectoryEntry) -> str:
    radius = f" r={entry.radius_mm:.0f}mm" if entry.radius_mm is not None else ""
    planner = f" {entry.planner}" if entry.planner else ""
    raw = "" if entry.raw_exists else " [raw missing]"
    return f"{index:03d} {entry.geometry_type}{radius}{planner} N={entry.n_samples:,} {entry.source_file}{raw}"


def _select_catalog_entries(
    catalog: list[TrajectoryEntry],
    trajectory_sources: list[str],
) -> list[TrajectoryEntry]:
    if not trajectory_sources:
        return catalog
    wanted = set(trajectory_sources)
    entries = [entry for entry in catalog if entry.source_file in wanted]
    found = {entry.source_file for entry in entries}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"trajectory source(s) not present in catalog: {', '.join(missing)}")
    return entries


def _catalog_indices(entries: list[TrajectoryEntry], n_samples_total: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    bad: list[str] = []
    for entry in entries:
        s = int(entry.start_idx)
        e = int(entry.end_idx_exclusive)
        if s < 0 or e <= s or e > n_samples_total:
            bad.append(f"{entry.source_file} [{s}, {e})")
            continue
        chunks.append(np.arange(s, e, dtype=np.int64))
    if bad:
        raise ValueError(
            "catalog contains invalid trajectory bounds for this split: "
            + "; ".join(bad[:4])
        )
    if not chunks:
        raise ValueError("no samples selected from trajectory catalog")
    return np.concatenate(chunks)


def _overlapping_catalog_entries(
    catalog: list[TrajectoryEntry],
    indices: np.ndarray,
) -> list[TrajectoryEntry]:
    if not catalog or indices.size == 0:
        return []
    lo = int(indices.min())
    hi = int(indices.max()) + 1
    return [entry for entry in catalog if entry.start_idx < hi and entry.end_idx_exclusive > lo]


def _source_files_for_indices(catalog: list[TrajectoryEntry], indices: np.ndarray) -> np.ndarray | None:
    if not catalog or indices.size == 0:
        return None
    out = np.full(len(indices), "", dtype=object)
    for entry in catalog:
        mask = (indices >= entry.start_idx) & (indices < entry.end_idx_exclusive)
        out[mask] = entry.source_file
    return out


def _trajectory_result_label(entries: list[TrajectoryEntry], catalog_mode: bool) -> str:
    if not entries:
        return "Sample/time window"
    if catalog_mode and len(entries) > 1:
        return f"All {len(entries)} catalog trajectories"
    if len(entries) == 1:
        entry = entries[0]
        return f"{entry.geometry_type}: {entry.source_file}"
    return f"Window across {len(entries)} catalog trajectories"


def _load_csv(split_dir: Path, name: str) -> np.ndarray:
    npy = split_dir / (Path(name).stem + ".npy")
    if npy.is_file():
        return np.load(npy)
    return np.loadtxt(split_dir / name, delimiter=",", skiprows=1, dtype=np.float32)


def _contiguous_indices(n: int, start: int, max_samples: int) -> np.ndarray:
    start = max(0, min(int(start), max(0, n - 1)))
    if max_samples <= 0:
        stop = n
    else:
        stop = min(n, start + int(max_samples))
    if stop <= start:
        stop = min(n, start + 1)
    return np.arange(start, stop, dtype=np.int64)


def _time_range_indices(time: np.ndarray, start_t: float, end_t: float) -> np.ndarray:
    lo, hi = sorted((float(start_t), float(end_t)))
    raw = np.flatnonzero((time >= lo) & (time <= hi))
    if raw.size == 0:
        return raw.astype(np.int64)
    breaks = np.flatnonzero(np.diff(raw) != 1) + 1
    groups = np.split(raw, breaks)
    longest = max(groups, key=len)
    return longest.astype(np.int64)


def _norm_from_checkpoint(ckpt: dict[str, Any], data_run_dir: Path) -> dict[str, np.ndarray]:
    norm = ckpt.get("norm_stats") or {}
    if not norm:
        with (data_run_dir / "metadata.json").open("r") as f:
            norm = (json.load(f).get("normalisation") or {})
    required = ("mean_q", "std_q", "mean_qd", "std_qd", "mean_qdd", "std_qdd", "mean_tau", "std_tau")
    missing = [k for k in required if k not in norm]
    if missing:
        raise ValueError(f"missing normalisation keys in checkpoint/dataset: {', '.join(missing)}")
    out = {k: np.asarray(norm[k], dtype=np.float32) for k in required}
    for k in ("std_q", "std_qd", "std_qdd", "std_tau"):
        out[k] = np.clip(out[k], 1e-8, None)
    return out


def _build_eval_arrays(data_run_dir: Path, split: str, indices: np.ndarray, norm: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_dir = data_run_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split not found: {split_dir}")
    q = _load_csv(split_dir, CSV_FILTERED_Q)[indices]
    qd = _load_csv(split_dir, CSV_FILTERED_QD)[indices]
    qdd = _load_csv(split_dir, CSV_FILTERED_QDD)[indices]
    target = _load_csv(split_dir, CSV_FILTERED_TAU_MEASURED)[indices]
    physics_raw = _load_csv(split_dir, CSV_FILTERED_TAU_DECOMPOSED)[indices]
    t_path = split_dir / CSV_T
    time = _load_csv(split_dir, CSV_T)[indices].reshape(-1) if t_path.is_file() else np.arange(len(indices), dtype=np.float32)

    features = np.concatenate(
        [
            (q - norm["mean_q"]) / norm["std_q"],
            (qd - norm["mean_qd"]) / norm["std_qd"],
            (qdd - norm["mean_qdd"]) / norm["std_qdd"],
        ],
        axis=-1,
    ).astype(np.float32)
    per_comp_mean = np.tile(norm["mean_tau"] / 4.0, 4)
    per_comp_std = np.tile(norm["std_tau"], 4)
    physics = ((physics_raw - per_comp_mean) / per_comp_std).astype(np.float32)
    physics_total = physics_raw.reshape(len(indices), 4, ACTIVE_JOINTS).sum(axis=1)
    return features, physics, target.astype(np.float32), time.astype(np.float32), physics_total.astype(np.float32)


def _load_edr_class():
    sys.path.insert(0, str(EDR_DIR))
    spec = importlib.util.spec_from_file_location("mtp_edr_model", EDR_DIR / "edr_model.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import EDRModel from {EDR_DIR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EDRModel


def build_model(ckpt: dict[str, Any]) -> torch.nn.Module:
    cls_name = str(ckpt.get("model_class") or "")
    hp = ckpt.get("hparams") or {}
    state = _checkpoint_model_state(ckpt)
    net_in_dim = _first_net_input_dim(state)
    model_type = cls_name.lower()
    if "blackbox" in model_type or cls_name == "BlackBoxFNN":
        return BlackBoxFNN(
            hidden_layers=list(hp.get("hidden_layers") or [256, 512, 256]),
            dropout=float(hp.get("dropout", 0.1)),
            activation=str(hp.get("activation", "silu")),
        )
    if "physicsregularized" in model_type:
        if net_in_dim in (ACTIVE_JOINTS * 3, ACTIVE_JOINTS * 4):
            compat_params = None
            if "tau_scale" in state or "tau_bias" in state:
                compat_params = "tau"
            elif "cal_scale" in state or "cal_bias" in state:
                compat_params = "cal"
            return LegacyPhysicsRegularizedFNN(
                hidden_layers=list(hp.get("hidden_layers") or [256, 512, 256]),
                dropout=float(hp.get("dropout", 0.1)),
                activation=str(hp.get("activation", "silu")),
                in_dim=net_in_dim,
                compat_params=compat_params,
            )
        return PhysicsRegularizedFNN(
            hidden_layers=list(hp.get("hidden_layers") or [128, 256, 128]),
            dropout=float(hp.get("dropout", 0.2)),
            activation=str(hp.get("activation", "silu")),
        )
    if "residualcorrection" in model_type:
        if net_in_dim in (ACTIVE_JOINTS * 3, ACTIVE_JOINTS * 4):
            return LegacyResidualCorrectionFNN(
                hidden_layers=list(hp.get("hidden_layers") or [256, 512, 256]),
                dropout=float(hp.get("dropout", 0.1)),
                activation=str(hp.get("activation", "silu")),
                in_dim=net_in_dim,
            )
        return ResidualCorrectionFNN(
            hidden_layers=list(hp.get("hidden_layers") or [128, 256, 128]),
            dropout=float(hp.get("dropout", 0.2)),
            activation=str(hp.get("activation", "silu")),
            correction_scale=float(hp.get("correction_scale", 0.5)),
        )
    if "edr" in model_type or cls_name == "EDRModel":
        EDRModel = _load_edr_class()
        norm_stats = ckpt.get("norm_stats") or {}
        q_mean = hp.get("_q_mean")
        q_std = hp.get("_q_std")
        if bool(hp.get("use_trig_features", False)) and (q_mean is None or q_std is None):
            q_mean = norm_stats.get("mean_q")
            q_std = norm_stats.get("std_q")
        model = EDRModel(
            n_joints=ACTIVE_JOINTS,
            gravity_hidden=list(hp.get("gravity_hidden") or [64, 64]),
            inertia_hidden=list(hp.get("inertia_hidden") or [64, 64]),
            coriolis_hidden=list(hp.get("coriolis_hidden") or [64, 64]),
            friction_hidden=list(hp.get("friction_hidden") or [32, 32]),
            activation=str(hp.get("activation", "silu")),
            correction_dropout=float(hp.get("correction_dropout", 0.0)),
            q_mean=q_mean,
            q_std=q_std,
        )
        model.set_phase(2)
        return model
    raise ValueError(f"unsupported model class in checkpoint: {cls_name!r}")


def run_inference(
    record: ModelRecord,
    split: str,
    start_sample: int,
    max_samples: int,
    start_time: float | None,
    end_time: float | None,
    override_data_dir: Path | None,
    trajectory_sources: list[str] | None,
) -> PredictionResult:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = _safe_torch_load(record.model_path, device)
    data_run_dir = override_data_dir or _resolve_existing_data_dir(record.data_run_dir)
    if data_run_dir is None:
        missing = str(record.data_run_dir) if record.data_run_dir is not None else "(not recorded)"
        raise ValueError(
            f"{record.run_id}: data_run_dir is not available locally:\n{missing}\n\n"
            "Choose an existing Dataset in the Evaluation panel, or select a checkpoint marked data=ok."
        )
    data_run_dir = Path(data_run_dir)
    if not data_run_dir.is_dir():
        raise FileNotFoundError(f"{record.run_id}: data_run_dir not found: {data_run_dir}")
    split_dir = data_run_dir / split
    n = len(_load_csv(split_dir, CSV_FILTERED_TAU_MEASURED))
    t_path = split_dir / CSV_T
    time_all = _load_csv(split_dir, CSV_T).reshape(-1) if t_path.is_file() else np.arange(n, dtype=np.float32)
    catalog = load_trajectory_catalog(data_run_dir, split)
    catalog_mode = trajectory_sources is not None
    if catalog_mode:
        if not catalog:
            raise FileNotFoundError(f"{data_run_dir.name}/{split}: no trajectories_catalog.csv found.")
        selected_entries = _select_catalog_entries(catalog, trajectory_sources)
        missing_raw = [entry.source_file for entry in selected_entries if not entry.raw_exists]
        if missing_raw:
            raise FileNotFoundError(
                f"{data_run_dir.name}/{split}: catalog trajectory raw JSON missing under "
                f"{_raw_dir_for_data_run(data_run_dir)}:\n" + "\n".join(missing_raw[:8])
            )
        indices = _catalog_indices(selected_entries, n)
    else:
        if start_time is not None and end_time is not None:
            indices = _time_range_indices(time_all, start_time, end_time)
            if indices.size == 0:
                raise ValueError(f"No contiguous samples found in time range {start_time:g} to {end_time:g}.")
        else:
            indices = _contiguous_indices(n, start_sample, max_samples)
        selected_entries = _overlapping_catalog_entries(catalog, indices)
    norm = _norm_from_checkpoint(ckpt, data_run_dir)
    features, physics, target, time, physics_total = _build_eval_arrays(data_run_dir, split, indices, norm)

    model = build_model(ckpt).to(device)
    state = _checkpoint_model_state(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{record.run_id}: checkpoint mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.eval()
    preds_norm: list[np.ndarray] = []
    batch_size = 4096 if device.type == "cuda" else 1024
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            xb = torch.from_numpy(features[start:end]).to(device)
            pb = torch.from_numpy(physics[start:end]).to(device)
            yb = model(xb, pb).detach().cpu().numpy()
            preds_norm.append(yb)
    pred_norm = np.concatenate(preds_norm, axis=0)
    pred = pred_norm * norm["std_tau"] + norm["mean_tau"]
    metrics = compute_metrics(pred, target)
    residual = pred - target
    raw_paths = [entry.raw_path for entry in selected_entries if entry.raw_path is not None and entry.raw_exists]
    return PredictionResult(
        record=record,
        pred=pred,
        target=target,
        residual=residual,
        time=time,
        dataset_sample_indices=indices,
        source_file_per_sample=_source_files_for_indices(catalog, indices),
        trajectory_label=_trajectory_result_label(selected_entries, catalog_mode),
        raw_paths=raw_paths,
        physics_total=physics_total,
        metrics=metrics,
    )


class TorqueInferenceApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Torque Model Inference and Comparison")
        self.geometry("1450x900")
        self.minsize(1120, 720)
        self.records: list[ModelRecord] = []
        self.filtered_records: list[ModelRecord] = []
        self.selected_model_paths: set[Path] = set()
        self.legend_label_vars: dict[str, tk.StringVar] = {}
        self.model_groups: list[ModelGroup] = []
        self.extra_model_groups: list[ModelGroup] = []
        self.results: list[PredictionResult] = []
        self.trajectory_catalog: list[TrajectoryEntry] = []
        self.trajectory_choice_by_label: dict[str, str | None] = {}
        self.joint_vars = [tk.BooleanVar(value=True) for _ in range(ACTIVE_JOINTS)]
        self._build_style()
        self._build_ui()
        self.refresh_models()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(bg="#f4f5f7")
        style.configure("TFrame", background="#f4f5f7")
        style.configure("TLabelframe", background="#f4f5f7")
        style.configure("TLabelframe.Label", background="#f4f5f7", font=("TkDefaultFont", 10, "bold"))
        style.configure("TButton", padding=(8, 5))
        style.configure("Accent.TButton", padding=(10, 6), font=("TkDefaultFont", 10, "bold"))
        style.configure("TLabel", padding=(2, 2))
        style.configure("Treeview", rowheight=24)
        style.configure("Treeview.Heading", font=("TkDefaultFont", 9, "bold"))

    def _build_ui(self) -> None:
        root = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True)

        left_shell = ttk.Frame(root, padding=0)
        root.add(left_shell, weight=0)
        right = ttk.Frame(root, padding=(0, 8, 8, 8))
        root.add(right, weight=1)

        left_canvas = tk.Canvas(left_shell, width=690, highlightthickness=0, bg="#f4f5f7")
        left_scroll = ttk.Scrollbar(left_shell, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        left = ttk.Frame(left_canvas, padding=8)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda e: left_canvas.itemconfigure(left_window, width=e.width))

        ttk.Label(left, text="Models").pack(anchor="w")
        filters = ttk.LabelFrame(left, text="Filter", padding=8)
        filters.pack(fill=tk.X, pady=(4, 8))
        self.group_var = tk.StringVar()
        self.search_var = tk.StringVar()
        self.type_filter_var = tk.StringVar(value="All")
        self.max_rmse_filter_var = tk.StringVar()
        self.top_n_filter_var = tk.StringVar(value="")
        self.runnable_only_var = tk.BooleanVar(value=True)
        ttk.Label(filters, text="Model group").grid(row=0, column=0, sticky="w")
        self.group_box = ttk.Combobox(filters, textvariable=self.group_var, values=[], state="readonly", width=42)
        self.group_box.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=2)
        self.group_box.bind("<<ComboboxSelected>>", lambda _e: self._on_model_group_selected())
        ttk.Label(filters, text="Search").grid(row=1, column=0, sticky="w")
        search_entry = ttk.Entry(filters, textvariable=self.search_var, width=24)
        search_entry.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=2)
        ttk.Label(filters, text="Type").grid(row=2, column=0, sticky="w")
        self.type_filter_box = ttk.Combobox(filters, textvariable=self.type_filter_var, values=["All"], state="readonly", width=22)
        self.type_filter_box.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=2)
        ttk.Label(filters, text="Max RMSE").grid(row=3, column=0, sticky="w")
        ttk.Entry(filters, textvariable=self.max_rmse_filter_var, width=12).grid(row=3, column=1, sticky="ew", padx=(6, 0), pady=2)
        ttk.Label(filters, text="Show top").grid(row=4, column=0, sticky="w")
        ttk.Entry(filters, textvariable=self.top_n_filter_var, width=12).grid(row=4, column=1, sticky="ew", padx=(6, 0), pady=2)
        ttk.Checkbutton(filters, text="Runnable data only", variable=self.runnable_only_var, command=self.apply_model_filter).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(filters, text="Apply Filter", command=self.apply_model_filter).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.filter_summary_var = tk.StringVar(value="Filter: runnable data only; no search/type/RMSE/top limit")
        ttk.Label(filters, textvariable=self.filter_summary_var, wraplength=620).grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))
        filters.columnconfigure(1, weight=1)
        self.search_var.trace_add("write", lambda *_: self.apply_model_filter())
        self.type_filter_var.trace_add("write", lambda *_: self.apply_model_filter())

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        columns = ("use", "type", "data", "rmse", "frac", "tag", "run")
        self.model_table = ttk.Treeview(list_frame, columns=columns, show="headings", height=13, selectmode="browse")
        for col, text, width, anchor in [
            ("use", "Use", 44, "center"),
            ("type", "Type", 88, "w"),
            ("data", "Data", 48, "center"),
            ("rmse", "TEST RMSE", 72, "e"),
            ("frac", "Frac", 50, "e"),
            ("tag", "HP", 88, "w"),
            ("run", "Run", 292, "w"),
        ]:
            self.model_table.heading(col, text=text)
            self.model_table.column(col, width=width, anchor=anchor, stretch=(col == "run"))
        yscroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.model_table.yview)
        self.model_table.configure(yscrollcommand=yscroll.set)
        self.model_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_table.bind("<ButtonRelease-1>", self._on_model_table_click)
        self.model_table.bind("<Double-1>", self._on_model_table_click)

        button_row = ttk.Frame(left)
        button_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(button_row, text="Refresh", command=self.refresh_models).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Add Folder", command=self.add_model_folder).pack(side=tk.LEFT, padx=6)
        ttk.Button(button_row, text="Select Visible", command=self.select_visible_models).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Clear", command=self.clear_selected_models).pack(side=tk.LEFT, padx=(6, 0))

        quick_row = ttk.Frame(left)
        quick_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(quick_row, text="Best FNN", command=lambda: self.select_best_by_type("BlackBoxFNN")).pack(side=tk.LEFT)
        ttk.Button(quick_row, text="Best PINN", command=lambda: self.select_best_by_type("PhysicsRegularizedFNN")).pack(side=tk.LEFT, padx=6)
        ttk.Button(quick_row, text="Best Residual", command=lambda: self.select_best_by_type("ResidualCorrectionFNN")).pack(side=tk.LEFT)
        ttk.Button(quick_row, text="Best EDR", command=lambda: self.select_best_by_type("EDR")).pack(side=tk.LEFT, padx=6)

        action_row = ttk.Frame(left)
        action_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(action_row, text="Run Comparison", style="Accent.TButton", command=self.start_inference).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(action_row, text="Save Figure", command=self.save_figure).pack(side=tk.LEFT, padx=6)
        ttk.Button(action_row, text="Save CSV", command=self.save_predictions_csv).pack(side=tk.LEFT)

        legends = ttk.LabelFrame(left, text="Legend Labels", padding=8)
        legends.pack(fill=tk.X, pady=(0, 8))
        self.legend_editor = ttk.Frame(legends)
        self.legend_editor.pack(fill=tk.X)
        ttk.Label(legends, text="Edit these names before or after running comparison.", wraplength=620).pack(anchor="w", pady=(4, 0))

        settings = ttk.LabelFrame(left, text="Evaluation", padding=8)
        settings.pack(fill=tk.X, pady=(0, 8))
        self.split_var = tk.StringVar(value="test")
        ttk.Label(settings, text="Split").grid(row=0, column=0, sticky="w")
        self.split_box = ttk.Combobox(settings, textvariable=self.split_var, values=["test", "val", "train"], state="readonly", width=14)
        self.split_box.grid(row=0, column=1, sticky="ew", pady=2)
        self.split_box.bind("<<ComboboxSelected>>", lambda _e: self._on_split_selected())

        self.trajectory_var = tk.StringVar(value=TRAJECTORY_ALL_LABEL)
        ttk.Label(settings, text="Trajectory").grid(row=1, column=0, sticky="w")
        self.trajectory_box = ttk.Combobox(
            settings,
            textvariable=self.trajectory_var,
            values=[TRAJECTORY_ALL_LABEL],
            state="readonly",
            width=60,
        )
        self.trajectory_box.grid(row=1, column=1, sticky="ew", pady=2)
        self.trajectory_box.bind("<<ComboboxSelected>>", lambda _e: self.update_time_info())

        self.time_info_var = tk.StringVar(value="Select model(s) to load their raw-backed catalog trajectories")
        ttk.Label(settings, textvariable=self.time_info_var, wraplength=620).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        settings.columnconfigure(1, weight=1)

        plots = ttk.LabelFrame(left, text="Plot", padding=8)
        plots.pack(fill=tk.X, pady=(0, 8))
        self.plot_kind_var = tk.StringVar(value="Prediction overlay")
        ttk.Combobox(
            plots,
            textvariable=self.plot_kind_var,
            values=["Prediction overlay", "Residuals", "Scatter", "Metrics bars"],
            state="readonly",
            width=30,
        ).pack(fill=tk.X)
        self.plot_kind_var.trace_add("write", lambda *_: self.redraw_plot())
        self.grid_layout_var = tk.StringVar(value="Auto grid")
        self.legend_cols_var = tk.StringVar(value="5")
        self.smooth_var = tk.StringVar(value="7")
        self.line_width_var = tk.StringVar(value="2.6")
        self.show_physics_var = tk.BooleanVar(value=False)
        style_grid = ttk.Frame(plots)
        style_grid.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(style_grid, text="Layout").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            style_grid,
            textvariable=self.grid_layout_var,
            values=["Auto grid", "1 column", "2 columns", "3 columns"],
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=2)
        ttk.Label(style_grid, text="Smooth").grid(row=1, column=0, sticky="w")
        ttk.Entry(style_grid, textvariable=self.smooth_var, width=10).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=2)
        ttk.Label(style_grid, text="Line width").grid(row=2, column=0, sticky="w")
        ttk.Entry(style_grid, textvariable=self.line_width_var, width=10).grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=2)
        ttk.Label(style_grid, text="Legend cols").grid(row=3, column=0, sticky="w")
        ttk.Entry(style_grid, textvariable=self.legend_cols_var, width=10).grid(row=3, column=1, sticky="ew", padx=(6, 0), pady=2)
        style_grid.columnconfigure(1, weight=1)
        for v in (self.grid_layout_var, self.smooth_var, self.line_width_var, self.legend_cols_var):
            v.trace_add("write", lambda *_: self.redraw_plot())

        joints = ttk.Frame(plots)
        joints.pack(fill=tk.X, pady=(8, 0))
        for j, var in enumerate(self.joint_vars):
            ttk.Checkbutton(joints, text=f"J{j + 1}", variable=var, command=self.redraw_plot).grid(row=j // 3, column=j % 3, sticky="w", padx=4)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, wraplength=520).pack(fill=tk.X, pady=(8, 0))

        self.figure = Figure(figsize=(12, 9), dpi=100, constrained_layout=False)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        metrics_frame = ttk.LabelFrame(right, text="Metrics", padding=6)
        metrics_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        columns = ("model", "trajectory", "ckpt_rmse", "rmse", "mae", "r2", "nrmse")
        self.metrics_tree = ttk.Treeview(metrics_frame, columns=columns, show="headings", height=5)
        for col, text, width in [
            ("model", "Model", 330),
            ("trajectory", "Selection", 300),
            ("ckpt_rmse", "TEST RMSE", 120),
            ("rmse", "Selected RMSE", 115),
            ("mae", "MAE mean", 110),
            ("r2", "R2 overall", 110),
            ("nrmse", "NRMSE mean", 110),
        ]:
            self.metrics_tree.heading(col, text=text)
            self.metrics_tree.column(col, width=width, anchor="w" if col in {"model", "trajectory"} else "e")
        self.metrics_tree.pack(fill=tk.X)

    def _selected_model_group(self) -> ModelGroup | None:
        label = self.group_var.get() if hasattr(self, "group_var") else ""
        for group in self.model_groups:
            if group.label == label:
                return group
        return self.model_groups[0] if self.model_groups else None

    def _active_stored_metric_split(self) -> str:
        split = self.split_var.get() if hasattr(self, "split_var") else "test"
        return split if split in {"val", "test"} else "avg"

    def _active_stored_metric_heading(self) -> str:
        split = self._active_stored_metric_split()
        return f"{split.upper()} RMSE"

    def _record_stored_rmse(self, rec: ModelRecord) -> float | None:
        split = self._active_stored_metric_split()
        return rec.rmse_by_split.get(split) or rec.rmse

    def _record_stored_r2(self, rec: ModelRecord) -> float | None:
        split = self._active_stored_metric_split()
        return rec.r2_by_split.get(split) or rec.r2

    def _refresh_metric_headings(self) -> None:
        heading = self._active_stored_metric_heading()
        if hasattr(self, "model_table"):
            self.model_table.heading("rmse", text=heading)
        if hasattr(self, "metrics_tree"):
            self.metrics_tree.heading("ckpt_rmse", text=heading)

    def _on_split_selected(self) -> None:
        self.refresh_trajectory_choices(preserve_choice=False)
        self.apply_model_filter()
        self._refresh_metric_headings()

    def _on_model_group_selected(self) -> None:
        self.selected_model_paths.clear()
        self.legend_label_vars.clear()
        self.results = []
        self.refresh_models()
        self.redraw_plot()

    def refresh_models(self) -> None:
        previous_group = self.group_var.get() if hasattr(self, "group_var") else ""
        self.model_groups = discover_model_groups(self.extra_model_groups)
        group_labels = [g.label for g in self.model_groups]
        self.group_box.configure(values=group_labels)
        if previous_group in group_labels:
            self.group_var.set(previous_group)
        elif group_labels:
            self.group_var.set(group_labels[0])
            self.selected_model_paths.clear()
        group = self._selected_model_group()
        self.records = discover_models(list(group.roots)) if group is not None else []
        current_paths = {r.model_path for r in self.records}
        self.selected_model_paths.intersection_update(current_paths)
        types = ["All"] + sorted({r.model_type for r in self.records})
        self.type_filter_box.configure(values=types)
        if self.type_filter_var.get() not in types:
            self.type_filter_var.set("All")
        self.apply_model_filter()
        self._refresh_legend_editor()
        self.refresh_trajectory_choices(preserve_choice=False)
        group_label = group.label if group is not None else "no model group"
        self.status_var.set(f"{group_label}: found {len(self.records)} model checkpoints. Check Use for multiple models.")

    def add_model_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select a folder containing trained models")
        if not folder:
            return
        path = Path(folder)
        label = f"Folder: {path.name}"
        existing = {g.label for g in self.extra_model_groups}
        if label in existing:
            label = f"Folder: {path.name} ({len(existing) + 1})"
        self.extra_model_groups.append(ModelGroup(label, (path,), "custom"))
        self.group_var.set(label)
        self.selected_model_paths.clear()
        self.legend_label_vars.clear()
        self.refresh_models()

    def apply_model_filter(self) -> None:
        query = self.search_var.get().strip().lower()
        type_filter = self.type_filter_var.get()
        try:
            max_rmse = float(self.max_rmse_filter_var.get()) if self.max_rmse_filter_var.get().strip() else None
        except ValueError:
            max_rmse = None
        try:
            top_n = int(self.top_n_filter_var.get()) if self.top_n_filter_var.get().strip() else 0
        except ValueError:
            top_n = 150

        filtered: list[ModelRecord] = []
        for rec in self.records:
            haystack = f"{rec.model_type} {rec.run_id} {rec.physics_tag} {rec.data_run_dir or ''}".lower()
            if query and query not in haystack:
                continue
            if type_filter != "All" and rec.model_type != type_filter:
                continue
            if self.runnable_only_var.get() and not rec.data_available:
                continue
            rmse = self._record_stored_rmse(rec)
            if max_rmse is not None and (rmse is None or rmse > max_rmse):
                continue
            filtered.append(rec)
        filtered.sort(key=lambda r: (float("inf") if self._record_stored_rmse(r) is None else self._record_stored_rmse(r), r.model_type, r.run_id))
        if top_n > 0:
            filtered = filtered[:top_n]
        self.filtered_records = filtered
        self._populate_model_table()
        self._refresh_metric_headings()
        parts = []
        parts.append(f"group={self.group_var.get() or 'none'}")
        parts.append("runnable data only" if self.runnable_only_var.get() else "including missing-data checkpoints")
        parts.append(f"type={type_filter}")
        parts.append(f"search={query!r}" if query else "search=none")
        parts.append(f"stored metric={self._active_stored_metric_heading()}")
        parts.append(f"max RMSE={max_rmse:g}" if max_rmse is not None else "max RMSE=none")
        parts.append(f"top={top_n}" if top_n > 0 else "top=all")
        self.filter_summary_var.set(f"Filter: {', '.join(parts)}. Showing {len(filtered)} of {len(self.records)} checkpoints.")

    def _populate_model_table(self) -> None:
        for iid in self.model_table.get_children():
            self.model_table.delete(iid)
        for idx, rec in enumerate(self.filtered_records):
            mark = "[x]" if rec.model_path in self.selected_model_paths else "[ ]"
            run = rec.run_id if len(rec.run_id) <= 64 else rec.run_id[:61] + "..."
            self.model_table.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    mark,
                    _model_alias(rec.model_type),
                    "ok" if rec.data_available else "missing",
                    _fmt_optional_float(self._record_stored_rmse(rec), 5),
                    _fmt_optional_float(rec.data_fraction, 2),
                    rec.physics_tag,
                    run,
                ),
            )

    def _default_legend_label(self, rec: ModelRecord) -> str:
        return _model_alias(rec.model_type)

    def _refresh_legend_editor(self) -> None:
        for child in self.legend_editor.winfo_children():
            child.destroy()
        selected = [r for r in self.records if r.model_path in self.selected_model_paths]
        selected = selected[:8]
        if not selected:
            ttk.Label(self.legend_editor, text="No models selected. Check Use in the table.").grid(row=0, column=0, sticky="w")
            return
        for row, rec in enumerate(selected):
            key = str(rec.model_path)
            if key not in self.legend_label_vars:
                suffix = "" if len([r for r in selected if r.model_type == rec.model_type]) == 1 else f" {row + 1}"
                self.legend_label_vars[key] = tk.StringVar(value=self._default_legend_label(rec) + suffix)
                self.legend_label_vars[key].trace_add("write", lambda *_: self.redraw_plot())
            ttk.Label(self.legend_editor, text=_model_alias(rec.model_type), width=12).grid(row=row, column=0, sticky="w", pady=1)
            ttk.Entry(self.legend_editor, textvariable=self.legend_label_vars[key], width=28).grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=1)
        self.legend_editor.columnconfigure(1, weight=1)

    def _on_model_table_click(self, event) -> None:
        iid = self.model_table.identify_row(event.y)
        if not iid:
            return
        try:
            rec = self.filtered_records[int(iid)]
        except (ValueError, IndexError):
            return
        if rec.model_path in self.selected_model_paths:
            self.selected_model_paths.remove(rec.model_path)
        else:
            self.selected_model_paths.add(rec.model_path)
        self._populate_model_table()
        self._refresh_legend_editor()
        self.refresh_trajectory_choices(preserve_choice=False)
        self.status_var.set(f"Selected {len(self.selected_model_paths)} model(s)")

    def select_visible_models(self) -> None:
        max_visible_select = 8
        for rec in self.filtered_records[:max_visible_select]:
            self.selected_model_paths.add(rec.model_path)
        self._populate_model_table()
        self._refresh_legend_editor()
        self.refresh_trajectory_choices(preserve_choice=False)
        self.status_var.set(f"Selected first {min(max_visible_select, len(self.filtered_records))} visible model(s)")

    def clear_selected_models(self) -> None:
        self.selected_model_paths.clear()
        self._populate_model_table()
        self._refresh_legend_editor()
        self.refresh_trajectory_choices(preserve_choice=False)
        self.status_var.set("Selection cleared")

    def select_best_by_type(self, model_type: str) -> None:
        matches = [r for r in self.records if r.model_type == model_type and self._record_stored_rmse(r) is not None and r.data_available]
        if not matches:
            self.status_var.set(f"No runnable {model_type} checkpoint found in {self.group_var.get() or 'selected group'}.")
            return
        best = min(matches, key=lambda r: self._record_stored_rmse(r) if self._record_stored_rmse(r) is not None else float("inf"))
        self.selected_model_paths.add(best.model_path)
        self._populate_model_table()
        self._refresh_legend_editor()
        self.refresh_trajectory_choices(preserve_choice=False)
        self.status_var.set(
            f"Selected {_model_alias(best.model_type)} by {self._active_stored_metric_heading()}: {best.run_id}"
        )

    def _selected_records(self) -> list[ModelRecord]:
        selected = [r for r in self.records if r.model_path in self.selected_model_paths]
        if selected:
            return selected
        focused = self.model_table.focus()
        if focused:
            try:
                return [self.filtered_records[int(focused)]]
            except (ValueError, IndexError):
                pass
        return []

    def _selected_trajectory_sources(self) -> list[str] | None:
        if not hasattr(self, "trajectory_var"):
            return []
        choice = self.trajectory_var.get()
        if choice == TRAJECTORY_ALL_LABEL:
            return []
        source = self.trajectory_choice_by_label.get(choice)
        return [source] if source else []

    def _record_data_dir(self, rec: ModelRecord) -> Path | None:
        return _resolve_existing_data_dir(rec.data_run_dir)

    def _record_data_dirs(self, records: list[ModelRecord]) -> list[Path]:
        dirs: list[Path] = []
        seen: set[Path] = set()
        for rec in records:
            p = self._record_data_dir(rec)
            if p is None:
                continue
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                dirs.append(p)
        return dirs

    def _time_info_data_dir(self) -> Path | None:
        dirs = self._record_data_dirs(self._selected_records())
        if dirs:
            return dirs[0]
        for rec in self.records:
            p = _resolve_existing_data_dir(rec.data_run_dir)
            if p is not None:
                return p
        return None

    def refresh_trajectory_choices(self, preserve_choice: bool = True) -> None:
        if not hasattr(self, "trajectory_box"):
            return
        old_choice = self.trajectory_var.get()
        data_dir = self._time_info_data_dir()
        split = self.split_var.get() if hasattr(self, "split_var") else "test"
        choices = [TRAJECTORY_ALL_LABEL]
        choice_by_label: dict[str, str | None] = {
            TRAJECTORY_ALL_LABEL: None,
        }
        catalog: list[TrajectoryEntry] = []
        if data_dir is not None:
            try:
                catalog = load_trajectory_catalog(data_dir, split)
            except Exception:
                catalog = []
            for idx, entry in enumerate(catalog):
                label = _trajectory_choice_label(idx, entry)
                choices.append(label)
                choice_by_label[label] = entry.source_file
        self.trajectory_catalog = catalog
        self.trajectory_choice_by_label = choice_by_label
        self.trajectory_box.configure(values=choices)
        if preserve_choice and old_choice in choices:
            self.trajectory_var.set(old_choice)
        else:
            self.trajectory_var.set(TRAJECTORY_ALL_LABEL)
        self.update_time_info()

    def update_time_info(self) -> None:
        if not hasattr(self, "time_info_var"):
            return
        data_dir = self._time_info_data_dir()
        if data_dir is None:
            self.time_info_var.set("Auto dataset: no runnable selected checkpoint data found")
            return
        selected_dirs = self._record_data_dirs(self._selected_records())
        if len(selected_dirs) > 1:
            dataset_text = (
                f"Auto dataset: {data_dir.name} shown. "
                f"{len(selected_dirs)} different selected train_data runs; select models from one run."
            )
        else:
            dataset_text = f"Auto dataset: {data_dir.name}"
        split = self.split_var.get() if hasattr(self, "split_var") else "test"
        split_dir = data_dir / split
        try:
            t = _load_csv(split_dir, CSV_T).reshape(-1)
            n = len(t)
            catalog = load_trajectory_catalog(data_dir, split)
            raw_ok = sum(1 for entry in catalog if entry.raw_exists)
            selected_sources = self._selected_trajectory_sources()
            if not catalog:
                selected_text = "Selected: catalog mode, but no trajectory catalog was found"
            else:
                try:
                    selected_entries = _select_catalog_entries(catalog, selected_sources)
                    selected_n = sum(entry.end_idx_exclusive - entry.start_idx for entry in selected_entries)
                    if selected_sources:
                        selected_text = (
                            f"Selected: {selected_entries[0].geometry_type} "
                            f"{selected_entries[0].source_file}, N={selected_n:,}"
                        )
                    else:
                        selected_text = f"Selected: all {len(selected_entries)} catalog trajectories, N={selected_n:,}"
                except Exception as exc:
                    selected_text = f"Selected: invalid trajectory choice ({exc})"
            self.time_info_var.set(
                f"{dataset_text}. {split}: {n:,} samples, "
                f"t=[{float(np.nanmin(t)):.4g}, {float(np.nanmax(t)):.4g}]. "
                f"Catalog: {len(catalog)} trajectories, raw JSON {raw_ok}/{len(catalog)}. "
                f"{selected_text}."
            )
        except Exception as exc:
            self.time_info_var.set(f"Time range unavailable for {data_dir.name}/{split}: {exc}")

    def start_inference(self) -> None:
        records = self._selected_records()
        if not records:
            messagebox.showinfo("No model selected", "Select one or more models first.")
            return
        if len(records) > 8:
            messagebox.showwarning(
                "Too many models",
                "Plotting more than 8 models becomes unreadable. Clear some selections or use filters.",
            )
            return
        data_dirs = self._record_data_dirs(records)
        missing_data = [r for r in records if self._record_data_dir(r) is None]
        if missing_data:
            names = "\n".join(r.run_id for r in missing_data[:4])
            messagebox.showerror(
                "Dataset missing",
                "Some selected checkpoints point to datasets that are not present locally.\n\n"
                f"{names}\n\nSelect checkpoints marked data=ok.",
            )
            return
        if len(data_dirs) > 1:
            names = "\n".join(p.name for p in data_dirs[:6])
            messagebox.showerror(
                "Mixed datasets",
                "Selected checkpoints point to different train_data runs.\n\n"
                f"{names}\n\nSelect models from the same data run for a valid trajectory comparison.",
            )
            return
        trajectory_sources = self._selected_trajectory_sources()
        start_sample = 0
        max_samples = 0
        start_time = end_time = None
        self.status_var.set(f"Running inference for {len(records)} model(s)...")
        threading.Thread(
            target=self._inference_worker,
            args=(records, self.split_var.get(), start_sample, max_samples, start_time, end_time, None, trajectory_sources),
            daemon=True,
        ).start()

    def _inference_worker(
        self,
        records: list[ModelRecord],
        split: str,
        start_sample: int,
        max_samples: int,
        start_time: float | None,
        end_time: float | None,
        override_data_dir: Path | None,
        trajectory_sources: list[str] | None,
    ) -> None:
        try:
            results = []
            for idx, rec in enumerate(records, start=1):
                self.after(0, self.status_var.set, f"[{idx}/{len(records)}] Loading {rec.run_id}")
                results.append(
                    run_inference(
                        rec,
                        split,
                        start_sample,
                        max_samples,
                        start_time,
                        end_time,
                        override_data_dir,
                        trajectory_sources,
                    )
                )
            self.after(0, self._set_results, results)
        except Exception as exc:
            tb = traceback.format_exc()
            self.after(0, self._show_error, str(exc), tb)

    def _show_error(self, msg: str, tb: str) -> None:
        self.status_var.set("Inference failed")
        print(tb, file=sys.stderr)
        messagebox.showerror("Inference failed", msg)

    def _set_results(self, results: list[PredictionResult]) -> None:
        self.results = results
        self._refresh_metric_headings()
        for row in self.metrics_tree.get_children():
            self.metrics_tree.delete(row)
        for res in self.results:
            m = res.metrics
            label = self.legend_label_vars.get(str(res.record.model_path))
            self.metrics_tree.insert(
                "",
                tk.END,
                values=(
                    label.get().strip() if label is not None and label.get().strip() else res.record.run_id,
                    res.trajectory_label,
                    _fmt_optional_float(self._record_stored_rmse(res.record), 6),
                    f"{m['rmse_pooled']:.6f}",
                    f"{m['mae_mean']:.6f}",
                    f"{m['r2_overall']:.6f}",
                    f"{m['nrmse_mean']:.6f}",
                ),
            )
        trajectory = results[0].trajectory_label if results else "-"
        self.status_var.set(f"Completed {len(results)} model(s) on {trajectory}")
        self.redraw_plot()

    def _selected_joints(self) -> list[int]:
        joints = [i for i, v in enumerate(self.joint_vars) if v.get()]
        return joints or list(range(ACTIVE_JOINTS))

    def _plot_indices(self, n: int) -> np.ndarray:
        return _contiguous_indices(n, 0, 0)

    def _smooth_series(self, values: np.ndarray) -> np.ndarray:
        try:
            window = int(self.smooth_var.get())
        except ValueError:
            window = 1
        if window <= 1 or len(values) < window:
            return values
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=np.float64) / float(window)
        return np.convolve(values, kernel, mode="same")

    def _line_width(self) -> float:
        try:
            return max(0.5, float(self.line_width_var.get()))
        except ValueError:
            return 2.4

    def _legend_cols(self) -> int:
        try:
            return max(1, int(self.legend_cols_var.get()))
        except ValueError:
            return 5

    def _grid_shape(self, n: int) -> tuple[int, int]:
        layout = self.grid_layout_var.get()
        if layout == "1 column":
            return n, 1
        if layout == "2 columns":
            return int(np.ceil(n / 2)), 2
        if layout == "3 columns":
            return int(np.ceil(n / 3)), 3
        cols = 2 if n <= 4 else 3
        return int(np.ceil(n / cols)), cols

    def _prepare_axes_grid(self, joints: list[int]) -> tuple[list[Any], list[Any]]:
        self.figure.clear()
        rows, cols = self._grid_shape(len(joints))
        if len(joints) == 5 and self.grid_layout_var.get() in {"Auto grid", "3 columns"}:
            gs = self.figure.add_gridspec(2, 12, hspace=0.38, wspace=0.26)
            axes = [
                self.figure.add_subplot(gs[0, 0:4]),
                self.figure.add_subplot(gs[0, 4:8]),
                self.figure.add_subplot(gs[0, 8:12]),
                self.figure.add_subplot(gs[1, 2:6]),
                self.figure.add_subplot(gs[1, 6:10]),
            ]
            return axes, axes
        axes = self.figure.subplots(rows, cols, squeeze=False).flatten().tolist()
        for ax in axes[len(joints):]:
            ax.axis("off")
        return axes[:len(joints)], axes

    def _axis_label_flags(self, plot_idx: int, n_axes: int) -> tuple[bool, bool]:
        rows, cols = self._grid_shape(n_axes)
        if n_axes == 5 and self.grid_layout_var.get() in {"Auto grid", "3 columns"}:
            return plot_idx >= 3, plot_idx in {0, 3}
        return plot_idx >= (rows - 1) * cols, (plot_idx % cols) == 0

    def _format_joint_axis(self, ax, title: str, show_xlabel: bool, show_ylabel: bool = True) -> None:
        ax.set_title(title, fontsize=9.5, pad=4)
        ax.set_ylabel(r"$\tau$ (N m)" if show_ylabel else "", fontsize=9, labelpad=4)
        ax.set_xlabel("Time (s)" if show_xlabel else "", fontsize=9, labelpad=3)
        ax.grid(True, color="#b8b8b8", alpha=0.32, linewidth=0.75)
        ax.tick_params(axis="both", labelsize=8.5, pad=2)
        if not show_xlabel:
            ax.tick_params(axis="x", labelbottom=False)
        ax.margins(x=0.01)
        for spine in ax.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(0.8)

    def _shared_legend(self, handles: list[Any], labels: list[str]) -> None:
        unique_handles: list[Any] = []
        unique_labels: list[str] = []
        seen: set[str] = set()
        for h, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            unique_handles.append(h)
            unique_labels.append(label)
        self.figure.legend(
            unique_handles,
            unique_labels,
            loc="upper center",
            ncol=min(self._legend_cols(), max(1, len(unique_labels))),
            frameon=True,
            fancybox=True,
            fontsize=9,
            columnspacing=1.0,
            handlelength=2.4,
            borderaxespad=0.1,
            bbox_to_anchor=(0.5, 1.008),
        )
        self.figure.subplots_adjust(left=0.07, right=0.99, bottom=0.085, top=0.92, hspace=0.38, wspace=0.26)
        try:
            self.figure.align_ylabels()
        except Exception:
            pass

    def _result_label(self, res: PredictionResult, seen: dict[str, int]) -> str:
        key = str(res.record.model_path)
        if key in self.legend_label_vars:
            label = self.legend_label_vars[key].get().strip()
            if label:
                return label
        base = _model_alias(res.record.model_type)
        seen[base] = seen.get(base, 0) + 1
        if seen[base] == 1:
            return base
        rmse = f"{res.metrics['rmse_pooled']:.4f}"
        return f"{base} {seen[base]} ({rmse})"

    def redraw_plot(self) -> None:
        if not self.results:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Select model(s), then run comparison", ha="center", va="center", fontsize=13)
            ax.axis("off")
            self.canvas.draw_idle()
            return
        kind = self.plot_kind_var.get()
        if kind == "Prediction overlay":
            self._draw_overlay()
        elif kind == "Residuals":
            self._draw_residuals()
        elif kind == "Scatter":
            self._draw_scatter()
        else:
            self._draw_metrics()
        self.canvas.draw_idle()

    def _draw_overlay(self) -> None:
        joints = self._selected_joints()
        axes, all_axes = self._prepare_axes_grid(joints)
        base = self.results[0]
        idx = self._plot_indices(len(base.time))
        handles: list[Any] = []
        labels: list[str] = []
        label_counts: dict[str, int] = {}
        result_labels = {id(res): self._result_label(res, label_counts) for res in self.results}
        line_width = self._line_width()
        time0 = base.time[idx] - base.time[idx][0] if len(idx) else base.time[idx]
        for plot_idx, (ax, j) in enumerate(zip(axes, joints)):
            target_line = ax.plot(
                time0,
                self._smooth_series(base.target[idx, j]),
                color="#666666",
                linewidth=line_width + 0.4,
                label="Ground Truth",
                zorder=4,
            )[0]
            if plot_idx == 0:
                handles.append(target_line)
                labels.append("Ground Truth")
            if self.show_physics_var.get() and base.physics_total is not None:
                phys_line = ax.plot(
                    time0,
                    self._smooth_series(base.physics_total[idx, j]),
                    color="#111111",
                    linewidth=max(1.0, line_width - 1.1),
                    linestyle="--",
                    alpha=0.55,
                    label="Physics",
                )[0]
                if plot_idx == 0:
                    handles.append(phys_line)
                    labels.append("Physics")
            for model_idx, res in enumerate(self.results):
                idx_r = self._plot_indices(len(res.time))
                time_r = res.time[idx_r] - res.time[idx_r][0] if len(idx_r) else res.time[idx_r]
                line = ax.plot(
                    time_r,
                    self._smooth_series(res.pred[idx_r, j]),
                    linewidth=line_width,
                    alpha=0.98,
                    color=_model_color(res.record.model_type, model_idx),
                    label=result_labels[id(res)],
                    zorder=3,
                )[0]
                if plot_idx == 0:
                    handles.append(line)
                    labels.append(result_labels[id(res)])
            show_xlabel, show_ylabel = self._axis_label_flags(plot_idx, len(joints))
            self._format_joint_axis(ax, f"Joint {j + 1}", show_xlabel, show_ylabel)
        self._shared_legend(handles, labels)

    def _draw_residuals(self) -> None:
        joints = self._selected_joints()
        axes, _ = self._prepare_axes_grid(joints)
        handles: list[Any] = []
        labels: list[str] = []
        label_counts: dict[str, int] = {}
        result_labels = {id(res): self._result_label(res, label_counts) for res in self.results}
        line_width = self._line_width()
        for plot_idx, (ax, j) in enumerate(zip(axes, joints)):
            for model_idx, res in enumerate(self.results):
                idx = self._plot_indices(len(res.time))
                time_r = res.time[idx] - res.time[idx][0] if len(idx) else res.time[idx]
                line = ax.plot(
                    time_r,
                    self._smooth_series(res.residual[idx, j]),
                    linewidth=line_width,
                    alpha=0.95,
                    color=_model_color(res.record.model_type, model_idx),
                    label=result_labels[id(res)],
                )[0]
            if plot_idx == 0:
                handles.append(line)
                labels.append(result_labels[id(res)])
            ax.axhline(0.0, color="#444444", linewidth=1.0)
            show_xlabel, show_ylabel = self._axis_label_flags(plot_idx, len(joints))
            self._format_joint_axis(ax, f"Joint {j + 1} Residual", show_xlabel, show_ylabel)
        self._shared_legend(handles, labels)

    def _draw_scatter(self) -> None:
        joints = self._selected_joints()
        axes, _ = self._prepare_axes_grid(joints)
        handles: list[Any] = []
        labels: list[str] = []
        label_counts: dict[str, int] = {}
        result_labels = {id(res): self._result_label(res, label_counts) for res in self.results}
        for plot_idx, (ax, j) in enumerate(zip(axes, joints)):
            for model_idx, res in enumerate(self.results):
                idx = self._plot_indices(len(res.target))
                sc = ax.scatter(
                    res.target[idx, j],
                    res.pred[idx, j],
                    s=10,
                    alpha=0.42,
                    color=_model_color(res.record.model_type, model_idx),
                    label=result_labels[id(res)],
                    edgecolors="none",
                )
                if plot_idx == 0:
                    handles.append(sc)
                    labels.append(result_labels[id(res)])
            lo = min(float(res.target[:, j].min()) for res in self.results)
            hi = max(float(res.target[:, j].max()) for res in self.results)
            ax.plot([lo, hi], [lo, hi], color="#666666", linewidth=1.2)
            show_xlabel, show_ylabel = self._axis_label_flags(plot_idx, len(joints))
            self._format_joint_axis(ax, f"Joint {j + 1}", show_xlabel, show_ylabel)
            ax.set_xlabel(r"Target $\tau$ (N m)" if show_xlabel else "", fontsize=9, labelpad=3)
            ax.set_ylabel(r"Predicted $\tau$ (N m)" if show_ylabel else "", fontsize=9, labelpad=3)
        self._shared_legend(handles, labels)

    def _draw_metrics(self) -> None:
        self.figure.clear()
        axes = self.figure.subplots(1, 2, squeeze=False).flatten()
        seen: dict[str, int] = {}
        labels = [self._result_label(r, seen) for r in self.results]
        x = np.arange(len(self.results))
        rmse = [r.metrics["rmse_pooled"] for r in self.results]
        r2 = [r.metrics["r2_overall"] for r in self.results]
        colors = [_model_color(r.record.model_type, i) for i, r in enumerate(self.results)]
        axes[0].bar(x, rmse, color=colors)
        axes[0].set_xticks(x, labels, rotation=25, ha="right")
        axes[0].set_ylabel("RMSE pooled (N m)")
        axes[0].grid(True, axis="y", color="#b8b8b8", alpha=0.35)
        axes[1].bar(x, r2, color=colors)
        axes[1].set_xticks(x, labels, rotation=25, ha="right")
        axes[1].set_ylabel("R2 overall")
        axes[1].grid(True, axis="y", color="#b8b8b8", alpha=0.35)
        self.figure.subplots_adjust(left=0.055, right=0.99, top=0.94, bottom=0.18, wspace=0.18)

    def save_figure(self) -> None:
        if not self.results:
            messagebox.showinfo("No figure", "Run comparison before saving.")
            return
        path = filedialog.asksaveasfilename(
            title="Save high-quality figure",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("TIFF", "*.tiff")],
        )
        if not path:
            return
        dpi = 300
        self.figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        self.status_var.set(f"Saved figure: {path}")

    def save_predictions_csv(self) -> None:
        if not self.results:
            messagebox.showinfo("No predictions", "Run comparison before exporting.")
            return
        path = filedialog.asksaveasfilename(
            title="Save predictions CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        joints = range(ACTIVE_JOINTS)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["model", "checkpoint_rmse", "trajectory_selection", "sample", "dataset_sample", "time", "source_file"]
            for j in joints:
                header += [f"target_J{j+1}", f"pred_J{j+1}", f"residual_J{j+1}"]
            writer.writerow(header)
            for res in self.results:
                for i in range(len(res.time)):
                    source_file = ""
                    if res.source_file_per_sample is not None and i < len(res.source_file_per_sample):
                        source_file = str(res.source_file_per_sample[i])
                    row = [
                        res.record.run_id,
                        "" if res.record.rmse is None else float(res.record.rmse),
                        res.trajectory_label,
                        i,
                        int(res.dataset_sample_indices[i]),
                        float(res.time[i]),
                        source_file,
                    ]
                    for j in joints:
                        row += [float(res.target[i, j]), float(res.pred[i, j]), float(res.residual[i, j])]
                    writer.writerow(row)
        self.status_var.set(f"Saved predictions: {path}")


def main() -> None:
    app = TorqueInferenceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
