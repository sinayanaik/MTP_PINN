#!/usr/bin/env python3
"""Evaluate the best registered model of each type on a target test split.

Usage (from repository root):
    PYTHONPATH=. python3 Neural_Networks/eval_best_models.py [--data-dir PATH] [--split test]

Defaults to the new run_train22 dataset test split.  No GUI dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NN_ROOT   = _REPO_ROOT / "Neural_Networks"
_EDR_DIR   = _NN_ROOT / "models" / "Equivariant-Decomposed-Residual"

for _p in [str(_REPO_ROOT), str(_EDR_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Neural_Networks.loader import (  # noqa: E402
    ACTIVE_JOINTS,
    CSV_FILTERED_Q,
    CSV_FILTERED_QD,
    CSV_FILTERED_QDD,
    CSV_FILTERED_TAU_DECOMPOSED,
    CSV_FILTERED_TAU_MEASURED,
)
from Neural_Networks.models.torque_models import (  # noqa: E402
    BlackBoxFNN,
    PhysicsRegularizedFNN,
    ResidualCorrectionFNN,
    build_mlp,
    reduce_physics_to_total,
)
from Neural_Networks.models.shared.metrics_numpy import compute_metrics  # noqa: E402

REGISTRY_PATH = _NN_ROOT / "Trained_Models" / "models_registry.yaml"
DEFAULT_DATA_DIR = _NN_ROOT / "train_data" / "run_train22_q0_qd91_qdd21_tau51_rnea15"

BATCH_SIZE_GPU = 4096
BATCH_SIZE_CPU = 1024


# ---------------------------------------------------------------------------
# Legacy model wrappers (defined in torque_inference.py, copied here to avoid
# importing the full GUI module)
# ---------------------------------------------------------------------------

class LegacyPhysicsRegularizedFNN(nn.Module):
    def __init__(self, hidden_layers, dropout, activation, in_dim, compat_params=None):
        super().__init__()
        self.n_joints = ACTIVE_JOINTS
        self.in_dim = in_dim
        self.net = build_mlp(in_dim, hidden_layers, ACTIVE_JOINTS, activation, dropout)
        if compat_params == "tau":
            self.tau_scale = nn.Parameter(torch.ones(ACTIVE_JOINTS))
            self.tau_bias  = nn.Parameter(torch.zeros(ACTIVE_JOINTS))
        elif compat_params == "cal":
            self.cal_scale = nn.Parameter(torch.ones(ACTIVE_JOINTS))
            self.cal_bias  = nn.Parameter(torch.zeros(ACTIVE_JOINTS))

    def forward(self, features: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        if self.in_dim == ACTIVE_JOINTS * 3:
            x = features
        else:
            tau_phys = reduce_physics_to_total(physics, self.n_joints)
            x = torch.cat([features, tau_phys], dim=-1)
        return self.net(x)


class LegacyResidualCorrectionFNN(nn.Module):
    def __init__(self, hidden_layers, dropout, activation, in_dim):
        super().__init__()
        self.n_joints = ACTIVE_JOINTS
        self.in_dim = in_dim
        self.net = build_mlp(in_dim, hidden_layers, ACTIVE_JOINTS, activation, dropout)

    def forward(self, features: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        tau_phys = reduce_physics_to_total(physics, self.n_joints)
        x = features if self.in_dim == ACTIVE_JOINTS * 3 else torch.cat([features, tau_phys], dim=-1)
        return tau_phys + self.net(x)


# ---------------------------------------------------------------------------
# EDR dynamic import
# ---------------------------------------------------------------------------

def _load_edr_class():
    spec = importlib.util.spec_from_file_location("mtp_edr_model", _EDR_DIR / "edr_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EDRModel


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _first_net_input_dim(state: dict) -> int:
    for k, v in state.items():
        if k.endswith(".weight") and hasattr(v, "shape") and len(v.shape) == 2:
            return int(v.shape[1])
    return ACTIVE_JOINTS * 3


def build_model(ckpt: dict[str, Any]) -> torch.nn.Module:
    cls_name = str(ckpt.get("model_class") or "")
    hp = ckpt.get("hparams") or {}
    state = ckpt.get("model_state") or ckpt
    model_type = cls_name.lower()
    net_in_dim = _first_net_input_dim(state)

    if "blackbox" in model_type or cls_name == "BlackBoxFNN":
        return BlackBoxFNN(
            hidden_layers=list(hp.get("hidden_layers") or [256, 512, 256]),
            dropout=float(hp.get("dropout", 0.1)),
            activation=str(hp.get("activation", "silu")),
        )
    if "physicsregularized" in model_type:
        if net_in_dim in (ACTIVE_JOINTS * 3, ACTIVE_JOINTS * 4):
            compat = None
            if "tau_scale" in state or "tau_bias" in state:
                compat = "tau"
            elif "cal_scale" in state or "cal_bias" in state:
                compat = "cal"
            return LegacyPhysicsRegularizedFNN(
                hidden_layers=list(hp.get("hidden_layers") or [256, 512, 256]),
                dropout=float(hp.get("dropout", 0.1)),
                activation=str(hp.get("activation", "silu")),
                in_dim=net_in_dim,
                compat_params=compat,
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
        q_std  = hp.get("_q_std")
        if bool(hp.get("use_trig_features", False)) and (q_mean is None or q_std is None):
            q_mean = norm_stats.get("mean_q")
            q_std  = norm_stats.get("std_q")
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
    raise ValueError(f"unsupported model class: {cls_name!r}")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _load_norm(ckpt: dict[str, Any], data_dir: Path) -> dict[str, np.ndarray]:
    norm = ckpt.get("norm_stats") or {}
    if not norm:
        with (data_dir / "metadata.json").open() as f:
            norm = json.load(f).get("normalisation") or {}
    required = ("mean_q", "std_q", "mean_qd", "std_qd", "mean_qdd", "std_qdd", "mean_tau", "std_tau")
    missing = [k for k in required if k not in norm]
    if missing:
        raise ValueError(f"missing norm keys: {', '.join(missing)}")
    out = {k: np.asarray(norm[k], dtype=np.float32) for k in required}
    for k in ("std_q", "std_qd", "std_qdd", "std_tau"):
        out[k] = np.clip(out[k], 1e-8, None)
    return out


# ---------------------------------------------------------------------------
# Data loading & inference
# ---------------------------------------------------------------------------

def _load_csv(split_dir: Path, name: str) -> np.ndarray:
    return np.loadtxt(split_dir / name, delimiter=",", skiprows=1, dtype=np.float32)


def run_inference_on_split(
    model_path: str,
    data_dir: Path,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    norm = _load_norm(ckpt, data_dir)

    split_dir = data_dir / split
    q            = _load_csv(split_dir, CSV_FILTERED_Q)
    qd           = _load_csv(split_dir, CSV_FILTERED_QD)
    qdd          = _load_csv(split_dir, CSV_FILTERED_QDD)
    target       = _load_csv(split_dir, CSV_FILTERED_TAU_MEASURED)
    physics_raw  = _load_csv(split_dir, CSV_FILTERED_TAU_DECOMPOSED)

    features = np.concatenate(
        [
            (q   - norm["mean_q"])   / norm["std_q"],
            (qd  - norm["mean_qd"])  / norm["std_qd"],
            (qdd - norm["mean_qdd"]) / norm["std_qdd"],
        ],
        axis=-1,
    ).astype(np.float32)
    per_comp_mean = np.tile(norm["mean_tau"] / 4.0, 4)
    per_comp_std  = np.tile(norm["std_tau"], 4)
    physics = ((physics_raw - per_comp_mean) / per_comp_std).astype(np.float32)

    model = build_model(ckpt).to(device)
    state = ckpt.get("model_state") or ckpt
    model.load_state_dict(state, strict=True)   # strict=True to catch arch mismatches early
    model.eval()

    bs = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), bs):
            end = min(start + bs, len(features))
            xb = torch.from_numpy(features[start:end]).to(device)
            pb = torch.from_numpy(physics[start:end]).to(device)
            yb = model(xb, pb).detach().cpu().numpy()
            preds.append(yb)

    pred_norm = np.concatenate(preds, axis=0)
    pred = pred_norm * norm["std_tau"] + norm["mean_tau"]
    return compute_metrics(pred, target)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_sorted_per_type(registry_path: Path) -> dict[str, list[dict]]:
    """Return {model_type: [entries sorted by headline test RMSE asc]}.

    Headline = ``rmse_traj_macro`` (trajectory-macro, the estimator matching
    the live val_rmse / return value); falls back to ``rmse_pooled`` for
    older runs whose metadata predates that key.
    """
    with registry_path.open() as f:
        reg = yaml.safe_load(f)

    groups: dict[str, list] = defaultdict(list)
    for m in reg["models"]:
        tm = m.get("test_metrics") or {}
        rmse = (
            tm.get("rmse_traj_macro")
            or tm.get("rmse_pooled")
            or (m.get("metrics") or {}).get("test_rmse_pooled")
            or float("inf")
        )
        groups[m["model_type"]].append((rmse, m))

    return {mtype: [e[1] for e in sorted(entries)] for mtype, entries in groups.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate best model per type on a test split.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"ERROR: data-dir not found: {data_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Dataset: {data_dir.name}")
    print(f"Split  : {args.split}\n")

    sorted_models = load_sorted_per_type(REGISTRY_PATH)

    col = [24, 52, 14, 14, 38]
    header = (
        f"{'Model Type':<{col[0]}} {'Run ID':<{col[1]}} {'Stored RMSE':>{col[2]}} "
        f"{'New RMSE':>{col[3]}} {'Per-joint RMSE (J1..J5)':<{col[4]}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    results: list[tuple[str, float, float, np.ndarray]] = []

    for mtype in sorted(sorted_models):
        candidates = sorted_models[mtype]
        succeeded = False
        for m in candidates:
            model_path = m["model_path"]
            tm = m.get("test_metrics") or {}
            stored_rmse = tm.get("rmse_pooled") or (m.get("metrics") or {}).get("test_rmse_pooled") or float("nan")
            run_id_short = m["run_id"][:50]
            try:
                metrics = run_inference_on_split(model_path, data_dir, args.split, device)
            except RuntimeError as e:
                if "size mismatch" in str(e) or "Missing key" in str(e):
                    # legacy architecture — try next
                    continue
                raise
            new_rmse = float(metrics["rmse_pooled"])
            per_joint = np.asarray(metrics["rmse"])   # shape (n_joints,)
            pj_str = "[" + ", ".join(f"{v:.4f}" for v in per_joint) + "]"
            print(
                f"{mtype:<{col[0]}} {run_id_short:<{col[1]}} "
                f"{stored_rmse:>{col[2]}.5f} {new_rmse:>{col[3]}.5f} {pj_str:<{col[4]}}"
            )
            results.append((mtype, stored_rmse, new_rmse, per_joint))
            succeeded = True
            break
        if not succeeded:
            print(f"  WARNING: no compatible checkpoint found for {mtype}")

    if results:
        print(sep)
        avg_new = float(np.mean([r[2] for r in results]))
        avg_old = float(np.mean([r[1] for r in results if np.isfinite(r[1])]))
        print(
            f"{'Average (all types)':<{col[0]}} {'':<{col[1]}} "
            f"{avg_old:>{col[2]}.5f} {avg_new:>{col[3]}.5f}"
        )
        print(f"\nAverage new test RMSE across {len(results)} model types: {avg_new:.5f} N·m")


if __name__ == "__main__":
    main()
