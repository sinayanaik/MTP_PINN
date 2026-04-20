"""Plots, architecture dump, and models_registry.yaml updates."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from Neural_Networks.models.shared.checkpointing import _make_serializable

logger = logging.getLogger(__name__)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def save_comparison_plot(
    pred: np.ndarray,
    target: np.ndarray,
    metrics: dict,
    save_path: str,
    model_name: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    joint_names = ["J1 (yaw)", "J2 (shoulder)", "J3 (elbow)", "J4 (wrist)", "J5 (wrist roll)"]
    n_show = min(3000, len(pred))
    idx = np.linspace(0, len(pred) - 1, n_show, dtype=int)
    for j in range(5):
        ax = axes[j]
        ax.plot(target[idx, j], color="darkorange", linewidth=1.0, label="Ground Truth")
        ax.plot(pred[idx, j], color="steelblue", linewidth=1.0, alpha=0.8, label=model_name)
        r2_j = metrics.get("r2", [0] * 5)[j]
        ax.set_title(
            f"{joint_names[j]}  RMSE={metrics['rmse'][j]:.4f}  "
            f"R²={r2_j:.4f}  NRMSE={metrics['nrmse'][j]:.3f}"
        )
        ax.set_xlabel("sample")
        ax.set_ylabel("torque (N·m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax = axes[5]
    ax.axis("off")
    _rp = metrics.get("rmse_pooled", metrics["rmse_mean"])
    _r2o = metrics.get("r2_overall", metrics.get("r2_mean", 0.0))
    lines = [
        f"Model: {model_name}",
        "",
        f"Pooled RMSE:  {_rp:.5f} N·m  (all τ, same as val_rmse)",
        f"Macro RMSE:   {metrics['rmse_mean']:.5f} N·m  (mean of joint RMSE)",
        f"Mean MAE:     {metrics.get('mae_mean', 0):.5f} N·m",
        f"R2 overall:   {_r2o:.5f}   R2 macro: {metrics.get('r2_mean', 0):.5f}",
        f"Mean Pearson: {metrics.get('pearson_r_mean', 0):.5f}",
        f"Mean NRMSE:   {metrics['nrmse_mean']:.4f}",
        "",
        "Per-joint   RMSE      R2      MAE",
    ]
    for j in range(5):
        r2_j = metrics.get("r2", [0] * 5)[j]
        mae_j = metrics.get("mae", [0] * 5)[j]
        lines.append(f"  J{j+1}:  {metrics['rmse'][j]:.5f}  {r2_j:.4f}  {mae_j:.5f}")
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=9, verticalalignment="top", fontfamily="monospace")
    plt.suptitle(f"Prediction vs Ground Truth — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_architecture_summary(model: nn.Module, save_path: str) -> None:
    lines = [
        f"Model: {model.__class__.__name__}",
        f"Params: {model.count_parameters():,}" if hasattr(model, "count_parameters") else "",
        "",
        "Hyperparameters:",
    ]
    _cfg = getattr(model, "hparams", None) or getattr(model, "config", None)
    if _cfg:
        for k, v in _cfg.items():
            lines.append(f"  {k}: {v}")
    lines += ["", "PyTorch repr:", str(model)]
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def update_registry(
    registry_file: str,
    model_key: str,
    run_id: str,
    run_dir: str,
    hp: dict,
    metrics: dict,
    model_path: str,
    training_time_s: float = 0.0,
    device_str: str = "cpu",
    stopped_early: bool = False,
    epochs_ran: int = 0,
    epochs_max: int = 0,
    final_train_loss: float = 0.0,
    final_val_loss: float = 0.0,
    num_train_samples: int = 0,
    num_val_samples: int = 0,
    val_metrics: dict | None = None,
    test_metrics: dict | None = None,
    num_test_samples: int = 0,
) -> None:
    try:
        registry: dict = {}
        if os.path.exists(registry_file):
            try:
                with open(registry_file, "r") as f:
                    registry = yaml.safe_load(f) or {}
            except Exception as load_err:
                backup = registry_file + ".bak"
                logger.warning("Registry unreadable (%s) — backing up to %s.", load_err, backup)
                try:
                    shutil.copy2(registry_file, backup)
                except Exception:
                    pass
                registry = {}
        if "models" not in registry:
            registry["models"] = []
        entry = {
            "model_type": model_key,
            "run_id": run_id,
            "run_dir": run_dir,
            "trained_at": datetime.now().isoformat(),
            "model_path": model_path,
            "training": {
                "time_seconds": round(training_time_s, 1),
                "time_formatted": _fmt_time(training_time_s),
                "epochs_ran": epochs_ran,
                "epochs_max": epochs_max,
                "stopped_early": stopped_early,
                "final_train_loss": round(final_train_loss, 7) if final_train_loss else None,
                "final_val_loss": round(final_val_loss, 7) if final_val_loss else None,
            },
            "hardware": {"device": device_str, "torch_version": str(torch.__version__)},
            "data": {
                "num_train_samples": num_train_samples,
                "num_val_samples": num_val_samples,
                "num_test_samples": num_test_samples,
            },
            "hyperparams": _make_serializable(hp),
            "metrics": {
                "avg_rmse_mean": metrics.get("rmse_mean"),
                "avg_rmse_pooled": metrics.get("rmse_pooled"),
                "avg_r2_overall": metrics.get("r2_overall"),
                "avg_r2_mean": metrics.get("r2_mean"),
                "avg_nrmse_mean": metrics.get("nrmse_mean"),
                "avg_mse_mean": metrics.get("mse_mean"),
                "avg_pearson_mean": metrics.get("pearson_r_mean"),
                "per_joint_rmse": metrics.get("rmse"),
                "per_joint_r2": metrics.get("r2"),
                "test_rmse_mean": (test_metrics or metrics).get("rmse_mean"),
                "test_rmse_pooled": (test_metrics or metrics).get("rmse_pooled"),
                "test_r2_overall": (test_metrics or metrics).get("r2_overall"),
                "test_nrmse_mean": (test_metrics or metrics).get("nrmse_mean"),
                "test_mse_mean": (test_metrics or metrics).get("mse_mean"),
            },
        }
        if val_metrics is not None:
            entry["val_metrics"] = {
                "rmse_pooled": val_metrics.get("rmse_pooled"),
                "rmse_mean": val_metrics.get("rmse_mean"),
                "r2_overall": val_metrics.get("r2_overall"),
                "r2_mean": val_metrics.get("r2_mean"),
                "nrmse_mean": val_metrics.get("nrmse_mean"),
                "per_joint_rmse": val_metrics.get("rmse"),
            }
        if test_metrics is not None:
            entry["test_metrics"] = {
                "rmse_pooled": test_metrics.get("rmse_pooled"),
                "rmse_mean": test_metrics.get("rmse_mean"),
                "r2_overall": test_metrics.get("r2_overall"),
                "r2_mean": test_metrics.get("r2_mean"),
                "nrmse_mean": test_metrics.get("nrmse_mean"),
                "per_joint_rmse": test_metrics.get("rmse"),
            }
        registry["models"].insert(0, entry)
        registry_out = {
            "total_models": len(registry["models"]),
            "last_updated": datetime.now().isoformat(),
            "models": registry["models"],
        }
        os.makedirs(os.path.dirname(registry_file), exist_ok=True)
        with open(registry_file, "w") as f:
            yaml.dump(_make_serializable(registry_out), f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.warning("Could not update registry: %s", e)
