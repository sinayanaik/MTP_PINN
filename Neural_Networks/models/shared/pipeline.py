"""Single training loop for torque models (delegates per-step logic to a strategy)."""

from __future__ import annotations

import csv
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Neural_Networks.loader import make_dataloaders
from Neural_Networks.models.shared.artifacts import (
    save_architecture_summary,
    save_comparison_plot,
    update_registry,
)
from Neural_Networks.models.shared.checkpointing import (
    build_run_id,
    dump_yaml,
    exhaustive_hparams,
    save_checkpoints,
)
from Neural_Networks.models.shared.metrics_numpy import compute_metrics, macro_rmse_numpy
from Neural_Networks.models.shared.optim import build_scheduler
from Neural_Networks.models.shared.strategies import TorqueTrainStrategy

logger = logging.getLogger(__name__)


@dataclass
class TrainJob:
    """Everything needed for one training run."""

    run_dir: str
    models_dir: str
    registry_file: str
    model_type: str
    save_subdir: str
    hp: dict[str, Any]
    strategy: TorqueTrainStrategy
    run_help: str


def _is_valid_dataset(p: Path) -> bool:
    if not p.is_dir():
        return False
    meta = p / "metadata.json"
    if not meta.is_file():
        return False
    return all((p / s).is_dir() for s in ("train", "val", "test"))


def run_training(
    job: TrainJob,
    *,
    log: logging.Logger | None = None,
    progress_callback: Callable[[int, int, float, int, int], None] | None = None,
) -> int:
    """Train one model; return 0 on success, 1 on invalid dataset.

    Args:
        progress_callback: optional callable invoked after every epoch with
            signature ``(epoch, total_epochs, val_rmse_phys, patience_counter,
            patience)``.  When provided, per-epoch ``log.info`` lines are
            suppressed so an external progress bar can own the display.
    """
    log = log or logger
    if not _is_valid_dataset(Path(job.run_dir)):
        log.error(
            "RUN_DIR must contain metadata.json plus train/, val/, test/. "
            "Edit RUN_DIR in %s.",
            job.run_help,
        )
        return 1

    import psutil

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    hp = job.hp
    _seed = int(hp.get("seed", 42))
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)

    pin_memory = device.type == "cuda"
    _ncpu = os.cpu_count() or 4
    _ram_gb = psutil.virtual_memory().total / 1e9
    _max_workers = 4 if _ram_gb < 20 else 8
    _nw_env = os.environ.get("NN_NUM_WORKERS", "").strip()
    num_workers = (
        max(0, int(_nw_env))
        if _nw_env.isdigit()
        else (max(2, min(_max_workers, _ncpu // 2)) if device.type == "cuda" else 0)
    )
    _pf_env = os.environ.get("NN_PREFETCH", "").strip()
    _prefetch = max(2, int(_pf_env)) if _pf_env.isdigit() else (6 if _ram_gb < 20 else 10)

    loaders = make_dataloaders(
        run_dir=job.run_dir,
        batch_size=int(hp.get("batch_size", 2048)),
        mode="pointwise",
        seq_len=int(hp.get("seq_len", 50)),
        stride=int(hp.get("stride", 1)),
        normalise=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=_prefetch,
        drop_last=True,
        data_train_fraction=float(hp.get("data_train_fraction", 1.0)),
        data_train_seed=int(hp.get("data_train_seed", hp.get("_grid_seed", 0)) or 0),
    )

    model = job.strategy.make_model(device, hp)
    _model_cls_name = model.__class__.__name__

    _compiled = False
    if device.type == "cuda" and bool(hp.get("torch_compile", False)):
        try:
            mode = str(hp.get("torch_compile_mode", "default")).strip().lower()
            if mode not in ("default", "reduce-overhead", "max-autotune"):
                mode = "default"
            model = torch.compile(model, mode=mode)
            _compiled = True
        except Exception as exc:
            log.debug("torch.compile failed, falling back to eager mode: %s", exc)

    optimizer = job.strategy.build_optimizer(model, hp)
    n_train_batches = len(loaders["train"])
    onecycle_sched = None
    if str(hp.get("lr_scheduler", "")).lower() == "onecycle":
        onecycle_sched = build_scheduler(optimizer, hp, n_train_batches)
        scheduler = None
    else:
        scheduler = build_scheduler(optimizer, hp, n_train_batches)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    _min_delta = float(hp.get("min_delta", 1e-4))
    best_val_loss = math.inf
    best_val_rmse = math.inf
    best_val_rmse_phys = math.inf
    best_state: dict | None = None
    best_epoch_num = 0
    patience_counter = 0
    patience = int(hp.get("patience", 100))
    use_early_stop = bool(hp.get("early_stopping", True))
    epochs = int(hp.get("epochs", 500))
    _early_metric = str(hp.get("early_stop_metric", "val_rmse")).strip().lower()
    if _early_metric not in ("val_rmse", "val_loss"):
        _early_metric = "val_rmse"
    best_val_loss_track = math.inf
    _snapshot_every = int(hp.get("snapshot_every", 0))
    _print_every = max(1, int(hp.get("print_every", 10)))
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": []}
    _tau_std_val = loaders["val"].dataset.std_tau
    _tau_mean_val = loaders["val"].dataset.mean_tau
    _val_trajectories: list[dict] = (
        loaders["val"].dataset.metadata.get("split", {}).get("stats", {}).get("val", {}).get("trajectories", [])
    )
    stopped_early = False
    t0 = time.time()

    # Per-component correction-magnitude history (optional; populated when the
    # strategy's train_epoch returns a trailing telemetry dict).
    history["correction_magnitudes"] = []

    for epoch in range(1, epochs + 1):
        _train_ret = job.strategy.train_epoch(
            model, loaders["train"], optimizer, device, hp, epoch, onecycle_sched, scaler
        )
        # Backward-compatible unpacking: strategies may return 3, 5, or 6
        # elements.  The 6-element form adds a trailing ``correction_magnitudes``
        # dict with per-δ-network mean magnitudes.
        _corr_mags: dict[str, float] | None = None
        if len(_train_ret) == 6:
            (train_loss, _grad_norm, train_rmse_unw,
             train_l_data_m, train_l_corr_m, _corr_mags) = _train_ret
        elif len(_train_ret) == 5:
            train_loss, _grad_norm, train_rmse_unw, train_l_data_m, train_l_corr_m = _train_ret
        else:
            train_loss, _grad_norm, train_rmse_unw = _train_ret
            train_l_data_m = train_l_corr_m = None
        history["correction_magnitudes"].append(_corr_mags)
        val_loss, _val_pred, _val_tgt = job.strategy.eval_epoch(model, loaders["val"], device)
        _val_rmse_unw = macro_rmse_numpy(_val_pred, _val_tgt, _val_trajectories)
        _val_pred_phys = _val_pred * _tau_std_val + _tau_mean_val
        _val_tgt_phys = _val_tgt * _tau_std_val + _tau_mean_val
        _val_rmse_phys = macro_rmse_numpy(_val_pred_phys, _val_tgt_phys, _val_trajectories)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_rmse_unw)
        history["val_rmse"].append(_val_rmse_phys)

        # Duck-typed hook: if the model supports it, let it record the latest
        # unnormalised val_rmse.  Used by EDR's adaptive phase-2 plateau detector.
        _model_target = model._orig_mod if hasattr(model, "_orig_mod") else model
        if hasattr(_model_target, "record_val_rmse"):
            _model_target.record_val_rmse(_val_rmse_unw)
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss if _early_metric == "val_loss" else _val_rmse_unw)
            elif onecycle_sched is None:
                scheduler.step()
        if _early_metric == "val_loss":
            improved = val_loss < (best_val_loss_track - 1e-7)
        else:
            improved = _val_rmse_unw < (best_val_rmse - _min_delta)
        if improved:
            if _early_metric == "val_loss":
                best_val_loss_track = val_loss
            best_val_loss = val_loss
            best_val_rmse = _val_rmse_unw
            best_val_rmse_phys = _val_rmse_phys
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch_num = int(epoch)
            patience_counter = 0
        else:
            patience_counter += 1
        if _snapshot_every > 0 and epoch % _snapshot_every == 0 and best_state is not None:
            log.debug(
                "snapshot epoch=%d  best_epoch=%d  val_rmse_phys=%.5f",
                epoch, best_epoch_num, _val_rmse_phys,
            )
        # ── Progress callback (e.g. tqdm bar update) ────────────────────────
        if progress_callback is not None:
            progress_callback(epoch, epochs, _val_rmse_phys, patience_counter, patience)
        elif epoch == 1 or epoch % _print_every == 0 or epoch == epochs:
            if train_l_data_m is not None:
                _corr_tag = ""
                if _corr_mags is not None:
                    _corr_tag = (
                        f"  |δg|={_corr_mags['mean_abs_delta_g']:.3e}"
                        f"  ||δM||_F={_corr_mags['mean_frob_delta_M']:.3e}"
                        f"  |δC·q̇|={_corr_mags['mean_abs_delta_C_qd']:.3e}"
                        f"  |δτ_f|={_corr_mags['mean_abs_delta_tau_f']:.3e}"
                    )
                log.info(
                    "epoch %4d/%d  train_loss=%.5f  train_l_data=%.5f  train_l_corr=%.5f  "
                    "val_loss=%.5f  val_rmse_phys=%.5f N·m%s  best_ep=%d  patience=%d/%d",
                    epoch,
                    epochs,
                    train_loss,
                    train_l_data_m,
                    train_l_corr_m,
                    val_loss,
                    _val_rmse_phys,
                    _corr_tag,
                    best_epoch_num,
                    patience_counter,
                    patience,
                )
            else:
                log.info(
                    "epoch %4d/%d  train_loss=%.5f  val_loss=%.5f  val_rmse_phys=%.5f N·m  "
                    "best_ep=%d  patience=%d/%d",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                    _val_rmse_phys,
                    best_epoch_num,
                    patience_counter,
                    patience,
                )
        if use_early_stop and patience_counter >= patience:
            stopped_early = True
            break

    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    final_epoch_trained = len(history["train_loss"])
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0

    _, _val_pred_norm, _val_tgt_norm = job.strategy.eval_epoch(model, loaders["val"], device)
    _val_pred_phys = _val_pred_norm * _tau_std_val + _tau_mean_val
    _val_tgt_phys = _val_tgt_norm * _tau_std_val + _tau_mean_val
    val_metrics_final = compute_metrics(_val_pred_phys, _val_tgt_phys)
    _, test_pred, test_target = job.strategy.eval_epoch(model, loaders["test"], device)
    _tau_std = loaders["test"].dataset.std_tau
    _tau_mean = loaders["test"].dataset.mean_tau
    test_pred_phys = test_pred * _tau_std + _tau_mean
    test_target_phys = test_target * _tau_std + _tau_mean
    test_metrics = compute_metrics(test_pred_phys, test_target_phys)
    _eval_pred = np.concatenate([_val_pred_phys, test_pred_phys], axis=0)
    _eval_target = np.concatenate([_val_tgt_phys, test_target_phys], axis=0)
    avg_metrics = compute_metrics(_eval_pred, _eval_target)
    avg_metrics["_n_val"] = int(len(_val_pred_phys))
    avg_metrics["_n_test"] = int(len(test_pred_phys))

    epochs_trained = len(history["train_loss"])
    rmse_val = float(avg_metrics["rmse_pooled"])
    run_id = build_run_id(
        job.model_type,
        epochs_trained=epochs_trained,
        rmse=rmse_val,
        hp=hp,
        run_id_hp_keys=job.strategy.run_id_hp_keys,
    )
    save_dir = os.path.join(job.models_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    _train_ds = loaders["train"].dataset

    def _to_list(arr):
        return arr.tolist() if hasattr(arr, "tolist") else list(arr)

    _norm_stats = {
        "mean_tau": _to_list(_train_ds.mean_tau),
        "std_tau": _to_list(_train_ds.std_tau),
        "mean_q": _to_list(_train_ds.mean_q),
        "std_q": _to_list(_train_ds.std_q),
        "mean_qd": _to_list(_train_ds.mean_qd),
        "std_qd": _to_list(_train_ds.std_qd),
        "mean_qdd": _to_list(_train_ds.mean_qdd),
        "std_qdd": _to_list(_train_ds.std_qdd),
    }
    _unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    _hparams_blob = getattr(_unwrapped, "hparams", None) or {}
    model_path, _ = save_checkpoints(
        save_dir,
        model=model,
        final_state=final_state,
        best_epoch=best_epoch_num,
        epochs_trained=final_epoch_trained,
        model_cls_name=_model_cls_name,
        hparams_blob=_hparams_blob,
        norm_stats=_norm_stats,
        avg_metrics=avg_metrics,
        val_metrics=val_metrics_final,
        test_metrics=test_metrics,
    )
    meta_path = os.path.join(save_dir, "metadata.yaml")
    dump_yaml(
        {
            "model_type": job.model_type,
            "run_id": run_id,
            "data_run_dir": job.run_dir,
            "trained_at": datetime.now().isoformat(),
            "device": str(device),
            "epochs_trained": int(epochs_trained),
            "best_val_loss": float(best_val_loss),
            "best_val_rmse": float(best_val_rmse),
            "hyperparams": dict(hp),
            "exhaustive_hyperparams": exhaustive_hparams(hp, job.strategy.default_exhaustive_hp),
            "physics_sched_config": job.strategy.physics_sched_metadata(hp),
            "metrics": avg_metrics,
            "val_metrics": val_metrics_final,
            "test_metrics": test_metrics,
            # Per-component correction magnitudes at the best epoch (if available).
            "correction_magnitudes_at_best": (
                history.get("correction_magnitudes", [None])[best_epoch_num - 1]
                if best_epoch_num >= 1 and best_epoch_num <= len(history.get("correction_magnitudes", []))
                else None
            ),
            "torch_compile": _compiled,
            "save_subdir": job.save_subdir,
        },
        meta_path,
    )
    save_comparison_plot(
        _eval_pred, _eval_target, avg_metrics, os.path.join(save_dir, "comparison_plot.png"), job.model_type
    )
    save_architecture_summary(_unwrapped, os.path.join(save_dir, "architecture.txt"))
    _hist_png = os.path.join(save_dir, "training_history.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="train loss", color="steelblue")
    ax.plot(history["val_loss"], label="val loss", color="darkorange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"Training History — {job.model_type}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_hist_png, dpi=100)
    plt.close(fig)
    with open(os.path.join(save_dir, "training_history.csv"), "w", newline="") as csvf:
        w = csv.writer(csvf)
        # Per-component correction-magnitude columns are added only if at least
        # one epoch recorded them (strategies that don't emit the telemetry
        # dict will store None).
        _corr_hist = history.get("correction_magnitudes", [])
        _has_corr = any(x is not None for x in _corr_hist)
        header = ["epoch", "train_loss", "val_loss", "train_rmse", "val_rmse"]
        if _has_corr:
            header += [
                "mean_abs_delta_g",
                "mean_frob_delta_M",
                "mean_abs_delta_C_qd",
                "mean_abs_delta_tau_f",
            ]
        w.writerow(header)
        for i in range(len(history["train_loss"])):
            row = [
                i + 1,
                f"{history['train_loss'][i]:.8f}",
                f"{history['val_loss'][i]:.8f}",
                f"{history['train_rmse'][i]:.8f}",
                f"{history['val_rmse'][i]:.8f}",
            ]
            if _has_corr:
                c = _corr_hist[i] if i < len(_corr_hist) else None
                if c is None:
                    row += ["", "", "", ""]
                else:
                    row += [
                        f"{c['mean_abs_delta_g']:.8e}",
                        f"{c['mean_frob_delta_M']:.8e}",
                        f"{c['mean_abs_delta_C_qd']:.8e}",
                        f"{c['mean_abs_delta_tau_f']:.8e}",
                    ]
            w.writerow(row)
    device_str = f"cuda:{torch.cuda.get_device_name(0)}" if device.type == "cuda" else "cpu"
    update_registry(
        registry_file=job.registry_file,
        model_key=job.model_type,
        run_id=run_id,
        run_dir=save_dir,
        hp=hp,
        metrics=avg_metrics,
        val_metrics=val_metrics_final,
        test_metrics=test_metrics,
        model_path=model_path,
        training_time_s=elapsed,
        device_str=device_str,
        stopped_early=stopped_early,
        epochs_ran=epochs_trained,
        epochs_max=epochs,
        final_train_loss=history["train_loss"][-1] if history["train_loss"] else 0.0,
        final_val_loss=history["val_loss"][-1] if history["val_loss"] else 0.0,
        num_train_samples=len(loaders["train"].dataset),
        num_val_samples=len(loaders["val"].dataset),
        num_test_samples=len(loaders["test"].dataset),
    )
    log.info("Saved to %s  rmse_pooled=%.5f", save_dir, avg_metrics["rmse_pooled"])
    return 0


def main_cli(job: TrainJob, *, log_name: str = "training") -> None:
    """Configure logging and ``sys.exit`` for ``python -m`` entrypoints."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(log_name)
    try:
        sys.exit(run_training(job, log=log))
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
