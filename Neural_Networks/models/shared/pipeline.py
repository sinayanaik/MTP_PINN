"""Single training loop for torque models (delegates per-step logic to a strategy)."""

from __future__ import annotations

import csv
import gc
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
    apply_model_training_extras,
    build_run_id,
    dump_yaml,
    exhaustive_hparams,
    load_training_state,
    save_checkpoints,
    save_training_state,
    set_rng_state_bundle,
    TRAINING_STATE_NAME,
    TRAINING_STATE_SCHEMA,
)
from Neural_Networks.models.shared.metrics_numpy import compute_metrics, macro_rmse_numpy
from Neural_Networks.models.shared.optim import build_scheduler
from Neural_Networks.models.shared.strategies import TorqueTrainStrategy, TrainEpochMetrics

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


def _gc_every_epoch_enabled(hp: dict[str, Any]) -> bool:
    if bool(hp.get("gc_every_epoch", False)):
        return True
    v = str(os.environ.get("NN_GC_EVERY_EPOCH", "")).strip().lower()
    return v in ("1", "true", "yes", "on")


def _dataset_memmap_enabled(hp: dict[str, Any]) -> bool:
    if bool(hp.get("dataset_memmap", False)):
        return True
    v = str(os.environ.get("NN_DATASET_MEMMAP", "")).strip().lower()
    return v in ("1", "true", "yes", "on")


def run_training(
    job: TrainJob,
    *,
    log: logging.Logger | None = None,
    progress_callback: Callable[[int, int, float, int, int], None] | None = None,
) -> float | None:
    """Train one model.

    Returns the held-out test ``rmse_traj_macro`` (N·m) on full completion —
    the trajectory-macro RMSE, identical in definition to the live per-epoch
    ``val_rmse`` and the early-stopping criterion (so val and test are the
    same estimator and are directly comparable),
    ``float('nan')`` on a successful-but-incomplete segment (resumable HPC
    mode), and ``None`` on failure (invalid dataset / unrecoverable resume
    state).  Callers treat ``None`` as failure and any float as success.

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
        return None

    import psutil

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    hp = job.hp
    _use_memmap = _dataset_memmap_enabled(hp)
    _run_gc = _gc_every_epoch_enabled(hp)
    _seed = int(hp.get("seed", 42))
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)

    pin_memory = device.type == "cuda"
    _ncpu = os.cpu_count() or 4
    _vm = psutil.virtual_memory()
    _ram_total_gb = _vm.total / 1e9
    # Low-RAM: cap dataloader fan-out to reduce duplicate RSS (workers × prefetch).
    _low_ram = _ram_total_gb < 32.0
    _max_workers = 2 if _low_ram else (4 if _ram_total_gb < 20.0 else 8)
    _nw_env = os.environ.get("NN_NUM_WORKERS", "").strip()
    if _nw_env.isdigit():
        num_workers = max(0, int(_nw_env))
    elif device.type == "cuda":
        if _low_ram:
            num_workers = min(2, max(1, _ncpu // 2))
        else:
            num_workers = max(2, min(_max_workers, _ncpu // 2))
    else:
        num_workers = 0
    _pf_env = os.environ.get("NN_PREFETCH", "").strip()
    if _pf_env.isdigit():
        _prefetch = max(2, int(_pf_env))
    else:
        _prefetch = 2 if _low_ram else (6 if _ram_total_gb < 20.0 else 10)

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
        use_memmap=_use_memmap,
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
    # Patience must not count during LR warm-up: the LR is intentionally low
    # then, so per-epoch improvements are smaller than min_delta even for a
    # well-converging model.  patience_counter is held at 0 through warmup so
    # the full patience budget begins at peak LR.
    _warmup_ep = (
        max(1, epochs // 20)
        if str(hp.get("lr_scheduler", "")).lower() == "warmup_cosine"
        else 0
    )
    _early_metric = str(hp.get("early_stop_metric", "val_rmse")).strip().lower()

    # ── EMA (exponential moving average of weights) ───────────────────────
    # Per-epoch weight EMA.  The best-by-raw-val checkpoint sits on a noisy
    # val curve and tends to be an over-fit point (observed directly: a long
    # run kept crawling val down while TEST worsened).  An EMA of the weights
    # is a flatter, lower-variance solution that generalises better, with no
    # data or architecture change.  ema_decay<=0 (or absent) ⇒ fully OFF,
    # exact back-compat.  The EMA model is evaluated on val each epoch and
    # the best EMA-by-val state competes with the raw best at the end.
    _ema_decay = float(hp.get("ema_decay", 0.0) or 0.0)
    if not 0.0 <= _ema_decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {_ema_decay!r}")
    _ema_on = _ema_decay > 0.0
    _ema_state: dict | None = None
    _best_ema_state: dict | None = None
    _best_ema_val = math.inf
    _best_ema_epoch = 0
    if _ema_on:
        _ema_state = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
    if _early_metric not in ("val_rmse", "val_loss"):
        _early_metric = "val_rmse"
    best_val_loss_track = math.inf
    _snapshot_every = int(hp.get("snapshot_every", 0))
    _print_every = max(1, int(hp.get("print_every", 10)))
    # History keys with consistent semantics across train/val:
    #   train_loss / val_loss : unweighted per-element MSE on (tau_hat, target)
    #                           in normalised target space — directly comparable.
    #   train_rmse / val_rmse : per-joint RMSE in physical N·m, then macro-mean
    #                           over joints — directly comparable.
    #   train_loss_obj        : the *actual* objective being optimised (joint-
    #                           weighted, blended for PhysReg, includes the δ
    #                           regulariser for Residual, etc.).  Tracked for
    #                           diagnostics but not plotted against val_loss.
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_rmse": [], "val_rmse": [],
        "train_loss_obj": [],
    }
    _tau_std_train = np.asarray(loaders["train"].dataset.std_tau, dtype=np.float64)
    _tau_std_val = loaders["val"].dataset.std_tau
    _tau_mean_val = loaders["val"].dataset.mean_tau
    _val_trajectories: list[dict] = (
        loaders["val"].dataset.metadata.get("split", {}).get("stats", {}).get("val", {}).get("trajectories", [])
    )
    _test_trajectories: list[dict] = (
        loaders["test"].dataset.metadata.get("split", {}).get("stats", {}).get("test", {}).get("trajectories", [])
    )
    # The headline RMSE (live val_rmse, early-stop, final val/test) is the
    # trajectory-macro estimator: per-joint RMSE within each trajectory,
    # averaged over joints, then averaged over trajectories.  If the split
    # metadata carries no trajectory boundaries ``macro_rmse_numpy`` silently
    # collapses to a single pooled slice — a different (and the previously
    # buggy) estimator — so fail loudly rather than degrade quietly.
    if not _val_trajectories:
        log.warning(
            "val split metadata has no trajectory boundaries — val_rmse will "
            "collapse to a single pooled slice (NOT trajectory-macro)."
        )
    if not _test_trajectories:
        log.warning(
            "test split metadata has no trajectory boundaries — test_rmse will "
            "collapse to a single pooled slice (NOT trajectory-macro) and will "
            "not be comparable to val_rmse."
        )
    stopped_early = False
    t0 = time.time()

    # Per-component correction-magnitude history (optional; populated when the
    # strategy's train_epoch returns extras["correction_magnitudes"]).
    history["correction_magnitudes"] = []

    seg_ep = int(hp.get("segment_epochs", 0) or 0)
    resume_path = str(hp.get("resume_from", "") or "").strip()
    seg_path = str(hp.get("segment_save_path", "") or "").strip() or os.path.join(
        job.models_dir, "_in_progress", TRAINING_STATE_NAME
    )
    start_epoch = 1
    if resume_path and os.path.isfile(resume_path):
        st = load_training_state(resume_path, str(device))
        if int(st.get("schema", 0) or 0) < int(TRAINING_STATE_SCHEMA):
            log.error("Unrecognised training state schema in %s", resume_path)
            return None
        model.load_state_dict(st["model_state"])
        optimizer.load_state_dict(st["optimizer_state"])
        if scheduler is not None and st.get("scheduler_state") is not None:
            scheduler.load_state_dict(st["scheduler_state"])
        if onecycle_sched is not None and st.get("onecycle_state") is not None:
            onecycle_sched.load_state_dict(st["onecycle_state"])
        if scaler is not None and st.get("scaler_state") is not None:
            scaler.load_state_dict(st["scaler_state"])
        h_in = st.get("history", {})
        for k in (
            "train_loss", "val_loss", "train_rmse", "val_rmse",
            "train_loss_obj", "correction_magnitudes",
        ):
            if k in h_in:
                history[k] = list(h_in[k])
        _bst = st.get("best_state")
        if _bst is not None:
            best_state = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in _bst.items()}
        best_epoch_num = int(st.get("best_epoch_num", 0))
        patience_counter = int(st.get("patience_counter", 0))
        best_val_loss = float(st.get("best_val_loss", math.inf))
        best_val_rmse = float(st.get("best_val_rmse", math.inf))
        best_val_loss_track = float(st.get("best_val_loss_track", math.inf))
        best_val_rmse_phys = float(st.get("best_val_rmse_phys", math.inf))
        set_rng_state_bundle(st.get("rng"))
        apply_model_training_extras(model, st.get("model_extras"))
        start_epoch = int(st.get("next_epoch", 1))
        if start_epoch < 1 or start_epoch > epochs:
            log.error("Invalid resume next_epoch %s (target epochs=%d)", st.get("next_epoch"), epochs)
            return None
        log.info("Resuming training from %s at epoch %d (target %d).", resume_path, start_epoch, epochs)

    if seg_ep > 0:
        epoch_end = min(start_epoch + seg_ep - 1, epochs)
    else:
        epoch_end = epochs
    if start_epoch > epoch_end:
        log.error("No epochs to run: start=%d  end=%d (segment_epochs=%d)", start_epoch, epoch_end, seg_ep)
        return None

    for epoch in range(start_epoch, epoch_end + 1):
        _tm: TrainEpochMetrics = job.strategy.train_epoch(
            model, loaders["train"], optimizer, device, hp, epoch, onecycle_sched, scaler
        )
        _grad_norm = _tm.grad_norm
        train_loss_obj = _tm.loss_total
        train_loss_data = _tm.loss_data_unw
        # Convert per-joint SSE in normalised target space to a physical-units
        # RMSE — per-joint MSE × std_tau[j]² *before* the sqrt, then averaged
        # across joints.  NOTE: this is a *pooled* per-joint estimate over all
        # train samples (running SSE), NOT the trajectory-macro estimator used
        # for the canonical val_rmse/test_rmse.  A true trajectory-macro train
        # RMSE would need an extra non-shuffled full-train eval pass each epoch
        # (the train loader shuffles and weights change mid-epoch).  This curve
        # is DIAGNOSTIC ONLY (history plot/CSV) — never used for ranking,
        # model selection, early-stopping, or the returned headline — so it is
        # an accepted, documented asymmetry: read it as a rough guide against
        # the val_rmse curve, expecting a small concavity (Jensen) offset.
        _sse_j = np.asarray(_tm.sse_per_joint, dtype=np.float64)
        _n_train = max(1, int(_tm.n_samples))
        _train_rmse_phys_per_joint = np.sqrt(_sse_j / _n_train) * _tau_std_train
        train_rmse_phys = float(_train_rmse_phys_per_joint.mean())
        _extras = _tm.extras or {}
        train_l_data_m = _extras.get("l_data_jw")
        train_l_corr_m = _extras.get("l_corr")
        _corr_mags = _extras.get("correction_magnitudes")
        history["correction_magnitudes"].append(_corr_mags)
        val_loss, _val_pred, _val_tgt = job.strategy.eval_epoch(model, loaders["val"], device)
        _val_pred_phys = _val_pred * _tau_std_val + _tau_mean_val
        _val_tgt_phys = _val_tgt * _tau_std_val + _tau_mean_val
        _val_rmse_phys = macro_rmse_numpy(_val_pred_phys, _val_tgt_phys, _val_trajectories)
        history["train_loss"].append(train_loss_data)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_rmse_phys)
        history["val_rmse"].append(_val_rmse_phys)
        history["train_loss_obj"].append(train_loss_obj)

        # Duck-typed hook: if the model supports it, let it record the latest
        # physical-units val_rmse. Used by EDR's adaptive phase-2 plateau detector.
        _model_target = model._orig_mod if hasattr(model, "_orig_mod") else model
        if hasattr(_model_target, "record_val_rmse"):
            _model_target.record_val_rmse(_val_rmse_phys)
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss if _early_metric == "val_loss" else _val_rmse_phys)
            elif onecycle_sched is None:
                scheduler.step()
        if _early_metric == "val_loss":
            improved = val_loss < (best_val_loss_track - 1e-7)
        else:
            improved = _val_rmse_phys < (best_val_rmse - _min_delta)
        if improved:
            if _early_metric == "val_loss":
                best_val_loss_track = val_loss
            best_val_loss = val_loss
            best_val_rmse = _val_rmse_phys
            best_val_rmse_phys = _val_rmse_phys
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch_num = int(epoch)
            patience_counter = 0
        elif epoch <= _warmup_ep:
            # LR warm-up in progress — hold patience at 0 so the full budget
            # starts once the scheduler reaches peak learning rate.
            patience_counter = 0
        else:
            patience_counter += 1

        # ── EMA weight update + val eval (no data/architecture change) ────
        if _ema_on:
            with torch.no_grad():
                _msd = model.state_dict()
                for _k, _v in _ema_state.items():
                    _src = _msd[_k].detach()
                    if _v.dtype.is_floating_point:
                        _v.mul_(_ema_decay).add_(_src.to(_v.device),
                                                 alpha=1.0 - _ema_decay)
                    else:
                        _v.copy_(_src)            # int buffers: just track
            # Evaluate the EMA weights on val; restore live weights after.
            _live_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(_ema_state)
            _e_loss, _e_pred, _e_tgt = job.strategy.eval_epoch(
                model, loaders["val"], device
            )
            model.load_state_dict(_live_sd)
            _ema_val_rmse = macro_rmse_numpy(
                _e_pred * _tau_std_val + _tau_mean_val,
                _e_tgt * _tau_std_val + _tau_mean_val,
                _val_trajectories,
            )
            history.setdefault("ema_val_rmse", []).append(_ema_val_rmse)
            if _ema_val_rmse < (_best_ema_val - _min_delta):
                _best_ema_val = _ema_val_rmse
                _best_ema_state = {
                    k: v.detach().cpu().clone() for k, v in _ema_state.items()
                }
                _best_ema_epoch = int(epoch)
        if _snapshot_every > 0 and epoch % _snapshot_every == 0 and best_state is not None:
            log.debug(
                "snapshot epoch=%d  best_epoch=%d  val_rmse_phys=%.5f",
                epoch, best_epoch_num, _val_rmse_phys,
            )
        # ── Progress callback (e.g. tqdm bar update) ────────────────────────
        if progress_callback is not None:
            progress_callback(epoch, epochs, _val_rmse_phys, patience_counter, patience)
        elif epoch == 1 or epoch % _print_every == 0 or epoch == epochs or epoch == epoch_end:
            # Strategy-specific tail: only build it when extras carry the
            # relevant fields.  Any missing key collapses cleanly to "".
            _extra_tag = ""
            if train_l_data_m is not None and train_l_corr_m is not None:
                _extra_tag = f"  l_data={train_l_data_m:.5f}  l_corr={train_l_corr_m:.5f}"
            elif train_l_data_m is not None:
                _alpha = _extras.get("alpha_eff")
                _l_phys = _extras.get("l_phys_jw")
                if _alpha is not None and _l_phys is not None:
                    _extra_tag = (
                        f"  l_data={train_l_data_m:.5f}  l_phys={_l_phys:.5f}  α={_alpha:.3f}"
                    )
            if _corr_mags is not None:
                _extra_tag += (
                    f"  |δg|={_corr_mags['mean_abs_delta_g']:.3e}"
                    f"  ||δM||_F={_corr_mags['mean_frob_delta_M']:.3e}"
                    f"  |δC·q̇|={_corr_mags['mean_abs_delta_C_qd']:.3e}"
                    f"  |δτ_f|={_corr_mags['mean_abs_delta_tau_f']:.3e}"
                )
            log.info(
                "epoch %4d/%d  train_loss=%.5f (obj=%.5f)  val_loss=%.5f  "
                "train_rmse=%.5f  val_rmse=%.5f N·m%s  best_ep=%d  patience=%d/%d",
                epoch,
                epochs,
                train_loss_data,
                train_loss_obj,
                val_loss,
                train_rmse_phys,
                _val_rmse_phys,
                _extra_tag,
                best_epoch_num,
                patience_counter,
                patience,
            )
        if _run_gc:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if use_early_stop and patience_counter >= patience:
            stopped_early = True
            break

    if (
        seg_ep > 0
        and (not stopped_early)
        and epoch_end < epochs
    ):
        _sg = os.path.dirname(os.path.abspath(seg_path))
        if _sg:
            os.makedirs(_sg, exist_ok=True)
        save_training_state(
            seg_path,
            next_epoch=epoch_end + 1,
            epochs_max=epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            onecycle_sched=onecycle_sched,
            scaler=scaler,
            history=history,
            best_state=best_state,
            best_epoch_num=best_epoch_num,
            patience_counter=patience_counter,
            best_val_loss=best_val_loss,
            best_val_rmse=best_val_rmse,
            best_val_loss_track=best_val_loss_track,
            best_val_rmse_phys=best_val_rmse_phys,
            stopped_early=False,
        )
        log.info(
            "Segment complete — saved resume state to %s (next epoch %d / %d). "
            "Re-run with resume_from= that path to continue.",
            seg_path, epoch_end + 1, epochs,
        )
        return float("nan")

    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    final_epoch_trained = len(history["train_loss"])

    # EMA vs raw-best selection: take whichever has the lower held-out-val
    # trajectory-macro RMSE.  EMA wins when the raw best was an over-fit
    # spike on the noisy val curve (the seed-1 failure mode).
    _selected = "raw-best-by-val"
    if _ema_on and _best_ema_state is not None and _best_ema_val < best_val_rmse:
        best_state = _best_ema_state
        best_epoch_num = _best_ema_epoch
        log.info(
            "EMA selected: ema_val_rmse=%.5f < raw best_val_rmse=%.5f "
            "(decay=%.4g, ema_best_epoch=%d) — using EMA weights.",
            _best_ema_val, best_val_rmse, _ema_decay, _best_ema_epoch,
        )
        best_val_rmse = _best_ema_val
        _selected = "ema"
    elif _ema_on:
        log.info(
            "EMA not selected: best ema_val_rmse=%.5f ≥ raw best_val_rmse=%.5f "
            "— keeping raw best.",
            _best_ema_val, best_val_rmse,
        )
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

    # ── Canonical trajectory-macro RMSE (the single headline estimator) ──────
    # Identical definition to the live per-epoch ``val_rmse`` (line ~321) and
    # the early-stopping criterion: per-joint RMSE within each trajectory,
    # mean over joints, then mean over trajectories.  ``compute_metrics``
    # above is kept only for the rich per-joint diagnostics (r²/mae/nrmse/
    # pooled) — its ``rmse_macro_mean`` pools over all samples and is NOT the
    # same estimator as the live val_rmse (per-trajectory sqrt-then-mean +
    # equal-per-trajectory weighting vs pooled sqrt + equal-per-sample
    # weighting; the two diverge, sign data-dependent).  Reporting test with
    # the pooled estimator while val used the macro one was the apparent
    # val/test gap.
    val_rmse_macro = macro_rmse_numpy(_val_pred_phys, _val_tgt_phys, _val_trajectories)
    test_rmse_macro = macro_rmse_numpy(test_pred_phys, test_target_phys, _test_trajectories)
    val_metrics_final["rmse_traj_macro"] = float(val_rmse_macro)
    test_metrics["rmse_traj_macro"] = float(test_rmse_macro)
    # Combined val+test: reuse the same estimator by concatenating the two
    # trajectory lists with the test indices offset past the val rows.
    _n_val_rows = int(len(_val_pred_phys))
    _combined_trajs = list(_val_trajectories) + [
        {
            **t,
            "start_idx": int(t["start_idx"]) + _n_val_rows,
            "end_idx_exclusive": int(t["end_idx_exclusive"]) + _n_val_rows,
        }
        for t in _test_trajectories
    ]
    avg_metrics["rmse_traj_macro"] = float(
        macro_rmse_numpy(_eval_pred, _eval_target, _combined_trajs)
    )

    # Internal-consistency check: identical weights (best_state) + identical
    # val data (shuffle=False) + deterministic eval() + identical metric must
    # reproduce the tracked best_val_rmse.  A mismatch means this fix (or the
    # checkpoint path) has regressed — surface it loudly.
    if best_state is not None and math.isfinite(best_val_rmse):
        _d = abs(val_rmse_macro - best_val_rmse)
        if _d > 1e-4:
            log.warning(
                "val_rmse_macro=%.6f != tracked best_val_rmse=%.6f (Δ=%.2e) — "
                "metric/checkpoint inconsistency",
                val_rmse_macro, best_val_rmse, _d,
            )

    epochs_trained = len(history["train_loss"])
    # Headline metric = trajectory-macro test RMSE — the SAME estimator as the
    # live val_rmse and the early-stopping criterion, so val and test are now
    # directly comparable.  Both ``rmse_macro_mean`` (pooled per-joint) and
    # ``rmse_pooled`` remain persisted in test_metrics for diagnostics; we
    # rank/return/name by the trajectory-macro value.
    rmse_val = float(test_rmse_macro)
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
    # Two-panel history: comparable MSE losses (left) and comparable RMSEs in
    # physical N·m (right).  Both panels now plot quantities computed on the
    # same target space and same MSE formulation, so train/val curves are
    # directly comparable.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax_loss, ax_rmse = axes
    ax_loss.plot(history["train_loss"], label="train (data MSE)", color="steelblue")
    ax_loss.plot(history["val_loss"], label="val (data MSE)", color="darkorange")
    if any(v is not None for v in history.get("train_loss_obj", [])):
        ax_loss.plot(
            history["train_loss_obj"],
            label="train (objective)",
            color="steelblue", linestyle="--", alpha=0.6,
        )
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("MSE (normalised target)")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    ax_rmse.plot(history["train_rmse"], label="train RMSE (N·m)", color="steelblue")
    ax_rmse.plot(history["val_rmse"], label="val RMSE (N·m)", color="darkorange")
    ax_rmse.set_xlabel("epoch")
    ax_rmse.set_ylabel("macro RMSE (N·m)")
    ax_rmse.set_title("RMSE (physical units)")
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3)
    fig.suptitle(f"Training History — {job.model_type}")
    plt.tight_layout()
    plt.savefig(_hist_png, dpi=100)
    plt.close(fig)
    with open(os.path.join(save_dir, "training_history.csv"), "w", newline="") as csvf:
        w = csv.writer(csvf)
        # train_loss / val_loss : unweighted MSE in normalised target space.
        # train_rmse / val_rmse : physical N·m macro RMSE.
        # train_loss_obj        : the actual training objective (joint-weighted
        #                         and possibly blended/regularised, depending
        #                         on the strategy).  Diagnostics only.
        _corr_hist = history.get("correction_magnitudes", [])
        _has_corr = any(x is not None for x in _corr_hist)
        header = [
            "epoch", "train_loss", "val_loss",
            "train_rmse", "val_rmse", "train_loss_obj",
        ]
        if _has_corr:
            header += [
                "mean_abs_delta_g",
                "mean_frob_delta_M",
                "mean_abs_delta_C_qd",
                "mean_abs_delta_tau_f",
            ]
        w.writerow(header)
        _obj_hist = history.get("train_loss_obj", [])
        for i in range(len(history["train_loss"])):
            row = [
                i + 1,
                f"{history['train_loss'][i]:.8f}",
                f"{history['val_loss'][i]:.8f}",
                f"{history['train_rmse'][i]:.8f}",
                f"{history['val_rmse'][i]:.8f}",
                f"{_obj_hist[i]:.8f}" if i < len(_obj_hist) else "",
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
    log.info(
        "Saved to %s  test_rmse_traj_macro=%.5f  "
        "(pooled-per-joint=%.5f  rmse_pooled=%.5f)",
        save_dir, test_rmse_macro,
        test_metrics["rmse_macro_mean"], test_metrics["rmse_pooled"],
    )

    # ── End-of-training summary (explicit, single block) ──────────────────
    # epoch ran / best epoch / test rmse / val rmse @best / train rmse @best,
    # all on the canonical trajectory-macro metric so val and test compare
    # like-for-like.  The gap (val@best vs test) is the true generalisation
    # gap; a large train↔val@best gap flags overfitting before the best epoch.
    _bi = best_epoch_num - 1
    _train_rmse_at_best = (
        history["train_rmse"][_bi]
        if 0 <= _bi < len(history["train_rmse"]) else float("nan")
    )
    # Selected model's val (EMA-best or raw-best) — best_val_rmse holds it.
    _val_rmse_at_best = best_val_rmse
    log.info(
        "════════ TRAINING SUMMARY [%s] ════════\n"
        "  epochs ran      : %d / %d%s\n"
        "  model selected   : %s\n"
        "  best epoch       : %d\n"
        "  test  rmse (N·m) : %.5f   (trajectory-macro)\n"
        "  val   rmse @best : %.5f\n"
        "  train rmse @best : %.5f\n"
        "  gen. gap (test−val@best): %+.5f   |  overfit gap (val−train @best): %+.5f\n"
        "═══════════════════════════════════════",
        job.model_type,
        epochs_trained, epochs,
        " (early-stopped)" if stopped_early else "",
        _selected,
        best_epoch_num,
        test_rmse_macro,
        _val_rmse_at_best,
        _train_rmse_at_best,
        test_rmse_macro - _val_rmse_at_best,
        _val_rmse_at_best - _train_rmse_at_best,
    )
    return float(test_rmse_macro)


def main_cli(job: TrainJob, *, log_name: str = "training") -> None:
    """Configure logging and ``sys.exit`` for ``python -m`` entrypoints."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(log_name)
    try:
        _rc = run_training(job, log=log)
        sys.exit(1 if _rc is None else 0)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
