"""Regenerate per-timestep predictions for a model, with an on-disk cache.

Reuses the *exact* model-building and normalisation logic from
``Neural_Networks/eval_best_models.py`` (``build_model``, ``_load_norm``,
``_load_csv``) so the prediction contract cannot drift.  ``run_inference_on_split``
there only returns metrics; we need the raw arrays plus trajectory boundaries,
so the inference body is reproduced here with a single private ``_make_inputs``
that mirrors eval_best_models.py lines 232-242 verbatim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import bootstrap  # noqa: F401  (ensures sys.path is patched first)
from .config import PlotConfig
from .dataio import resolve_dataset_dir

from Neural_Networks.eval_best_models import (  # noqa: E402
    BATCH_SIZE_CPU,
    BATCH_SIZE_GPU,
    _load_csv,
    _load_edr_class,
    _load_norm,
    build_model,
)
from Neural_Networks.loader import (  # noqa: E402
    CSV_FILTERED_Q,
    CSV_FILTERED_QD,
    CSV_FILTERED_QDD,
    CSV_FILTERED_TAU_DECOMPOSED,
    CSV_FILTERED_TAU_MEASURED,
)
from Neural_Networks.models.shared.metrics_numpy import compute_metrics  # noqa: E402


def _device(cfg: PlotConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _build_model(ckpt: dict[str, Any]):
    """Build the model from a checkpoint.

    ``eval_best_models.build_model`` is correct for FNN / PhysReg / Legacy
    variants but its EDR branch predates ``use_phys_cond`` /
    ``coriolis_matrix_form`` / ``inertia_psd`` and silently drops them, causing
    a state-dict size mismatch.  EDR is therefore reconstructed here from the
    full ``hparams`` (which the checkpoint stores in their entirety).
    """
    cls = str(ckpt.get("model_class") or "")
    if "edr" not in cls.lower():
        return build_model(ckpt)

    # NOTE: the EDR model was rewritten in the robustness overhaul — the old
    # two-phase ``set_phase`` curriculum was replaced by a smooth capacity gate
    # ``set_correction_gain`` (γ), and ``coriolis_structural`` selects whether
    # δC is derived from δM (no independent net) or an independent net. Both
    # eval_best_models.build_model and earlier copies of this function predate
    # that, so EDR is rebuilt here from the checkpoint's full ``hparams``.

    hp = ckpt.get("hparams") or {}
    norm_stats = ckpt.get("norm_stats") or {}
    q_mean = q_std = None
    if bool(hp.get("use_trig_features", False)):
        q_mean = hp.get("_q_mean") or norm_stats.get("mean_q")
        q_std = hp.get("_q_std") or norm_stats.get("std_q")

    EDRModel = _load_edr_class()
    model = EDRModel(
        n_joints=int(hp.get("n_joints", 5)),
        gravity_hidden=list(hp.get("gravity_hidden") or [64, 64]),
        inertia_hidden=list(hp.get("inertia_hidden") or [64, 64]),
        coriolis_hidden=list(hp.get("coriolis_hidden") or [64, 64]),
        friction_hidden=list(hp.get("friction_hidden") or [32, 32]),
        activation=str(hp.get("activation", "silu")),
        correction_dropout=float(hp.get("correction_dropout", 0.0)),
        q_mean=q_mean,
        q_std=q_std,
        use_friction_qdd=bool(hp.get("use_friction_qdd", False)),
        use_phys_cond=bool(hp.get("use_phys_cond", False)),
        coriolis_matrix_form=bool(hp.get("coriolis_matrix_form", True)),
        coriolis_structural=bool(hp.get("coriolis_structural", True)),
        friction_form=str(hp.get("friction_form", "mlp")),
        inertia_psd=bool(hp.get("inertia_psd", False)),
        spectral_norm=bool(hp.get("spectral_norm", False)),
    )
    # Evaluate at full correction capacity. New model: γ-gate; legacy: two-phase.
    if hasattr(model, "set_correction_gain"):
        model.set_correction_gain(1.0)
    elif hasattr(model, "set_phase"):
        model.set_phase(2)
    return model


def _make_inputs(norm: dict, q, qd, qdd, physics_raw):
    """Verbatim mirror of eval_best_models.run_inference_on_split lines 232-242."""
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
    return features, physics


def _trajectories(dataset_dir: Path, split: str) -> list[tuple[int, int, str]]:
    with (dataset_dir / "metadata.json").open() as f:
        meta = json.load(f)
    trajs = meta["split"]["stats"][split]["trajectories"]
    return [
        (int(t["start_idx"]), int(t["end_idx_exclusive"]), str(t.get("geometry_type", "?")))
        for t in trajs
    ]


def _cache_path(cfg: PlotConfig, record: dict[str, Any]) -> Path:
    key = hashlib.sha1(f"{record['run_id']}|{cfg.split}".encode()).hexdigest()[:16]
    return Path(cfg.cache_dir) / f"{record['arch']}_{key}_{cfg.split}.npz"


def predict_split(record: dict[str, Any], cfg: PlotConfig) -> dict[str, Any]:
    """Return predictions/targets/trajectory layout for one model.

    Result keys: ``pred`` (N,5), ``target`` (N,5), ``qd`` (N,5),
    ``traj`` list[(start,end,geom)], ``metrics`` (compute_metrics dict),
    ``arch``, ``run_id``.  Cached to ``cfg.cache_dir`` as ``.npz``,
    invalidated when ``model.pt`` is newer than the cache.
    """
    model_path = Path(record["model_path"])
    cache = _cache_path(cfg, record)
    model_mtime = model_path.stat().st_mtime if model_path.exists() else 0.0

    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if float(z["model_mtime"]) >= model_mtime:
            traj = [(int(s), int(e), str(g)) for s, e, g in z["traj"]]
            metrics = compute_metrics(z["pred"], z["target"])
            return {
                "pred": z["pred"], "target": z["target"], "qd": z["qd"],
                "traj": traj, "metrics": metrics,
                "arch": record["arch"], "run_id": record["run_id"],
            }

    dataset_dir = resolve_dataset_dir(record)
    split = cfg.split
    device = _device(cfg)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    norm = _load_norm(ckpt, dataset_dir)

    split_dir = dataset_dir / split
    q = _load_csv(split_dir, CSV_FILTERED_Q)
    qd = _load_csv(split_dir, CSV_FILTERED_QD)
    qdd = _load_csv(split_dir, CSV_FILTERED_QDD)
    target = _load_csv(split_dir, CSV_FILTERED_TAU_MEASURED)
    physics_raw = _load_csv(split_dir, CSV_FILTERED_TAU_DECOMPOSED)

    features, physics = _make_inputs(norm, q, qd, qdd, physics_raw)

    model = _build_model(ckpt).to(device)
    state = ckpt.get("model_state") or ckpt
    # torch.compile wraps the module, prefixing every key with "_orig_mod.".
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    bs = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(features), bs):
            e = min(s + bs, len(features))
            xb = torch.from_numpy(features[s:e]).to(device)
            pb = torch.from_numpy(physics[s:e]).to(device)
            preds.append(model(xb, pb).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0) * norm["std_tau"] + norm["mean_tau"]

    # Hard guard: silent dataset/sample mismatch is the #1 failure mode here.
    n_expected = record.get("n_test", len(pred))
    if n_expected and len(pred) != n_expected:
        raise AssertionError(
            f"{record['run_id']}: got {len(pred)} samples, "
            f"registry expects {n_expected} (wrong dataset?)"
        )

    traj = _trajectories(dataset_dir, split)
    metrics = compute_metrics(pred, target)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        pred=pred.astype(np.float32),
        target=target.astype(np.float32),
        qd=qd.astype(np.float32),
        traj=np.array(traj, dtype=object),
        model_mtime=np.array(model_mtime),
    )
    return {
        "pred": pred, "target": target, "qd": qd,
        "traj": traj, "metrics": metrics,
        "arch": record["arch"], "run_id": record["run_id"],
    }


def prefetch_all(cfg: PlotConfig) -> dict[str, dict[str, Any]]:
    """Warm the cache for every champion once; return {arch: result}."""
    from .dataio import champions
    out: dict[str, dict[str, Any]] = {}
    for arch, rec in champions(cfg).items():
        out[arch] = predict_split(rec, cfg)
    return out


# ---------------------------------------------------------------------------
# inference cost measurement (FLOPs)
# ---------------------------------------------------------------------------


def _benchmark_cache_path(cfg: PlotConfig) -> Path:
    return Path(cfg.cache_dir) / "inference_benchmark.json"


def benchmark_inference(
    cfg: PlotConfig,
    *,
    force: bool = False,
) -> dict[str, float]:
    """Measure per-sample FLOPs for each champion model.

    Uses ``torch.utils.flop_counter.FlopCounterMode`` to count the actual
    floating-point operations in a single forward pass.  FLOPs are the
    hardware-independent measure of computational cost — they reflect the
    true work the model does, not PyTorch kernel-launch overhead (which
    penalises structured architectures like EDR that dispatch many small
    sub-networks sequentially).

    Returns ``{arch: flops_per_sample}``.
    Cached to ``<cache_dir>/inference_benchmark.json``.
    """
    import json as _json
    cache = _benchmark_cache_path(cfg)
    if cache.exists() and not force:
        with cache.open() as f:
            data = _json.load(f)
        from .dataio import champions
        champs = champions(cfg)
        if all(a in data for a in champs):
            return data

    from .dataio import champions, resolve_dataset_dir

    champs = champions(cfg)
    device = torch.device("cpu")  # FLOPs don't depend on device
    results: dict[str, float] = {}

    for arch, rec in champs.items():
        model_path = Path(rec["model_path"])
        if not model_path.exists():
            results[arch] = float("nan")
            continue

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        norm = _load_norm(ckpt, resolve_dataset_dir(rec))

        dataset_dir = resolve_dataset_dir(rec)
        split_dir = dataset_dir / cfg.split
        q = _load_csv(split_dir, CSV_FILTERED_Q)
        qd = _load_csv(split_dir, CSV_FILTERED_QD)
        qdd = _load_csv(split_dir, CSV_FILTERED_QDD)
        physics_raw = _load_csv(split_dir, CSV_FILTERED_TAU_DECOMPOSED)
        features, physics = _make_inputs(norm, q, qd, qdd, physics_raw)

        # Single sample for per-sample FLOP count
        feat_t = torch.from_numpy(features[:1])
        phys_t = torch.from_numpy(physics[:1])

        model = _build_model(ckpt).to(device)
        state = ckpt.get("model_state") or ckpt
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()

        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad():
            with FlopCounterMode(display=False) as fcm:
                model(feat_t, phys_t)
        flops = fcm.get_total_flops()
        results[arch] = float(flops)
        print(f"  {arch:>10s}: {flops:>10,} FLOPs/sample")

    # Cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        _json.dump(results, f, indent=2)

    return results


def benchmark_inference_time(
    cfg: PlotConfig,
    *,
    force: bool = False,
    n_warmup: int = 50,
    n_timed: int = 200,
) -> dict[str, float]:
    """Measure real wall-clock per-sample inference time (seconds) for each champion.

    Uses CPU timing with warm-up passes to get stable measurements.
    Returns ``{arch: seconds_per_sample}``.
    Cached to ``<cache_dir>/inference_time.json``.
    """
    import json as _json
    import time

    cache = Path(cfg.cache_dir) / "inference_time.json"
    if cache.exists() and not force:
        with cache.open() as f:
            data = _json.load(f)
        from .dataio import champions
        champs = champions(cfg)
        if all(a in data for a in champs):
            return data

    from .dataio import champions, resolve_dataset_dir

    champs = champions(cfg)
    device = torch.device("cpu")
    results: dict[str, float] = {}

    for arch, rec in champs.items():
        model_path = Path(rec["model_path"])
        if not model_path.exists():
            results[arch] = float("nan")
            continue

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        norm = _load_norm(ckpt, resolve_dataset_dir(rec))

        dataset_dir = resolve_dataset_dir(rec)
        split_dir = dataset_dir / cfg.split
        q = _load_csv(split_dir, CSV_FILTERED_Q)
        qd = _load_csv(split_dir, CSV_FILTERED_QD)
        qdd = _load_csv(split_dir, CSV_FILTERED_QDD)
        physics_raw = _load_csv(split_dir, CSV_FILTERED_TAU_DECOMPOSED)
        features, physics = _make_inputs(norm, q, qd, qdd, physics_raw)

        # Single sample
        feat_t = torch.from_numpy(features[:1])
        phys_t = torch.from_numpy(physics[:1])

        model = _build_model(ckpt).to(device)
        state = ckpt.get("model_state") or ckpt
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()

        # Warm-up
        with torch.no_grad():
            for _ in range(n_warmup):
                model(feat_t, phys_t)

        # Timed runs
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(n_timed):
                model(feat_t, phys_t)
            t1 = time.perf_counter()

        per_sample = (t1 - t0) / n_timed
        results[arch] = per_sample
        print(f"  {arch:>10s}: {per_sample*1e6:.1f} µs/sample")

    # Cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        _json.dump(results, f, indent=2)

    return results


