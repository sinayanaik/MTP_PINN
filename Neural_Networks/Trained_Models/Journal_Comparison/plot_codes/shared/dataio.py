"""Read-only access to the grid-study artefacts.

Everything here returns *local* paths (foreign training-box paths are remapped)
and canonical architecture keys.  The single sources of truth are:

  * ``models_registry.yaml``  -> 48 trained models w/ val/test metrics + hparams
  * ``grid_results.csv``      -> the 30-run data-fraction sweep
  * ``<run_dir>/training_history.csv``  -> per-epoch curves
  * ``<run_dir>/architecture.txt``      -> parameter count
  * ``<run_dir>/metadata.yaml``         -> per-model ``data_run_dir``
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .bootstrap import GRID_CSV, REGISTRY_PATH, REPO_ROOT, remap
from .palette import canon_arch

# Verified correct training dataset (registry num_test_samples == 87596).
_FALLBACK_DATASET = (
    REPO_ROOT / "Neural_Networks" / "train_data"
    / "run_abl_q0_qd91_qdd91_tau51_lk_3to1_20260515_1923"
)


@lru_cache(maxsize=1)
def load_registry() -> list[dict[str, Any]]:
    with REGISTRY_PATH.open() as f:
        reg = yaml.safe_load(f)
    return list(reg["models"])


def _rec(m: dict[str, Any]) -> dict[str, Any]:
    """Flatten one registry entry into a plotting-friendly record."""
    tm = m.get("test_metrics") or {}
    vm = m.get("val_metrics") or {}
    tr = m.get("training") or {}
    dat = m.get("data") or {}
    hp = m.get("hyperparams") or {}
    return {
        "arch": canon_arch(m["model_type"]),
        "model_type": m["model_type"],
        "run_id": m["run_id"],
        "run_dir": remap(m["run_dir"]),
        "model_path": remap(m["model_path"]),
        "data_train_fraction": float(hp.get("data_train_fraction", 1.0)),
        "seed": int(hp.get("seed", 0)) if hp.get("seed") is not None else 0,
        "physics_weight": hp.get("physics_weight"),
        "n_test": int(dat.get("num_test_samples", 0)),
        "n_train": int(dat.get("num_train_samples", 0)),
        "time_seconds": float(tr.get("time_seconds", float("nan"))),
        "epochs_ran": int(tr.get("epochs_ran", 0)),
        "test_rmse": float(tm.get("rmse_traj_macro", tm.get("rmse_pooled", float("nan")))),
        "test_rmse_pooled": float(tm.get("rmse_pooled", float("nan"))),
        "test_r2": float(tm.get("r2_overall", float("nan"))),
        "test_r2_mean": float(tm.get("r2_mean", float("nan"))),
        "val_rmse": float(vm.get("rmse_traj_macro", vm.get("rmse_pooled", float("nan")))),
        "test_per_joint_rmse": list(tm.get("per_joint_rmse") or []),
        "_raw": m,
    }


@lru_cache(maxsize=1)
def registry_records() -> list[dict[str, Any]]:
    return [_rec(m) for m in load_registry()]


def registry_df() -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != "_raw"}
                         for r in registry_records()])


def champions(metric: str = "rmse_traj_macro") -> dict[str, dict[str, Any]]:
    """Best model per architecture (lowest test ``metric``)."""
    best: dict[str, dict[str, Any]] = {}
    for r in registry_records():
        a = r["arch"]
        if a not in best or r["test_rmse"] < best[a]["test_rmse"]:
            best[a] = r
    return best


def _config_sig(hp: dict[str, Any]) -> tuple:
    """Seed/fraction-independent config fingerprint for one registry entry.

    Two runs of the *same* model configuration that differ only in seed or
    ``data_train_fraction`` (i.e. the data-efficiency study) share this key.
    """
    skip = {"seed", "data_train_seed", "_grid_seed", "data_train_fraction"}
    items = []
    for k, v in sorted((hp or {}).items()):
        if k in skip or str(k).startswith("_"):
            continue
        items.append((k, tuple(v) if isinstance(v, list) else v))
    return tuple(items)


def sweep_df() -> pd.DataFrame:
    """Data-efficiency curve — seed-averaged, like-for-like across fractions.

    Honest construction (replaces the old ``idxmin`` over the full registry):

      1. **Like-for-like**: the data-efficiency study is ONE fixed config per
         arch trained at every fraction.  DETAILED ablation rows live only at
         fraction=1.0, so a naive per-(arch,fraction) ``idxmin`` mixed the
         100% point ("best of all configs") with the 10–90% points ("one
         config") — a distorted, non-comparable curve.  We isolate the true
         data-efficiency config per arch as the config-signature that spans
         the most distinct fractions (it appears at all 10; ablations at one).
      2. **Seed-averaged**: with multi-seed DATAEFF, each (arch,fraction) is
         the **mean over seeds** (single-seed noise — a ~0.0008 N·m wiggle —
         was what made PhysReg look like its test RMSE "increases" with more
         data; it is actually ~flat).  ``test_rmse_std`` carries the spread so
         the figures can draw an honest ±1σ band.
    """
    recs = registry_records()
    rows = []
    for r in recs:
        hp = (r["_raw"].get("hyperparams") or {})
        rows.append({
            "arch": r["arch"], "run_id": r["run_id"], "run_dir": r["run_dir"],
            "data_train_fraction": r["data_train_fraction"],
            "seed": r["seed"], "test_rmse": r["test_rmse"],
            "val_rmse": r["val_rmse"], "_sig": _config_sig(hp),
        })
    df = pd.DataFrame(rows)
    keep = []
    for a, g in df.groupby("arch"):
        # The data-efficiency config = the signature covering the most
        # distinct fractions (ties → most rows).  Robust to DETAILED/QUICK
        # rows leaking into the same registry.
        span = g.groupby("_sig")["data_train_fraction"].nunique()
        cnt = g.groupby("_sig").size()
        best_sig = sorted(span.index, key=lambda s: (span[s], cnt[s]))[-1]
        keep.append(g[g["_sig"] == best_sig])
    sel = pd.concat(keep, ignore_index=True)
    agg = (
        sel.groupby(["arch", "data_train_fraction"])
        .agg(test_rmse=("test_rmse", "mean"),
             test_rmse_std=("test_rmse", "std"),
             val_rmse=("val_rmse", "mean"),
             n_seeds=("seed", "nunique"))
        .reset_index()
        .sort_values(["arch", "data_train_fraction"])
        .reset_index(drop=True)
    )
    agg["test_rmse_std"] = agg["test_rmse_std"].fillna(0.0)
    return agg


@lru_cache(maxsize=1)
def grid_df() -> pd.DataFrame:
    df = pd.read_csv(GRID_CSV)
    df["arch"] = df["arch"].map(canon_arch)
    return df


def load_history(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "training_history.csv")


_PARAM_RE = re.compile(r"Params:\s*([\d,]+)")


def param_count(run_dir: Path) -> int:
    txt = (Path(run_dir) / "architecture.txt").read_text()
    m = _PARAM_RE.search(txt)
    if not m:
        raise ValueError(f"no 'Params:' line in {run_dir}/architecture.txt")
    return int(m.group(1).replace(",", ""))


def resolve_dataset_dir(record: dict[str, Any]) -> Path:
    """Local dataset dir for a model (from its metadata.yaml, remapped)."""
    meta_path = Path(record["run_dir"]) / "metadata.yaml"
    if meta_path.is_file():
        with meta_path.open() as f:
            meta = yaml.safe_load(f)
        cand = remap(meta.get("data_run_dir", ""))
        if cand and cand.is_dir():
            return cand
    if _FALLBACK_DATASET.is_dir():
        return _FALLBACK_DATASET
    raise FileNotFoundError(
        f"cannot resolve a local dataset dir for {record['run_id']}"
    )
