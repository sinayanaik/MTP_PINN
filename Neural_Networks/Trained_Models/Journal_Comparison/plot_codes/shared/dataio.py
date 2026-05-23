"""Authoritative, complete model index for the Journal_Comparison suite.

The single source of truth is the **curated grid record** — the two CSVs the
training pipeline reconstructs from disk:

  * ``grid_results.csv``          -> final architecture / HP sweep (frac = 1.0)
  * ``grid_results_dataeff.csv``  -> data-efficiency sweep (frac 0.1 .. 1.0)

Each CSV row is the *curated* identity of one run (``test_rmse``,
``elapsed_sec``, hyperparams, fraction, seed, status); the matching on-disk
directory under ``EDR/`` | ``FNN/`` | ``PhysicsRegularized/`` holds the
artefacts needed for the figures:

  * ``model.pt``               -> weights (inference)
  * ``metadata.yaml``          -> full val / test metrics (incl. per-joint),
                                  hyperparams, ``data_run_dir``
  * ``training_history.csv``   -> per-epoch curves (+ EDR delta magnitudes)
  * ``architecture.txt``       -> ``Params:`` line

Why not ``models_registry.yaml``?  It is stale/incomplete (the FNN and most
PhysReg *detailed* runs were never registered), so it under-represents the
baselines.  Why not scan the dirs directly?  ``EDR/`` also contains ~55 stale
experimental runs from earlier rounds — scanning blindly would cherry-pick
EDR's best from a much larger pool than the baselines.  Joining *grid row ->
dir* uses the CSVs as the curation filter and the dirs as the artefact store.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .bootstrap import GRID_CSV, JOURNAL_DIR, REPO_ROOT, remap
from .config import ARCH_EDR, ARCH_FNN, ARCH_PHYSREG
from .palette import canon_arch

GRID_DATAEFF_CSV = JOURNAL_DIR / "grid_results_dataeff.csv"

# Canonical architecture key -> on-disk sub-directory holding its run folders.
_ARCH_SUBDIR = {
    ARCH_EDR: "EDR",
    ARCH_FNN: "FNN",
    ARCH_PHYSREG: "PhysicsRegularized",
}

# Verified-correct training dataset (used only as a last-resort fallback when a
# model's metadata.yaml cannot be read; num_test_samples == 87596).
_FALLBACK_DATASET = (
    REPO_ROOT / "Neural_Networks" / "train_data"
    / "run_abl_q0_qd91_qdd91_tau51_lk_3to1_20260515_1923"
)

# Run-dir names encode the rounded test RMSE and the data fraction, e.g.
# ``EDR_ep415_rmse0.09048_frac1_seed42_...`` / ``..._frac0.3_...``.
_NAME_RE = re.compile(r"_rmse([0-9.]+)_frac([0-9.]+(?:e-?\d+)?)")


# ---------------------------------------------------------------------------
# on-disk scan + grid-row -> dir resolution
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _arch_candidates(arch: str) -> tuple[tuple[Path, float, float], ...]:
    """All run dirs for ``arch`` as (dir, frac_from_name, rmse_from_name)."""
    base = JOURNAL_DIR / _ARCH_SUBDIR[arch]
    out: list[tuple[Path, float, float]] = []
    if not base.is_dir():
        return tuple(out)
    for d in base.iterdir():
        if not d.is_dir():
            continue
        m = _NAME_RE.search(d.name)
        if not m:
            continue
        out.append((d, float(m.group(2)), float(m.group(1))))
    return tuple(out)


@lru_cache(maxsize=None)
def _meta_rmse(run_dir: str) -> float:
    """Full-precision test ``rmse_traj_macro`` from a dir's metadata.yaml."""
    meta = _read_meta(Path(run_dir))
    return float((meta.get("test_metrics") or {}).get("rmse_traj_macro", float("nan")))


def _resolve_dir(arch: str, frac: float, test_rmse: float) -> Path | None:
    """Map one grid row to its on-disk run directory.

    Match on (arch, fraction, rounded RMSE).  Disambiguate collisions and
    rounding edge-cases by the full-precision test RMSE in each candidate's
    metadata.yaml (robust to the lossy 5-dp RMSE in the dir name).
    """
    cands = [c for c in _arch_candidates(arch) if abs(c[1] - frac) < 1e-6]
    if not cands:
        return None
    target5 = round(test_rmse, 5)
    exact = [c for c in cands if round(c[2], 5) == target5]
    pool = exact if exact else cands
    if len(pool) == 1:
        return pool[0][0]
    # Tie / rounding: pick the dir whose true metadata RMSE is closest.
    return min(pool, key=lambda c: abs(_meta_rmse(str(c[0])) - test_rmse))[0]


@lru_cache(maxsize=None)
def _read_meta(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "metadata.yaml"
    if not path.is_file():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# record construction
# ---------------------------------------------------------------------------
def _record(arch: str, frac: float, seed: int, elapsed_sec: float,
            run_dir: Path) -> dict[str, Any]:
    meta = _read_meta(run_dir)
    tm = meta.get("test_metrics") or {}
    vm = meta.get("val_metrics") or {}
    hp = meta.get("hyperparams") or {}
    # The reconstructed detailed grid stores elapsed_sec=0 ("unrecorded"); only
    # the data-efficiency sweep has real wall-times. Treat 0 / NaN as missing so
    # nothing fakes a zero training cost.
    t = float(elapsed_sec)
    time_seconds = t if (t == t and t > 0.0) else float("nan")
    return {
        "arch": arch,
        "model_type": meta.get("model_type", arch),
        "run_id": meta.get("run_id", run_dir.name),
        "run_dir": run_dir,
        "model_path": run_dir / "model.pt",
        "data_train_fraction": float(frac),
        "seed": int(seed),
        "physics_weight": hp.get("physics_weight"),
        "n_test": 0,  # not stored in metadata; inference falls back to len(pred)
        "time_seconds": time_seconds,
        "epochs_ran": int(meta.get("epochs_trained", 0) or 0),
        "test_rmse": float(tm.get("rmse_traj_macro", float("nan"))),
        "test_rmse_pooled": float(tm.get("rmse_pooled", float("nan"))),
        "test_r2": float(tm.get("r2_overall", float("nan"))),
        "test_r2_mean": float(tm.get("r2_mean", float("nan"))),
        "val_rmse": float(vm.get("rmse_traj_macro", float("nan"))),
        "test_per_joint_rmse": list(tm.get("rmse") or []),
        "hyperparams": hp,
        "_raw": meta,
    }


def _ok_grid_rows() -> pd.DataFrame:
    """Both grid CSVs concatenated, status==ok, canonical arch."""
    frames = []
    for csv in (GRID_CSV, GRID_DATAEFF_CSV):
        if Path(csv).is_file():
            frames.append(pd.read_csv(csv))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    df["arch"] = df["arch"].map(canon_arch)
    return df


@lru_cache(maxsize=1)
def model_index() -> list[dict[str, Any]]:
    """Every curated run, one record per on-disk dir (detailed + dataeff)."""
    df = _ok_grid_rows()
    seen: dict[Path, dict[str, Any]] = {}
    for _, r in df.iterrows():
        arch = r["arch"]
        frac = float(r["data_train_fraction"])
        d = _resolve_dir(arch, frac, float(r["test_rmse"]))
        if d is None or d in seen:
            continue
        seen[d] = _record(arch, frac, int(r["seed"]),
                          float(r.get("elapsed_sec", float("nan"))), d)
    return list(seen.values())


def registry_records() -> list[dict[str, Any]]:
    """Back-compat alias: the authoritative model index."""
    return model_index()


def registry_df() -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != "_raw"}
                         for r in model_index()])


# ---------------------------------------------------------------------------
# champions (global / train / val / test) -- the user wants all bases
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def train_at_best(run_dir: str) -> tuple[float, int]:
    """(train_rmse at the best-validation epoch, that epoch)."""
    h = load_history(Path(run_dir))
    i = int(h["val_rmse"].to_numpy().argmin())
    return float(h["train_rmse"].iloc[i]), int(h["epoch"].iloc[i])


def champion_by_basis(basis: str = "test", *, full_data_only: bool = True
                       ) -> dict[str, dict[str, Any]]:
    """Best run per architecture under one selection ``basis``.

    ``basis``:
      * ``test``   -> lowest test ``rmse_traj_macro``
      * ``val``    -> lowest validation ``rmse_traj_macro`` (no test leakage)
      * ``train``  -> lowest train RMSE at the best-val epoch
      * ``global`` -> lowest test RMSE over *all* fractions (forces
                      ``full_data_only=False``)
    """
    recs = model_index()
    if basis == "global":
        full_data_only = False
    if full_data_only:
        recs = [r for r in recs if abs(r["data_train_fraction"] - 1.0) < 1e-9]

    def key(r: dict[str, Any]) -> float:
        if basis == "val":
            return r["val_rmse"]
        if basis == "train":
            return train_at_best(str(r["run_dir"]))[0]
        return r["test_rmse"]            # test / global

    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for r in recs:
        k = key(r)
        if k != k:                        # NaN guard
            continue
        a = r["arch"]
        if a not in best or k < best[a][0]:
            best[a] = (k, r)
    return {a: kr[1] for a, kr in best.items()}


def champions(cfg) -> dict[str, dict[str, Any]]:
    """Champions for the figures, per ``cfg.champion_basis`` / fraction policy."""
    return champion_by_basis(cfg.champion_basis,
                             full_data_only=cfg.champion_full_data_only)


# ---------------------------------------------------------------------------
# data-efficiency sweep (fig02 / fig03)
# ---------------------------------------------------------------------------
def sweep_df() -> pd.DataFrame:
    """Per (arch, fraction): seed-mean test/val RMSE from the dataeff sweep.

    Built straight from ``grid_results_dataeff.csv`` (the data-efficiency runs
    are one fixed config per arch across all fractions, so no config-signature
    disambiguation is needed).  ``test_rmse`` comes from the curated CSV;
    ``val_rmse`` from each resolved dir's metadata.  ``test_rmse_std`` carries
    the across-seed spread (0 here -- a single seed -- but kept for an honest
    ±1σ band if seeds are added later).
    """
    df = pd.read_csv(GRID_DATAEFF_CSV)
    df = df[df["status"] == "ok"].copy()
    df["arch"] = df["arch"].map(canon_arch)
    rows = []
    for _, r in df.iterrows():
        arch = r["arch"]
        frac = float(r["data_train_fraction"])
        d = _resolve_dir(arch, frac, float(r["test_rmse"]))
        val = float("nan")
        if d is not None:
            val = float((_read_meta(d).get("val_metrics") or {})
                        .get("rmse_traj_macro", float("nan")))
        rows.append({"arch": arch, "data_train_fraction": frac,
                     "seed": int(r["seed"]), "test_rmse": float(r["test_rmse"]),
                     "val_rmse": val})
    sel = pd.DataFrame(rows)
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


# ---------------------------------------------------------------------------
# grid distribution (fig01) + per-run helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def grid_df() -> pd.DataFrame:
    """The detailed architecture/HP sweep (frac=1.0) for the fig01 violin."""
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
    meta = _read_meta(Path(record["run_dir"]))
    cand = remap(meta.get("data_run_dir", ""))
    if cand and Path(cand).is_dir():
        return Path(cand)
    if _FALLBACK_DATASET.is_dir():
        return _FALLBACK_DATASET
    raise FileNotFoundError(
        f"cannot resolve a local dataset dir for {record['run_id']}"
    )
