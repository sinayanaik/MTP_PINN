"""Unit tests for the combined two-stage grid orchestration.

Covers the NEW logic added for `MTP_GRID_MODE=combined`:
  * Stage-A winner selection (min test_rmse, fold reconstructed, omit missing)
  * Stage-B trial construction (winner × fractions × seeds, no re-expansion)
  * `_config_key` seed-invariance for the expanded EDR axes
  * `warmup_cosine_min_factor` LR-floor (EDR-only; default unchanged)
  * `_result_basenames` legacy mirror only for detailed
  * the trimmed-210 EDR grid + new EDR horizon HP
  * the runner refactor (collect + behavior-preserving wrapper)
"""
from __future__ import annotations

import math

import pytest
import torch

from Neural_Networks.models import run_journal_grid_3model as G
from Neural_Networks.models.shared.optim import build_scheduler


def _R(arch, rmse, status="ok", **hp):
    return G._Result(n=0, arch=arch, config="c", status=status,
                      rmse=rmse, elapsed=0.0, hp={"epochs": 10, **hp})


# ── Stage-A winners ─────────────────────────────────────────────────────────

def test_winner_is_min_test_rmse_per_arch():
    ex = [
        _R("fnn", 0.10, lr=1), _R("fnn", 0.08, lr=2), _R("fnn", 0.12, lr=3),
        _R("edr", 0.09, w=1), _R("edr", 0.07, w=2),
    ]
    w = G._stage_a_winners(ex, [], "/tmp/nonexistent_xyz", G.logging.getLogger("t"))
    assert w["fnn"]["lr"] == 2
    assert w["edr"]["w"] == 2
    # winner hp is a *copy*
    w["fnn"]["lr"] = 999
    assert ex[1].hp["lr"] == 2


def test_winner_ignores_failed_and_nan():
    ex = [
        _R("fnn", None, status="fail", lr=1),
        _R("fnn", None, status="err", lr=2),
        _R("fnn", 0.11, status="ok", lr=3),
    ]
    w = G._stage_a_winners(ex, [], "/tmp/nonexistent_xyz", G.logging.getLogger("t"))
    assert w["fnn"]["lr"] == 3


def test_winner_missing_arch_omitted():
    ex = [_R("fnn", 0.10)]
    w = G._stage_a_winners(ex, [], "/tmp/nonexistent_xyz", G.logging.getLogger("t"))
    assert "fnn" in w and "edr" not in w and "physreg" not in w


def test_winner_skip_status_counts_as_completed():
    ex = [_R("edr", 0.06, status="skip", w=1), _R("edr", 0.09, status="ok", w=2)]
    w = G._stage_a_winners(ex, [], "/tmp/nonexistent_xyz", G.logging.getLogger("t"))
    assert w["edr"]["w"] == 1


# ── Stage-B trial construction ──────────────────────────────────────────────

def test_stage_b_trials_shape_and_no_reexpansion():
    win = {
        "fnn": {"hidden_layers": [128, 128], "seed": 42,
                "data_train_fraction": 1.0},
        "edr": {"gravity_hidden": [96, 96], "inertia_hidden": [96, 96],
                "coriolis_hidden": [96, 96], "friction_hidden": [48, 48],
                "seed": 42, "data_train_fraction": 1.0},
    }
    trials = G._build_stage_b_trials(win)
    per_arch = len(G._DATAEFF_FRACTIONS) * len(G._DATAEFF_SEEDS)
    assert len(trials) == 2 * per_arch
    by = {}
    for t in trials:
        by.setdefault(t["arch"], []).append(t)
    assert len(by["fnn"]) == per_arch and len(by["edr"]) == per_arch
    for t in by["edr"]:
        hp = t["hp"]
        assert "edr_width" not in hp                       # not re-expanded
        assert hp["gravity_hidden"] == [96, 96]            # widths preserved
        assert hp["data_train_seed"] == hp["seed"]
        assert hp["seed"] in G._DATAEFF_SEEDS
        assert hp["data_train_fraction"] in G._DATAEFF_FRACTIONS
    # fraction=1.0 × 3 seeds present => the multi-seed headline subset exists
    head = [t for t in by["edr"] if t["hp"]["data_train_fraction"] == 1.0]
    assert sorted(t["hp"]["seed"] for t in head) == sorted(G._DATAEFF_SEEDS)


def test_stage_b_injects_trig_stats_for_edr_when_available():
    if not G._EDR_Q_STATS:                  # dataset metadata absent in CI
        pytest.skip("no _EDR_Q_STATS available")
    win = {"edr": {"gravity_hidden": [64, 64], "seed": 42,
                   "data_train_fraction": 1.0}}
    t = G._build_stage_b_trials(win)[0]
    assert "_q_mean" in t["hp"] and "_q_std" in t["hp"]


# ── _config_key seed-invariance for the expanded EDR axes ───────────────────

def test_config_key_seed_invariant_but_axis_sensitive():
    a = {"lambda_correction_reg": 0.1, "correction_dropout": 0.15,
         "seed": 42, "data_train_seed": 42}
    b = {**a, "seed": 1, "data_train_seed": 1}
    c = {**a, "lambda_correction_reg": 0.2}
    assert G._config_key(a) == G._config_key(b)            # seed ignored
    assert G._config_key(a) != G._config_key(c)            # real axis matters


# ── warmup_cosine LR floor (EDR-only) ───────────────────────────────────────

def _floor(hp):
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sch = build_scheduler(opt, hp, n_train_batches=10)
    # LambdaLR multiplies base lr; at ep==epochs the cosine term is 0 ⇒ factor
    # equals min_factor exactly.
    return sch.lr_lambdas[0](hp["epochs"])


def test_min_factor_default_is_one_percent():
    assert math.isclose(_floor({"lr_scheduler": "warmup_cosine", "epochs": 1000}),
                         0.01, rel_tol=1e-9)


def test_min_factor_edr_override_is_five_percent():
    assert math.isclose(
        _floor({"lr_scheduler": "warmup_cosine", "epochs": 1500,
                "warmup_cosine_min_factor": 0.05}),
        0.05, rel_tol=1e-9)


def test_fnn_physreg_fixed_hp_keep_default_floor():
    assert "warmup_cosine_min_factor" not in G.FIXED_HP_FNN
    assert "warmup_cosine_min_factor" not in G.FIXED_HP_PHYSREG
    assert G.FIXED_HP_EDR["warmup_cosine_min_factor"] == 0.05


# ── artifact basenames: legacy mirror only for detailed ─────────────────────

def test_result_basenames_mirror_only_detailed():
    old = G.RUN_MODE
    try:
        G.RUN_MODE = "detailed"
        names = [c for c, _ in G._result_basenames()]
        assert "grid_results.csv" in names and \
               "grid_results_detailed.csv" in names
        G.RUN_MODE = "dataeff"
        names = [c for c, _ in G._result_basenames()]
        assert names == ["grid_results_dataeff.csv"]
    finally:
        G.RUN_MODE = old


# ── new EDR grid + horizon ──────────────────────────────────────────────────

def test_edr_detailed_grid_is_tightened():
    # 2026-05-22: grid tightened to 16 = 2 widths × 2 λ × 2 cdrop × 2 fqdd × 1 seed.
    # Rationale: per-HP analysis of the prior 56-run grid showed
    # correction_dropout=0.05 dragged test_rmse from 0.092→0.106 (gap_mean
    # 0.022 vs 0.013) — dropped entirely. See project_edr_dropout_root_cause.
    # Single seed=[42] per user request; multi-seed verification lives in DATAEFF.
    assert len(G._cartesian(G.GRID_EDR_DETAILED)) == 16
    # The dropped values must remain dropped — guard against accidental restore.
    assert 0.05 not in G.GRID_EDR_DETAILED["correction_dropout"]
    assert 0.15 not in G.GRID_EDR_DETAILED["correction_dropout"]
    # Single seed in DETAILED — multi-seed coverage now lives in DATAEFF.
    assert G.GRID_EDR_DETAILED["seed"] == [42]


def test_horizon_per_arch():
    # 2026-05-22: horizon is no longer uniform — EDR was wastefully budgeted
    # at 1500 epochs but no run ever reached that (median 330, p90 612).
    # FNN/PhysReg keep their original horizon; EDR trimmed to 1000.
    assert G.FIXED_HP_FNN["epochs"] == 1500
    assert G.FIXED_HP_PHYSREG["epochs"] == 1500
    assert G.FIXED_HP_EDR["epochs"] == 1000
    # 2026-05-22: patience is now uniform at 50 across archs (was 150 for EDR).
    assert G.FIXED_HP_FNN["patience"] == 50
    assert G.FIXED_HP_PHYSREG["patience"] == 50
    assert G.FIXED_HP_EDR["patience"] == 50
    # structural levers stay OFF (no re-opened experiments)
    for k in ("coriolis_structural", "inertia_psd", "spectral_norm"):
        assert G.FIXED_HP_EDR[k] is False


def test_write_resume_status_roundtrip(tmp_path):
    # Resume-status writer (2026-05-22) records what is done/pending so the
    # user can read a single JSON between HPC jobs.
    import json
    skipped = [
        {"arch": "edr", "hp": {"seed": 42, "correction_dropout": 0.30,
                               "torch_compile": True, "_q_mean": [0.0]*5}},
        {"arch": "fnn", "hp": {"seed": 42}},
    ]
    pending = [
        {"arch": "edr", "hp": {"seed": 1, "correction_dropout": 0.30}},
    ]
    path = str(tmp_path / "status.json")
    G._write_resume_status(skipped, pending, path)
    with open(path) as f:
        blob = json.load(f)
    assert blob["total_planned"] == 3
    assert blob["completed_count"] == 2
    assert blob["pending_count"] == 1
    assert blob["completed_by_arch"] == {"edr": 1, "fnn": 1}
    assert blob["pending_by_arch"] == {"edr": 1}
    # Skip keys (torch_compile, _q_mean) are excluded from the pending signature
    # so the file stays human-readable.
    assert blob["pending_trials"][0]["hp_signature"] == {"seed": 1, "correction_dropout": 0.30}


def test_default_mode_is_combined(monkeypatch):
    log = G.logging.getLogger("t")
    monkeypatch.delenv("MTP_GRID_MODE", raising=False)
    # non-interactive (HPC): no env, no prompt → combined
    monkeypatch.setattr(G.sys.stdin, "isatty", lambda: False)
    assert G._select_run_mode(log) == "combined"
    # interactive but empty input (just hit Enter) → combined
    monkeypatch.setattr(G.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert G._select_run_mode(log) == "combined"


# ── runner refactor: behavior-preserving wrappers ───────────────────────────

def test_parallel_wrapper_collects_then_finalizes(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        G, "_run_parallel_collect",
        lambda *a, **k: (["RES"], ["SK"], (3, 1, 0)))
    monkeypatch.setattr(
        G, "_finalize",
        lambda res, sk, root, log: calls.setdefault("fin", (res, sk)))
    out = G._run_parallel_dynamic([], 4, 1, "cpu",
                                  G.logging.getLogger("t"), 0.0, n_gpus=1)
    assert out == (3, 1, 0)
    assert calls["fin"] == (["RES"], ["SK"])


def test_sequential_wrapper_collects_then_finalizes(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        G, "_run_sequential_collect",
        lambda *a, **k: ([], ["ALL"], (2, 0, 1)))
    monkeypatch.setattr(
        G, "_finalize",
        lambda res, sk, root, log: calls.setdefault("fin", (res, sk)))
    out = G._run_sequential([], G.logging.getLogger("t"), 0.0)
    assert out == (2, 0, 1)
    assert calls["fin"] == ([], ["ALL"])


# ── _run_combined end-to-end orchestration (training fully stubbed) ─────────

def test_run_combined_chains_stage_a_to_b(monkeypatch, tmp_path):
    """Stage A → winners → Stage B → both finalized → summary written,
    with NO real training and writing only into a tmp dir."""
    out = str(tmp_path)
    monkeypatch.setattr(G, "DATASET_OUT_ROOT", out)
    monkeypatch.setattr(G, "DRY_RUN", False)
    monkeypatch.setattr(G, "_probe_runtime",
                        lambda log: {"cuda_ok": False, "n_gpus": 1,
                                     "pool_size": 1, "threads_per_worker": 1,
                                     "device_label": "cpu"})
    monkeypatch.setattr(G, "_run_metrics_for", lambda *a, **k: {})

    fin_calls, seen_trials = [], {}

    def fake_collect(trials, log, t_start):
        seen_trials[G.RUN_MODE] = list(trials)
        if G.RUN_MODE == "detailed":           # Stage A: fabricate results
            res = [
                G._Result(0, "fnn", "c", "ok", 0.10, 0.0,
                          {"hidden_layers": [128, 128]}),
                G._Result(0, "fnn", "c", "ok", 0.08, 0.0,
                          {"hidden_layers": [256, 256]}),   # fnn winner
                G._Result(0, "edr", "c", "ok", 0.07, 0.0,
                          {"gravity_hidden": [96, 96]}),     # edr winner
                G._Result(0, "edr", "c", "ok", 0.09, 0.0,
                          {"gravity_hidden": [64, 64]}),
                G._Result(0, "physreg", "c", "ok", 0.11, 0.0,
                          {"physics_weight": 0.5}),
            ]
            return res, [], (5, 0, 0)
        return [], [], (0, 0, 0)               # Stage B

    monkeypatch.setattr(G, "_run_sequential_collect", fake_collect)
    monkeypatch.setattr(
        G, "_finalize",
        lambda res, sk, root, log: fin_calls.append(G.RUN_MODE))
    # keep the real _build_trials but shrink Stage A so it's instant-ish; we
    # never actually train (collect is stubbed) so size only affects logging.
    monkeypatch.setattr(G, "_build_trials", lambda: [{"arch": "fnn"}])

    G._run_combined(G.logging.getLogger("t"))

    # finalize called once per stage, in order
    assert fin_calls == ["detailed", "dataeff"]
    # Stage B trials were built from the Stage-A winners (min test_rmse)
    sb = seen_trials["dataeff"]
    per_arch = len(G._DATAEFF_FRACTIONS) * len(G._DATAEFF_SEEDS)
    assert len(sb) == 3 * per_arch
    edr = [t for t in sb if t["arch"] == "edr"][0]
    assert edr["hp"]["gravity_hidden"] == [96, 96]      # the 0.07 winner
    fnn = [t for t in sb if t["arch"] == "fnn"][0]
    assert fnn["hp"]["hidden_layers"] == [256, 256]     # the 0.08 winner
    # additive summary written into the tmp out root, grid_* untouched
    import os
    assert os.path.isfile(os.path.join(out, "combined_summary.md"))


def test_run_combined_aborts_b_when_no_winner(monkeypatch, tmp_path):
    monkeypatch.setattr(G, "DATASET_OUT_ROOT", str(tmp_path))
    monkeypatch.setattr(G, "DRY_RUN", False)
    monkeypatch.setattr(G, "_probe_runtime",
                        lambda log: {"cuda_ok": False, "n_gpus": 1,
                                     "pool_size": 1, "threads_per_worker": 1,
                                     "device_label": "cpu"})
    monkeypatch.setattr(G, "_build_trials", lambda: [{"arch": "fnn"}])
    monkeypatch.setattr(G, "_run_sequential_collect",
                        lambda *a, **k: ([], [], (0, 0, 3)))   # all failed
    fin = []
    monkeypatch.setattr(G, "_finalize",
                        lambda *a, **k: fin.append(G.RUN_MODE))
    G._run_combined(G.logging.getLogger("t"))
    # Stage A finalized; Stage B never runs (no winners)
    assert fin == ["detailed"]
