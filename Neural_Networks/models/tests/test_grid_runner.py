"""Robustness + grid-shape locks for run_journal_grid_3model.py.

These tests are pure (no training, no CUDA): they pin the DETAILED grid
trial counts, the fairness invariants, the A1 recycle-safe GPU-id contract,
and the incremental-CSV helper.  Run:

    PYTHONPATH=. pytest Neural_Networks/models/tests/test_grid_runner.py -q
"""
import csv
import inspect
import importlib

import pytest

GR = importlib.import_module("Neural_Networks.models.run_journal_grid_3model")


# ── B1/B2: DETAILED grid shape + fairness ───────────────────────────────────

def _detailed_trials():
    """Build every DETAILED trial without training (mutates module globals
    exactly the way main() does, then restores them)."""
    saved_grid = GR._ARCH_GRID
    saved_arch = GR.ARCH
    try:
        GR._ARCH_GRID = dict(GR._ARCH_GRID_DETAILED)
        GR.ARCH = "all"
        return GR._build_trials()
    finally:
        GR._ARCH_GRID = saved_grid
        GR.ARCH = saved_arch


def test_detailed_trial_counts():
    trials = _detailed_trials()
    by_arch: dict[str, int] = {}
    for t in trials:
        by_arch[t["arch"]] = by_arch.get(t["arch"], 0) + 1
    assert by_arch == {"fnn": 60, "physreg": 96, "edr": 288}
    assert len(trials) == 444


def test_edr_drops_proven_bad_widths():
    """Moderate trim: no δ-net wider than [96,96] survives the sweep."""
    bad = {(128, 128), (192, 192)}
    for t in _detailed_trials():
        if t["arch"] != "edr":
            continue
        # edr_width was expanded into the three δ-net hidden lists.
        for key in ("gravity_hidden", "inertia_hidden", "coriolis_hidden"):
            assert tuple(t["hp"][key]) not in bad, (key, t["hp"][key])


def test_edr_fairness_flags_are_fixed_not_swept():
    """Every EDR trial shares the same robustness protocol (fair compare):
    the structural flags must NOT vary across the cartesian product."""
    fixed_expected = {
        "use_phys_cond": True,
        "coriolis_matrix_form": False,
        "friction_form": "mlp",
        "inertia_psd": False,
        "spectral_norm": False,
        "joint_loss_weights": None,
        "lambda_correction_reg_per_component": None,
    }
    edr = [t["hp"] for t in _detailed_trials() if t["arch"] == "edr"]
    assert edr, "no EDR trials built"
    for hp in edr:
        for k, v in fixed_expected.items():
            assert hp[k] == v, f"{k}={hp[k]!r} expected {v!r} (must be fixed)"


def test_physreg_grid_extended_high_and_trimmed_dropout():
    pr = [t["hp"] for t in _detailed_trials() if t["arch"] == "physreg"]
    assert {hp["physics_weight"] for hp in pr} == {0.05, 0.1, 0.25, 0.5, 1.0, 2.0}
    assert {hp["dropout"] for hp in pr} == {0.1, 0.3}


def test_fnn_feature_noise_axis_dropped():
    fnn = [t["hp"] for t in _detailed_trials() if t["arch"] == "fnn"]
    # Single fixed value inherited from FIXED_HP_FNN — not a swept axis.
    assert len({hp.get("feature_noise_std") for hp in fnn}) == 1


# ── A1: GPU-id assignment is recycle-safe (no shared draining queue) ─────────

def test_pool_init_takes_fixed_gpu_id_not_a_queue():
    """The recycle deadlock was a fixed-size ticket queue + blocking get in
    _pool_init.  Lock the fix: the param is a plain ``gpu_id`` scalar and the
    body never calls a blocking queue ``.get()`` for the GPU id."""
    params = list(inspect.signature(GR._pool_init).parameters)
    assert params == ["progress_queue", "threads_per_worker", "gpu_id", "is_hpc"]
    src = inspect.getsource(GR._pool_init)
    # Strip the docstring so its prose (which *describes* the old queue bug)
    # doesn't trip the body checks.
    body = src.split('"""')[-1]
    assert "gpu_ticket_q" not in body
    assert ".get(" not in body          # no blocking ticket pop in the body
    assert "_POOL_GPU_ID = int(gpu_id)" in body


def test_starvation_timeout_constant_present():
    assert isinstance(GR.STARVATION_TIMEOUT_SEC, (int, float))
    assert GR.STARVATION_TIMEOUT_SEC > 0


# ── A4: incremental CSV flush helper ────────────────────────────────────────

def test_flush_results_csv_roundtrip(tmp_path):
    R = GR._Result
    results = [
        R(n=1, arch="edr", config="c1", status="ok", rmse=0.0912,
          elapsed=12.3, hp={"seed": 42, "gravity_hidden": [64, 64],
                            "lambda_correction_reg": 0.05}),
        R(n=2, arch="fnn", config="c2", status="fail", rmse=None,
          elapsed=4.0, hp={"seed": 42, "dropout": 0.3}),
    ]
    GR._flush_results_csv(results, str(tmp_path))
    csv_path = tmp_path / "grid_results.csv"
    assert csv_path.exists()
    rows = list(csv.reader(csv_path.open()))
    assert rows[0][:5] == ["n", "arch", "status", "test_rmse", "elapsed_sec"]
    assert len(rows) == 1 + len(results)
    # List-valued HPs are serialised with "-".join (dash-joined), not Python repr.
    body = csv_path.read_text()
    assert "64-64" in body
    assert "[" not in body  # no list repr leaked into the CSV


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
