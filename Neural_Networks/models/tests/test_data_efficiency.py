"""Data-efficiency regression: data_train_fraction must subsample ONLY the
train split, deterministically by data_train_seed, with val/test always full —
and via the single shared make_dataloaders path (uniform for FNN/PhysReg/EDR).

    PYTHONPATH=. pytest Neural_Networks/models/tests/test_data_efficiency.py -q
"""
import importlib
import inspect
import os

import pytest

from Neural_Networks.loader import make_dataloaders

GR = importlib.import_module("Neural_Networks.models.run_journal_grid_3model")
RUN_DIR = GR.TRAIN_DATA_RUN_DIR

pytestmark = pytest.mark.skipif(
    not os.path.isdir(RUN_DIR), reason=f"dataset {RUN_DIR} not present"
)


def _sizes(frac, seed=0):
    d = make_dataloaders(RUN_DIR, batch_size=8192, num_workers=0,
                          data_train_fraction=frac, data_train_seed=seed)
    return {s: len(d[s].dataset) for s in ("train", "val", "test")}, d


def test_val_test_full_regardless_of_fraction():
    full, _ = _sizes(1.0)
    for frac in (0.1, 0.5):
        sz, _ = _sizes(frac)
        assert sz["val"] == full["val"], (frac, sz, full)
        assert sz["test"] == full["test"], (frac, sz, full)
        # train shrinks ≈ proportionally (loader uses round(), min 1)
        assert sz["train"] == max(1, round(full["train"] * frac)), (frac, sz)


def test_train_subsample_deterministic_by_seed():
    _, d_a = _sizes(0.25, seed=42)
    _, d_b = _sizes(0.25, seed=42)
    _, d_c = _sizes(0.25, seed=7)
    idx_a = list(d_a["train"].dataset.indices)
    idx_b = list(d_b["train"].dataset.indices)
    idx_c = list(d_c["train"].dataset.indices)
    assert idx_a == idx_b                 # same (frac, seed) ⇒ identical subset
    assert idx_a != idx_c                 # different seed ⇒ different subset
    assert len(set(idx_a)) == len(idx_a)  # no duplicate samples


def test_fraction_bounds_validated():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            make_dataloaders(RUN_DIR, data_train_fraction=bad)


def test_subsampling_path_is_shared_not_per_arch():
    """The only data-loading entry is pipeline.run_training → make_dataloaders;
    no strategy/EDR module loads data itself, so data-efficiency is uniform
    across all three architectures by construction."""
    import pathlib
    import Neural_Networks.models.shared.pipeline as P
    import Neural_Networks.models.shared.strategies as S
    assert "make_dataloaders" in inspect.getsource(P.run_training)
    assert "make_dataloaders" not in inspect.getsource(S)
    edr_src = (pathlib.Path(GR.__file__).parent
               / "Equivariant-Decomposed-Residual" / "edr_strategy.py").read_text()
    assert "make_dataloaders" not in edr_src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
