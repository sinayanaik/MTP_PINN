"""B1 non-finite guard: a NaN/Inf loss must abort the run FAST and LOUD —
never silently burn the (now 200-epoch) patience budget poisoning a model.

Two real cases, run through the actual pipeline on a tiny data subsample:
  • NaN from epoch 1  → no finite checkpoint → RuntimeError within ~1 epoch
  • NaN after a good epoch → keep best (treated like early stop), no raise

    PYTHONPATH=. pytest Neural_Networks/models/tests/test_pipeline_robustness.py -q
"""
import dataclasses
import importlib
import os

import numpy as np
import pytest

GR = importlib.import_module("Neural_Networks.models.run_journal_grid_3model")
import Neural_Networks.models.shared.strategies as S
from Neural_Networks.models.shared.pipeline import TrainJob, run_training
from Neural_Networks.models.shared.strategies import TrainEpochMetrics

RUN_DIR = GR.TRAIN_DATA_RUN_DIR
pytestmark = pytest.mark.skipif(
    not os.path.isdir(RUN_DIR), reason=f"dataset {RUN_DIR} not present"
)

_NAN_METRICS = TrainEpochMetrics(
    loss_total=float("nan"), loss_data_unw=float("nan"), grad_norm=0.0,
    sse_per_joint=np.zeros(5, dtype=np.float64), n_samples=1, extras={},
)


def _hp():
    hp = dict(GR.FIXED_HP_FNN)
    hp.update(
        hidden_layers=[8], dropout=0.0, epochs=50, patience=20,
        batch_size=16384, data_train_fraction=0.03, data_train_seed=0,
        feature_noise_std=0.0, early_stopping=True, torch_compile=False,
        snapshot_every=0, print_every=999,
    )
    return hp


def _job(tmp_path, strategy):
    return TrainJob(
        run_dir=RUN_DIR,
        models_dir=str(tmp_path / "models"),
        registry_file=str(tmp_path / "registry.yaml"),
        model_type="FNN",
        save_subdir="fnn",
        hp=_hp(),
        strategy=strategy,
        run_help="",
    )


def test_nan_from_epoch1_fails_fast_and_loud(tmp_path):
    calls = []

    def nan_train_epoch(*a, **k):
        calls.append(1)
        return _NAN_METRICS

    strat = dataclasses.replace(S.PLAIN_STRATEGY, train_epoch=nan_train_epoch)
    with pytest.raises(RuntimeError, match="diverged"):
        run_training(_job(tmp_path, strat))
    # The whole point: it aborted in ~1 epoch, NOT after the 20-epoch patience.
    assert len(calls) <= 2, f"ran {len(calls)} epochs — patience budget burned"


def test_nan_after_good_epoch_keeps_best_no_raise(tmp_path):
    calls = []
    real = S.PLAIN_STRATEGY.train_epoch

    def late_nan_train_epoch(model, loader, optimizer, device, hp, epoch,
                             onecycle_sched, scaler):
        calls.append(epoch)
        if epoch <= 1:                       # one genuine, finite epoch
            return real(model, loader, optimizer, device, hp, epoch,
                        onecycle_sched, scaler)
        return _NAN_METRICS                  # then diverge

    strat = dataclasses.replace(S.PLAIN_STRATEGY, train_epoch=late_nan_train_epoch)
    out = run_training(_job(tmp_path, strat))   # must NOT raise
    assert isinstance(out, float) and np.isfinite(out), out
    # Stopped at the divergence (epoch 2), not after patience-many epochs.
    assert len(calls) <= 3, f"ran {len(calls)} epochs — should stop at divergence"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
