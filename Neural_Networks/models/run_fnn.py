#!/usr/bin/env python3
"""Plain black-box torque MLP training.

From repository root::

    PYTHONPATH=. python3 -m Neural_Networks.models.run_fnn
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Neural_Networks.models.shared.pipeline import TrainJob, main_cli
from Neural_Networks.models.shared.strategies import PLAIN_STRATEGY

_NN_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# Set TRAIN_DATA_RUN_DIR to the dataset folder you want to train on.
# ------------------------------------------------------------------
TRAIN_DATA_RUN_DIR = str(
    _NN_ROOT / "train_data"
    / "run_0419_1338_qraw_d25p3i_ddL_mraw_a1_R_70v15t15_f1p0t1p0_f6a3df"
)

MODELS_DIR = str(_NN_ROOT / "Trained_Models" / "FNN")
REGISTRY_FILE = str(_NN_ROOT / "Trained_Models" / "models_registry.yaml")

HP: dict[str, Any] = {
    "epochs": 1000,
    "batch_size": 512,
    "learning_rate": 3e-4,
    "weight_decay": 5e-3,
    "dropout": 0.4,
    "activation": "silu",
    "hidden_layers": [256, 512, 256],
    "optimizer": "adamw",
    "lr_scheduler": "warmup_cosine",
    "early_stopping": True,
    "early_stop_metric": "val_rmse",
    "patience": 50,
    "min_delta": 1e-4,
    "grad_clip_norm": 5.0,
    "feature_noise_std": 0.02,
    "data_train_fraction": 1.0,
    "data_train_seed": 0,
    "stride": 1,
    "seed": 42,
    "torch_compile": False,
    "torch_compile_mode": "default",
    "snapshot_every": 0,
}

MODEL_TYPE = "BlackBoxFNN"
SAVE_SUBDIR = "FNN"


def main() -> None:
    job = TrainJob(
        run_dir=TRAIN_DATA_RUN_DIR,
        models_dir=MODELS_DIR,
        registry_file=REGISTRY_FILE,
        model_type=MODEL_TYPE,
        save_subdir=SAVE_SUBDIR,
        hp=HP,
        strategy=PLAIN_STRATEGY,
        run_help="Neural_Networks/models/run_fnn.py",
    )
    main_cli(job, log_name="run_fnn")


if __name__ == "__main__":
    main()
