"""Constants shared across the analyzer package."""

from __future__ import annotations

from pathlib import Path

_NN_ROOT = Path(__file__).resolve().parent.parent
_GRID_ROOT = _NN_ROOT / "Trained_Models" / "Grid_Searches"
DEFAULT_MODELS_DIR = str(_GRID_ROOT)

N_JOINTS = 5
JOINT_NAMES = ["J1 (yaw)", "J2 (shoulder)", "J3 (elbow)", "J4 (wrist)", "J5 (wrist roll)"]
JOINT_NAMES_SHORT = ["J1", "J2", "J3", "J4", "J5"]

TRAIN_METRICS_CACHE_VERSION = 3

_TYPE_ABBREV: dict[str, str] = {
    "BlackBoxFNN": "FNN",
    "PhysicsRegularizedFNN": "PhysReg",
    "ResidualCorrectionFNN": "ResCorr",
    "EDR": "EDR",
}

_ARCH_COLOR_ORDER: list[str] = [
    "BlackBoxFNN",
    "PhysicsRegularizedFNN",
    "ResidualCorrectionFNN",
    "EDR",
]

_ARCH_DIR_NAMES: set[str] = {
    "FNN", "BlackBoxFNN",
    "PhysicsRegularizedFNN", "ResidualCorrectionFNN", "EDR",
}

_GRID_HP_KEYS_FNN: list[str] = [
    "hidden_layers", "dropout", "learning_rate", "weight_decay",
    "batch_size", "activation",
]
_GRID_HP_KEYS_PHYSREG: list[str] = [
    "hidden_layers", "dropout", "learning_rate", "batch_size",
    "physics_weight", "physics_warmup_fraction", "phi_lr_ratio",
]
_GRID_HP_KEYS_RESIDUAL: list[str] = [
    "hidden_layers", "dropout", "learning_rate", "weight_decay",
    "batch_size", "alpha_reg_weight",
]
_ARCH_HP_KEYS: dict[str, list[str]] = {
    "BlackBoxFNN":           _GRID_HP_KEYS_FNN,
    "PhysicsRegularizedFNN": _GRID_HP_KEYS_PHYSREG,
    "ResidualCorrectionFNN": _GRID_HP_KEYS_RESIDUAL,
}

_ARCH_PICKER_ABBREV: list[tuple[str, str]] = [
    ("BlackBoxFNN",           "FNN"),
    ("PhysicsRegularizedFNN", "PhysReg"),
    ("ResidualCorrectionFNN", "ResCorr"),
    ("EDR",                   "EDR"),
]
