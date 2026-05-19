"""Local smoke harness for the EDR robustness overhaul (not a journal artifact).

Trains EDR (new) and optionally PhysReg for a short horizon on the default
journal dataset and prints the TRAINING SUMMARY so we can verify:
  * EDR trains end-to-end (Christoffel jacrev+vmap, γ-gate, spectral-norm,
    EMA, checkpointing) without crashing,
  * val keeps improving (best epoch late, not stuck ~ep30 then flat),
  * gen-gap (test−val@best) is small,
  * EDR test ≤ PhysReg test (directional at this short horizon).

Usage:  python -m Neural_Networks.models._smoke_edr_robust edr 60
"""
from __future__ import annotations

import logging
import sys
import tempfile

from Neural_Networks.models.run_journal_grid_3model import (
    FIXED_HP_EDR, FIXED_HP_PHYSREG, TRAIN_DATA_RUN_DIR, REGISTRY_FILE,
)
from pathlib import Path

from Neural_Networks.models.shared.pipeline import TrainJob, run_training
from Neural_Networks.models.shared.strategies import PHYSICS_REG_STRATEGY

_EDR_DIR = str(Path(__file__).resolve().parent / "Equivariant-Decomposed-Residual")
if _EDR_DIR not in sys.path:
    sys.path.insert(0, _EDR_DIR)
import edr_strategy as _edr  # noqa: E402


def main() -> None:
    arch = sys.argv[1] if len(sys.argv) > 1 else "edr"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if arch == "edr":
        hp = dict(FIXED_HP_EDR)
        strategy, mt, sub = _edr.EDR_STRATEGY, "edr", "EDR"
    else:
        hp = dict(FIXED_HP_PHYSREG)
        strategy, mt, sub = PHYSICS_REG_STRATEGY, "physreg", "PhysicsRegularized"
    hp["epochs"] = epochs
    hp["seed"] = 42
    out = tempfile.mkdtemp(prefix=f"smoke_{arch}_")
    job = TrainJob(
        run_dir=TRAIN_DATA_RUN_DIR, models_dir=out, registry_file=REGISTRY_FILE,
        model_type=mt, save_subdir=sub, hp=hp, strategy=strategy,
        run_help="smoke",
    )
    rc = run_training(job)
    print(f"\nSMOKE[{arch}] returned test_rmse_traj_macro = {rc}  (out={out})")


if __name__ == "__main__":
    main()
