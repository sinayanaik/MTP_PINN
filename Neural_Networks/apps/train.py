#!/usr/bin/env python3
"""Interactive Rich training entrypoint (dataset, model, hyperparameters)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_APPS = Path(__file__).resolve().parent
_NN_ROOT = _APPS.parent
_PROJECT_ROOT = _NN_ROOT.parent
TRAIN_DATA_DIR = _NN_ROOT / "train_data"
MODELS_DIR = _NN_ROOT / "Trained_Models"
REGISTRY_FILE = MODELS_DIR / "models_registry.yaml"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from Neural_Networks.apps.train_ui import run_interactive_train

    return run_interactive_train(
        train_data_dir=TRAIN_DATA_DIR,
        models_dir=MODELS_DIR,
        registry_file=REGISTRY_FILE,
        nn_dir=_NN_ROOT,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
