"""Path bootstrap for the Journal_Comparison plotting suite.

Two jobs:

1. Make the repo importable regardless of CWD (the repo is normally run as
   ``PYTHONPATH=. python ...`` from the root; these scripts live 5 levels deep).
   This mirrors the sys.path injection done by
   ``Neural_Networks/eval_best_models.py`` so that ``build_model`` can in turn
   dynamically import the EDR model.

2. Remap *foreign* absolute paths.  The registry and per-model ``metadata.yaml``
   were written on the training box and embed
   ``/home/sinayan_iitp/MTP_PINN/...`` paths.  Locally the repo lives at a
   different root, so every path read out of those files must pass through
   :func:`remap` before use.
"""

from __future__ import annotations

import sys
from pathlib import Path

# REPO_ROOT = .../MTP_PINN
#   bootstrap.py -> shared -> plot_codes -> Journal_Comparison
#   -> Trained_Models -> Neural_Networks -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[5]
assert (REPO_ROOT / "Neural_Networks").is_dir(), (
    f"repo-root detection failed: {REPO_ROOT} has no Neural_Networks/"
)

_EDR_DIR = REPO_ROOT / "Neural_Networks" / "models" / "Equivariant-Decomposed-Residual"

for _p in (str(REPO_ROOT), str(_EDR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Absolute prefixes that belong to the training machine, not this one.
FOREIGN_PREFIXES = (
    "/home/sinayan_iitp/MTP_PINN",
    "/home/sinayan_iitp/mtp_pinn",
)

JOURNAL_DIR = REPO_ROOT / "Neural_Networks" / "Trained_Models" / "Journal_Comparison"
REGISTRY_PATH = JOURNAL_DIR / "models_registry.yaml"
GRID_CSV = JOURNAL_DIR / "grid_results.csv"
PLOT_CODES_DIR = JOURNAL_DIR / "plot_codes"
FIGURES_DIR = PLOT_CODES_DIR / "figures"
TABLES_DIR = PLOT_CODES_DIR / "tables"
CACHE_DIR = PLOT_CODES_DIR / "_cache"


def remap(path: str | Path) -> Path:
    """Rewrite a training-box absolute path onto the local repo root.

    Non-foreign paths are returned unchanged (as :class:`~pathlib.Path`).
    """
    s = str(path)
    for prefix in FOREIGN_PREFIXES:
        if s.startswith(prefix):
            rel = s[len(prefix):].lstrip("/")
            return REPO_ROOT / rel
    return Path(s)
