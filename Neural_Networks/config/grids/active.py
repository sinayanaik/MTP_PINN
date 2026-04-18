"""Active grid config — edit the single import line below to switch grids.

``grid_search`` always loads this module — there are no CLI arguments.
Everything (which grid, dry-run, model filter, resume folder) is expressed
here.  To add a new preset: drop a new file next to ``laptop.py`` / ``hpc.py``
and change the import line.
"""
from __future__ import annotations

# --- Pick which preset to use (edit this line) -----------------------------
from Neural_Networks.config.grids.laptop import *  # noqa: F401,F403
# from Neural_Networks.config.grids.hpc import *   # <- uncomment for HPC

# --- Run-time knobs (override defaults from the preset if you want) --------
# Set to True to list trials without training.
DRY_RUN: bool = False

# Optional subset of models to actually run.  None -> use MODELS from preset.
MODELS_FILTER: list[str] | None = None

# Resume an existing timestamped run folder (absolute or relative to the
# study root Trained_Models_GridSearch/<study_name>/).  None -> start a new
# timestamped folder each invocation.
RESUME: str | None = None

# Maximum parallel training workers.  "auto" detects from VRAM at runtime
# (e.g. 1 on a 4 GB laptop, ~12 on an 80 GB A100).  Set to 1 to force
# sequential execution.  Inherits from preset unless overridden here.
# MAX_PARALLEL_TRIALS: int | str = "auto"

# Save lightweight checkpoints every N epochs during grid training (0 = off).
# Inherits from preset unless overridden here.
# SNAPSHOT_EVERY: int = 0

# --- Reporting knobs (used by grid_report) ---------------------------------
# Which study run to aggregate.  None -> pick the newest timestamp under
# Trained_Models_GridSearch/<STUDY_NAME>/.  Otherwise pass the timestamp
# folder name (e.g. "20260415_015608") or an absolute path.
REPORT_TARGET: str | None = None

# Skip matplotlib plots.
REPORT_NO_PLOTS: bool = False
