"""
Neural_Networks.data
====================
Pure data-utility layer — no Rich, no CLI, no Tkinter.

Exports:
  labels.py  — compact param-tag builders and one-liner summaries
  scanner.py — filesystem scan for processed train-data directories
"""

from Neural_Networks.data.labels import (    # noqa: F401
    build_param_tag,
    compact_summary_one_line,
    dataset_stats_one_liner,
    default_run_dir,
    param_tag_from_metadata,
    pipeline_compact_codes,
    preprocessing_pipeline_table_rows,
    sg_mode_token,
    train_val_test_split_table_rows,
)
from Neural_Networks.data.scanner import (   # noqa: F401
    load_run_metadata,
    scan_existing_datasets,
)
