"""
Neural_Networks.tui
====================
Rich terminal UI layer — console singleton, section headings,
HP documentation tables, and dataset summary tables.

All functions in this package write to ``console`` (a Rich Console instance).
Zero ML / math logic lives here; every function is a display helper.

Sub-modules
-----------
console.py         — global Console singleton + section() / subsection() helpers
hp_display.py      — print_hp_docs()   (hyperparameter documentation table)
dataset_display.py — show_dataset_table(), print_dataset_summary(),
                     human_summary_lines(), preprocessing_pipeline_table_rows(),
                     train_val_test_split_table_rows(), preprocessing_filters_detailed_lines(),
                     dataset_stats_one_liner(), _compact_count()
"""

from Neural_Networks.tui.console import (          # noqa: F401
    console,
    section,
    subsection,
)
from Neural_Networks.tui.hp_display import (       # noqa: F401
    print_hp_docs,
)
from Neural_Networks.tui.dataset_display import (  # noqa: F401
    print_dataset_summary,
    show_dataset_table,
)
