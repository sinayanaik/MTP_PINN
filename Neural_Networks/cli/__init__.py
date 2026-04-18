"""
Neural_Networks.cli
====================
Interactive command-line interface components.

Sub-modules
-----------
prompts   — ask() / ask_list() — low-level Rich-styled prompt helpers
hp_wizard — gather_hp() / gather_hp_for_models() / get_default_hp()
menus     — dataset picker, model selection, batch results table, quick-test-all

Public API (everything you need for the training CLI):
"""

from Neural_Networks.cli.prompts  import ask, ask_list
from Neural_Networks.cli.hp_wizard import (
    gather_hp,
    gather_hp_for_models,
    get_default_hp,
)
from Neural_Networks.cli.menus import (
    QUICK_TEST_SENTINEL,
    data_preparation_step,
    select_existing_dataset,
    _build_model_menu,
    _select_model_types,
    print_batch_results_table,
    _run_quick_test_all,
)

__all__ = [
    # prompts
    "ask",
    "ask_list",
    # hp_wizard
    "gather_hp",
    "gather_hp_for_models",
    "get_default_hp",
    # menus
    "QUICK_TEST_SENTINEL",
    "data_preparation_step",
    "select_existing_dataset",
    "_build_model_menu",
    "_select_model_types",
    "print_batch_results_table",
    "_run_quick_test_all",
]
