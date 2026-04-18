"""
Neural_Networks.config
======================
Pure configuration layer — no Rich, no CLI, no GUI.

Exports:
  detect_hardware()      — hardware profile dict  (hardware.py)
  All HP doc-dicts and registry helpers             (hp_registry.py)
"""

from Neural_Networks.config.hardware import detect_hardware  # noqa: F401
from Neural_Networks.config.hp_registry import (             # noqa: F401
    COMMON_HP_DOCS,
    COMMON_STYLE_MODELS,
    DEFAULT_PROFILE,
    DECOMPOSED_FNN_HP,
    EC_PINN_HP,
    FNN_SPECIFIC_HP,
    GROUP_ORDER,
    HP_KEY_GROUPS,
    KEY_TO_GROUP,
    LNN_SPECIFIC_HP,
    MODEL_HP_LAYERS,
    NORMAL_MODE_HIDDEN_KEYS,
    PHYSICS_WEIGHT_HP,
    RESIDUAL_CORRECTION_HP,
    STRUCTURED_MODELS,
    activation_prompt_split_needed,
    apply_accurate_nominal_to_docs,
    apply_profile_to_hp_dict,
    dropout_prompt_split_needed,
    get_model_hp_docs,
    lookup_hp_doc,
    merge_doc_dicts_for_prompt,
    merge_shared_into_model_hp,
    model_needs_group,
    should_prompt_key,
    union_groups,
)
