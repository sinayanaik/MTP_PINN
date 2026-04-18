"""
Neural_Networks.cli.hp_wizard
================================
Interactive hyperparameter gathering wizard.

Functions
---------
_prompt_one_hp(name, info)           — prompt for a single HP based on its doc dict
_n_train_samples_from_dataset_meta   — helper to extract n_train from metadata dict
_inject_hw_into_doc_maps             — mutate HP doc dicts with hardware-profile defaults
gather_hp(model_type, hw)             → dict   — single-model HP prompt
gather_hp_for_models(model_types, hw) → {m: hp} — multi-model shared-group prompt
get_default_hp(model_type, ...)       → dict   — headless default HPs (no prompts)

The actual HP documentation / metadata lives in ``Neural_Networks.hp_registry``
(re-exported via ``Neural_Networks.config.hp_registry``).  This module only owns
the *prompt* logic (type dispatch, batch-prompt shared groups, etc.).
"""

from __future__ import annotations

from typing import Any

from rich.panel import Panel
from rich.markup import escape

from Neural_Networks.tui.console import console, section, subsection
from Neural_Networks.tui.hp_display import print_hp_docs
from Neural_Networks.cli.prompts import ask, ask_list
from Neural_Networks.config.hp_registry import (
    COMMON_STYLE_MODELS,
    GROUP_ORDER,
    HP_KEY_GROUPS,
    STRUCTURED_MODELS,
    activation_prompt_split_needed,
    apply_accurate_nominal_to_docs,
    apply_profile_to_hp_dict,
    dropout_prompt_split_needed,
    get_model_hp_docs,
    merge_doc_dicts_for_prompt,
    merge_shared_into_model_hp,
    model_needs_group,
    should_prompt_key,
    union_groups,
)


# =============================================================================
# Internal helpers
# =============================================================================

def _n_train_samples_from_dataset_meta(meta: dict | None) -> int:
    """Extract the training-set sample count from a metadata.json dict."""
    if not meta:
        return 0
    ss = (meta.get("split") or {}).get("stats") or {}
    return int(ss.get("train", {}).get("n_samples", 0) or 0)


def _inject_hw_into_doc_maps(
    common_map: dict[str, dict],
    specific_map: dict[str, dict[str, dict]],
    hw: dict,
) -> None:
    """Apply hardware-profile defaults to HP doc dicts **in place**.

    When the user presses Enter on any HP prompt, the shown default comes from
    the HP doc dict's ``"default"`` key.  Mutating those dicts with hardware-
    appropriate values means the user gets sensible defaults for their GPU.

    common_map  : common HP doc dict (shared by all models)
    specific_map: {model_type → specific HP doc dict}
    hw          : hardware profile dict from detect_hardware()
    """
    hw_batch  = hw.get("batch_size")
    hw_hidden = hw.get("hidden_size")
    hw_fc     = hw.get("fc_layers")
    hw_epochs = hw.get("epochs")
    hw_stride = hw.get("stride")

    if hw_batch  is not None and "batch_size"       in common_map:
        common_map["batch_size"]["default"]       = hw_batch
    if hw_epochs is not None and "epochs"           in common_map:
        common_map["epochs"]["default"]           = hw_epochs

    for _m, spec in specific_map.items():
        if hw_hidden is not None and "hidden_size" in spec:
            spec["hidden_size"]["default"] = hw_hidden
        if hw_fc     is not None and "fc_layers"   in spec:
            spec["fc_layers"]["default"]   = hw_fc
        if hw_stride is not None and "stride"       in spec:
            spec["stride"]["default"]      = hw_stride


# =============================================================================
# Single-HP prompt dispatcher
# =============================================================================

def _prompt_one_hp(name: str, info: dict) -> Any:
    """Dispatch to ask() or ask_list() based on the HP's default type.

    Handles: list (comma-separated int list), bool (true/false), choices (enum),
    float, int, and plain string.  A ``None`` default means the HP is optional
    and should auto-resolve downstream — the prompt shows ``auto`` and returns
    ``None`` on blank/``auto`` input.  Reached only in expert mode; normal-mode
    callers suppress the prompt via ``NORMAL_MODE_HIDDEN_KEYS``.
    """
    default = info["default"]
    choices = info.get("choices", None)
    if default is None:
        raw = ask(f"  {name} [dim](blank = auto)[/dim]", default="auto")
        if raw in (None, "", "auto"):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    if isinstance(default, list):
        return ask_list(f"  {name}", default=default, cast=int)
    if isinstance(default, bool):
        return ask(
            f"  {name}",
            default=default,
            cast=lambda x: x.lower() in ("true", "1", "yes"),
            choices=["true", "false"],
        )
    if choices:
        return ask(f"  {name}", default=default, choices=choices)
    if isinstance(default, float):
        return ask(f"  {name}", default=default, cast=float)
    if isinstance(default, int):
        return ask(f"  {name}", default=default, cast=int)
    return ask(f"  {name}", default=default)


# =============================================================================
# Single-model HP gathering
# =============================================================================

def gather_hp(model_type: str, hw: dict | None = None) -> dict:
    """Interactively gather hyperparameters for one model type.

    Prompts for every HP group in order, using docs from ``get_model_hp_docs()``.
    Hardware defaults are injected when ``hw`` is provided so that pressing Enter
    always yields a hardware-appropriate default.

    Parameters
    ----------
    model_type : str
        One of the 6 registered model type keys.
    hw : dict | None
        Hardware profile from ``detect_hardware()``.

    Returns
    -------
    dict of {hp_name: value}
    """
    specific, common = get_model_hp_docs(model_type)

    # Shallow-copy so hardware injection never mutates module-level constants
    if hw is not None:
        common   = {k: dict(v) for k, v in common.items()}
        specific = {k: dict(v) for k, v in specific.items()}
        _inject_hw_into_doc_maps(common, {model_type: specific}, hw)

    all_docs = {**common, **specific}
    _ep = int(all_docs.get("epochs", {}).get("default", 500))
    apply_accurate_nominal_to_docs(
        model_type, all_docs, n_train_samples=None, epochs=_ep,
    )

    section("Hyperparameter Configuration")
    hw_tag = (
        f"  [dim]Hardware profile: [bold]{hw['profile'].upper()}[/bold] "
        f"({hw['gpu_name']}, {hw['vram_gb']} GB VRAM)[/dim]"
    ) if hw is not None else ""
    console.print(Panel(
        f"[bold]Model:[/bold] [bright_cyan]{model_type}[/bright_cyan]\n"
        "[dim]Press ENTER on any prompt to accept the default value shown in brackets.[/dim]"
        + ("\n" + hw_tag if hw_tag else ""),
        border_style="cyan", padding=(0, 2),
    ))
    print_hp_docs(all_docs)

    hp = {}
    subsection("Enter Hyperparameters")
    for name, info in specific.items():
        hp[name] = _prompt_one_hp(name, info)
    for name, info in common.items():
        hp[name] = _prompt_one_hp(name, info)

    hp["_n_train_samples"] = 0
    apply_profile_to_hp_dict(model_type, hp)
    return hp


# =============================================================================
# Multi-model shared-group gathering
# =============================================================================

def _queue_uses_key(model_types: list[str], key: str) -> bool:
    """Return True if any model in the queue defines ``key`` in its HP docs."""
    for m in model_types:
        sp, cm = get_model_hp_docs(m)
        if key in sp or key in cm:
            return True
    return False


def _ref_doc_for_key(
    model_types: list[str], key: str
) -> tuple[str, dict[str, dict]]:
    """Find a reference model that defines ``key``; return (model_name, merged_doc)."""
    for m in model_types:
        sp, cm = get_model_hp_docs(m)
        merged = {
            **{k: dict(v) for k, v in cm.items()},
            **{k: dict(v) for k, v in sp.items()},
        }
        if key in merged:
            return m, merged
    return model_types[0], {}


def gather_hp_for_models(
    model_types: list[str],
    hw: dict | None = None,
    dataset_meta: dict | None = None,
) -> dict[str, dict]:
    """Gather HPs for a batch of models: shared groups asked once, then merged per model.

    Single-model queues delegate to ``gather_hp()``.
    Multi-model queues collect the union of HP key groups, prompt each group
    once, then use ``merge_shared_into_model_hp()`` to build per-model dicts.

    Parameters
    ----------
    model_types   : list of model type strings
    hw            : hardware profile from detect_hardware()
    dataset_meta  : metadata.json dict for the selected dataset (used to set
                    ``apply_accurate_nominal_to_docs`` sample counts)

    Returns
    -------
    {model_type: hp_dict}
    """
    if len(model_types) == 1:
        return {model_types[0]: gather_hp(model_types[0], hw=hw)}

    n_train = _n_train_samples_from_dataset_meta(dataset_meta)
    expert  = (
        ask(
            "Hyperparameter prompts", default="normal",
            choices=["normal", "expert"],
        ).lower() == "expert"
    )

    per_spec:   dict[str, dict[str, dict]] = {}
    per_common: dict[str, dict[str, dict]] = {}
    for m in model_types:
        sp, cm         = get_model_hp_docs(m)
        per_spec[m]    = {k: dict(v) for k, v in sp.items()}
        per_common[m]  = {k: dict(v) for k, v in cm.items()}

    if hw is not None:
        for m in model_types:
            _inject_hw_into_doc_maps(per_common[m], {m: per_spec[m]}, hw)

    ugroups   = union_groups(model_types)
    shared:   dict[str, Any] = {}
    act_split = activation_prompt_split_needed(model_types)
    do_split  = dropout_prompt_split_needed(model_types)

    section("Hyperparameter Configuration (batch)")
    console.print(Panel(
        f"[bold]Models:[/bold] [bright_cyan]{', '.join(model_types)}[/bright_cyan]\n"
        f"[dim]Detail: {'expert (all parameters)' if expert else 'normal (recommended subset)'}[/dim]",
        border_style="cyan", padding=(0, 2),
    ))

    def _silent_default(key: str) -> None:
        """Apply default without prompting (used for non-expert fast path)."""
        ref_m, merged = _ref_doc_for_key(model_types, key)
        if key not in merged:
            return
        ep   = int(shared.get("epochs", merged.get("epochs", {}).get("default", 500)))
        frag = {key: dict(merged[key])}
        apply_accurate_nominal_to_docs(ref_m, frag, n_train_samples=n_train, epochs=ep)
        shared[key] = frag[key]["default"]

    for group in GROUP_ORDER:
        if group not in ugroups:
            continue
        subsection(f"Shared — {group}")
        for key in HP_KEY_GROUPS.get(group, []):
            if not any(model_needs_group(m, group) for m in model_types):
                continue
            if not _queue_uses_key(model_types, key):
                continue
            if key == "activation" and act_split:
                continue
            if key == "dropout" and do_split:
                continue
            if not should_prompt_key(key, expert, shared):
                _silent_default(key)
                continue

            doc   = merge_doc_dicts_for_prompt(model_types, key)
            ref_m, merged = _ref_doc_for_key(model_types, key)
            if key in merged:
                ep   = int(shared.get("epochs", merged.get("epochs", {}).get("default", 500)))
                frag = {key: dict(merged[key])}
                apply_accurate_nominal_to_docs(ref_m, frag, n_train_samples=n_train, epochs=ep)
                doc = dict(frag[key])
            shared[key] = _prompt_one_hp(key, doc)

    # Per-type activation split when models differ in activation API (MLP vs structured)
    if act_split:
        subsection("Shared — activation (split)")
        m_cb = next(m for m in model_types if m in COMMON_STYLE_MODELS)
        m_st = next(m for m in model_types if m in STRUCTURED_MODELS)
        _, ccb = get_model_hp_docs(m_cb)
        sst,  _ = get_model_hp_docs(m_st)
        shared["activation_mlp"]        = _prompt_one_hp(
            "activation (MLP / black-box-style)", dict(ccb["activation"]),
        )
        shared["activation_structured"] = _prompt_one_hp(
            "activation (Lagrangian / Decomposed)", dict(sst["activation"]),
        )

    # Per-type dropout split when models differ in dropout API
    if do_split:
        subsection("Shared — dropout (split)")
        m_cb = next(m for m in model_types if m in COMMON_STYLE_MODELS)
        m_st = next(m for m in model_types if m in STRUCTURED_MODELS)
        _, ccb = get_model_hp_docs(m_cb)
        sst,  _ = get_model_hp_docs(m_st)
        shared["dropout_mlp"]        = _prompt_one_hp(
            "dropout (MLP / black-box-style)", dict(ccb["dropout"]),
        )
        shared["dropout_structured"] = _prompt_one_hp(
            "dropout (Lagrangian / Decomposed)", dict(sst["dropout"]),
        )

    out: dict[str, dict] = {}
    for m in model_types:
        hp_m = merge_shared_into_model_hp(m, shared)
        hp_m["_n_train_samples"] = n_train
        apply_profile_to_hp_dict(m, hp_m)
        out[m] = hp_m
    return out


# =============================================================================
# Headless default HPs (no prompts — for programmatic or Quick-Test use)
# =============================================================================

def get_default_hp(
    model_type: str,
    n_train_samples: int | None = None,
    epochs: int | None = None,
) -> dict:
    """Return default hyperparameters without any interactive prompts.

    Applies accurate-nominal profile patches (cycle length, warmup epochs, etc.)
    that depend on the dataset size and epoch budget.

    Parameters
    ----------
    model_type       : model registry key
    n_train_samples  : number of training samples (used for CosineAnnealingLR, etc.)
    epochs           : max epoch budget

    Returns
    -------
    dict of {hp_name: value}
    """
    specific, common = get_model_hp_docs(model_type)
    all_docs = {k: dict(v) for k, v in {**common, **specific}.items()}
    _ep = (
        int(epochs)
        if epochs is not None
        else int(all_docs.get("epochs", {}).get("default", 500))
    )
    apply_accurate_nominal_to_docs(
        model_type, all_docs, n_train_samples=n_train_samples, epochs=_ep,
    )
    hp = {name: info["default"] for name, info in all_docs.items()}
    hp["_n_train_samples"] = int(n_train_samples or 0)
    apply_profile_to_hp_dict(model_type, hp)
    return hp
