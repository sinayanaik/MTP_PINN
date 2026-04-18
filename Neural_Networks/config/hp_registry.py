"""
Neural_Networks.config.hp_registry
====================================
Single source of truth for hyperparameter documentation, groupings,
profile defaults, and per-model HP merging helpers.

This module carries NO UI or CLI dependencies — it exports only plain
Python dicts, sets, and (Config → dict) functions.  All Rich display
logic for HP tables lives in Neural_Networks.tui.hp_display; all
interactive prompting lives in Neural_Networks.cli.hp_wizard.

Sections
--------
COMMON_HP_DOCS          — 13 keys shared by every model (batch_size, lr, …)
FNN_SPECIFIC_HP         — hidden_layers for FNN-style models (A/B/C)
EC_PINN_HP              — phi_lr_ratio, lambda_collocation, n_collocation
RESIDUAL_CORRECTION_HP  — alpha_reg_weight
PHYSICS_WEIGHT_HP       — physics_weight + warmup_fraction (fixed α mixture)
LNN_SPECIFIC_HP         — Lagrangian sub-network config
DECOMPOSED_FNN_HP       — Lagrangian + decomposed extras (uncertainty, SPD, friction)
HP_KEY_GROUPS / GROUP_ORDER — ordered key lists per group
MODEL_HP_LAYERS         — per-model group membership
Helper functions — get_model_hp_docs, merge_shared_into_model_hp, apply_accurate_nominal_to_docs, …
"""

from __future__ import annotations

from typing import Any

from Neural_Networks.models import (
    MODEL_REGISTRY,
    DECOMPOSED_MODELS,
    EQUATION_CONSTRAINED_MODELS,
    FNN_MODELS,
    LAGRANGIAN_MODELS,
    PHYSICS_WEIGHT_MODELS,
)

DEFAULT_PROFILE = "accurate_nominal"

# ---------------------------------------------------------------------------
# Doc dicts (single source of truth for defaults and Rich descriptions)
# ---------------------------------------------------------------------------

COMMON_HP_DOCS: dict[str, dict[str, Any]] = {
    "batch_size": {
        "default": 512,
        "desc": "Number of samples per gradient update step.",
        "effect": "Larger → more stable gradients, better GPU utilisation. "
                  "2048 is optimal for A100/H100 with AMP. "
                  "On RTX 3050/local GPU: 512-1024. On CPU: 64-256.",
    },
    "epochs": {
        "default": 500,
        "desc": "Maximum number of full passes over the training set.",
        "effect": "More → longer training; with early stopping, "
                  "actual epochs may be much less. "
                  "Physics-informed models typically need 500-1000 to fully converge. "
                  "NEVER set below 200 — PINN models need ≥200 epochs to surpass black-box.",
    },
    "learning_rate": {
        "default": 3e-4,
        "desc": "Step size for the optimiser.",
        "effect": "3e-4 is the recommended default for AdamW on MLP regression tasks. "
                  "With warmup_cosine_restarts and grad_clip_norm=5.0, this yields "
                  "effective LR ~1e-4 to 3e-4 with periodic resets. "
                  "1e-4 is too conservative — causes early plateau at 30-40 epochs. "
                  "Try 5e-4 only for very large datasets (>500K samples).",
    },
    "optimizer": {
        "default": "adamw",
        "choices": ["adam", "adamw", "sgd", "rmsprop"],
        "desc": "Gradient descent algorithm.",
        "effect": "adamw: adam + weight decay decoupled (better generalisation — recommended). "
                  "adam: adaptive lr, best default choice. "
                  "sgd: classic, needs lr schedule. "
                  "rmsprop: good for RNNs.",
    },
    "lr_scheduler": {
        "default": "warmup_cosine",
        "choices": ["none", "cosine", "warmup_cosine", "warmup_cosine_restarts",
                    "step", "reduce_on_plateau", "onecycle", "exponential", "cyclic"],
        "desc": "Learning rate schedule.",
        "effect": "warmup_cosine: linear ramp 5%% epochs, then cosine decay to 1%% LR "
                  "(RECOMMENDED).  Combined with curriculum physics schedule, the LR decay "
                  "aligns with w_p decay — both reduce in the data-refinement phase. "
                  "warmup_cosine_restarts: periodic restarts — can destabilise fine-tuning. "
                  "reduce_on_plateau: AVOID for physics models — halves LR on noise. "
                  "cosine: smooth decay to lr_min. "
                  "none: constant lr.",
    },
    "weight_decay": {
        "default": 5e-3,
        "desc": "L2 regularisation coefficient (AdamW: decoupled from LR).",
        "effect": "Penalises large weights → reduces overfitting. "
                  "5e-3 recommended — prevents the early plateau at epoch 20-30 "
                  "by slowing memorisation of training data, giving physics losses "
                  "time to constrain the solution space. "
                  "Too low (<2e-3) causes overfitting before physics can help.",
    },
    "dropout": {
        "default": 0.1,
        "desc": "Fraction of neurons randomly zeroed during training.",
        "effect": "0.1 = light (RECOMMENDED for structured physics models). "
                  "0.3+ destabilises Cholesky SPD constraint in Lagrangian/Decomposed models. "
                  "LayerNorm in hidden layers already provides regularisation.",
    },
    "early_stopping": {
        "default": True,
        "desc": "Stop training when the monitored validation metric stops improving.",
        "effect": "Prevents overfitting. Recommended: always on. Pair with early_stop_metric.",
    },
    "early_stop_metric": {
        "default": "val_rmse",
        "choices": ["val_rmse", "val_loss"],
        "desc": "Metric for early stopping and best-checkpoint selection.",
        "effect": "val_rmse: unweighted RMSE of τ̂ vs τ_meas in normalised space (no physics terms, "
                  "no J2 joint weighting) — recommended for fair comparison. "
                  "val_loss: full training objective on the validation loader.",
    },
    "patience": {
        "default": 60,
        "desc": "Epochs to wait before stopping after last improvement.",
        "effect": "60 recommended — with warm restarts and early physics engagement, "
                  "the optimizer gets multiple chances to improve. "
                  "Higher values (100+) waste compute on plateau epochs. "
                  "Setting <30 risks premature stop before restart benefits kick in.",
    },
    "min_delta": {
        "default": 1e-4,
        "desc": "Minimum improvement in val_rmse to count as 'improved'.",
        "effect": "Prevents noise-driven patience resets that keep training alive "
                  "for 100+ plateau epochs. 1e-4 works well for normalized RMSE. "
                  "Also used by PhysicsWeightScheduler for plateau detection.",
    },
    "activation": {
        "default": "silu",
        "choices": ["relu", "tanh", "silu", "gelu", "elu", "leaky_relu"],
        "desc": "Non-linear activation (black-box / PR / residual / EC style nets).",
        "effect": "silu/gelu: smooth MLP default. tanh: bounded — used for residual correction default. "
                  "Lagrangian/Decomposed use a separate tanh-oriented activation in their own block.",
    },
    "feature_noise_std": {
        "default": 0.02,
        "desc": "Gaussian noise std added to input features during training only.",
        "effect": "Acts as data augmentation preventing overfitting. "
                  "0.02 = moderate (recommended) — adds 2%% noise to z-scored features, "
                  "enough to slow memorisation and let physics constraints differentiate. "
                  "0.01 was too mild and models still overfit by epoch 30. 0 = disabled.",
    },
    "data_train_fraction": {
        "default": 1.0,
        "desc": "Fraction of train samples to actually use (val/test unchanged).",
        "effect": "Data-efficiency knob.  1.0 = use every training sample; "
                  "0.5 = keep 50%% of train samples; etc.  The subset is drawn with a "
                  "deterministic RNG seeded by `data_train_seed` (falls back to the "
                  "grid seed) so different models see the same samples at the same "
                  "fraction — required for fair PINN-vs-BlackBox data-efficiency "
                  "comparisons.  Validation and test sets are NEVER subsampled so "
                  "RMSE remains comparable across fractions.  Range: (0, 1].",
    },
    "data_train_seed": {
        "default": 0,
        "desc": "RNG seed used when sampling the train subset (see data_train_fraction).",
        "effect": "Only matters when data_train_fraction < 1.  Fix this across a "
                  "data-efficiency sweep so every cell trains on the *same* subset "
                  "for a given fraction (noise shrinks, signal sharpens).",
    },
    "grad_clip_norm": {
        "default": 5.0,
        "desc": "Maximum gradient L2 norm before clipping.",
        "effect": "Prevents gradient explosion from physics losses. "
                  "5.0 is standard for MLP models with LayerNorm — rarely clips "
                  "during normal training but catches pathological spikes. "
                  "1.0 was the previous hardcoded value and is too aggressive — "
                  "it scales down EVERY update by ~5× for a 270K-param model, "
                  "effectively reducing LR to 1/5th of nominal.",
    },
}

FNN_SPECIFIC_HP: dict[str, dict[str, Any]] = {
    "hidden_layers": {
        "default": [256, 512, 256],
        "desc": "Hidden layer widths (comma-separated list).",
        "effect": "All FNN variants (A/B/C) use [256,512,256] by default so that architecture "
                  "is identical across categories — only the loss function differs.",
    },
}

EC_PINN_HP: dict[str, dict[str, Any]] = {
    "phi_lr_ratio": {
        "default": 0.1,
        "desc": "Learning-rate multiplier for learnable τ_eq calibration φ (vs main network LR).",
        "effect": "~10× smaller LR for φ is standard when jointly optimising with θ.",
    },
    "lambda_collocation": {
        "default": 0.05,
        "desc": "Weight of collocation loss MSE(τ̂^c, τ_p^c_eff) on synthetic (q,q̇,q̈) points.",
        "effect": "0.05 enables collocation with moderate physics anchoring (recommended). "
                  "Set 0 to disable. Requires Pinocchio at training time; "
                  "silently skipped if Pinocchio unavailable.",
    },
    "n_collocation": {
        "default": 32,
        "desc": "Number of collocation samples per epoch.",
        "effect": "More samples → stronger physics anchor; slightly higher CPU cost per epoch.",
    },
}

RESIDUAL_CORRECTION_HP: dict[str, dict[str, Any]] = {
    "alpha_reg_weight": {
        "default": 0.05,
        "desc": "λ_α · mean((α − 1)²) on residual-correction scale factors.",
        "effect": "Higher values (e.g. 0.05) keep τ̂ anchored on τ_phys when the nominal model is accurate.",
    },
}

PHYSICS_WEIGHT_HP: dict[str, dict[str, Any]] = {
    "physics_weight": {
        "default": 0.10,
        "desc": "Physics mixture weight α ∈ [0, 1]  (w_data = 1 − α, w_physics = α).",
        "effect": (
            "Convex mixture:  L = (1 − α) · L_data + α · L_physics\n"
            "α is the fraction of the combined objective attributed to physics.\n\n"
            "Empirical guidance on this task (500k samples, calibrated τ_nom):\n"
            "  α = 0.00        → data-only ablation (equivalent to BlackBox).\n"
            "  α = 0.05 - 0.10 → SWEET SPOT — physics as a soft regulariser.\n"
            "                    Matches the best historical runs.\n"
            "  α = 0.15 - 0.25 → stronger anchor; useful when over-fitting is\n"
            "                    visible (val_rmse rising while train_loss falls).\n"
            "  α ≥ 0.40        → physics begins to dominate; the imperfect RNEA\n"
            "                    bias starts pulling predictions off ground truth\n"
            "                    and val_rmse degrades after ~10 epochs.\n\n"
            "α is held constant throughout training (after a short linear warmup\n"
            "to avoid optimiser shock).  To sweep, run several values manually —\n"
            "0.05, 0.10, 0.20 — and compare val_rmse on the ranking table.\n"
            "Values above 1.0 are silently clamped."
        ),
    },
    "physics_warmup_fraction": {
        "default": 0.03,
        "desc": "Fraction of total epochs to linearly ramp α from 0 to physics_weight.",
        "effect": (
            "During warmup, α ramps linearly 0 → physics_weight so early-epoch\n"
            "physics residuals (large on an uninitialised network) don't shock\n"
            "the optimiser.  After warmup, α is held constant.\n"
            "0.03 = 15 epochs for 500 max epochs (RECOMMENDED).  Lower to 0.01\n"
            "for quick experiments; raise to 0.05-0.10 if you see an early train-\n"
            "loss spike when physics kicks in."
        ),
    },
}

LNN_SPECIFIC_HP: dict[str, dict[str, Any]] = {
    "activation": {
        "default": "tanh",
        "choices": ["tanh", "silu", "gelu", "elu"],
        "desc": "Activation for all Lagrangian / Decomposed sub-networks.",
        "effect": "tanh: bounded output — recommended for stable Cholesky inertia. "
                  "AVOID relu/leaky_relu: unbounded outputs destabilize SPD constraint.",
    },
    "inertia_layers": {
        "default": [256, 512, 256],
        "desc": "InertiaNet hidden layers. q(5) → Cholesky entries(15) → M(q) SPD.",
        "effect": "Recommended: 3 layers, 128-256 wide.",
    },
    "coriolis_layers": {
        "default": [256, 512, 256],
        "desc": "CoriolisNet hidden layers. [q,qd](10) → C(q,qd) vector(5).",
        "effect": "Coriolis/centrifugal terms are quadratic in qd — needs >1 layer.",
    },
    "gravity_layers": {
        "default": [256, 512, 256],
        "desc": "GravityNet hidden layers. q(5) → g(q)(5).",
        "effect": "Gravity is smooth trigonometric of q. 2-3 layers usually sufficient.",
    },
    "friction_layers": {
        "default": [128, 128],
        "desc": "FrictionNet hidden layers. qd(5) → friction torque(5).",
        "effect": "Keep shallow — friction is a simple monotonic function of velocity.",
    },
    "dropout": {
        "default": 0.05,
        "desc": "Dropout rate for all Lagrangian/Decomposed sub-networks.",
        "effect": "0.05 recommended — keep very low. LayerNorm already regularises. "
                  "Dropout>0.1 destabilises the Cholesky SPD constraint and "
                  "creates discontinuous physics corrections.",
    },
    "spd_weight": {
        "default": 0.01,
        "desc": "λ_s — weight of SPD loss: penalises eigenvalues below ε.",
        "effect": "Keeps inertia matrix well-conditioned. 0.001-0.01 recommended.",
    },
    "friction_weight": {
        "default": 0.01,
        "desc": "λ_f — weight of friction dissipation loss: max(0, τ_f·qd).",
        "effect": "Enforces that friction removes energy (never adds it). 0.01 recommended.",
    },
    "nominal_consistency_weight": {
        "default": 1.0,
        "desc": "λ_n — weight of nominal consistency loss: MSE(τ̂, φ(τ_physics_nom)).",
        "effect": "Anchor to the calibrated nominal physics model φ(τ_nom).  "
                  "Without this, SPD and friction losses are near-zero by construction "
                  "so the Lagrangian model trains data-only with no real physics gradient "
                  "signal.  1.0 makes the effective physics weight comparable to "
                  "PhysicsReg/EC-PINN.  0 disables.",
    },
}

DECOMPOSED_FNN_HP: dict[str, dict[str, Any]] = {
    **LNN_SPECIFIC_HP,
    "spd_weight": {
        "default": 0.01,
        "desc": "λ_s — weight of SPD loss: penalises off-diagonal entries of M_hat.",
        "effect": "Keeps inertia matrix well-conditioned. 0.001-0.01 recommended.",
    },
    "friction_weight": {
        "default": 0.01,
        "desc": "λ_f — weight of friction dissipation loss: max(0, -f·qd).",
        "effect": "Enforces that friction removes energy (never adds it). 0.01 recommended.",
    },
    "correction_reg_weight": {
        "default": 0.001,
        "desc": "λ_c — weight of correction magnitude regularisation.",
        "effect": "Penalises large corrections (delta_c, delta_g, delta_f) to keep the model "
                  "close to nominal physics.  Acts as a light Occam's razor.  "
                  "Keep small (0.001) — the model's strength is learning corrections; "
                  "too high penalises the very thing it should learn.  0 disables.",
    },
    "nominal_consistency_weight": {
        "default": 1.0,
        "desc": "λ_n — weight of nominal consistency loss: MSE(τ̂, φ(τ_physics_nom)).",
        "effect": "Anchor to the calibrated nominal physics model φ(τ_nom).  "
                  "The structural constraints (SPD, dissipation) alone are near-zero "
                  "by construction — this term provides the main physics gradient.  "
                  "1.0 makes the effective physics weight comparable to "
                  "PhysicsReg/EC-PINN.  0 disables.",
    },
}

# Ordered keys per group (batch gather phase 1)
HP_KEY_GROUPS: dict[str, list[str]] = {
    "global_train": list(COMMON_HP_DOCS.keys()),
    "fnn_backbone": ["hidden_layers"],
    "physics": list(PHYSICS_WEIGHT_HP.keys()),  # physics_weight + warmup_fraction
    "lagrangian": [
        "activation",
        "inertia_layers",
        "coriolis_layers",
        "gravity_layers",
        "friction_layers",
        "dropout",
        "spd_weight",
        "friction_weight",
        "nominal_consistency_weight",
    ],
    "decomposed_extra": [
        "correction_reg_weight",
        "nominal_consistency_weight",
    ],
    "ec": ["phi_lr_ratio", "lambda_collocation", "n_collocation"],
    "residual": ["alpha_reg_weight"],
}

KEY_TO_GROUP: dict[str, str] = {}
for _g, _keys in HP_KEY_GROUPS.items():
    for _k in _keys:
        if _k in KEY_TO_GROUP and KEY_TO_GROUP[_k] != _g:
            # activation and dropout exist in both common and lagrangian — resolved by skip_common
            pass
        KEY_TO_GROUP.setdefault(_k, _g)

GROUP_ORDER: list[str] = [
    "global_train",
    "fnn_backbone",
    "physics",
    "lagrangian",
    "decomposed_extra",
    "ec",
    "residual",
]

MODEL_HP_LAYERS: dict[str, list[str]] = {
    "BlackBoxFNN":                ["global_train", "fnn_backbone"],
    "PhysicsRegularizedFNN":      ["global_train", "fnn_backbone", "physics"],
    "ResidualCorrectionFNN":      ["global_train", "fnn_backbone", "residual"],
    "LagrangianStructuredFNN":    ["global_train", "lagrangian", "physics"],
    "EquationConstrainedPINNFNN":   ["global_train", "fnn_backbone", "physics", "ec"],
    "DecomposedStructuredPINNFNN": ["global_train", "lagrangian", "decomposed_extra", "physics"],
}

STRUCTURED_MODELS: set[str] = set(LAGRANGIAN_MODELS) | set(DECOMPOSED_MODELS)
COMMON_STYLE_MODELS: set[str] = (
    set(MODEL_REGISTRY.keys()) - STRUCTURED_MODELS
)


def model_needs_group(model_type: str, group: str) -> bool:
    return group in MODEL_HP_LAYERS.get(model_type, [])


def union_groups(model_types: list[str]) -> set[str]:
    g: set[str] = set()
    for m in model_types:
        g.update(MODEL_HP_LAYERS.get(m, []))
    return g


def _specific_dict_for_model(model_type: str) -> dict[str, dict[str, Any]]:
    is_decomposed = model_type in DECOMPOSED_MODELS
    is_physics_w = model_type in PHYSICS_WEIGHT_MODELS
    is_ec = model_type in EQUATION_CONSTRAINED_MODELS

    if model_type == "LagrangianStructuredFNN":
        return {**LNN_SPECIFIC_HP, **PHYSICS_WEIGHT_HP}
    if model_type == "DecomposedStructuredPINNFNN":
        return {**DECOMPOSED_FNN_HP, **PHYSICS_WEIGHT_HP}
    if is_ec:
        return {**FNN_SPECIFIC_HP, **PHYSICS_WEIGHT_HP, **EC_PINN_HP}
    if model_type == "ResidualCorrectionFNN":
        return {**FNN_SPECIFIC_HP, **RESIDUAL_CORRECTION_HP}
    if is_physics_w:
        return {**FNN_SPECIFIC_HP, **PHYSICS_WEIGHT_HP}
    return dict(FNN_SPECIFIC_HP)


def get_model_hp_docs(model_type: str) -> tuple[dict, dict]:
    """Return (specific_hp_docs, common_hp_docs) for a model type."""
    specific = _specific_dict_for_model(model_type)
    is_lagrangian = model_type == "LagrangianStructuredFNN"
    is_decomposed = model_type in DECOMPOSED_MODELS

    skip_common: set[str] = set()
    if is_lagrangian or is_decomposed:
        skip_common.add("activation")
    if is_lagrangian:
        skip_common.add("dropout")

    common = {k: v for k, v in COMMON_HP_DOCS.items() if k not in skip_common}
    return specific, common


def lookup_hp_doc(model_type: str, key: str) -> dict[str, Any] | None:
    """Doc entry for a key for this model (specific overrides common)."""
    spec, common = get_model_hp_docs(model_type)
    if key in spec:
        return dict(spec[key])
    if key in common:
        return dict(common[key])
    return None


def activation_prompt_split_needed(model_types: list[str]) -> bool:
    has_common = any(m in COMMON_STYLE_MODELS for m in model_types)
    has_struct = any(m in STRUCTURED_MODELS for m in model_types)
    return has_common and has_struct


def dropout_prompt_split_needed(model_types: list[str]) -> bool:
    return activation_prompt_split_needed(model_types)


NORMAL_MODE_HIDDEN_KEYS: frozenset[str] = frozenset({
    "min_delta",
})


def should_prompt_key(
    key: str,
    expert: bool,
    shared_values: dict[str, Any],
) -> bool:
    if expert:
        return True
    if key in NORMAL_MODE_HIDDEN_KEYS:
        return False
    if key == "n_collocation":
        return float(shared_values.get("lambda_collocation", 0.0) or 0.0) > 0.0
    return True


def apply_accurate_nominal_to_docs(
    model_type: str,
    docs: dict[str, dict[str, Any]],
    *,
    n_train_samples: int | None,
    epochs: int | None,
) -> None:
    """Mutate doc dict copies in place (defaults only)."""
    if DEFAULT_PROFILE != "accurate_nominal":
        return

    # Physics-regularized: wider FNN
    if model_type == "PhysicsRegularizedFNN":
        if "hidden_layers" in docs:
            docs["hidden_layers"] = {**docs["hidden_layers"], "default": [256, 512, 256]}

    # Residual: [128,128] MLP — hidden_layers in FNN
    if model_type == "ResidualCorrectionFNN":
        if "hidden_layers" in docs:
            docs["hidden_layers"] = {**docs["hidden_layers"], "default": [128, 128]}
        if "activation" in docs:
            docs["activation"] = {**docs["activation"], "default": "tanh"}

    # Physics models need longer patience than BlackBox: the curriculum decay
    # phase (where w_p drops from peak → floor) is where EC-PINN/PhysicsReg
    # realise most of their gains, and it starts well after the first plateau.
    # Empirical: EC-PINN ep451 run reached its best val_rmse at epoch 399 with
    # w_p fully decayed to ~0.26; patience=60 cuts this off at ~170 epochs
    # (leaving ~70% of the benefit on the table).  BlackBox keeps patience=60
    # since it has no curriculum phase to wait for.
    if model_type in PHYSICS_WEIGHT_MODELS and "patience" in docs:
        docs["patience"] = {**docs["patience"], "default": 80}

    # EC-PINN / PhysicsRegularized — slightly longer warmup for pure-MLP PINNs.
    if model_type in EQUATION_CONSTRAINED_MODELS | {"PhysicsRegularizedFNN"}:
        if "physics_warmup_fraction" in docs:
            docs["physics_warmup_fraction"] = {
                **docs["physics_warmup_fraction"], "default": 0.05,
            }

    # ---- Low-data regime (n_train < 250k) doc-default bumps --------------
    # Keep the displayed wizard defaults in sync with the post-merge floors
    # applied in apply_profile_to_hp_dict() so the user sees (and can confirm)
    # the exact values that will actually be used.
    _n = int(n_train_samples or 0)
    if 0 < _n < 250_000:
        if "feature_noise_std" in docs:
            _cur = float(docs["feature_noise_std"].get("default", 0.01) or 0)
            if _cur < 0.025:
                docs["feature_noise_std"] = {**docs["feature_noise_std"], "default": 0.025}
        if "weight_decay" in docs:
            _cur = float(docs["weight_decay"].get("default", 0.002) or 0)
            if _cur < 0.008:
                docs["weight_decay"] = {**docs["weight_decay"], "default": 0.008}
        # Structured models keep their tuned lower dropout; only MLP-family
        # get the 0.12 floor here (matches apply_profile_to_hp_dict).
        if model_type not in LAGRANGIAN_MODELS and "dropout" in docs:
            _cur = float(docs["dropout"].get("default", 0.1) or 0)
            if _cur < 0.12:
                docs["dropout"] = {**docs["dropout"], "default": 0.12}
        # PINN scheduling: longer warmup, fixed schedule, physics-weight caps
        if "physics_warmup_fraction" in docs:
            _cur = float(docs["physics_warmup_fraction"].get("default", 0.03) or 0)
            if _cur < 0.05:
                docs["physics_warmup_fraction"] = {
                    **docs["physics_warmup_fraction"], "default": 0.05,
                }


def apply_profile_to_hp_dict(model_type: str, hp: dict[str, Any]) -> None:
    """Per-model post-merge tweaks (in place)."""
    if DEFAULT_PROFILE != "accurate_nominal":
        return

    if model_type == "PhysicsRegularizedFNN":
        hp["hidden_layers"] = hp.get("hidden_layers") or [256, 512, 256]

    # Patience floor for physics models (see doc comment in
    # apply_accurate_nominal_to_docs).  Only raise, never lower — user-set
    # higher values are preserved.
    if model_type in PHYSICS_WEIGHT_MODELS:
        if int(hp.get("patience", 0) or 0) < 80:
            hp["patience"] = 80

    # Physics weight is now a convex-mixture α ∈ [0, 1].  Clamp any legacy /
    # user-supplied value that violates the invariant (e.g. α = 2.0 from an
    # older config).  Values above 1 would mean negative data weight, which
    # is mathematically invalid.
    if model_type in PHYSICS_WEIGHT_MODELS:
        _pw = float(hp.get("physics_weight", 0.5) or 0.5)
        if _pw < 0.0:
            hp["physics_weight"] = 0.0
        elif _pw > 1.0:
            hp["physics_weight"] = 1.0

    # EC-PINN / PhysicsRegularized — pure MLP architecture; slightly longer
    # warmup (5% vs 3%) since physics is the only inductive bias.
    if model_type in EQUATION_CONSTRAINED_MODELS | {"PhysicsRegularizedFNN"}:
        if float(hp.get("physics_warmup_fraction", 0.03) or 0) < 0.05:
            hp["physics_warmup_fraction"] = 0.05

    if model_type == "ResidualCorrectionFNN":
        hp.setdefault("alpha_reg_weight", 0.05)
        hp.setdefault("activation", "tanh")

    # ---- Reduced-data regime tuning (n_train < 250k) -----------------------
    # With less data the physics prior is MORE valuable — increase physics
    # constraint strength and regularisation to prevent overfitting.
    _n_train = int(hp.get("_n_train_samples", 0) or 0)
    _is_low_data = 0 < _n_train < 250_000
    if _is_low_data:
        # Stronger regularisation on less data.
        if float(hp.get("feature_noise_std", 0.02) or 0) < 0.03:
            hp["feature_noise_std"] = 0.03
        if float(hp.get("weight_decay", 0.005) or 0) < 0.008:
            hp["weight_decay"] = 0.008
        # Dropout floor — only for MLP-family models; structured Lagrangian/
        # Decomposed are kept at their own tuned (lower) dropout to avoid
        # destabilising the Cholesky SPD branch.
        if model_type not in LAGRANGIAN_MODELS:
            if float(hp.get("dropout", 0.1) or 0) < 0.12:
                hp["dropout"] = 0.12

        # Slightly longer physics warmup for low-data (5% vs 3% normal).
        if "physics_warmup_fraction" in hp:
            if float(hp.get("physics_warmup_fraction", 0.03) or 0) < 0.05:
                hp["physics_warmup_fraction"] = 0.05

    hp.pop("_n_train_samples", None)


def merge_doc_dicts_for_prompt(
    model_types: list[str],
    key: str,
) -> dict[str, Any]:
    """Prefer model-specific (specific dict) over common so structured vs MLP defaults differ."""
    for m in model_types:
        spec, _com = get_model_hp_docs(m)
        if key in spec:
            return dict(spec[key])
    for m in model_types:
        _spec, com = get_model_hp_docs(m)
        if key in com:
            return dict(com[key])
    return {"default": None, "desc": key, "effect": ""}


def merge_shared_into_model_hp(model_type: str, shared: dict[str, Any]) -> dict[str, Any]:
    """Copy only keys this model uses; map split activation/dropout."""
    spec, common = get_model_hp_docs(model_type)
    valid_keys = set(spec.keys()) | set(common.keys())
    hp: dict[str, Any] = {}
    skip = {
        "activation_mlp",
        "activation_structured",
        "dropout_mlp",
        "dropout_structured",
    }
    for k, v in shared.items():
        if k in skip:
            continue
        if k in valid_keys:
            hp[k] = v

    if "activation_mlp" in shared or "activation_structured" in shared:
        if "activation" in valid_keys:
            if model_type in STRUCTURED_MODELS:
                hp["activation"] = shared["activation_structured"]
            else:
                hp["activation"] = shared["activation_mlp"]
    elif "activation" in shared and "activation" in valid_keys:
        hp["activation"] = shared["activation"]

    if "dropout_mlp" in shared or "dropout_structured" in shared:
        if "dropout" in valid_keys:
            if model_type in STRUCTURED_MODELS:
                hp["dropout"] = shared["dropout_structured"]
            else:
                hp["dropout"] = shared["dropout_mlp"]
    elif "dropout" in shared and "dropout" in valid_keys:
        hp["dropout"] = shared["dropout"]

    return hp
