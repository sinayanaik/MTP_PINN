"""
Neural Network Model Registry — 6 FNN architectures (guide Arch 1–6).

Signatures:
  • Black-box, physics-regularised, equation-constrained: forward(features) → tau_hat
  • Residual correction: forward(features, physics) → tau_hat
  • Lagrangian / Decomposed: forward(features) or (features, physics) → (tau_hat, components)

Equation-Constrained vs Physics-Regularised: both use nominal torques, but EC applies a
learnable affine φ on the summed equation torque, penalises the residual τ̂ − φ(τ_eq),
and can add collocation off data (see train.py HP `lambda_collocation`). PR uses
MSE(τ̂, τ_physics) as a soft target only.
"""

from Neural_Networks.models.unified_abc_fnn import (
    BlackBoxFNN,
    PhysicsRegularizedFNN,
    ResidualCorrectionFNN,
)
from Neural_Networks.models.lagrangian_structured_fnn import LagrangianStructuredFNN
from Neural_Networks.models.equation_constrained_pinn_fnn import EquationConstrainedPINNFNN
from Neural_Networks.models.decomposed_structured_pinn_fnn import DecomposedStructuredPINNFNN

# ---------------------------------------------------------------------------
# Registry: maps class name string → class object
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {
    "BlackBoxFNN":                BlackBoxFNN,
    "PhysicsRegularizedFNN":      PhysicsRegularizedFNN,
    "ResidualCorrectionFNN":      ResidualCorrectionFNN,
    "LagrangianStructuredFNN":    LagrangianStructuredFNN,
    "EquationConstrainedPINNFNN": EquationConstrainedPINNFNN,
    "DecomposedStructuredPINNFNN": DecomposedStructuredPINNFNN,
}

# ---------------------------------------------------------------------------
# Category groupings for CLI display
# ---------------------------------------------------------------------------

MODEL_CATEGORIES: dict[str, list[str]] = {
    "A — Black Box (Arch 1)":             ["BlackBoxFNN"],
    "B — Physics-Regularized (Arch 2)":   ["PhysicsRegularizedFNN"],
    "C — Residual Correction (Arch 3)":   ["ResidualCorrectionFNN"],
    "D — Lagrangian Structured (Arch 4)": ["LagrangianStructuredFNN"],
    "E.1 — Equation-Constrained PINN (Arch 5)": ["EquationConstrainedPINNFNN"],
    "E.2 — Decomposed Structured PINN (Arch 6)": ["DecomposedStructuredPINNFNN"],
}

# ---------------------------------------------------------------------------
# Model type sets for conditional logic in train.py
# ---------------------------------------------------------------------------

FNN_MODELS: set[str] = {
    "BlackBoxFNN",
    "PhysicsRegularizedFNN",
    "ResidualCorrectionFNN",
    "LagrangianStructuredFNN",
    "EquationConstrainedPINNFNN",
    "DecomposedStructuredPINNFNN",
}

# Models that accept a physics weight lambda for a physics loss term
PHYSICS_WEIGHT_MODELS: set[str] = {
    "PhysicsRegularizedFNN",
    "LagrangianStructuredFNN",
    "EquationConstrainedPINNFNN",
    "DecomposedStructuredPINNFNN",
}

# Models that return (tau_hat, components) from forward()
LAGRANGIAN_MODELS: set[str] = {
    "LagrangianStructuredFNN",
    "DecomposedStructuredPINNFNN",
}

# Models whose forward() requires physics as input
PHYSICS_INPUT_MODELS: set[str] = {
    "ResidualCorrectionFNN",
}

# Models using decomposed multi-term loss (compute_loss() method)
DECOMPOSED_MODELS: set[str] = {
    "DecomposedStructuredPINNFNN",
}

# Models using equation-constrained residual loss (compute_loss() method)
EQUATION_CONSTRAINED_MODELS: set[str] = {
    "EquationConstrainedPINNFNN",
}

# Save directory names (one subdirectory per model class)
MODEL_SAVE_DIRS: dict[str, str] = {name: name for name in MODEL_REGISTRY}

__all__ = [
    # Classes
    "BlackBoxFNN",
    "PhysicsRegularizedFNN",
    "ResidualCorrectionFNN",
    "LagrangianStructuredFNN",
    "EquationConstrainedPINNFNN",
    "DecomposedStructuredPINNFNN",
    # Registry dicts/sets
    "MODEL_REGISTRY", "MODEL_CATEGORIES", "MODEL_SAVE_DIRS",
    "FNN_MODELS", "PHYSICS_WEIGHT_MODELS",
    "LAGRANGIAN_MODELS", "PHYSICS_INPUT_MODELS",
    "DECOMPOSED_MODELS", "EQUATION_CONSTRAINED_MODELS",
]
