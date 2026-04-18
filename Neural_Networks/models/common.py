"""Shared building blocks for all model architectures."""

from __future__ import annotations

import torch.nn as nn


ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu":       nn.ReLU,
    "tanh":       nn.Tanh,
    "silu":       nn.SiLU,
    "gelu":       nn.GELU,
    "elu":        nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def build_mlp(
    in_dim: int,
    hidden_layers: list[int],
    out_dim: int,
    activation: str,
    dropout: float,
) -> nn.Sequential:
    """LayerNorm MLP with configurable activation and dropout."""
    Act = ACTIVATION_MAP.get(activation, nn.SiLU)
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_layers:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), Act(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)
