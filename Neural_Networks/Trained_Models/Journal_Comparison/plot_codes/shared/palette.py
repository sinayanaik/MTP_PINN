"""Architecture-name normalisation and per-arch styling lookups.

The registry uses class names (``EDR``, ``BlackBoxFNN``,
``PhysicsRegularizedFNN``); ``grid_results.csv`` uses short tags
(``edr``, ``fnn``, ``physreg``).  Everything downstream uses the canonical
keys defined in :mod:`config`.
"""

from __future__ import annotations

from .config import ARCH_EDR, ARCH_FNN, ARCH_PHYSREG, PlotConfig

_CANON = {
    "edr": ARCH_EDR,
    "edrmodel": ARCH_EDR,
    "fnn": ARCH_FNN,
    "blackboxfnn": ARCH_FNN,
    "physreg": ARCH_PHYSREG,
    "physicsregularized": ARCH_PHYSREG,
    "physicsregularizedfnn": ARCH_PHYSREG,
}


def canon_arch(name: str) -> str:
    """Map any spelling of an architecture to its canonical key."""
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key not in _CANON:
        raise KeyError(f"unknown architecture name: {name!r}")
    return _CANON[key]


def ordered_archs(cfg: PlotConfig) -> list[str]:
    return list(cfg.arch_order)


def color(cfg: PlotConfig, arch: str) -> str:
    return cfg.arch_colors[canon_arch(arch)]


def label(cfg: PlotConfig, arch: str) -> str:
    return cfg.arch_labels[canon_arch(arch)]


def marker(cfg: PlotConfig, arch: str) -> str:
    return cfg.arch_markers[canon_arch(arch)]


def line_width(cfg: PlotConfig, arch: str) -> float:
    """EDR gets a heavier stroke so it reads as the protagonist."""
    base = cfg.line_w
    if canon_arch(arch) == cfg.emphasis_arch:
        return base * cfg.emphasis_line_scale
    return base


def zorder(cfg: PlotConfig, arch: str) -> int:
    """Draw EDR on top of the baselines."""
    return 5 if canon_arch(arch) == cfg.emphasis_arch else 3
