"""Hydrogen spin-isomer presets.

The orbital depot baseline is :func:`parahydrogen` (liquefied H₂ is catalysed
to ≥ 99.8 % para to minimise long-term boil-off from spontaneous
ortho → para conversion, which releases ~699 J/mol).
"""

from __future__ import annotations

from .fluid_properties import FluidProperties


def parahydrogen() -> FluidProperties:
    """ParaHydrogen preset (default for orbital depots)."""
    return FluidProperties("ParaHydrogen")


def orthohydrogen() -> FluidProperties:
    """OrthoHydrogen preset."""
    return FluidProperties("OrthoHydrogen")


def normal_hydrogen() -> FluidProperties:
    """Normal (equilibrium) hydrogen — CoolProp identifier ``Hydrogen``."""
    return FluidProperties("Hydrogen")
