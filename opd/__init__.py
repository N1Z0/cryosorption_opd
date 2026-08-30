"""Orbital Propellant Depot — lumped-parameter thermodynamic model.

Top-level package. Sub-packages:

    opd.constants            physical constants (SI)
    opd.fluids               CoolProp wrapper (ParaHydrogen default)
    opd.adsorbents           equilibrium isotherms + material definitions
    opd.catalysts            reserved for Para-Ortho catalyst models (M5)

All physics code imports strictly in the direction
``fluids / adsorbents  →  nodes  →  tank  →  simulation``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
