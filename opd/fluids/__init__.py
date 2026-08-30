"""Fluid-property layer.

All CoolProp access in the OPD codebase is funnelled through
:class:`opd.fluids.fluid_properties.FluidProperties`. No other module
imports CoolProp directly.
"""

from __future__ import annotations

from .fluid_properties import FluidProperties, KNOWN_FLUIDS
from .hydrogen import normal_hydrogen, orthohydrogen, parahydrogen

__all__ = [
    "FluidProperties",
    "KNOWN_FLUIDS",
    "parahydrogen",
    "orthohydrogen",
    "normal_hydrogen",
]
