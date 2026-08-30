"""Equilibrium adsorption models and concrete adsorbent materials.

Public API
----------
* :class:`IsothermModel` — abstract isotherm
* :class:`DubininAstakhov`, :class:`HybridDA` — concrete isotherm forms
* :class:`AdsorbentMaterial` — material carrying an isotherm plus its
  skeletal and thermal data
* :func:`ActivatedCarbon208C` — factory for the baseline AC sample
* :func:`MIL101` — MIL-101(Cr) high-capacity MOF (M6)
* :func:`IRMOF20` — IRMOF-20 Zn-based MOF (M6)
* :func:`get_adsorbent` — string-key factory (``get_adsorbent("MIL-101")``)
* :func:`list_adsorbents` — enumerate registered materials
* :func:`register_adsorbent` — add custom materials at runtime
"""

from __future__ import annotations

from .activated_carbon import ActivatedCarbon208C
from .base import AdsorbentMaterial, constant_isosteric_heat
from .factory import get_adsorbent, list_adsorbents, register_adsorbent
from .isotherm_models import (
    DubininAstakhov,
    HybridDA,
    IsothermModel,
    amankwah_pressure,
    constant_pressure_Pa,
    constant_pressure_torr,
    exponential_pressure,
    exponential_pressure_torr,
    linear_energy,
    saturation_pressure,
)
from .mof_irmof20 import IRMOF20
from .mof_mil101 import MIL101

__all__ = [
    "IsothermModel",
    "DubininAstakhov",
    "HybridDA",
    "AdsorbentMaterial",
    "constant_isosteric_heat",
    "linear_energy",
    "constant_pressure_Pa",
    "constant_pressure_torr",
    "exponential_pressure",
    "exponential_pressure_torr",
    "amankwah_pressure",
    "saturation_pressure",
    "ActivatedCarbon208C",
    "MIL101",
    "IRMOF20",
    "get_adsorbent",
    "list_adsorbents",
    "register_adsorbent",
]
