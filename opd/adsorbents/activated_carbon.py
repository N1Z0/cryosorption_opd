"""Activated-carbon sample NZ208V3 (208C).

Parameters are empirical Dubinin–Astakhov fits to volumetric H₂ isotherms.
Two branches join at :math:`T = 32` K:

* **Subcritical** (:math:`T < 32` K): linear :math:`E(T)`, exponential
  :math:`p_0(T)`.
* **Supercritical** (:math:`T \\geq 32` K): linear :math:`E(T)`,
  constant :math:`p_0`.

Isotherm data were originally in cc(STP)/g; conversion ``/ 22400 · 1000``
yields mol/kg. All parameters are in SI.
"""

from __future__ import annotations

from .base import AdsorbentMaterial, constant_isosteric_heat
from .isotherm_models import (
    DubininAstakhov,
    HybridDA,
    constant_pressure_torr,
    exponential_pressure_torr,
    linear_energy,
)

__all__ = ["ActivatedCarbon208C", "cp_cryo_j_kg_k", "_CP_CRYO_A", "_CP_CRYO_B"]

_N_MAX_MOL_PER_KG: float = 552.71294209 * (1.0 / 22400.0) * 1000.0
_V_A_M3_PER_KG: float = 0.0004838173512692576

_SUBCRITICAL_E = linear_energy(-1384.82301956, 181.31452794)
_SUBCRITICAL_P0 = exponential_pressure_torr(0.0074801, 0.59807858)

_SUPERCRITICAL_E = linear_energy(-4776.95770, -4.64880774)
_SUPERCRITICAL_P0 = constant_pressure_torr(3050558.0367990825)

_T_SWITCH_K: float = 32.0

_ISOSTERIC_HEAT_JMOL: float = 1929.5548734380006
"""Constant isosteric heat of adsorption used in the 1-Temperature model."""

_CP_SKELETON_J_PER_KG_K: float = 850.0
"""Effective constant skeleton heat capacity for the 1-Temperature model."""

_CP_CRYO_A: float = 0.0878    # J/(kg K²)
_CP_CRYO_B: float = 3.05e-5   # J/(kg K⁴)


def _cp_constant(T_sorb: float) -> float:
    return _CP_SKELETON_J_PER_KG_K


def cp_cryo_j_kg_k(T: float) -> float:
    """Temperature-dependent skeleton c_p for the 2-Temperature model."""
    T = max(1.0, float(T))
    return _CP_CRYO_A * T + _CP_CRYO_B * T ** 3


def ActivatedCarbon208C() -> AdsorbentMaterial:
    """Return the 208C activated-carbon adsorbent preset."""
    subcritical = DubininAstakhov(
        n_max=_N_MAX_MOL_PER_KG,
        micropore_volume=_V_A_M3_PER_KG,
        characteristic_energy=_SUBCRITICAL_E,
        pseudo_saturation_pressure=_SUBCRITICAL_P0,
        exponent=2.0,
    )
    supercritical = DubininAstakhov(
        n_max=_N_MAX_MOL_PER_KG,
        micropore_volume=_V_A_M3_PER_KG,
        characteristic_energy=_SUPERCRITICAL_E,
        pseudo_saturation_pressure=_SUPERCRITICAL_P0,
        exponent=2.0,
    )
    isotherm = HybridDA(
        subcritical=subcritical,
        supercritical=supercritical,
        T_switch=_T_SWITCH_K,
    )
    return AdsorbentMaterial(
        name="Activated Carbon 208C",
        skeletal_density=2150.0,
        micropore_volume=_V_A_M3_PER_KG,
        isotherm=isotherm,
        cp_skeleton=_cp_constant,
        isosteric_heat_fn=constant_isosteric_heat(_ISOSTERIC_HEAT_JMOL),
    )
