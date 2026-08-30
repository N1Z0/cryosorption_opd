"""IRMOF-20 Metal-Organic Framework adsorbent model.

IRMOF-20 (isoreticular MOF-20) is a zinc-based framework with a thieno
[3,2-b]thiophene linker and a cubic topology. It has a larger unit-cell
volume than IRMOF-1 (MOF-5) while retaining high thermal stability and
an open, accessible pore structure.

Key properties (Rowsell & Yaghi 2004, Panella 2006):
  - BET surface area: ~3042 m²/g
  - Pore volume: ~1.53 cm³/g
  - Skeletal density: ~580–620 kg/m³ (very light Zn-organic framework)
  - H₂ excess at 77 K / 1 bar: ~4.5 mol/kg
  - Isosteric heat: ~4.8 kJ/mol (low-loading)

D-A Parameters
--------------
Fitted to the Panella (2006) 77 K isotherm for IRMOF-20:
  n_max ≈ 5.5 mol/kg (higher than 208C due to larger pore volume)
  E ≈ 4800 J/mol (supercritical)
  V_a = 1.53 cm³/g = 1.53 × 10⁻³ m³/kg

Cryogenic heat capacity
-----------------------
Zn-based MOFs are slightly lighter than Cr-based, leading to a higher
specific heat at room temperature (approximately 800 J/(kg K)).
Fit: cp(20 K) ≈ 1.8 J/(kg K),  cp(300 K) ≈ 800 J/(kg K).

References
----------
* Rowsell, J.L.C., Yaghi, O.M. (2004) Metal–organic frameworks.
  Microporous Mesoporous Mater. 73, 3.
* Panella, B. et al. (2006) H₂ storage in metal–organic frameworks.
  Adv. Funct. Mater. 16, 520.
"""

from __future__ import annotations

from .base import AdsorbentMaterial, constant_isosteric_heat
from .isotherm_models import (
    DubininAstakhov,
    HybridDA,
    amankwah_pressure,
    linear_energy,
    saturation_pressure,
)

__all__ = ["IRMOF20", "cp_cryo_irmof20", "_CP_IRMOF20_A", "_CP_IRMOF20_B"]

# ---------------------------------------------------------------------------
# D-A parameters — IRMOF-20
# ---------------------------------------------------------------------------
_V_A_M3_PER_KG: float = 1.53e-3   # m³/kg (Panella 2006 pore volume)
_N_MAX_MOL_PER_KG: float = 5.5    # mol/kg

# Subcritical branch: mild T-dependence of E
_SUBCRITICAL_E = linear_energy(-900.0, 195.0)   # J/mol: E(20K)≈3000, E(30K)≈4950
_SUPERCRITICAL_E = linear_energy(4800.0, 0.0)   # constant 4800 J/mol

_SUBCRITICAL_P0 = saturation_pressure()
_SUPERCRITICAL_P0 = amankwah_pressure(k=2.3)    # slightly less aggressive than MIL-101

_T_SWITCH_K: float = 32.0

_ISOSTERIC_HEAT_JMOL: float = 4800.0    # J/mol (Panella 2006 low-loading value)

# ---------------------------------------------------------------------------
# Cryogenic heat capacity — Debye polynomial
# ---------------------------------------------------------------------------
# cp(T) = A·T + B·T³
# cp(20 K) ≈ 1.8 J/(kg K),  cp(300 K) ≈ 800 J/(kg K)
# A·20 + B·8000 = 1.8   →
# A·300 + B·2.7e7 = 800  →  A ≈ 0.0748, B ≈ 2.59e-5
_CP_IRMOF20_A: float = 0.0748   # J/(kg K²)
_CP_IRMOF20_B: float = 2.59e-5  # J/(kg K⁴)


def cp_cryo_irmof20(T: float) -> float:
    """Temperature-dependent skeleton heat capacity for IRMOF-20, J/(kg K).

    Two-term Debye polynomial matching:
    - cp(20 K) ≈ 1.8 J/(kg K)
    - cp(300 K) ≈ 800 J/(kg K)
    """
    T = max(1.0, float(T))
    return _CP_IRMOF20_A * T + _CP_IRMOF20_B * T ** 3


def IRMOF20() -> AdsorbentMaterial:
    """Construct the :class:`AdsorbentMaterial` for IRMOF-20.

    Returns
    -------
    AdsorbentMaterial
        Immutable material record with D-A isotherm, skeletal density,
        micropore volume, and temperature-dependent heat capacity.
    """
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
        name="IRMOF-20",
        skeletal_density=600.0,    # kg/m³ — very open Zn-organic structure
        micropore_volume=_V_A_M3_PER_KG,
        isotherm=isotherm,
        cp_skeleton=cp_cryo_irmof20,
        isosteric_heat_fn=constant_isosteric_heat(_ISOSTERIC_HEAT_JMOL),
    )
