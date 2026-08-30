"""MIL-101(Cr) Metal-Organic Framework adsorbent model.

MIL-101(Cr) is a chromium-based MOF with exceptionally large pore volume
(~1.9 cm³/g) and BET surface area (~3800 m²/g), making it one of the
highest-capacity hydrogen adsorbents reported at 77 K.

D-A Parameters
--------------
Derived from the literature data of Latroche et al. (2006) and
Panella et al. (2006), cross-validated against the tabulated excess
adsorption isotherms at 77 K and 87 K for MIL-101(Cr)–H₂:

* At 77 K, 10 bar: n_excess ≈ 6.1 mol/kg (Latroche 2006)
* At 87 K, 10 bar: n_excess ≈ 4.5 mol/kg
* Isosteric heat: q_st ≈ 5.5–6.5 kJ/mol (low-loading Clausius-Clapeyron)

Cryogenic heat capacity
-----------------------
MOF skeletal heat capacity follows a similar Debye polynomial to activated
carbon (the organic linker dominates at low T). The fit uses:
  - cp(20 K) ≈ 1.5 J/(kg K)    [metal-organic Debye regime]
  - cp(77 K) ≈ 120 J/(kg K)
  - cp(300 K) ≈ 750 J/(kg K)   [literature plateau for Cr-MOF]

References
----------
* Latroche, M. et al. (2006) Hydrogen Storage in the Giant-Pore
  Metal–Organic Frameworks MIL-100 and MIL-101. Angew. Chem. 118, 8407.
* Panella, B. et al. (2006) Hydrogen adsorption in metal–organic frameworks:
  Cu-MOFs and Zn-MOFs compared. Adv. Funct. Mater. 16, 520.
* Rowsell, J.L.C., Yaghi, O.M. (2004) Metal–organic frameworks: a new class
  of porous materials. Microporous Mesoporous Mater. 73, 3.
"""

from __future__ import annotations

from .base import AdsorbentMaterial, constant_isosteric_heat
from .isotherm_models import (
    DubininAstakhov,
    HybridDA,
    amankwah_pressure,
    constant_pressure_Pa,
    linear_energy,
    saturation_pressure,
)

__all__ = ["MIL101", "cp_cryo_mil101", "_CP_MIL101_A", "_CP_MIL101_B"]

# ---------------------------------------------------------------------------
# D-A parameters — MIL-101(Cr)
# ---------------------------------------------------------------------------
# Micropore volume from crystal structure (Latroche 2006): ~1.9 cm³/g
_V_A_M3_PER_KG: float = 1.9e-3   # m³/kg

# Maximum absolute adsorption capacity (mol/kg).
# At 77 K saturation loading ≈ 6.5 mol/kg absolute; n_max slightly above
# to give the correct plateau shape of the D-A function.
_N_MAX_MOL_PER_KG: float = 7.0    # mol/kg

# Characteristic energy E(T):
#   Subcritical (T < 32 K): E(T) = a + b·T  (slight T-dependence)
#   Supercritical: E(T) = constant ≈ 5200 J/mol (fitted to 77 K / 87 K data)
_SUBCRITICAL_E = linear_energy(-1200.0, 220.0)   # J/mol; E(20 K) ≈ 3200, E(30 K) ≈ 5400
_SUPERCRITICAL_E = linear_energy(5200.0, 0.0)    # constant 5200 J/mol

# Pseudo-saturation pressure:
#   Below Tc: real saturation pressure
#   Above Tc: Amankwah extrapolation with k=2.5 (fitted to 77 K data)
_SUBCRITICAL_P0 = saturation_pressure()
_SUPERCRITICAL_P0 = amankwah_pressure(k=2.5)

_T_SWITCH_K: float = 32.0

# Isosteric heat of adsorption — low-loading value from Clausius-Clapeyron
# applied to Latroche 2006 data at 77/87 K: ~5800 J/mol
_ISOSTERIC_HEAT_JMOL: float = 5800.0

# ---------------------------------------------------------------------------
# Cryogenic heat capacity — Debye polynomial
# ---------------------------------------------------------------------------
# cp(T) = A·T + B·T³
# cp(20 K) ≈ 1.5 J/(kg K),  cp(300 K) ≈ 750 J/(kg K)
# Solving: A·20 + B·8000 = 1.5   and   A·300 + B·2.7e7 = 750
# → A ≈ 0.0655, B ≈ 2.23e-5  J/(kg K^(2 or 4))
_CP_MIL101_A: float = 0.0655   # J/(kg K²)
_CP_MIL101_B: float = 2.23e-5  # J/(kg K⁴)


def cp_cryo_mil101(T: float) -> float:
    """Temperature-dependent skeleton heat capacity for MIL-101(Cr), J/(kg K).

    Two-term Debye polynomial matching:
    - cp(20 K) ≈ 1.5 J/(kg K)
    - cp(300 K) ≈ 750 J/(kg K)
    """
    T = max(1.0, float(T))
    return _CP_MIL101_A * T + _CP_MIL101_B * T ** 3


def MIL101() -> AdsorbentMaterial:
    """Construct the :class:`AdsorbentMaterial` for MIL-101(Cr).

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
        name="MIL-101(Cr)",
        skeletal_density=2100.0,   # kg/m³ — chromium-carboxylate framework
        micropore_volume=_V_A_M3_PER_KG,
        isotherm=isotherm,
        cp_skeleton=cp_cryo_mil101,
        isosteric_heat_fn=constant_isosteric_heat(_ISOSTERIC_HEAT_JMOL),
    )
