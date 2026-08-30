"""AX-21 superactivated carbon (Anderson AX-21 / Maxsorb class).

Unlike the 208C parameterisation, which is fitted to isotherms limited to
about \\SI{1.07}{bar} by the measurement apparatus, this material's
modified Dubinin--Astakhov parameters were fitted by Richard, B\\'enard
and Chahine against hydrogen isotherms spanning **30--298 K and
0--6 MPa** [1, Table 3].  AX-21 therefore serves as an independent,
high-pressure-validated cross-check on any conclusion drawn from an
extrapolated low-pressure fit.

Model form (identical to :mod:`opd.adsorbents.activated_carbon`):

.. math::

    n_{\\mathrm{abs}} = n_{\\max}\\,
    \\exp\\!\\left[-\\left(\\frac{RT}{\\alpha + \\beta T}\\right)^{2}
    \\ln^{2}\\frac{p_0}{p}\\right]

with a single branch valid over the whole range --- no sub/supercritical
switch is required, because the fit already spans both regimes.

Parameters (Ref. [1], Table 3, ``H2`` column)
---------------------------------------------
==================  ==========================
:math:`n_{\\max}`     \\SI{71.6}{mol/kg}
:math:`p_0`          \\SI{1470}{MPa}
:math:`\\alpha`       \\SI{3080}{J/mol}
:math:`\\beta`        \\SI{18.9}{J/(mol.K)}
:math:`V_a`          \\SI{1.43e-3}{m^3/kg}
==================  ==========================

The skeletal density is **not** given in Ref. [1]; the value used here is
the conventional graphitic-carbon figure and is flagged as an assumption
because it enters the capacity criterion
:math:`\\rho_{\\mathrm{eff}} = n_{\\mathrm{abs}} M/(1/\\rho_{\\mathrm{skel}} + V_a)`.
For this material :math:`V_a \\gg 1/\\rho_{\\mathrm{skel}}`, so
:math:`\\rho_{\\mathrm{eff}}` is dominated by the micropore volume and the
assumption is not critical.

References
----------
[1] M.-A. Richard, P. B\\'enard, R. Chahine, "Gas adsorption process in
    activated carbon over a wide temperature range above the critical
    point. Part 1: modified Dubinin-Astakhov model", Adsorption 15
    (2009) 43--51.
"""

from __future__ import annotations

from .base import AdsorbentMaterial, constant_isosteric_heat
from .isotherm_models import DubininAstakhov, constant_pressure_Pa, linear_energy

__all__ = ["AX21", "cp_cryo_ax21"]

# --- Ref. [1], Table 3, H2 column -----------------------------------------
_N_MAX_MOL_PER_KG: float = 71.6          # mol/kg
_P0_PA: float = 1470.0e6                 # Pa  (1470 MPa)
_ALPHA_J_PER_MOL: float = 3080.0         # J/mol
_BETA_J_PER_MOL_K: float = 18.9          # J/(mol K)
_V_A_M3_PER_KG: float = 1.43e-3          # m^3/kg

_SKELETAL_DENSITY: float = 2200.0
"""Assumed graphitic-carbon skeletal density, kg/m3 (not given in Ref. [1])."""

_ISOSTERIC_HEAT_JMOL: float = 4000.0
"""Representative low-loading isosteric heat for H2 on superactivated
carbon, J/mol.  Ref. [1] reports characteristic energies of
:math:`\\alpha + \\beta T \\approx` 3.6-4.6 kJ/mol over 30-77 K; the
constant value used here is of that order and is only relevant to
transient energy balances, not to the capacity results."""

# Debye-like skeleton heat capacity, matching the 208C treatment:
# cp(T) = A*T + B*T^3 with cp(20 K) ~ 2 and cp(300 K) ~ 850 J/(kg K).
_CP_A: float = 0.0878    # J/(kg K^2)
_CP_B: float = 3.05e-5   # J/(kg K^4)


def cp_cryo_ax21(T: float) -> float:
    """Skeleton heat capacity of AX-21, J/(kg K).

    Parameters
    ----------
    T
        Sorbent temperature, K.
    """
    T = max(1.0, float(T))
    return _CP_A * T + _CP_B * T ** 3


def AX21() -> AdsorbentMaterial:
    """Construct the :class:`AdsorbentMaterial` for AX-21.

    A single Dubinin--Astakhov branch covers 30-298 K, since the source
    fit spans the critical point without a switch.
    """
    isotherm = DubininAstakhov(
        n_max=_N_MAX_MOL_PER_KG,
        micropore_volume=_V_A_M3_PER_KG,
        characteristic_energy=linear_energy(_ALPHA_J_PER_MOL, _BETA_J_PER_MOL_K),
        pseudo_saturation_pressure=constant_pressure_Pa(_P0_PA),
        exponent=2.0,
    )
    return AdsorbentMaterial(
        name="AX-21",
        skeletal_density=_SKELETAL_DENSITY,
        micropore_volume=_V_A_M3_PER_KG,
        isotherm=isotherm,
        cp_skeleton=cp_cryo_ax21,
        isosteric_heat_fn=constant_isosteric_heat(_ISOSTERIC_HEAT_JMOL),
    )
