"""Physical constants, SI.

No mutable state, no CoolProp dependency. Values sourced from CODATA 2019
and NIST."""

from __future__ import annotations

from typing import Final

R_UNIVERSAL: Final[float] = 8.314462618
"""Universal gas constant, :math:`\\mathrm{J\\,mol^{-1}\\,K^{-1}}`."""

N_AVOGADRO: Final[float] = 6.02214076e23
"""Avogadro constant, :math:`\\mathrm{mol^{-1}}`."""

K_BOLTZMANN: Final[float] = 1.380649e-23
"""Boltzmann constant, :math:`\\mathrm{J\\,K^{-1}}`."""

P_STANDARD_ATM: Final[float] = 101325.0
"""Standard atmospheric pressure, Pa."""

TORR_TO_PA: Final[float] = 133.32236842105263
"""Torr → Pa conversion, exact up to 17 digits (1 Torr = 101325/760 Pa)."""

T_STP_CC_PER_MOL: Final[float] = 22413.96954
"""Molar volume at 1 atm and 273.15 K (STP, 'old' IUPAC), cm^3/mol.

Used solely for converting legacy adsorption measurements reported in
``cc(STP) / g`` to ``mol / kg``; no hot-path code should call this."""

LHV_H2_MASS: Final[float] = 119.96e6
"""Lower heating value of hydrogen, :math:`\\mathrm{J\\,kg^{-1}}` (NIST)."""

M_H2: Final[float] = 2.01588e-3
"""Molar mass of H₂, :math:`\\mathrm{kg\\,mol^{-1}}` (CODATA)."""

LHV_H2_MOLAR: Final[float] = LHV_H2_MASS * M_H2
"""Lower heating value of hydrogen, :math:`\\mathrm{J\\,mol^{-1}}` (≈ 241.8 kJ/mol)."""
