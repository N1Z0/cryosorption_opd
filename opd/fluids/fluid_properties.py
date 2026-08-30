"""Strictly-SI thin wrapper around :mod:`CoolProp`.

This module is the single point of contact between OPD and CoolProp.
Every other physics module must request thermodynamic properties through
a :class:`FluidProperties` instance so that the back-end can eventually
be swapped (to REFPROP, a cubic EOS, or a tabulated surrogate) without
touching the rest of the stack.

All inputs and outputs are SI: Pa, K, mol, kg, J, m, s.
"""

from __future__ import annotations

from functools import cached_property
from typing import Final

from CoolProp.CoolProp import PropsSI

__all__ = ["FluidProperties", "KNOWN_FLUIDS"]

KNOWN_FLUIDS: Final[frozenset[str]] = frozenset(
    {
        "ParaHydrogen",
        "OrthoHydrogen",
        "Hydrogen",
        "Nitrogen",
        "Helium",
        "Methane",
        "Oxygen",
        "Neon",
    }
)
"""Fluids the rest of OPD is aware of. This is *not* an exhaustive list of
CoolProp-supported fluids; it only documents which ones OPD has been
exercised against. Passing any CoolProp-recognised name still works."""


class FluidProperties:
    """Single-fluid thermodynamic property provider.

    Parameters
    ----------
    name
        CoolProp fluid identifier. Defaults to ``"ParaHydrogen"`` since the
        long-term orbital-depot baseline is liquefied hydrogen that has
        been converted to the para-spin state for minimum boil-off.

    Notes
    -----
    *   All methods take and return SI quantities.
    *   Methods ending in ``_molar`` return *molar* quantities
        (mol, J/mol, mol/m³); methods ending in ``_mass`` return
        per-unit-mass quantities (kg, J/kg, kg/m³).
    *   Inputs ``(p, T)`` must lie in a single-phase region. Inside the
        vapour dome, use :meth:`from_TQ` or the dedicated
        saturation-branch methods.
    """

    def __init__(self, name: str = "ParaHydrogen") -> None:
        if not isinstance(name, str):
            raise TypeError(
                f"FluidProperties name must be a str, got {type(name).__name__}"
            )
        if not name:
            raise ValueError("FluidProperties name must be non-empty")
        self._name = name

    # ------------------------------------------------------------------
    # Identity and fluid-specific constants
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """CoolProp fluid identifier."""
        return self._name

    @cached_property
    def molar_mass(self) -> float:
        """Molar mass :math:`M`, :math:`\\mathrm{kg\\,mol^{-1}}`."""
        return float(PropsSI("M", self._name))

    @cached_property
    def T_critical(self) -> float:
        """Critical temperature :math:`T_c`, K."""
        return float(PropsSI("Tcrit", self._name))

    @cached_property
    def p_critical(self) -> float:
        """Critical pressure :math:`p_c`, Pa."""
        return float(PropsSI("pcrit", self._name))

    @cached_property
    def T_triple(self) -> float:
        """Triple-point temperature :math:`T_t`, K."""
        return float(PropsSI("Ttriple", self._name))

    @cached_property
    def p_triple(self) -> float:
        """Triple-point pressure :math:`p_t`, Pa."""
        return float(PropsSI("ptriple", self._name))

    @cached_property
    def T_normal_boiling(self) -> float:
        """Normal boiling point (:math:`p=101\\,325` Pa), K."""
        return float(PropsSI("T", "P", 101325.0, "Q", 0.0, self._name))

    # ------------------------------------------------------------------
    # Single-phase state functions from (p, T)
    # ------------------------------------------------------------------

    def rho_molar(self, p: float, T: float) -> float:
        """Molar density :math:`\\rho`, :math:`\\mathrm{mol\\,m^{-3}}`."""
        return float(PropsSI("Dmolar", "P", p, "T", T, self._name))

    def rho_mass(self, p: float, T: float) -> float:
        """Mass density :math:`\\rho_m`, :math:`\\mathrm{kg\\,m^{-3}}`."""
        return float(PropsSI("Dmass", "P", p, "T", T, self._name))

    def u_molar(self, p: float, T: float) -> float:
        """Molar internal energy, J/mol."""
        return float(PropsSI("Umolar", "P", p, "T", T, self._name))

    def h_molar(self, p: float, T: float) -> float:
        """Molar enthalpy, J/mol."""
        return float(PropsSI("Hmolar", "P", p, "T", T, self._name))

    def s_molar(self, p: float, T: float) -> float:
        """Molar entropy, J/(mol·K)."""
        return float(PropsSI("Smolar", "P", p, "T", T, self._name))

    def cp_molar(self, p: float, T: float) -> float:
        """Isobaric molar heat capacity, J/(mol·K)."""
        return float(PropsSI("Cpmolar", "P", p, "T", T, self._name))

    def cv_molar(self, p: float, T: float) -> float:
        """Isochoric molar heat capacity, J/(mol·K)."""
        return float(PropsSI("Cvmolar", "P", p, "T", T, self._name))

    # ------------------------------------------------------------------
    # Saturation line
    # ------------------------------------------------------------------

    def p_saturation(self, T: float) -> float:
        """Saturation pressure at temperature T, Pa."""
        return float(PropsSI("P", "T", T, "Q", 0.0, self._name))

    def T_saturation(self, p: float) -> float:
        """Saturation temperature at pressure p, K."""
        return float(PropsSI("T", "P", p, "Q", 0.0, self._name))

    def rho_molar_saturated_liquid(self, T: float) -> float:
        """Saturated liquid molar density :math:`\\rho'`, mol/m³."""
        return float(PropsSI("Dmolar", "T", T, "Q", 0.0, self._name))

    def rho_molar_saturated_vapor(self, T: float) -> float:
        """Saturated vapour molar density :math:`\\rho''`, mol/m³."""
        return float(PropsSI("Dmolar", "T", T, "Q", 1.0, self._name))

    def u_molar_saturated_liquid(self, T: float) -> float:
        """Saturated liquid molar internal energy, J/mol."""
        return float(PropsSI("Umolar", "T", T, "Q", 0.0, self._name))

    def u_molar_saturated_vapor(self, T: float) -> float:
        """Saturated vapour molar internal energy, J/mol."""
        return float(PropsSI("Umolar", "T", T, "Q", 1.0, self._name))

    def h_molar_saturated_liquid(self, T: float) -> float:
        """Saturated liquid molar enthalpy, J/mol."""
        return float(PropsSI("Hmolar", "T", T, "Q", 0.0, self._name))

    def h_molar_saturated_vapor(self, T: float) -> float:
        """Saturated vapour molar enthalpy, J/mol."""
        return float(PropsSI("Hmolar", "T", T, "Q", 1.0, self._name))

    def h_vaporization(self, T: float) -> float:
        """Molar enthalpy of vaporization at T, :math:`h'' - h'`, J/mol."""
        return self.h_molar_saturated_vapor(T) - self.h_molar_saturated_liquid(T)

    # ------------------------------------------------------------------
    # Two-phase
    # ------------------------------------------------------------------

    def from_TQ(
        self, T: float, Q: float
    ) -> tuple[float, float, float, float]:
        """Return ``(p, rho_molar, u_molar, h_molar)`` at vapour quality ``Q``
        in ``[0, 1]``. Valid only inside the vapour dome (:math:`T_t \\leq T < T_c`)."""
        if not 0.0 <= Q <= 1.0:
            raise ValueError(f"Vapour quality must be in [0, 1], got Q={Q}")
        p = float(PropsSI("P", "T", T, "Q", Q, self._name))
        rho = float(PropsSI("Dmolar", "T", T, "Q", Q, self._name))
        u = float(PropsSI("Umolar", "T", T, "Q", Q, self._name))
        h = float(PropsSI("Hmolar", "T", T, "Q", Q, self._name))
        return p, rho, u, h

    # ------------------------------------------------------------------
    # Pseudo-saturation pressure (Amankwah) for supercritical adsorption
    # ------------------------------------------------------------------

    def amankwah_pseudo_saturation(self, T: float, k: float = 2.0) -> float:
        """Amankwah (1995) supercritical pseudo-saturation pressure.

        .. math::

            p_0(T) \\;=\\; p_c \\left( \\frac{T}{T_c} \\right)^{k}

        Valid for :math:`T \\geq T_c`; for subcritical temperatures the
        actual saturation pressure should be used instead. The exponent
        ``k`` is material-specific (Amankwah originally proposed ``k = 2``).
        """
        return self.p_critical * (T / self.T_critical) ** k

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"FluidProperties({self._name!r})"
