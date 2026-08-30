"""Abstract adsorbent-material container.

:class:`AdsorbentMaterial` is an immutable record carrying:

* an :class:`~opd.adsorbents.isotherm_models.IsothermModel`,
* the skeletal (bulk) density used for volume bookkeeping,
* the micropore volume :math:`V_a`,
* a **skeleton** heat capacity :math:`c_p(T_{\\text{sorb}})` expressed
  as a function of the sorbent's *own* temperature — this is the
  2-Temp-ready seam. In M1–M4 the 1-Temp approximation simply evaluates
  ``cp_skeleton(T_fluid)``; in M5 when the sorbent gains its own state
  :math:`T_{\\text{sorb}}`, nothing about this interface changes.
* an isosteric heat of adsorption, stored as ``q_st(p, T) -> J/mol``,
  **positive** by convention (heat released per mole adsorbed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..fluids.fluid_properties import FluidProperties
from .isotherm_models import IsothermModel

__all__ = ["AdsorbentMaterial", "constant_isosteric_heat"]

CpOfT = Callable[[float], float]
"""``c_p(T_sorb) -> J/(kg K)``."""

IsostericHeatFn = Callable[[float, float], float]
"""``q_st(p, T) -> J/mol``, positive = exothermic adsorption."""


@dataclass(frozen=True)
class AdsorbentMaterial:
    """Immutable description of a microporous adsorbent.

    Attributes
    ----------
    name
        Human-readable identifier for plots and logs.
    skeletal_density
        Bulk density of the consolidated sorbent bed, :math:`\\mathrm{kg\\,m^{-3}}`.
        Used by :class:`~opd.tank.tank.Tank` to convert sorbent mass to
        displaced volume.
    micropore_volume
        Specific adsorbed-phase volume :math:`V_a`, :math:`\\mathrm{m^3\\,kg^{-1}}`.
    isotherm
        Equilibrium isotherm.
    cp_skeleton
        ``c_p(T_sorb) -> J/(kg K)``. Takes the sorbent's own temperature,
        **not** the fluid's. The 1-Temp simulations of M1–M4 simply call
        ``cp_skeleton(T_fluid)``.
    isosteric_heat_fn
        Optional; ``q_st(p, T) -> J/mol``. Positive = exothermic. If
        absent, :meth:`isosteric_heat` raises.
    """

    name: str
    skeletal_density: float
    micropore_volume: float
    isotherm: IsothermModel
    cp_skeleton: CpOfT
    isosteric_heat_fn: Optional[IsostericHeatFn] = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("AdsorbentMaterial name must be a non-empty string")
        if self.skeletal_density <= 0.0:
            raise ValueError("skeletal_density must be positive")
        if self.micropore_volume < 0.0:
            raise ValueError("micropore_volume must be non-negative")
        if not callable(self.cp_skeleton):
            raise TypeError("cp_skeleton must be a callable T -> J/(kg K)")

    # ---- convenience delegation to the isotherm ---------------------------

    def n_absolute(
        self, p: float, T: float, fluid: FluidProperties
    ) -> float:
        """Absolute adsorption at (p, T), mol/kg."""
        return self.isotherm.n_absolute(p, T, fluid)

    def n_excess(
        self, p: float, T: float, fluid: FluidProperties
    ) -> float:
        """Gibbs excess at (p, T), mol/kg."""
        return self.isotherm.n_excess(p, T, fluid)

    # ---- heats ------------------------------------------------------------

    def isosteric_heat(self, p: float, T: float) -> float:
        """Isosteric heat of adsorption at (p, T), J/mol.

        Positive = exothermic (heat released on adsorption, absorbed on
        desorption). Raises if no ``isosteric_heat_fn`` was supplied at
        construction.
        """
        if self.isosteric_heat_fn is None:
            raise NotImplementedError(
                f"No isosteric_heat_fn supplied for adsorbent {self.name!r}"
            )
        return self.isosteric_heat_fn(p, T)


def constant_isosteric_heat(q_st_Jmol: float) -> IsostericHeatFn:
    """Build a constant ``q_st(p, T) = q_st_Jmol`` callable."""
    if q_st_Jmol < 0.0:
        raise ValueError(
            "isosteric heat must be non-negative with the 'positive = "
            "exothermic' convention"
        )
    return lambda p, T: q_st_Jmol
