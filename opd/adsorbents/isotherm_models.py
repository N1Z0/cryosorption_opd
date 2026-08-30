"""Equilibrium isotherm models.

Core interface is :class:`IsothermModel` with two methods,
:meth:`n_absolute` (thermodynamic absolute adsorbed amount) and
:meth:`n_excess` (Gibbs surface excess), both in mol per kg of sorbent.

The Dubinin-Astakhov (D-A) framework writes the absolute adsorbed amount as

.. math::

    n_{\\text{abs}}(p, T) \\;=\\; n_{\\max}\\,
        \\exp\\!\\left[-\\left(\\frac{A(p, T)}{E(T)}\\right)^{\\!m}\\right],
    \\qquad
    A(p, T) \\;=\\; R\\,T\\,\\ln\\!\\left(\\frac{p_0(T)}{p}\\right)

with adsorption potential :math:`A`, characteristic energy :math:`E(T)`, and
pseudo-saturation pressure :math:`p_0(T)`. For :math:`m = 2` this reduces to
the Dubinin-Radushkevich (DR) form, which is what the notebook uses.

Both :math:`E` and :math:`p_0` are passed in as callables so that a single
:class:`DubininAstakhov` class handles sub- and super-critical regimes; a
ready-made :class:`HybridDA` delegates to two branch models at a configurable
switch temperature.

Excess and absolute are related by

.. math::

    n_{\\text{exc}}(p, T) \\;=\\; n_{\\text{abs}}(p, T)
        \\;-\\; V_a \\,\\rho_{\\text{bulk}}(p, T)

where :math:`V_a` (``micropore_volume``) is the specific adsorbed-phase
volume (m³/kg) and :math:`\\rho_{\\text{bulk}}` is the bulk-fluid molar
density at the same :math:`(p, T)`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from ..constants import R_UNIVERSAL, TORR_TO_PA
from ..fluids.fluid_properties import FluidProperties

__all__ = [
    "IsothermModel",
    "DubininAstakhov",
    "HybridDA",
    "linear_energy",
    "constant_pressure_Pa",
    "constant_pressure_torr",
    "exponential_pressure",
    "exponential_pressure_torr",
    "saturation_pressure",
    "amankwah_pressure",
]

EnergyOfT = Callable[[float], float]
"""``E(T) -> J/mol``."""

PressureOfT = Callable[[float, FluidProperties], float]
"""``p0(T, fluid) -> Pa``."""


class IsothermModel(ABC):
    """Abstract equilibrium isotherm.

    All concrete subclasses return molar adsorption *per kg of sorbent*.
    """

    @abstractmethod
    def n_absolute(self, p: float, T: float, fluid: FluidProperties) -> float:
        """Absolute adsorbed amount at (p, T), mol/kg."""

    @abstractmethod
    def n_excess(self, p: float, T: float, fluid: FluidProperties) -> float:
        """Gibbs surface excess at (p, T), mol/kg."""


@dataclass(frozen=True)
class DubininAstakhov(IsothermModel):
    """Dubinin-Astakhov equilibrium isotherm.

    Parameters
    ----------
    n_max
        Maximum absolute adsorption, mol/kg.
    micropore_volume
        Specific adsorbed-phase volume :math:`V_a`, m³/kg, used for the
        excess/absolute conversion.
    characteristic_energy
        ``E(T) -> J/mol``. The canonical D-A form takes :math:`E > 0`;
        however the expression :math:`(A/E)^m` depends only on :math:`|E|`
        for even ``m`` (hence the empirical fits in the notebook, where
        :math:`E(T)` can be negative, still yield valid results).
    pseudo_saturation_pressure
        ``p0(T, fluid) -> Pa``. For :math:`T < T_c` this is typically the
        real saturation pressure; for :math:`T \\geq T_c` an Amankwah-type
        extrapolation is used.
    exponent
        D-A exponent :math:`m`. Defaults to 2 (Dubinin-Radushkevich).
    """

    n_max: float
    micropore_volume: float
    characteristic_energy: EnergyOfT
    pseudo_saturation_pressure: PressureOfT
    exponent: float = 2.0

    def __post_init__(self) -> None:
        if self.n_max <= 0.0:
            raise ValueError("n_max must be positive")
        if self.micropore_volume < 0.0:
            raise ValueError("micropore_volume must be non-negative")
        if self.exponent <= 0.0:
            raise ValueError("D-A exponent must be positive")

    def n_absolute(self, p: float, T: float, fluid: FluidProperties) -> float:
        if p <= 0.0:
            return 0.0
        E = self.characteristic_energy(T)
        if E == 0.0:
            raise ValueError(
                f"Characteristic energy vanishes at T={T} K; "
                "isotherm is undefined here."
            )
        p0 = self.pseudo_saturation_pressure(T, fluid)
        if p0 <= 0.0:
            raise ValueError(
                f"Non-positive pseudo-saturation pressure p0={p0} at T={T} K"
            )
        A = R_UNIVERSAL * T * math.log(p0 / p)
        return self.n_max * math.exp(-((A / E) ** self.exponent))

    def n_excess(self, p: float, T: float, fluid: FluidProperties) -> float:
        n_abs = self.n_absolute(p, T, fluid)
        if self.micropore_volume == 0.0 or p <= 0.0:
            return n_abs
        return n_abs - self.micropore_volume * fluid.rho_molar(p, T)


@dataclass(frozen=True)
class HybridDA(IsothermModel):
    """Piecewise isotherm switching between a sub- and super-critical branch.

    Parameters
    ----------
    subcritical, supercritical
        Branch models. Either may be any :class:`IsothermModel`; usually
        they are two :class:`DubininAstakhov` instances with different
        ``characteristic_energy`` / ``pseudo_saturation_pressure`` closures.
    T_switch
        Branch boundary. ``T < T_switch`` uses ``subcritical``, otherwise
        ``supercritical``. The small discontinuity at ``T_switch`` is a
        known property of this empirical construction; fitted branches
        should be chosen so it is negligible at the switching pressures
        of interest.
    """

    subcritical: IsothermModel
    supercritical: IsothermModel
    T_switch: float

    def _branch(self, T: float) -> IsothermModel:
        return self.subcritical if T < self.T_switch else self.supercritical

    def n_absolute(self, p: float, T: float, fluid: FluidProperties) -> float:
        return self._branch(T).n_absolute(p, T, fluid)

    def n_excess(self, p: float, T: float, fluid: FluidProperties) -> float:
        return self._branch(T).n_excess(p, T, fluid)


# ---------------------------------------------------------------------------
# Factory helpers for the two D-A callables
# ---------------------------------------------------------------------------


def linear_energy(a: float, b: float) -> EnergyOfT:
    """``E(T) = a + b T``, J/mol."""
    return lambda T: a + b * T


def constant_pressure_Pa(p0: float) -> PressureOfT:
    """Constant ``p0`` in Pa."""
    return lambda T, fluid: p0


def constant_pressure_torr(p0_torr: float) -> PressureOfT:
    """Constant ``p0`` supplied in Torr (converted internally)."""
    value = p0_torr * TORR_TO_PA
    return lambda T, fluid: value


def exponential_pressure(p0_Pa_at_T_zero: float, k: float) -> PressureOfT:
    """``p0(T) = p0_0 exp(k T)``, Pa."""
    return lambda T, fluid: p0_Pa_at_T_zero * math.exp(k * T)


def exponential_pressure_torr(
    p0_torr_at_T_zero: float, k: float
) -> PressureOfT:
    """``p0(T) = p0_0 exp(k T)``, with ``p0_0`` supplied in Torr."""
    base = p0_torr_at_T_zero * TORR_TO_PA
    return lambda T, fluid: base * math.exp(k * T)


def saturation_pressure() -> PressureOfT:
    """``p0(T) = fluid.p_saturation(T)`` — the real saturation curve.

    Valid only below the critical point; above :math:`T_c` use
    :func:`amankwah_pressure` or a custom closure."""
    return lambda T, fluid: fluid.p_saturation(T)


def amankwah_pressure(k: float = 2.0) -> PressureOfT:
    """``p0(T) = p_c (T / T_c)^k`` (Amankwah 1995)."""
    return lambda T, fluid: fluid.amankwah_pseudo_saturation(T, k=k)
