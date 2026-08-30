"""Cryocooler model classes.

All models expose a single method:

    ``Q_cryo(t, T_cold) -> float``   [W]

which returns the heat *extracted from the cold side* at time ``t`` and
cold-side temperature ``T_cold``.  A positive value means the cryocooler
is removing heat (cooling the tank).  The model does **not** apply any
duty-cycle modulation — that is the responsibility of the controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = ["CryocoolerModel", "ConstantCryocooler", "CarnotCryocooler"]


class CryocoolerModel(ABC):
    """Abstract base class for cryocooler heat-extraction models."""

    @abstractmethod
    def Q_cryo(self, t: float, T_cold: float) -> float:
        """Heat extracted from the cold side, W.

        Parameters
        ----------
        t
            Current simulation time, s.
        T_cold
            Cold-side temperature (tank fluid or sorbent), K.

        Returns
        -------
        float
            Positive value = heat removed from cold side, W.
        """

    @property
    @abstractmethod
    def P_input_max(self) -> float:
        """Maximum electrical input power, W."""


@dataclass(frozen=True)
class ConstantCryocooler(CryocoolerModel):
    """Cryocooler that always extracts a fixed power from the cold side.

    This is the simplest model, suitable for parametric studies.  It does
    not saturate at a temperature limit, so it can in principle cool the
    tank below the cold-side setpoint if the controller allows it.

    Parameters
    ----------
    Q_max
        Heat extraction capacity, W.  Must be > 0.
    """

    Q_max: float

    def __post_init__(self) -> None:
        if self.Q_max <= 0.0:
            raise ValueError(f"Q_max must be positive, got {self.Q_max}")

    def Q_cryo(self, t: float, T_cold: float) -> float:
        return self.Q_max

    @property
    def P_input_max(self) -> float:
        """Electrical input power (equal to Q_max for this ideal model)."""
        return self.Q_max


@dataclass(frozen=True)
class CarnotCryocooler(CryocoolerModel):
    """Carnot-limited cryocooler with an engineering efficiency de-rating.

    The heat extracted from the cold side is:

    .. math::

        \\dot{Q}_{\\mathrm{cryo}} = P_{\\mathrm{in}} \\cdot \\eta \\cdot
        \\mathrm{COP}_{\\mathrm{Carnot}}(T_{\\mathrm{cold}}, T_{\\mathrm{hot}})

    where the Carnot COP is

    .. math::

        \\mathrm{COP}_{\\mathrm{Carnot}} =
            \\frac{T_{\\mathrm{cold}}}{T_{\\mathrm{hot}} - T_{\\mathrm{cold}}}

    and :math:`\\eta \\in (0, 1]` is an engineering fraction that accounts
    for irreversibilities in a real machine.

    Typical values for space Brayton cryocoolers: :math:`\\eta \\approx 0.05`
    to :math:`0.15`.

    Parameters
    ----------
    P_input
        Electrical input power, W.  Must be > 0.
    T_hot
        Hot-side (radiator) temperature, K.  Must be > T_cold.
    eta_fraction
        Engineering efficiency relative to Carnot.  ``1.0`` = ideal Carnot.
    """

    P_input: float
    T_hot: float
    eta_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.P_input <= 0.0:
            raise ValueError(f"P_input must be positive, got {self.P_input}")
        if self.T_hot <= 0.0:
            raise ValueError(f"T_hot must be positive, got {self.T_hot}")
        if not 0.0 < self.eta_fraction <= 1.0:
            raise ValueError(
                f"eta_fraction must be in (0, 1], got {self.eta_fraction}"
            )

    def COP(self, T_cold: float) -> float:
        """Actual (de-rated Carnot) coefficient of performance.

        Parameters
        ----------
        T_cold
            Cold-side temperature, K.

        Returns
        -------
        float
            COP = Q_cold / P_electrical.  Returns 0 if T_cold >= T_hot.
        """
        if T_cold >= self.T_hot:
            return 0.0
        cop_carnot = T_cold / (self.T_hot - T_cold)
        return self.eta_fraction * cop_carnot

    def Q_cryo(self, t: float, T_cold: float) -> float:
        return self.P_input * self.COP(T_cold)

    @property
    def P_input_max(self) -> float:
        return self.P_input
