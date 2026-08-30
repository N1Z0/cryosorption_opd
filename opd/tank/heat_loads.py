"""Heat-leak models: external thermal power delivered to the tank contents.

All models return the net heat flux :math:`\\dot{Q}` into the fluid in Watts.
Positive means heat flows *into* the tank.

M3 implements two concrete models:

* :class:`ConstantHeatFlux` — fixed wattage regardless of fluid temperature.
  Matches the notebook's step-by-step ``dQ`` approach.
* :class:`UAEnvironmentCoupling` — Newton-cooling law connecting the fluid
  to an ambient reservoir at a time-varying (or constant) temperature.
  Required for the tank-wall node (M5).

Future models (M5):
    ``MLIHeatLeak(emissivity, n_layers, …)``
    ``MissionProfileHeatLeak(t → Q̇)``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

__all__ = ["HeatLeakModel", "ConstantHeatFlux", "UAEnvironmentCoupling"]


class HeatLeakModel(ABC):
    """Abstract heat-leak source."""

    @abstractmethod
    def Q_dot(self, t: float, T_fluid: float) -> float:
        """Net thermal power into the fluid, W.

        Parameters
        ----------
        t
            Simulation time, s.
        T_fluid
            Current fluid temperature, K.  Some models (e.g. Newton-cooling)
            depend on this; others (constant flux) do not.
        """


@dataclass(frozen=True)
class ConstantHeatFlux(HeatLeakModel):
    """Constant heat input, independent of temperature.

    Parameters
    ----------
    Q_leak
        Heat power, W.  Must be non-negative (heat flows into the tank).
    """

    Q_leak: float

    def __post_init__(self) -> None:
        if self.Q_leak < 0.0:
            raise ValueError(
                f"ConstantHeatFlux requires Q_leak ≥ 0, got {self.Q_leak}"
            )

    def Q_dot(self, t: float, T_fluid: float) -> float:
        return self.Q_leak


@dataclass(frozen=True)
class UAEnvironmentCoupling(HeatLeakModel):
    """Newton-cooling: :math:`\\dot{Q} = UA \\cdot (T_{\\mathrm{env}}(t) - T_{\\mathrm{fl}})`.

    Parameters
    ----------
    UA
        Overall heat-transfer coefficient times area, W K⁻¹.
    T_env
        Environment temperature as a callable ``T_env(t) → K``, or a
        plain float for a constant environment.
    """

    UA: float
    T_env: "float | Callable[[float], float]"

    def __post_init__(self) -> None:
        if self.UA < 0.0:
            raise ValueError(f"UA must be non-negative, got {self.UA}")

    def Q_dot(self, t: float, T_fluid: float) -> float:
        T_outside = (
            self.T_env(t) if callable(self.T_env) else float(self.T_env)
        )
        return self.UA * (T_outside - T_fluid)
