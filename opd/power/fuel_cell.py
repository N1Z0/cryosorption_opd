"""Hydrogen fuel-cell model for the Active Subcooling scenario.

A proton-exchange-membrane (PEM) fuel cell consumes gaseous H₂ withdrawn
from the tank ullage and delivers electrical power to the cryocooler
during a scheduled *subcooling burst* ahead of a refuelling event.

Energy bookkeeping
------------------
The fuel cell converts the chemical energy of hydrogen (lower heating
value, LHV) into electricity with efficiency :math:`\\eta_{\\mathrm{fc}}`:

.. math::

    \\dot{n}_{\\mathrm{fc}}
        = \\frac{P_{\\mathrm{el}}}{\\eta_{\\mathrm{fc}}\\,
          \\Delta h_{\\mathrm{LHV}}}
    \\qquad
    \\Delta h_{\\mathrm{LHV}} \\approx 241.8\\,\\mathrm{kJ\\,mol^{-1}}

The waste heat :math:`(1-\\eta_{\\mathrm{fc}})\\,P_{\\mathrm{el}} /
\\eta_{\\mathrm{fc}}` is rejected by the spacecraft radiator and does
**not** enter the tank energy balance; only the *enthalpy of the
withdrawn gas* leaves the tank (see :meth:`~opd.tank.tank.Tank.ode_rhs`).

Activation profile
------------------
The output power follows a step window ``[t_start, t_end]`` with a short
cosine ramp of width ``ramp_s`` at both edges so the stiff BDF solver
never sees a true discontinuity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import LHV_H2_MOLAR, M_H2

__all__ = ["FuelCell"]


@dataclass(frozen=True)
class FuelCell:
    """PEM fuel cell drawing H₂ from the tank on a fixed schedule.

    Parameters
    ----------
    P_electrical
        DC electrical output while active, W.
    t_start, t_end
        Activation window, s.  Must satisfy ``t_end > t_start``.
    eta_fc
        Electrical efficiency relative to the H₂ lower heating value.
        Space-rated PEM stacks: 0.45–0.60.  Default 0.55.
    ramp_s
        Cosine ramp width at the window edges, s.  Default 120 s.
    specific_power_W_kg
        Stack + balance-of-plant specific power, W/kg, used for the
        ESM mass estimate (not by the ODE).  Default 150 W/kg.
    """

    P_electrical: float
    t_start: float
    t_end: float
    eta_fc: float = 0.55
    ramp_s: float = 120.0
    specific_power_W_kg: float = 150.0

    def __post_init__(self) -> None:
        if self.P_electrical <= 0.0:
            raise ValueError(
                f"P_electrical must be positive, got {self.P_electrical}"
            )
        if self.t_end <= self.t_start:
            raise ValueError("t_end must be > t_start")
        if not 0.0 < self.eta_fc <= 1.0:
            raise ValueError(f"eta_fc must be in (0, 1], got {self.eta_fc}")
        if self.ramp_s < 0.0:
            raise ValueError("ramp_s must be non-negative")

    # ------------------------------------------------------------------
    # Activation profile
    # ------------------------------------------------------------------

    def activation(self, t: float) -> float:
        """Smooth activation factor in ``[0, 1]`` at time ``t``, s."""
        if t < self.t_start or t > self.t_end:
            return 0.0
        r = self.ramp_s
        if r > 0.0:
            if t < self.t_start + r:
                x = (t - self.t_start) / r
                return 0.5 * (1.0 - math.cos(math.pi * x))
            if t > self.t_end - r:
                x = (self.t_end - t) / r
                return 0.5 * (1.0 - math.cos(math.pi * x))
        return 1.0

    # ------------------------------------------------------------------
    # Instantaneous rates
    # ------------------------------------------------------------------

    def P_out(self, t: float) -> float:
        """Electrical output power at time ``t``, W."""
        return self.P_electrical * self.activation(t)

    def n_dot_H2(self, t: float) -> float:
        """Hydrogen consumption rate at time ``t``, mol/s."""
        return self.P_out(t) / (self.eta_fc * LHV_H2_MOLAR)

    # ------------------------------------------------------------------
    # Integral quantities (mission bookkeeping)
    # ------------------------------------------------------------------

    @property
    def duration_s(self) -> float:
        """Length of the activation window, s."""
        return self.t_end - self.t_start

    @property
    def m_H2_per_burst_kg(self) -> float:
        """Total H₂ consumed per activation window, kg.

        Accounts for the two cosine ramps (each contributes half of a
        full-power ramp interval).
        """
        t_eff = self.duration_s - self.ramp_s  # two ramps × r/2 each
        t_eff = max(t_eff, 0.0)
        n_dot_full = self.P_electrical / (self.eta_fc * LHV_H2_MOLAR)
        return n_dot_full * t_eff * M_H2

    @property
    def mass_kg(self) -> float:
        """Stack + balance-of-plant mass estimate, kg."""
        return self.P_electrical / self.specific_power_W_kg
