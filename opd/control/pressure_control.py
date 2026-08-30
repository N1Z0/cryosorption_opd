"""Pressure control strategies.

All controllers expose:

    ``duty(t, p) -> float``  in ``[0, 1]``

The duty factor multiplies the cryocooler's ``Q_cryo`` to give the
*actual* heat extraction rate at the current time and pressure.

Design note — discontinuities
------------------------------
:class:`BangBangController` has a discontinuous output.  When used with an
ODE solver, this can cause the solver to take very small steps near the
switching pressure.  The recommended practice is to add a ``solve_ivp``
event function that detects each crossing; :meth:`BangBangController.events`
returns a ready-to-use list of such events.

:class:`ProportionalController` has a continuously differentiable output and
is therefore preferred for stiff ODE solvers.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List

__all__ = [
    "PressureController",
    "AlwaysOnController",
    "BangBangController",
    "ProportionalController",
]


class PressureController(ABC):
    """Abstract base class for pressure controllers."""

    @abstractmethod
    def duty(self, t: float, p: float) -> float:
        """Return the cryocooler duty factor.

        Parameters
        ----------
        t
            Current simulation time, s.
        p
            Current tank pressure, Pa.

        Returns
        -------
        float
            Duty factor in ``[0, 1]``.
        """

    def events(self) -> List[Callable]:
        """Return a (possibly empty) list of ``solve_ivp`` event callables.

        Subclasses with discontinuous duty curves should override this to
        return switching-pressure detection events so that the ODE solver
        can resolve the discontinuity accurately.
        """
        return []


@dataclass(frozen=True)
class AlwaysOnController(PressureController):
    """Controller that always returns duty = 1 (cryocooler fully on)."""

    def duty(self, t: float, p: float) -> float:
        return 1.0


@dataclass
class BangBangController(PressureController):
    """Two-state hysteresis controller.

    Activates the cryocooler when ``p >= p_on`` and deactivates it when
    ``p <= p_off``.  The initial state can be set at construction.

    Parameters
    ----------
    p_on
        Pressure at which the cryocooler switches on, Pa.
    p_off
        Pressure at which the cryocooler switches off, Pa.
        Must satisfy ``p_off < p_on``.
    initial_state
        Starting duty: ``1.0`` (on) or ``0.0`` (off).
    """

    p_on: float
    p_off: float
    initial_state: float = 0.0
    _state: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.p_off >= self.p_on:
            raise ValueError(
                f"p_off ({self.p_off} Pa) must be < p_on ({self.p_on} Pa)"
            )
        self._state = self.initial_state

    def duty(self, t: float, p: float) -> float:
        if p >= self.p_on:
            self._state = 1.0
        elif p <= self.p_off:
            self._state = 0.0
        return self._state

    def events(self) -> List[Callable]:
        """Pressure-crossing events for ``solve_ivp``.

        Returns two non-terminal events (one for each switching pressure)
        so the solver places output points exactly at the transitions.
        The solver should be called with ``dense_output=True`` or with
        fine ``t_eval`` to capture the duty changes accurately.
        """
        p_on  = self.p_on
        p_off = self.p_off

        def _on(t: float, y) -> float:
            return float(y[1]) - p_on   # NOTE: caller must adapt for tank EOS

        def _off(t: float, y) -> float:
            return float(y[1]) - p_off

        _on.terminal  = False
        _off.terminal = False
        return [_on, _off]


@dataclass(frozen=True)
class ProportionalController(PressureController):
    """Smooth proportional controller.

    The duty ramps linearly from 0 at ``p_lo`` to 1 at ``p_hi``:

    .. math::

        d(p) = \\mathrm{clip}\\!\\left(
            \\frac{p - p_{\\mathrm{lo}}}{p_{\\mathrm{hi}} - p_{\\mathrm{lo}}},\\ 0,\\ 1
        \\right)

    This is continuously differentiable within ``(p_lo, p_hi)`` and
    therefore plays well with BDF solvers.

    Parameters
    ----------
    p_lo
        Pressure at which duty starts rising, Pa.
    p_hi
        Pressure at which duty reaches 1.0, Pa.
    """

    p_lo: float
    p_hi: float

    def __post_init__(self) -> None:
        if self.p_lo >= self.p_hi:
            raise ValueError(
                f"p_lo ({self.p_lo} Pa) must be < p_hi ({self.p_hi} Pa)"
            )

    def duty(self, t: float, p: float) -> float:
        span = self.p_hi - self.p_lo
        return max(0.0, min(1.0, (p - self.p_lo) / span))
