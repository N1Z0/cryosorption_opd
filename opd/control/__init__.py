"""Pressure control strategies for Smart Ullage Control.

A controller maps the current tank pressure to a cryocooler *duty factor*
:math:`d \\in [0, 1]`.  The actual heat extracted is then

    :math:`\\dot{Q}_{\\mathrm{act}} = d \\cdot \\dot{Q}_{\\mathrm{cryo}}(t, T_{\\mathrm{cold}})`

Available controllers
---------------------
:class:`PressureController`
    Abstract base class.
:class:`AlwaysOnController`
    Duty = 1 always (no control logic).
:class:`BangBangController`
    Hysteresis two-state switch with ``p_on`` and ``p_off`` thresholds.
:class:`ProportionalController`
    Smooth ramp from 0 to 1 between ``p_lo`` and ``p_hi``.
"""

from __future__ import annotations

from .pressure_control import (
    AlwaysOnController,
    BangBangController,
    PressureController,
    ProportionalController,
)

__all__ = [
    "PressureController",
    "AlwaysOnController",
    "BangBangController",
    "ProportionalController",
]
