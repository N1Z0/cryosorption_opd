"""Orbital and planetary environment heat-flux models.

These models implement the :class:`~opd.tank.heat_loads.HeatLeakModel`
interface so they can be dropped directly into any
:class:`~opd.tank.tank.Tank` or :class:`~opd.tank.two_temp_tank.TwoTempTank`
in place of :class:`~opd.tank.heat_loads.ConstantHeatFlux`.

Available models
----------------
:class:`LEOHeatFlux`
    Low-Earth-Orbit 90-minute Sun/eclipse cycling with MLI insulation.
:class:`LunarHeatFlux`
    14-day Lunar day / 14-day Lunar night thermal cycling.
:class:`GatewayHeatFlux`
    Lunar Gateway orbit: 7-day NRHO period with varying Earth/Sun geometry.
:class:`MLIHeatFlux`
    Thin-shell MLI model giving time-averaged specific heat flux W/m².
"""

from __future__ import annotations

from .orbital import GatewayHeatFlux, LEOHeatFlux, LunarHeatFlux, MLIHeatFlux

__all__ = [
    "LEOHeatFlux",
    "LunarHeatFlux",
    "GatewayHeatFlux",
    "MLIHeatFlux",
]
