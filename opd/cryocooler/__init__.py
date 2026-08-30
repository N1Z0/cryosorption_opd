"""Cryocooler models for active heat extraction from the tank.

A cryocooler extracts heat :math:`\\dot{Q}_{\\mathrm{cryo}}` [W] from the cold
side (tank contents or sorbent bed) and rejects it to a warm radiator.

Available models
----------------
:class:`CryocoolerModel`
    Abstract base class.
:class:`ConstantCryocooler`
    Fixed extraction rate regardless of cold-side temperature.
:class:`CarnotCryocooler`
    Extraction rate limited by Carnot COP with an engineering de-rating
    fraction :math:`\\eta`.
"""

from __future__ import annotations

from .base import CarnotCryocooler, ConstantCryocooler, CryocoolerModel

__all__ = ["CryocoolerModel", "ConstantCryocooler", "CarnotCryocooler"]
