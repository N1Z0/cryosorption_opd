"""Simulation layer — ODE setup, solving, and post-processing.

M2:  ThermalStateResolver, ResolvedState
M3:  TransientSimulator, SimulationResult, ConservationReport
M5:  TwoTempResolver, TwoTempState
"""

from __future__ import annotations

from .results import ConservationReport, SimulationResult
from .simulator import TransientSimulator
from .thermal_state import (
    PHASE_GAS,
    PHASE_LIQUID,
    PHASE_SUPERCRITICAL,
    PHASE_TWO_PHASE,
    ResolvedState,
    ThermalStateResolver,
)
from .two_temp_resolver import TwoTempResolver, TwoTempState

__all__ = [
    "ThermalStateResolver",
    "ResolvedState",
    "PHASE_TWO_PHASE",
    "PHASE_LIQUID",
    "PHASE_GAS",
    "PHASE_SUPERCRITICAL",
    "TransientSimulator",
    "SimulationResult",
    "ConservationReport",
    "TwoTempResolver",
    "TwoTempState",
]
