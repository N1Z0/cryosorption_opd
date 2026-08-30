"""Simulation output containers.

:class:`SimulationResult` is built by :class:`~opd.simulation.simulator.TransientSimulator`
after a successful :func:`scipy.integrate.solve_ivp` run.  It stores the raw
ODE time-series and the resolved thermodynamic state at each stored time step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConservationReport", "SimulationResult"]


@dataclass
class ConservationReport:
    """Summary of first-law and mass-conservation quality.

    Attributes
    ----------
    max_mass_error_rel
        ``max |n(t) − n(0)| / n(0)`` for a closed tank (no venting).
        Should be < 1e-9 for a well-resolved integration.
    max_energy_error_rel
        ``max |U(t) − U(0) − ∫₀ᵗ Q̇ dt'| / |∫₀ᵗ Q̇ dt'|`` where the
        integral is computed from the stored ``Q_accumulated`` array.
        Should be < 1e-6.
    """

    max_mass_error_rel: float = 0.0
    max_energy_error_rel: float = 0.0

    def __str__(self) -> str:
        return (
            f"ConservationReport("
            f"mass_err={self.max_mass_error_rel:.2e}, "
            f"energy_err={self.max_energy_error_rel:.2e})"
        )


@dataclass
class SimulationResult:
    """Time-series output from a transient simulation.

    All arrays have the same length ``N`` (the number of stored time steps).

    Parameters
    ----------
    t
        Time array, s. Shape ``(N,)``.
    n_total
        Total H₂ moles, mol. Shape ``(N,)``.
    U_total
        Total internal energy, J. Shape ``(N,)``.
    T
        Fluid temperature, K. Shape ``(N,)``.
    p
        Fluid pressure, Pa. Shape ``(N,)``.
    Q_vapor
        Vapour quality; ``NaN`` for single-phase points. Shape ``(N,)``.
    phase
        Phase label at each time step. Shape ``(N,)`` of strings.
    n_bulk
        Bulk-fluid moles (outside micropores), mol. Shape ``(N,)``.
    n_ads_abs
        Absolute adsorbed moles, mol. Shape ``(N,)``.
    Q_accumulated
        Cumulative heat added to the tank, :math:`\\int_0^t \\dot{Q}\\,dt'`,
        J. Shape ``(N,)``. Computed from a trapezoidal integral of the
        stored ``Q_dot`` array.
    Q_dot
        Instantaneous heat-leak power at each stored step, W. Shape ``(N,)``.
    n_dot_vent
        Venting mass-flow rate, mol/s. Shape ``(N,)``. Zero when no vent.
    conservation
        Post-run conservation quality report.
    """

    t: np.ndarray
    n_total: np.ndarray
    U_total: np.ndarray
    T: np.ndarray
    p: np.ndarray
    Q_vapor: np.ndarray
    phase: np.ndarray
    n_bulk: np.ndarray
    n_ads_abs: np.ndarray
    Q_accumulated: np.ndarray
    Q_dot: np.ndarray
    n_dot_vent: np.ndarray
    conservation: ConservationReport = field(
        default_factory=ConservationReport
    )
    T_sorb: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    """Sorbent temperature array, K.  All ``NaN`` for 1-Temp runs."""
    Q_cryo: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    """Instantaneous cryocooler extraction power, W.  Zero when no cryocooler."""
    X_ortho: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    """Ortho-H₂ mole fraction.  All ``NaN`` when no para-ortho catalyst."""
    n_dot_fc: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    """Fuel-cell H₂ draw, mol/s.  Zero / empty when no fuel cell."""
    P_fc: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    """Fuel-cell electrical output, W.  Zero / empty when no fuel cell."""

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def fill_fraction(self) -> np.ndarray:
        """Current H₂ inventory relative to the initial inventory, dimensionless."""
        if self.n_total[0] == 0.0:
            return np.zeros_like(self.n_total)
        return self.n_total / self.n_total[0]

    @property
    def Q_accumulated_MJ(self) -> np.ndarray:
        """Cumulative heat, MJ."""
        return self.Q_accumulated / 1e6

    @property
    def p_bar(self) -> np.ndarray:
        """Pressure in bar."""
        return self.p / 1e5

    def __len__(self) -> int:
        return len(self.t)

    def __repr__(self) -> str:
        return (
            f"SimulationResult("
            f"N={len(self)}, "
            f"t=[{self.t[0]:.2g}, {self.t[-1]:.2g}] s, "
            f"p=[{self.p[0]:.3g}, {self.p[-1]:.3g}] Pa)"
        )
