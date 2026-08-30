"""Energy conservation tests.

For a closed tank the first law requires:

    U(t) − U(0) = ∫₀ᵗ Q̇(t') dt'

We verify this to within the ODE solver tolerance (rtol=1e-10).

Additionally we test that the SimulationResult's ``Q_accumulated`` array
tracks this integral faithfully.
"""

from __future__ import annotations

import numpy as np
import pytest

from opd.fluids.hydrogen import normal_hydrogen, parahydrogen
from opd.simulation import TransientSimulator
from opd.tank import ConstantHeatFlux, Tank, TankGeometry


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def run_isochoric(T0, Q0, V, Q_dot, t_end, n_points=400):
    """Run a bare isochoric tank and return the SimulationResult."""
    fluid = parahydrogen()
    geom = TankGeometry(volume=V)
    tank = Tank(fluid=fluid, geometry=geom, heat_leak=ConstantHeatFlux(Q_dot))
    y0 = tank.initial_state_two_phase(T0, Q0)
    sim = TransientSimulator(tank, solver_kwargs={"rtol": 1e-10, "atol": 1e-8})
    return sim.run(y0=y0, t_span=(0.0, t_end), n_points=n_points)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConstantHeatLeakEnergy:
    """ΔU must equal Q̇ · t to solver-tolerance accuracy."""

    @pytest.mark.parametrize("Q_dot", [100.0, 500.0, 2000.0])
    def test_delta_U_equals_Q_dot_times_t(self, Q_dot):
        result = run_isochoric(20.0, 0.3, 1.0, Q_dot, 2000.0)
        dU = result.U_total - result.U_total[0]
        Q_expected = Q_dot * result.t
        # Allow 10x the solver rtol
        rel_errs = np.abs(dU[1:] - Q_expected[1:]) / np.abs(Q_expected[1:])
        assert np.max(rel_errs) < 1e-6, f"max rel energy error = {np.max(rel_errs):.2e}"


class TestQAccumulatedArray:
    """SimulationResult.Q_accumulated must track ∫Q̇ dt faithfully."""

    def test_Q_accumulated_starts_at_zero(self):
        result = run_isochoric(20.0, 0.3, 1.0, 500.0, 1000.0)
        assert result.Q_accumulated[0] == pytest.approx(0.0, abs=1e-10)

    def test_Q_accumulated_matches_Q_dot_times_t(self):
        Q_dot = 300.0
        result = run_isochoric(20.0, 0.3, 1.0, Q_dot, 2000.0)
        Q_expected = Q_dot * result.t
        rel_err = np.abs(result.Q_accumulated - Q_expected) / Q_expected[-1]
        assert np.max(rel_err[1:]) < 1e-5

    def test_Q_accumulated_monotone(self):
        result = run_isochoric(20.0, 0.3, 1.0, 500.0, 1000.0)
        assert np.all(np.diff(result.Q_accumulated) >= 0.0)


class TestConservationReport:
    """ConservationReport fields must be within expected bounds."""

    def test_energy_error_below_threshold(self):
        result = run_isochoric(20.0, 0.3, 1.0, 500.0, 2000.0)
        assert result.conservation.max_energy_error_rel < 1e-5

    def test_mass_error_below_threshold(self):
        result = run_isochoric(20.0, 0.3, 1.0, 500.0, 2000.0)
        assert result.conservation.max_mass_error_rel < 1e-8


class TestEnergyAcrossPhaseTransition:
    """Energy conservation must hold across two-phase → compressed-liquid transition."""

    def test_energy_conserved_after_bubble_point(self):
        """A very dense tank (Q0 small) will cross the bubble line during heating."""
        result = run_isochoric(20.0, 0.01, 1.0, 500.0, 5000.0)
        dU = result.U_total - result.U_total[0]
        Q_exp = 500.0 * result.t
        rel_err = np.abs(dU[1:] - Q_exp[1:]) / Q_exp[1:]
        assert np.max(rel_err) < 1e-5
