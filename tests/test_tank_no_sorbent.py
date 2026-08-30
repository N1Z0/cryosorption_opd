"""Regression tests for the bare-tank transient simulation against the
reference quasi-static ``pressure_build_up`` function (cell 14 of
``isotherm reference``).

The notebook advances the tank state by adding a fixed increment ``dQ``
to the molar internal energy while keeping the total moles constant
(Phase 1: isochoric pressurisation).  The ODE integration traces the
*same* thermodynamic path because Phase 1 is a simple isochoric process:

    dn/dt = 0       (closed tank, no venting)
    dU/dt = Q̇       (constant heat leak)

Mapping between formulations:
    notebook step ``dQ``  ↔  ODE integrand ``Q̇·dt`` over one step
    notebook ``Q`` axis   ↔  ODE ``Q_accumulated = Q̇·t``

Both traces the same isobaric path in (n, U) space, hence the same
``p(Q_cumulative)`` and ``T(Q_cumulative)`` trajectories (to integration
accuracy).

We use ``"Hydrogen"`` (normal hydrogen, CoolProp identifier) to match the
notebook.
"""

from __future__ import annotations

import numpy as np
import pytest

from CoolProp.CoolProp import PropsSI

from opd.fluids.hydrogen import normal_hydrogen
from opd.simulation import SimulationResult, TransientSimulator
from opd.simulation.thermal_state import ThermalStateResolver
from opd.tank import ConstantHeatFlux, Tank, TankGeometry

# ---------------------------------------------------------------------------
# Notebook constants
# ---------------------------------------------------------------------------
FLUID_NB = "Hydrogen"
T0_NB = 20.28
Q0_NB = 0.004562952718855845
V_NB = 1.0
DQ_NB = 1000.0      # J per quasi-static step
P_MAX_NB = 13e5     # Pa   (Phase-1 ends here)


# ---------------------------------------------------------------------------
# Reference: re-implement quasi-static loop exactly
# ---------------------------------------------------------------------------

def notebook_phase1_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Q_arr, p_arr, T_arr)`` from the reference quasi-static loop.

    Reproduces ``pressure_build_up(Q0, T0, V, p_max, *, dQ)`` Phase 1
    from ``isotherm reference`` cell 14 to machine precision.
    """
    D_mol = PropsSI("DMOLAR", "T", T0_NB, "Q", Q0_NB, FLUID_NB)
    u = PropsSI("UMOLAR", "T", T0_NB, "DMOLAR", D_mol, FLUID_NB)
    p_ = PropsSI("P", "T", T0_NB, "DMOLAR", D_mol, FLUID_NB)
    du = DQ_NB / (D_mol * V_NB)  # constant: n fixed

    Q_list = [0.0]
    p_list = [p_]
    T_list = [PropsSI("T", "P", p_, "DMOLAR", D_mol, FLUID_NB)]

    while p_ < P_MAX_NB:
        u += du
        p_ = PropsSI("P", "UMOLAR", u, "DMOLAR", D_mol, FLUID_NB)
        p_list.append(p_)
        T_list.append(PropsSI("T", "P", p_, "DMOLAR", D_mol, FLUID_NB))
        Q_list.append(Q_list[-1] + DQ_NB)

    return np.array(Q_list), np.array(p_list), np.array(T_list)


# ---------------------------------------------------------------------------
# ODE simulation helper
# ---------------------------------------------------------------------------

def run_ode_phase1(Q_dot: float = 1000.0) -> SimulationResult:
    """Run the ODE bare-tank simulation until p reaches P_MAX_NB."""
    fluid = normal_hydrogen()
    geom = TankGeometry(volume=V_NB)
    tank = Tank(
        fluid=fluid,
        geometry=geom,
        heat_leak=ConstantHeatFlux(Q_dot),
    )
    y0 = tank.initial_state_two_phase(T0_NB, Q0_NB)

    # t_end large enough that the event fires before we reach it
    t_max = 1e8 / Q_dot   # Q_total = Q_dot * t_max = 1e8 J >> 6.6 MJ needed

    sim = TransientSimulator(tank, solver_kwargs={"rtol": 1e-9, "atol": 1e-7})
    return sim.run(
        y0=y0,
        t_span=(0.0, t_max),
        n_points=2000,
        events=[tank.event_pressure_target(P_MAX_NB)],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nb_ref():
    return notebook_phase1_reference()


@pytest.fixture(scope="module")
def ode_result():
    return run_ode_phase1(Q_dot=1000.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitialConditions:
    """ODE initial state must reproduce the reference initial (n, p, T)."""

    def test_initial_n_matches_notebook(self):
        fluid = normal_hydrogen()
        resolver = ThermalStateResolver(fluid, V_free=V_NB)
        n, _ = resolver.encode_two_phase(T0_NB, Q0_NB)
        nb_n = PropsSI("DMOLAR", "T", T0_NB, "Q", Q0_NB, FLUID_NB) * V_NB
        assert n == pytest.approx(nb_n, rel=1e-8)

    def test_initial_p_matches_notebook(self, ode_result):
        nb_p0 = PropsSI("P", "T", T0_NB, "Q", Q0_NB, FLUID_NB)
        assert ode_result.p[0] == pytest.approx(nb_p0, rel=1e-4)

    def test_initial_T_matches_notebook(self, ode_result):
        assert ode_result.T[0] == pytest.approx(T0_NB, rel=1e-5)


class TestPhase1Trajectory:
    """ODE p(Q) and T(Q) must match the notebook quasi-static reference.

    We interpolate the ODE result onto the reference Q grid and compare
    at every 5th notebook step (avoiding interpolation noise at the ends).
    Tolerance: 0.3 % on pressure, 0.1 % on temperature.
    """

    def test_pressure_trajectory_matches(self, nb_ref, ode_result):
        Q_nb, p_nb, _ = nb_ref
        Q_ode = ode_result.Q_accumulated
        p_ode = ode_result.p

        # Interpolate the DENSE notebook reference onto the SPARSE ODE Q
        # grid (not the other way round) to avoid curvature-driven errors.
        Q_max = min(Q_nb[-1], Q_ode[-1])
        mask = (Q_ode >= Q_ode[1]) & (Q_ode <= Q_max * 0.995)
        p_nb_at_ode = np.interp(Q_ode[mask], Q_nb, p_nb)

        for p_ref, p_sim in zip(p_nb_at_ode, p_ode[mask]):
            assert p_sim == pytest.approx(p_ref, rel=3e-3)

    def test_temperature_trajectory_matches(self, nb_ref, ode_result):
        Q_nb, _, T_nb = nb_ref
        Q_ode = ode_result.Q_accumulated
        T_ode = ode_result.T

        Q_max = min(Q_nb[-1], Q_ode[-1])
        mask = (Q_ode >= Q_ode[1]) & (Q_ode <= Q_max * 0.995)
        T_nb_at_ode = np.interp(Q_ode[mask], Q_nb, T_nb)

        for T_ref, T_sim in zip(T_nb_at_ode, T_ode[mask]):
            assert T_sim == pytest.approx(T_ref, rel=1e-3)

    def test_terminal_pressure_near_target(self, ode_result):
        """Simulation must stop at or very close to p_max."""
        p_final = ode_result.p[-1]
        assert p_final == pytest.approx(P_MAX_NB, rel=1e-3)

    def test_terminal_T_matches_notebook(self, nb_ref, ode_result):
        """Final temperature must match reference final Phase-1 T."""
        T_nb_final = nb_ref[2][-1]
        T_ode_final = ode_result.T[-1]
        assert T_ode_final == pytest.approx(T_nb_final, rel=1e-3)


class TestPhase1Conservation:
    """During Phase 1 (no venting) n_total must be strictly constant."""

    def test_mass_exactly_conserved(self, ode_result):
        n = ode_result.n_total
        assert ode_result.conservation.max_mass_error_rel < 1e-8

    def test_energy_error_below_solver_tolerance(self, ode_result):
        assert ode_result.conservation.max_energy_error_rel < 1e-5


class TestSimulationResultMetadata:
    """Basic sanity checks on the SimulationResult object."""

    def test_arrays_all_same_length(self, ode_result):
        N = len(ode_result.t)
        for arr in [
            ode_result.n_total, ode_result.U_total, ode_result.T,
            ode_result.p, ode_result.Q_vapor, ode_result.n_bulk,
            ode_result.n_ads_abs, ode_result.Q_accumulated,
        ]:
            assert len(arr) == N

    def test_Q_accumulated_monotone(self, ode_result):
        assert np.all(np.diff(ode_result.Q_accumulated) >= 0.0)

    def test_pressure_monotone_in_phase1(self, ode_result):
        """Isochoric heating → pressure should be non-decreasing."""
        assert np.all(np.diff(ode_result.p) >= -1.0)  # -1 Pa tolerance

    def test_no_ads_in_bare_tank(self, ode_result):
        assert np.all(ode_result.n_ads_abs == 0.0)

    def test_fill_fraction_starts_at_one(self, ode_result):
        assert ode_result.fill_fraction[0] == pytest.approx(1.0, rel=1e-12)
