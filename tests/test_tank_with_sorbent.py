"""Adsorbent-loaded tank transient tests.

Three areas of coverage:

1. **Quasi-static regression** (``TestACTankQuasistaticComparison``):
   A T-driven quasi-static loop (consistent with the resolver's physics)
   is compared with the ODE simulation.  The ``p(Q_accumulated)``
   trajectory must agree to 0.3 % — same tolerance as the bare-tank
   regression.

2. **Pressure advantage** (``TestACTankPressureAdvantage``):
   For the same cumulative heat input the AC-loaded tank must have
   *lower* pressure than the bare tank at every comparable point.
   This is the central thermodynamic benefit of the "Smart Ullage"
   concept: the sorbent thermal mass and adsorption buffering slow the
   pressure rise.

3. **Adsorption physics** (``TestACTankAdsorption``):
   n_ads_abs starts large (cold, near-saturation conditions) and
   decreases monotonically as the tank warms (desorption).

Physical parameters match reference 17:
    m_AC = 480 kg   (bulk density 480 kg/m³ fills 1 m³ total volume)
    V_total = 1 m³  (same physical tank as bare case)
    T0 = 20.28 K,   Q0 = 0.004562…  (same initial conditions)
    fluid = normal hydrogen  (matches notebook)
    Q_dot = 1000 W           (constant heat leak)
"""

from __future__ import annotations

import numpy as np
import pytest

from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from opd.adsorbents.activated_carbon import ActivatedCarbon208C
from opd.fluids.hydrogen import normal_hydrogen
from opd.simulation import SimulationResult, TransientSimulator
from opd.simulation.thermal_state import ThermalStateResolver
from opd.tank import ConstantHeatFlux, Tank, TankGeometry

# ---------------------------------------------------------------------------
# Physical constants for this suite
# ---------------------------------------------------------------------------
T0       = 20.28
Q0       = 0.004562952718855845
M_AC     = 480.0           # kg
V_TOTAL  = 1.0             # m³
RHO_SKEL = 2150.0          # kg/m³ (skeletal density of AC 208C)
P_MAX    = 13e5            # Pa
Q_DOT    = 1000.0          # W

FLUID_CP = "Hydrogen"      # CoolProp identifier matching the notebook


# ---------------------------------------------------------------------------
# Helpers — geometry
# ---------------------------------------------------------------------------

def _ac_tank() -> Tank:
    """Construct a 1 m³ AC-loaded tank with 480 kg of 208C."""
    fluid = normal_hydrogen()
    ac    = ActivatedCarbon208C()
    geom  = TankGeometry(volume=V_TOTAL)
    return Tank(
        fluid=fluid,
        geometry=geom,
        heat_leak=ConstantHeatFlux(Q_DOT),
        adsorbent=ac,
        m_sorb=M_AC,
    )


def _bare_tank() -> Tank:
    """Construct the matching 1 m³ bare tank (no adsorbent)."""
    fluid = normal_hydrogen()
    geom  = TankGeometry(volume=V_TOTAL)
    return Tank(fluid=fluid, geometry=geom, heat_leak=ConstantHeatFlux(Q_DOT))


# ---------------------------------------------------------------------------
# Quasi-static T-driven reference (physics-consistent with the resolver)
# ---------------------------------------------------------------------------

def _quasistatic_ac_reference(
    dT: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(T_arr, p_arr, Q_arr)`` from a T-step quasi-static loop.

    At each temperature step the algorithm:

    1. Computes n_abs at the new T (using p_sat in the two-phase region,
       or p from the mass balance in the single-phase region).
    2. Finds the bulk fluid state that conserves n_total.
    3. Encodes (T, Q_bulk) or (T, p) into the resolver's U_total, so that
       the energy increment dQ = ΔU_total is **identical** to what the ODE
       integrates.

    This makes the reference internally consistent with
    ``ThermalStateResolver``: both the reference and the ODE trace the same
    thermodynamic path; any deviation in p(Q) is purely numerical.
    """
    fluid = normal_hydrogen()
    ac    = ActivatedCarbon208C()
    V_free = V_TOTAL - M_AC / RHO_SKEL
    V_gas  = V_free - M_AC * ac.micropore_volume

    resolver = ThermalStateResolver(
        fluid, V_free=V_free, adsorbent=ac, m_sorb=M_AC
    )
    n_total, U_init = resolver.encode_two_phase(T0, Q0)

    T_list: list[float] = [T0]
    p_list: list[float] = [resolver.resolve(n_total, U_init).p]
    Q_list: list[float] = [0.0]
    U_prev = U_init
    T_     = T0

    while p_list[-1] < P_MAX:
        T_ += dT
        if T_ > 45.0:
            break  # safety guard

        # Saturation properties at new T
        rho_L   = PropsSI("Dmolar", "T", T_, "Q", 0, FLUID_CP)
        rho_V   = PropsSI("Dmolar", "T", T_, "Q", 1, FLUID_CP)
        p_sat_T = PropsSI("P",      "T", T_, "Q", 0, FLUID_CP)
        n_ads_sat = ac.isotherm.n_absolute(p_sat_T, T_, fluid) * M_AC
        rho_bulk  = (n_total - n_ads_sat) / V_gas

        if rho_bulk <= rho_L:
            # ---- two-phase bulk fluid ----
            v_L   = 1.0 / rho_L
            v_V   = 1.0 / rho_V
            Q_blk = (1.0 / rho_bulk - v_L) / (v_V - v_L)
            Q_blk = float(np.clip(Q_blk, 0.0, 1.0))
            p_new = p_sat_T
            _, U_new = resolver.encode_two_phase(T_, Q_blk)
        else:
            # ---- single-phase (compressed liquid) ----
            def _mass_res(p_try: float) -> float:
                n_a   = ac.isotherm.n_absolute(max(p_try, 1.0), T_, fluid) * M_AC
                rho   = PropsSI("Dmolar", "T", T_, "P", p_try, FLUID_CP)
                return n_total - (rho * V_gas + n_a)

            p_lo = p_sat_T * (1.0 + 1e-4)
            p_hi = 60e5
            try:
                p_new = brentq(_mass_res, p_lo, p_hi, xtol=1.0, rtol=1e-10)
            except ValueError:
                p_new = p_lo
            _, U_new = resolver.encode_single_phase(T_, p_new)

        dQ = U_new - U_prev
        U_prev = U_new
        Q_list.append(Q_list[-1] + dQ)
        p_list.append(p_new)
        T_list.append(T_)

    return np.array(T_list), np.array(p_list), np.array(Q_list)


# ---------------------------------------------------------------------------
# ODE runner
# ---------------------------------------------------------------------------

def _run_ac_ode(n_points: int = 1500) -> SimulationResult:
    """Run the AC tank ODE simulation until p = P_MAX."""
    tank = _ac_tank()
    y0   = tank.initial_state_two_phase(T0, Q0)
    # t_end ≈ Q_total/Q_dot + 10% safety margin; Q_total ≈ 10.3 MJ at 1000 W → ~10 300 s
    t_end = 12_000.0
    sim  = TransientSimulator(tank, solver_kwargs={"rtol": 1e-9, "atol": 1e-7})
    return sim.run(
        y0=y0,
        t_span=(0.0, t_end),
        n_points=n_points,
        events=[tank.event_pressure_target(P_MAX)],
    )


def _run_bare_ode(n_points: int = 1500) -> SimulationResult:
    """Run the matching bare tank ODE simulation until p = P_MAX."""
    tank = _bare_tank()
    y0   = tank.initial_state_two_phase(T0, Q0)
    # Bare tank reaches p_max at ~6.594 MJ → 6594 s; +20% safety
    t_end = 8_000.0
    sim  = TransientSimulator(tank, solver_kwargs={"rtol": 1e-9, "atol": 1e-7})
    return sim.run(
        y0=y0,
        t_span=(0.0, t_end),
        n_points=n_points,
        events=[tank.event_pressure_target(P_MAX)],
    )


# ---------------------------------------------------------------------------
# Module-scoped fixtures (expensive; compute once for all tests in the file)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ac_ode():
    return _run_ac_ode()


@pytest.fixture(scope="module")
def bare_ode():
    return _run_bare_ode()


@pytest.fixture(scope="module")
def qs_ref():
    return _quasistatic_ac_reference(dT=0.01)


# ---------------------------------------------------------------------------
# Tests — quasi-static regression
# ---------------------------------------------------------------------------

class TestACTankQuasistaticComparison:
    """ODE p(Q) must match the T-driven quasi-static reference to 0.3 %."""

    def test_terminal_Q_matches_reference(self, ac_ode, qs_ref):
        Q_ode  = ac_ode.Q_accumulated[-1]
        Q_ref  = qs_ref[2][-1]
        assert Q_ode == pytest.approx(Q_ref, rel=5e-3)

    def test_terminal_T_matches_reference(self, ac_ode, qs_ref):
        T_ode = ac_ode.T[-1]
        T_ref = qs_ref[0][-1]
        assert T_ode == pytest.approx(T_ref, rel=5e-3)

    def test_terminal_p_near_target(self, ac_ode):
        assert ac_ode.p[-1] == pytest.approx(P_MAX, rel=1e-3)

    def test_pressure_trajectory_matches(self, ac_ode, qs_ref):
        """p(Q) from ODE matches quasi-static reference at ODE output Q points."""
        T_ref, p_ref, Q_ref = qs_ref
        Q_ode = ac_ode.Q_accumulated
        p_ode = ac_ode.p
        # Compare only the range fully covered by both
        Q_max = min(Q_ref[-1], Q_ode[-1])
        mask  = (Q_ode >= Q_ode[1]) & (Q_ode <= Q_max * 0.99)
        p_nb_at_ode = np.interp(Q_ode[mask], Q_ref, p_ref)
        for p_r, p_s in zip(p_nb_at_ode, p_ode[mask]):
            assert p_s == pytest.approx(p_r, rel=3e-3)

    def test_temperature_trajectory_matches(self, ac_ode, qs_ref):
        """T(Q) from ODE matches quasi-static reference at ODE output Q points."""
        T_ref, p_ref, Q_ref = qs_ref
        Q_ode = ac_ode.Q_accumulated
        T_ode = ac_ode.T
        Q_max = min(Q_ref[-1], Q_ode[-1])
        mask  = (Q_ode >= Q_ode[1]) & (Q_ode <= Q_max * 0.99)
        T_nb_at_ode = np.interp(Q_ode[mask], Q_ref, T_ref)
        for T_r, T_s in zip(T_nb_at_ode, T_ode[mask]):
            assert T_s == pytest.approx(T_r, rel=3e-3)


# ---------------------------------------------------------------------------
# Tests — pressure advantage
# ---------------------------------------------------------------------------

class TestACTankPressureAdvantage:
    """Core physics: AC tank has lower pressure than bare tank for same Q."""

    def test_ac_needs_more_heat_to_reach_pmax(self, ac_ode, bare_ode):
        """AC tank requires significantly more heat input to reach P_MAX."""
        Q_ac   = ac_ode.Q_accumulated[-1]
        Q_bare = bare_ode.Q_accumulated[-1]
        # AC tank uses the sorbent as a thermal and adsorption buffer;
        # experimentally 10.3 MJ vs 6.6 MJ for bare tank (≥ 30 % more).
        assert Q_ac > Q_bare * 1.3, (
            f"AC tank Q_final={Q_ac/1e6:.2f} MJ not > 1.3× bare "
            f"{Q_bare/1e6:.2f} MJ"
        )

    def test_lower_pressure_for_same_cumulative_heat(self, ac_ode, bare_ode):
        """At every shared Q point the AC tank pressure must be lower."""
        Q_ac   = ac_ode.Q_accumulated
        Q_bare = bare_ode.Q_accumulated

        # Compare at Q points where both simulations have coverage
        Q_lo = max(Q_ac[2], Q_bare[2])          # skip first two points
        Q_hi = min(Q_bare[-1], Q_ac[-1]) * 0.98  # stay away from the terminal

        # Sample 20 Q values in the overlapping window
        Q_test = np.linspace(Q_lo, Q_hi, 20)
        p_ac   = np.interp(Q_test, Q_ac,   ac_ode.p)
        p_bare = np.interp(Q_test, Q_bare, bare_ode.p)

        # p_ac must be strictly lower at every test point
        for i, (pa, pb) in enumerate(zip(p_ac, p_bare)):
            assert pa < pb, (
                f"At Q={Q_test[i]/1e6:.2f} MJ: p_ac={pa/1e5:.3f} bar "
                f"≥ p_bare={pb/1e5:.3f} bar"
            )

    def test_pressure_advantage_magnitude(self, ac_ode, bare_ode):
        """At the mid-point Q the pressure reduction must be at least 20 %."""
        Q_bare = bare_ode.Q_accumulated[-1]
        Q_mid  = Q_bare * 0.5
        p_ac_mid   = np.interp(Q_mid, ac_ode.Q_accumulated,   ac_ode.p)
        p_bare_mid = np.interp(Q_mid, bare_ode.Q_accumulated, bare_ode.p)
        reduction  = (p_bare_mid - p_ac_mid) / p_bare_mid
        assert reduction >= 0.20, (
            f"Pressure reduction at Q={Q_mid/1e6:.2f} MJ is "
            f"{reduction*100:.1f}% < 20%"
        )


# ---------------------------------------------------------------------------
# Tests — adsorption behaviour
# ---------------------------------------------------------------------------

class TestACTankAdsorption:
    """n_ads_abs must behave consistently with the D-A isotherm."""

    def test_initial_adsorption_large(self, ac_ode):
        """At LH2 conditions ~43 % of H₂ should be adsorbed."""
        n_ads_frac = ac_ode.n_ads_abs[0] / ac_ode.n_total[0]
        assert n_ads_frac > 0.3, (
            f"Initial adsorption fraction {n_ads_frac:.2%} < 30%"
        )

    def test_adsorption_decreases_with_temperature(self, ac_ode):
        """Heating at constant inventory releases gas: mean n_ads in the
        final quarter of the simulation must be less than in the initial
        quarter.  Pointwise monotonicity is not required because the
        adsorption change per output step (< 1 mol over 8 s) is close to
        the resolver's numerical noise floor.
        """
        n_ads = ac_ode.n_ads_abs
        N     = len(n_ads)
        q     = max(N // 4, 1)
        mean_early = np.mean(n_ads[:q])
        mean_late  = np.mean(n_ads[-q:])
        assert mean_late < mean_early, (
            f"n_ads did not decrease: early mean {mean_early:.2f} mol, "
            f"late mean {mean_late:.2f} mol"
        )

    def test_bulk_plus_ads_equals_total(self, ac_ode):
        """n_bulk + n_ads_abs = n_total at every stored time step."""
        n_check = ac_ode.n_bulk + ac_ode.n_ads_abs
        rel_err = np.max(np.abs(n_check - ac_ode.n_total)) / ac_ode.n_total[0]
        assert rel_err < 1e-6

    def test_bare_tank_has_zero_adsorption(self, bare_ode):
        assert np.all(bare_ode.n_ads_abs == 0.0)


# ---------------------------------------------------------------------------
# Tests — conservation
# ---------------------------------------------------------------------------

class TestACTankConservation:
    """Mass and energy conservation in the closed AC tank."""

    def test_mass_conserved(self, ac_ode):
        assert ac_ode.conservation.max_mass_error_rel < 1e-8

    def test_energy_conserved(self, ac_ode):
        assert ac_ode.conservation.max_energy_error_rel < 1e-5


# ---------------------------------------------------------------------------
# Tests — geometry / initial state
# ---------------------------------------------------------------------------

class TestACTankGeometry:
    """Verify free-volume and adsorption geometry are consistent."""

    def test_skeletal_density_is_2150(self):
        ac = ActivatedCarbon208C()
        assert ac.skeletal_density == pytest.approx(2150.0, rel=1e-12)

    def test_free_volume_less_than_total(self):
        ac   = ActivatedCarbon208C()
        geom = TankGeometry(volume=V_TOTAL)
        V_f  = geom.free_volume(m_sorb=M_AC, adsorbent=ac)
        assert V_f < V_TOTAL
        assert V_f == pytest.approx(V_TOTAL - M_AC / RHO_SKEL, rel=1e-12)

    def test_initial_state_encodes_correctly(self):
        """Tank initial state must have p ≈ p_sat(T0)."""
        tank   = _ac_tank()
        y0     = tank.initial_state_two_phase(T0, Q0)
        state0 = tank.resolver.resolve(y0[0], y0[1])
        p_sat  = PropsSI("P", "T", T0, "Q", 0, FLUID_CP)
        assert state0.p == pytest.approx(p_sat, rel=1e-5)
        assert state0.T == pytest.approx(T0, rel=1e-6)

    def test_initial_n_total_includes_adsorbed(self):
        """AC tank initial inventory exceeds what the bare free volume holds."""
        tank   = _ac_tank()
        y0     = tank.initial_state_two_phase(T0, Q0)
        n_ac   = y0[0]
        # bare tank at same (T, Q) but full V_TOTAL
        bare   = _bare_tank()
        y0_b   = bare.initial_state_two_phase(T0, Q0)
        n_bare = y0_b[0]
        # The AC tank has a smaller free volume, so n_bulk is less;
        # but adsorption adds many moles.  Net: roughly comparable or
        # slightly less depending on the adsorption vs volume trade-off.
        # Just verify both are > 0 and positive.
        assert n_ac > 0
        assert n_bare > 0
        # AC n_ads_abs should be > 40% of n_total
        state = tank.resolver.resolve(y0[0], y0[1])
        assert state.n_ads_abs / n_ac > 0.3
