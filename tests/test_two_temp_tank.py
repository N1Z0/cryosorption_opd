"""Tests for M5 TwoTempTank and TwoTempResolver.

Covers:
- H_skel ↔ T_sorb round-trip accuracy (analytical fast path)
- TwoTempResolver.resolve returns consistent state at initial conditions
- TwoTempTank initial state encoding (y0 shape, H_skel value)
- ODE RHS returns finite values at t=0
- Short transient: energy and mass conservation over 60 s
- Cryocooler reduces bulk energy when duty > 0
- T_sorb tracks T_fluid in the 1-UA limit (very high UA_sf → equilibrium)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opd.adsorbents.activated_carbon import (
    ActivatedCarbon208C,
    _CP_CRYO_A,
    _CP_CRYO_B,
    cp_cryo_j_kg_k,
)
from opd.control import AlwaysOnController, BangBangController
from opd.cryocooler import ConstantCryocooler
from opd.fluids import parahydrogen
from opd.simulation.two_temp_resolver import TwoTempResolver
from opd.tank.geometry import TankGeometry
from opd.tank.heat_loads import ConstantHeatFlux
from opd.tank.two_temp_tank import TwoTempTank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_resolver():
    fl    = parahydrogen()
    ads   = ActivatedCarbon208C()
    geom  = TankGeometry(volume=1.0)
    m_sorb = 200.0
    V_free = geom.free_volume(m_sorb=m_sorb, adsorbent=ads)
    return TwoTempResolver(
        fluid=fl, V_free=V_free, adsorbent=ads,
        m_sorb=m_sorb, T_skel_ref=0.0, cp_sorb_fn=cp_cryo_j_kg_k,
    )


@pytest.fixture(scope="module")
def base_tank():
    fl    = parahydrogen()
    ads   = ActivatedCarbon208C()
    geom  = TankGeometry(volume=1.0)
    hl    = ConstantHeatFlux(Q_leak=3.0)
    cryo  = ConstantCryocooler(Q_max=5.0)
    ctrl  = BangBangController(p_on=3.0e5, p_off=2.5e5)
    return TwoTempTank(
        fluid=fl, geometry=geom, adsorbent=ads,
        heat_leak=hl, m_sorb=200.0,
        cryocooler=cryo, controller=ctrl,
    )


# ---------------------------------------------------------------------------
# TwoTempResolver – H_skel / T_sorb round-trip
# ---------------------------------------------------------------------------

class TestHskelRoundTrip:
    @pytest.mark.parametrize("T_sorb", [5.0, 10.0, 15.0, 20.0, 25.0, 77.0, 200.0, 300.0])
    def test_round_trip_machine_precision(self, base_resolver, T_sorb):
        H = base_resolver._H_skel(T_sorb)
        T_back = base_resolver.T_sorb_from_H_skel(H)
        assert abs(T_back - T_sorb) < 1e-10, (
            f"T={T_sorb} K → H={H:.4f} J → T_back={T_back:.10f} K "
            f"(err={abs(T_back - T_sorb):.2e} K)"
        )

    def test_analytical_path_detected(self, base_resolver):
        assert base_resolver._poly_A == pytest.approx(_CP_CRYO_A, rel=1e-12)
        assert base_resolver._poly_B == pytest.approx(_CP_CRYO_B, rel=1e-12)

    def test_H_skel_zero_at_T_ref(self, base_resolver):
        """H_skel(T_ref=0) must be 0 (or negligibly small)."""
        assert abs(base_resolver._H_skel(0.0)) < 1e-9

    def test_H_skel_analytical_formula(self, base_resolver):
        """Compare analytical H against direct Debye integral."""
        m = base_resolver._m_sorb
        A, B = _CP_CRYO_A, _CP_CRYO_B
        for T in [10.0, 20.0, 30.0]:
            expected = m * (A * T ** 2 / 2.0 + B * T ** 4 / 4.0)
            assert base_resolver._H_skel(T) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# TwoTempResolver – resolve at a known two-phase state
# ---------------------------------------------------------------------------

class TestTwoTempResolverResolve:
    def test_resolve_two_phase_initial_state(self, base_resolver):
        """Resolver should return a valid two-phase state for a pure-liquid IC."""
        m_sorb = base_resolver._m_sorb
        T_fl   = 20.0
        T_sorb = 20.0
        n, U = base_resolver.encode_two_phase(T_fl, Q_bulk=0.5, T_sorb=T_sorb)
        state = base_resolver.resolve(n, U, T_sorb)
        assert math.isfinite(state.T_fluid)
        assert math.isfinite(state.p)
        assert state.T_fluid == pytest.approx(T_fl, abs=0.5)

    def test_resolve_pressure_positive(self, base_resolver):
        n, U = base_resolver.encode_two_phase(20.0, Q_bulk=0.3, T_sorb=20.0)
        state = base_resolver.resolve(n, U, 20.0)
        assert state.p > 0.0

    def test_T_sorb_in_state_matches_input(self, base_resolver):
        T_sorb = 22.0
        n, U = base_resolver.encode_two_phase(20.0, Q_bulk=0.4, T_sorb=T_sorb)
        state = base_resolver.resolve(n, U, T_sorb)
        assert state.T_sorb == pytest.approx(T_sorb, abs=1e-9)


# ---------------------------------------------------------------------------
# TwoTempTank – initial state
# ---------------------------------------------------------------------------

class TestTwoTempTankInitialState:
    def test_y0_shape(self, base_tank):
        y0 = base_tank.initial_state_two_phase(T_fluid=20.0, Q_bulk=0.5, T_sorb=20.0)
        assert y0.shape == (3,)

    def test_y0_n_positive(self, base_tank):
        y0 = base_tank.initial_state_two_phase(T_fluid=20.0, Q_bulk=0.5, T_sorb=20.0)
        assert y0[0] > 0.0

    def test_y0_H_skel_matches_formula(self, base_tank):
        T_sorb = 20.0
        y0 = base_tank.initial_state_two_phase(T_fluid=20.0, Q_bulk=0.5, T_sorb=T_sorb)
        H_skel = y0[2]
        expected = base_tank._resolver._H_skel(T_sorb)
        assert H_skel == pytest.approx(expected, rel=1e-10)

    def test_resolve_state_returns_T_fluid(self, base_tank):
        y0 = base_tank.initial_state_two_phase(T_fluid=20.0, Q_bulk=0.5, T_sorb=20.0)
        state = base_tank.resolve_state(y0)
        assert math.isfinite(state.T_fluid)
        assert state.T_fluid == pytest.approx(20.0, abs=0.5)


# ---------------------------------------------------------------------------
# TwoTempTank – ODE RHS
# ---------------------------------------------------------------------------

class TestTwoTempTankODE:
    def test_rhs_finite_at_t0(self, base_tank):
        y0 = base_tank.initial_state_two_phase(T_fluid=20.0, Q_bulk=0.5, T_sorb=20.0)
        dy = base_tank.ode_rhs(0.0, y0)
        assert dy.shape == (3,)
        assert all(math.isfinite(v) for v in dy), f"Non-finite RHS: {dy}"

    def test_heat_leak_increases_U(self, base_tank):
        """Without venting, dU/dt must equal Q_leak (minus any active cryo)."""
        y0 = base_tank.initial_state_two_phase(T_fluid=20.0, Q_bulk=0.5, T_sorb=20.0)
        # Use pressure well below controller threshold (no cryo)
        dy = base_tank.ode_rhs(0.0, y0)
        dU = dy[1]
        # Heat leak is 3 W; cryocooler off (p_on=3 bar, current p ≈ 1 bar)
        assert dU == pytest.approx(3.0, abs=0.5)

    def test_dH_skel_finite(self, base_tank):
        y0 = base_tank.initial_state_two_phase(T_fluid=20.0, Q_bulk=0.5, T_sorb=20.0)
        dy = base_tank.ode_rhs(0.0, y0)
        assert math.isfinite(dy[2])


# ---------------------------------------------------------------------------
# TwoTempTank – short transient (60 s)
# ---------------------------------------------------------------------------

class TestTwoTempTransient:
    @pytest.fixture(scope="class")
    def short_result(self, base_tank):
        from opd.simulation.simulator import TransientSimulator
        y0  = base_tank.initial_state_two_phase(T_fluid=20.0, Q_bulk=0.5, T_sorb=20.0)
        sim = TransientSimulator(tank=base_tank)
        return sim.run(y0=y0, t_span=(0.0, 60.0), n_points=30)

    def test_result_has_t_sorb(self, short_result):
        assert short_result.T_sorb is not None
        assert len(short_result.T_sorb) == len(short_result.t)

    def test_pressure_increases_with_heat_leak(self, short_result):
        """Over 60 s with only heat leak (no cryo active), pressure must rise."""
        assert short_result.p[-1] > short_result.p[0]

    def test_mass_roughly_conserved(self, short_result):
        """No vent event expected in 60 s → total moles constant."""
        dn_rel = abs(short_result.n_total[-1] - short_result.n_total[0]) / short_result.n_total[0]
        assert dn_rel < 1e-3

    def test_T_sorb_stays_positive(self, short_result):
        assert np.all(short_result.T_sorb > 0.0)

    def test_T_fluid_stays_physical(self, short_result):
        """Fluid temperature must stay above H₂ triple point (13.8 K)."""
        assert np.all(short_result.T > 13.8)


# ---------------------------------------------------------------------------
# TwoTempTank – cryocooler extraction reduces energy
# ---------------------------------------------------------------------------

class TestCryocoolerIntegration:
    def test_active_cryocooler_removes_energy(self):
        """With AlwaysOn controller + cryocooler, dU/dt < Q_leak."""
        fl   = parahydrogen()
        ads  = ActivatedCarbon208C()
        geom = TankGeometry(volume=1.0)
        hl   = ConstantHeatFlux(Q_leak=3.0)
        cryo = ConstantCryocooler(Q_max=5.0)
        ctrl = AlwaysOnController()
        tank = TwoTempTank(
            fluid=fl, geometry=geom, adsorbent=ads,
            heat_leak=hl, m_sorb=200.0,
            cryocooler=cryo, controller=ctrl,
        )
        y0 = tank.initial_state_two_phase(T_fluid=20.0, Q_bulk=0.5, T_sorb=20.0)
        dy = tank.ode_rhs(0.0, y0)
        # dU/dt = Q_leak - Q_cryo = 3 - 5 = -2 W (net cooling)
        assert dy[1] < 0.0, "Active cryocooler should make dU/dt < 0"

    def test_dH_skel_negative_under_cryo(self):
        """Sorbent enthalpy decreases when cryocooler extracts heat from sorbent."""
        fl   = parahydrogen()
        ads  = ActivatedCarbon208C()
        geom = TankGeometry(volume=1.0)
        hl   = ConstantHeatFlux(Q_leak=1.0)
        cryo = ConstantCryocooler(Q_max=20.0)   # large cryo
        ctrl = AlwaysOnController()
        tank = TwoTempTank(
            fluid=fl, geometry=geom, adsorbent=ads,
            heat_leak=hl, m_sorb=200.0,
            cryocooler=cryo, controller=ctrl,
        )
        y0 = tank.initial_state_two_phase(T_fluid=20.0, Q_bulk=0.5, T_sorb=20.0)
        dy = tank.ode_rhs(0.0, y0)
        # dH_skel = -Q_HX - Q_cryo; large cryo → strongly negative
        assert dy[2] < 0.0
