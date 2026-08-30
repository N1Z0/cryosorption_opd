"""Tests for :mod:`opd.simulation.thermal_state`.

Each test class corresponds to one physical scenario:

* :class:`TestBareBareTankTwoPhase`         — bare LH2 tank, vapour dome
* :class:`TestBareBareTankSinglePhase`      — bare H2 tank, single-phase
* :class:`TestACTankTwoPhase`               — AC-208C sorbent, two-phase
* :class:`TestACTankSinglePhase`            — AC-208C sorbent, single-phase
* :class:`TestConservation`                 — mass / energy round-trips
* :class:`TestPhaseDetection`               — correct phase labels
* :class:`TestConstructionValidation`       — constructor guard-rails
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest

from opd.adsorbents.activated_carbon import ActivatedCarbon208C
from opd.fluids.hydrogen import normal_hydrogen, parahydrogen
from opd.simulation.thermal_state import (
    PHASE_GAS,
    PHASE_SUPERCRITICAL,
    PHASE_TWO_PHASE,
    ThermalStateResolver,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

V_TANK = 1.0  # m³ — one cubic metre as in the notebook
M_SORB_AC = 100.0  # kg activated carbon (realistic lab-scale mass)


def bare_resolver(fluid=None):
    """ThermalStateResolver for a bare 1 m³ tank of parahydrogen."""
    if fluid is None:
        fluid = parahydrogen()
    return ThermalStateResolver(fluid=fluid, V_free=V_TANK)


def ac_resolver(fluid=None):
    """ThermalStateResolver for a 1 m³ tank with 100 kg of AC-208C.

    V_free is reduced by the skeleton volume first:
        rho_skel = 480 kg/m³  →  V_skel = 100/480 ≈ 0.208 m³
        V_free = 1.0 - 0.208 = 0.792 m³
    """
    if fluid is None:
        fluid = normal_hydrogen()  # match notebook fluid for AC regression
    ac = ActivatedCarbon208C()
    V_skel = M_SORB_AC / ac.skeletal_density
    V_free = V_TANK - V_skel
    return ThermalStateResolver(
        fluid=fluid, V_free=V_free, adsorbent=ac, m_sorb=M_SORB_AC
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pf():
    return parahydrogen()


@pytest.fixture(scope="module")
def nf():
    return normal_hydrogen()


@pytest.fixture(scope="module")
def bare(pf):
    return bare_resolver(pf)


@pytest.fixture(scope="module")
def bare_nf(nf):
    return bare_resolver(nf)


@pytest.fixture(scope="module")
def ac(nf):
    return ac_resolver(nf)


# ---------------------------------------------------------------------------
# 1. Bare tank — two-phase
# ---------------------------------------------------------------------------


class TestBareTankTwoPhase:
    """Round-trip (T, Q) → encode_two_phase → resolve → (T, Q)."""

    @pytest.mark.parametrize(
        "T, Q",
        [
            (20.0, 0.0),   # saturated liquid
            (20.0, 0.5),   # equal liquid/vapour volume fill
            (20.0, 1.0),   # saturated vapour
            (25.0, 0.3),
            (30.0, 0.8),
        ],
    )
    def test_round_trip_T(self, bare, T, Q):
        n, U = bare.encode_two_phase(T, Q)
        state = bare.resolve(n, U)
        assert state.T == pytest.approx(T, rel=1e-6)

    @pytest.mark.parametrize(
        "T, Q",
        [(20.0, 0.5), (25.0, 0.3), (30.0, 0.8)],
    )
    def test_round_trip_Q(self, bare, T, Q):
        n, U = bare.encode_two_phase(T, Q)
        state = bare.resolve(n, U)
        assert state.Q_vapor == pytest.approx(Q, abs=1e-6)

    def test_phase_label_two_phase(self, bare):
        n, U = bare.encode_two_phase(22.0, 0.4)
        state = bare.resolve(n, U)
        assert state.phase == PHASE_TWO_PHASE

    def test_Q_vapor_in_range(self, bare):
        for Q in (0.0, 0.25, 0.5, 0.75, 1.0):
            n, U = bare.encode_two_phase(20.0, Q)
            state = bare.resolve(n, U)
            assert 0.0 <= state.Q_vapor <= 1.0


# ---------------------------------------------------------------------------
# 2. Bare tank — single-phase (gas / supercritical)
# ---------------------------------------------------------------------------


class TestBareTankSinglePhase:
    """Round-trip (T, p) → encode_single_phase → resolve → (T, p)."""

    @pytest.mark.parametrize(
        "T, p",
        [
            (50.0, 2.0e5),   # low-pressure gas
            (100.0, 1.0e6),  # moderate gas
            (200.0, 5.0e5),  # warm gas
            (40.0, 3.0e6),   # supercritical
        ],
    )
    def test_round_trip_T(self, bare, T, p):
        n, U = bare.encode_single_phase(T, p)
        state = bare.resolve(n, U, T_guess=T, p_guess=p)
        assert state.T == pytest.approx(T, rel=1e-5)

    @pytest.mark.parametrize(
        "T, p",
        [
            (50.0, 2.0e5),
            (100.0, 1.0e6),
            (200.0, 5.0e5),
            (40.0, 3.0e6),
        ],
    )
    def test_round_trip_p(self, bare, T, p):
        n, U = bare.encode_single_phase(T, p)
        state = bare.resolve(n, U, T_guess=T, p_guess=p)
        assert state.p == pytest.approx(p, rel=1e-4)

    def test_Q_vapor_is_nan_for_single_phase(self, bare):
        n, U = bare.encode_single_phase(100.0, 1e6)
        state = bare.resolve(n, U, T_guess=100.0, p_guess=1e6)
        assert math.isnan(state.Q_vapor)

    def test_phase_label_supercritical(self, bare, pf):
        """State clearly above both T_c and p_c → SUPERCRITICAL."""
        T, p = 40.0, 3.0e6  # T=40>32.938 K, p=3 MPa > p_c=1.286 MPa
        n, U = bare.encode_single_phase(T, p)
        state = bare.resolve(n, U, T_guess=T, p_guess=p)
        assert state.phase == PHASE_SUPERCRITICAL

    def test_phase_label_gas(self, bare, pf):
        """Superheated gas (T >> T_c, p << p_c) → GAS."""
        T, p = 200.0, 5.0e5
        n, U = bare.encode_single_phase(T, p)
        state = bare.resolve(n, U, T_guess=T, p_guess=p)
        assert state.phase == PHASE_GAS


# ---------------------------------------------------------------------------
# 3. AC tank — two-phase
# ---------------------------------------------------------------------------


class TestACTankTwoPhase:
    """Same round-trips as bare tank, now with 100 kg AC-208C."""

    @pytest.mark.parametrize(
        "T, Q",
        [(20.0, 0.4), (25.0, 0.6), (30.0, 0.2)],
    )
    def test_round_trip_T(self, ac, T, Q):
        n, U = ac.encode_two_phase(T, Q)
        state = ac.resolve(n, U)
        assert state.T == pytest.approx(T, rel=1e-5)

    @pytest.mark.parametrize(
        "T, Q",
        [(20.0, 0.4), (25.0, 0.6), (30.0, 0.2)],
    )
    def test_round_trip_Q(self, ac, T, Q):
        n, U = ac.encode_two_phase(T, Q)
        state = ac.resolve(n, U)
        assert state.Q_vapor == pytest.approx(Q, abs=1e-5)

    def test_adsorption_contributes_nonzero_n_ads(self, ac):
        """The AC sorbent adsorbs a non-trivial amount of H₂ at 20 K."""
        T, Q = 20.0, 0.3
        n, U = ac.encode_two_phase(T, Q)
        state = ac.resolve(n, U)
        assert state.n_ads_abs > 0.0

    def test_n_ads_abs_is_inventory_gain_over_bulk_alone(self, ac, nf):
        """n_ads_abs represents hydrogen stored beyond what bare bulk holds
        in the same V_gas.  It must be positive at 20 K, 1 atm."""
        T, Q = 20.0, 0.3
        n, U = ac.encode_two_phase(T, Q)
        state = ac.resolve(n, U)
        # Absolute adsorption at the resolved (p_sat, T) state
        p = nf.p_saturation(T)
        n_ads_sp = ActivatedCarbon208C().n_absolute(p, T, nf)
        assert n_ads_sp > 0.0
        assert state.n_ads_abs == pytest.approx(
            M_SORB_AC * n_ads_sp, rel=1e-5
        )


# ---------------------------------------------------------------------------
# 4. AC tank — single-phase
# ---------------------------------------------------------------------------


class TestACTankSinglePhase:
    @pytest.mark.parametrize(
        "T, p",
        [(50.0, 2e5), (100.0, 5e5)],
    )
    def test_round_trip_T(self, ac, T, p):
        n, U = ac.encode_single_phase(T, p)
        state = ac.resolve(n, U, T_guess=T, p_guess=p)
        assert state.T == pytest.approx(T, rel=1e-4)

    @pytest.mark.parametrize(
        "T, p",
        [(50.0, 2e5), (100.0, 5e5)],
    )
    def test_round_trip_p(self, ac, T, p):
        n, U = ac.encode_single_phase(T, p)
        state = ac.resolve(n, U, T_guess=T, p_guess=p)
        assert state.p == pytest.approx(p, rel=1e-3)


# ---------------------------------------------------------------------------
# 5. Conservation — mass and energy
# ---------------------------------------------------------------------------


class TestConservation:
    """After encode + resolve the conserved quantities must be reproduced
    to machine precision (the whole point of the EOS inversion)."""

    def _check_conservation(self, resolver, n_in, U_in, state):
        # mass
        n_total_out = state.n_bulk + state.n_ads_abs
        assert n_total_out == pytest.approx(n_in, rel=1e-6)

        # energy: U_model = n * u_bulk - n_ads * q_st + H_skel
        q_st = (
            resolver._q_st(state.p, state.T) if resolver._ads is not None else 0.0
        )
        H_skel = resolver._H_skel(state.T)

        if state.phase == PHASE_TWO_PHASE:
            u = state.u_molar_bulk
        else:
            u = state.u_molar_bulk
        U_model = n_in * u - state.n_ads_abs * q_st + H_skel
        assert U_model == pytest.approx(U_in, rel=1e-5)

    @pytest.mark.parametrize(
        "T, Q",
        [(20.0, 0.3), (25.0, 0.7), (30.0, 0.5)],
    )
    def test_bare_two_phase(self, bare, T, Q):
        n, U = bare.encode_two_phase(T, Q)
        state = bare.resolve(n, U)
        self._check_conservation(bare, n, U, state)

    @pytest.mark.parametrize(
        "T, p",
        [(50.0, 2e5), (100.0, 1e6), (200.0, 5e5)],
    )
    def test_bare_single_phase(self, bare, T, p):
        n, U = bare.encode_single_phase(T, p)
        state = bare.resolve(n, U, T_guess=T, p_guess=p)
        self._check_conservation(bare, n, U, state)

    @pytest.mark.parametrize(
        "T, Q",
        [(20.0, 0.4), (25.0, 0.6)],
    )
    def test_ac_two_phase(self, ac, T, Q):
        n, U = ac.encode_two_phase(T, Q)
        state = ac.resolve(n, U)
        self._check_conservation(ac, n, U, state)

    @pytest.mark.parametrize(
        "T, p",
        [(50.0, 2e5), (100.0, 5e5)],
    )
    def test_ac_single_phase(self, ac, T, p):
        n, U = ac.encode_single_phase(T, p)
        state = ac.resolve(n, U, T_guess=T, p_guess=p)
        self._check_conservation(ac, n, U, state)


# ---------------------------------------------------------------------------
# 6. Phase detection
# ---------------------------------------------------------------------------


class TestPhaseDetection:
    def test_two_phase_detected(self, bare):
        n, U = bare.encode_two_phase(22.0, 0.5)
        assert bare.resolve(n, U).phase == PHASE_TWO_PHASE

    def test_supercritical_detected(self, bare, pf):
        """T > T_c and p > p_c."""
        T, p = 35.0, 2.0e6  # above both T_c=32.94 K and p_c=1.286 MPa
        n, U = bare.encode_single_phase(T, p)
        state = bare.resolve(n, U, T_guess=T, p_guess=p)
        assert state.phase == PHASE_SUPERCRITICAL

    def test_gas_detected(self, bare):
        T, p = 200.0, 1.0e5  # warm dilute gas
        n, U = bare.encode_single_phase(T, p)
        state = bare.resolve(n, U, T_guess=T, p_guess=p)
        assert state.phase == PHASE_GAS

    def test_Q_vapor_nan_in_single_phase(self, bare):
        n, U = bare.encode_single_phase(100.0, 1e6)
        state = bare.resolve(n, U, T_guess=100.0, p_guess=1e6)
        assert math.isnan(state.Q_vapor)

    def test_Q_vapor_not_nan_in_two_phase(self, bare):
        n, U = bare.encode_two_phase(22.0, 0.5)
        state = bare.resolve(n, U)
        assert not math.isnan(state.Q_vapor)


# ---------------------------------------------------------------------------
# 7. V_gas property
# ---------------------------------------------------------------------------


class TestVGas:
    def test_bare_V_gas_equals_V_free(self, pf):
        res = ThermalStateResolver(fluid=pf, V_free=0.75)
        assert res.V_gas == pytest.approx(0.75, rel=1e-14)

    def test_ac_V_gas_reduced_by_Va(self, nf):
        ac = ActivatedCarbon208C()
        m = 50.0
        V_free = 0.8
        res = ThermalStateResolver(
            fluid=nf, V_free=V_free, adsorbent=ac, m_sorb=m
        )
        expected = V_free - m * ac.micropore_volume
        assert res.V_gas == pytest.approx(expected, rel=1e-14)


# ---------------------------------------------------------------------------
# 8. Constructor validation
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_rejects_nonpositive_V_free(self, pf):
        with pytest.raises(ValueError, match="V_free"):
            ThermalStateResolver(fluid=pf, V_free=0.0)

    def test_rejects_negative_m_sorb(self, pf):
        with pytest.raises(ValueError, match="m_sorb"):
            ThermalStateResolver(fluid=pf, V_free=1.0, m_sorb=-1.0)

    def test_rejects_m_sorb_without_adsorbent(self, pf):
        with pytest.raises(ValueError, match="adsorbent"):
            ThermalStateResolver(fluid=pf, V_free=1.0, m_sorb=10.0)

    def test_rejects_V_gas_too_small(self, nf):
        ac = ActivatedCarbon208C()
        # 1e7 kg of AC → Va * m_sorb >> V_free
        with pytest.raises(ValueError, match="V_gas"):
            ThermalStateResolver(
                fluid=nf, V_free=1.0, adsorbent=ac, m_sorb=1e7
            )

    def test_encode_two_phase_rejects_invalid_Q(self, bare):
        with pytest.raises(ValueError):
            bare.encode_two_phase(20.0, -0.1)
        with pytest.raises(ValueError):
            bare.encode_two_phase(20.0, 1.5)
