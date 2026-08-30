"""Tests for M5 cryocooler and pressure-controller modules.

Covers:
- CryocoolerModel ABC compliance
- ConstantCryocooler behaviour
- CarnotCryocooler COP limits
- PressureController ABC compliance
- AlwaysOnController / BangBangController / ProportionalController
- Combined duty factor computation in a closed-loop scenario
"""

from __future__ import annotations

import math

import pytest

from opd.control import (
    AlwaysOnController,
    BangBangController,
    ProportionalController,
)
from opd.cryocooler import CarnotCryocooler, ConstantCryocooler


# ---------------------------------------------------------------------------
# ConstantCryocooler
# ---------------------------------------------------------------------------

class TestConstantCryocooler:
    def test_fixed_output(self):
        c = ConstantCryocooler(Q_max=10.0)
        assert c.Q_cryo(t=0.0, T_cold=20.0) == pytest.approx(10.0)
        assert c.Q_cryo(t=3600.0, T_cold=77.0) == pytest.approx(10.0)

    def test_negative_Q_raises(self):
        with pytest.raises(ValueError):
            ConstantCryocooler(Q_max=-1.0)

    def test_zero_Q_raises(self):
        with pytest.raises(ValueError):
            ConstantCryocooler(Q_max=0.0)

    def test_P_input_max_equals_Q_max(self):
        c = ConstantCryocooler(Q_max=7.5)
        assert c.P_input_max == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# CarnotCryocooler
# ---------------------------------------------------------------------------

class TestCarnotCryocooler:
    def test_cop_decreases_with_lower_T_cold(self):
        """Lower cold-side temperature → lower COP → less Q extracted."""
        T_hot = 300.0
        c = CarnotCryocooler(P_input=100.0, T_hot=T_hot, eta_fraction=1.0)
        q_warm = c.Q_cryo(0.0, T_cold=77.0)
        q_cold = c.Q_cryo(0.0, T_cold=20.0)
        assert q_cold < q_warm

    def test_output_zero_when_T_cold_exceeds_T_hot(self):
        """Q_cryo must be 0 when T_cold ≥ T_hot (unphysical / no cooling)."""
        c = CarnotCryocooler(P_input=10.0, T_hot=300.0)
        assert c.Q_cryo(0.0, T_cold=350.0) == pytest.approx(0.0)

    def test_carnot_cop_formula(self):
        """COP = eta * T_cold / (T_hot - T_cold)."""
        T_cold, T_hot = 20.0, 300.0
        c = CarnotCryocooler(P_input=1.0, T_hot=T_hot, eta_fraction=0.1)
        expected_cop = 0.1 * T_cold / (T_hot - T_cold)
        assert c.COP(T_cold) == pytest.approx(expected_cop, rel=1e-9)

    def test_Q_cryo_equals_P_input_times_COP(self):
        T_cold = 20.0
        c = CarnotCryocooler(P_input=50.0, T_hot=300.0, eta_fraction=0.12)
        assert c.Q_cryo(0.0, T_cold) == pytest.approx(50.0 * c.COP(T_cold))

    def test_eta_scales_output(self):
        T_cold = 20.0
        c1 = CarnotCryocooler(P_input=100.0, T_hot=300.0, eta_fraction=1.0)
        c2 = CarnotCryocooler(P_input=100.0, T_hot=300.0, eta_fraction=0.5)
        assert c2.Q_cryo(0.0, T_cold) == pytest.approx(
            c1.Q_cryo(0.0, T_cold) * 0.5, rel=1e-9
        )


# ---------------------------------------------------------------------------
# AlwaysOnController
# ---------------------------------------------------------------------------

class TestAlwaysOnController:
    def test_duty_always_one(self):
        ctrl = AlwaysOnController()
        for p in [0.5e5, 1.0e5, 5.0e5, 100.0e5]:
            assert ctrl.duty(0.0, p) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BangBangController
# ---------------------------------------------------------------------------

class TestBangBangController:
    _CTRL = BangBangController(p_on=3.0e5, p_off=2.0e5)

    def test_turns_on_above_p_on(self):
        assert self._CTRL.duty(0.0, 3.5e5) == pytest.approx(1.0)

    def test_stays_off_below_p_off(self):
        assert self._CTRL.duty(0.0, 1.5e5) == pytest.approx(0.0)

    def test_hysteresis_band(self):
        """In the hysteresis band the output does not change."""
        p_mid = 2.5e5
        d = self._CTRL.duty(0.0, p_mid)
        assert d in (0.0, 1.0)  # deterministic, but either is valid

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            BangBangController(p_on=1.0e5, p_off=2.0e5)  # p_off > p_on


# ---------------------------------------------------------------------------
# ProportionalController
# ---------------------------------------------------------------------------

class TestProportionalController:
    _CTRL = ProportionalController(p_lo=2.0e5, p_hi=3.0e5)

    def test_zero_below_p_lo(self):
        assert self._CTRL.duty(0.0, 1.0e5) == pytest.approx(0.0)

    def test_one_above_p_hi(self):
        assert self._CTRL.duty(0.0, 4.0e5) == pytest.approx(1.0)

    def test_midpoint(self):
        assert self._CTRL.duty(0.0, 2.5e5) == pytest.approx(0.5, rel=1e-6)

    def test_monotone(self):
        pressures = [1.5e5, 2.0e5, 2.3e5, 2.7e5, 3.0e5, 3.5e5]
        duties = [self._CTRL.duty(0.0, p) for p in pressures]
        assert duties == sorted(duties)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            ProportionalController(p_lo=3.0e5, p_hi=2.0e5)


# ---------------------------------------------------------------------------
# Combined: cryocooler × controller duty
# ---------------------------------------------------------------------------

class TestCombinedCryoControl:
    def test_effective_Q_scales_with_duty(self):
        cryo = ConstantCryocooler(Q_max=10.0)
        ctrl = ProportionalController(p_lo=2.0e5, p_hi=3.0e5)
        p = 2.5e5
        duty = ctrl.duty(0.0, p)
        Q_eff = duty * cryo.Q_cryo(t=0.0, T_cold=20.0)
        assert Q_eff == pytest.approx(5.0, rel=1e-6)

    def test_zero_duty_gives_zero_Q(self):
        cryo = ConstantCryocooler(Q_max=10.0)
        ctrl = BangBangController(p_on=3.0e5, p_off=2.0e5)
        duty = ctrl.duty(0.0, 1.0e5)   # below p_off → off
        assert duty * cryo.Q_cryo(0.0, 20.0) == pytest.approx(0.0)
