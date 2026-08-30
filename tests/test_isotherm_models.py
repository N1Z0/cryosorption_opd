"""Tests for :mod:`opd.adsorbents.isotherm_models`."""

from __future__ import annotations

import math

import pytest

from opd.adsorbents.isotherm_models import (
    DubininAstakhov,
    HybridDA,
    IsothermModel,
    amankwah_pressure,
    constant_pressure_Pa,
    constant_pressure_torr,
    exponential_pressure,
    exponential_pressure_torr,
    linear_energy,
    saturation_pressure,
)
from opd.constants import R_UNIVERSAL, TORR_TO_PA


# ---------------------------------------------------------------------------
# D-A limits and closed-form checks
# ---------------------------------------------------------------------------


class TestDubininAstakhovLimits:
    def _make(self, Va=0.0, n_max=10.0, E=5000.0, p0=1e8, m=2.0):
        return DubininAstakhov(
            n_max=n_max,
            micropore_volume=Va,
            characteristic_energy=lambda T: E,
            pseudo_saturation_pressure=lambda T, fluid: p0,
            exponent=m,
        )

    def test_zero_pressure_gives_zero_adsorption(self, h2_notebook):
        iso = self._make()
        assert iso.n_absolute(0.0, 77.0, h2_notebook) == 0.0

    def test_negative_pressure_gives_zero_adsorption(self, h2_notebook):
        iso = self._make()
        assert iso.n_absolute(-1e3, 77.0, h2_notebook) == 0.0

    def test_p_equal_p0_gives_n_max(self, h2_notebook):
        iso = self._make(Va=0.0)
        assert iso.n_absolute(1e8, 77.0, h2_notebook) == pytest.approx(
            10.0, rel=1e-12
        )

    def test_closed_form_at_reference_point(self, h2_notebook):
        n_max, E, p0 = 15.0, 5000.0, 1e8
        iso = self._make(n_max=n_max, E=E, p0=p0, m=2.0)
        T, p = 77.0, 1e5
        A = R_UNIVERSAL * T * math.log(p0 / p)
        expected = n_max * math.exp(-((A / E) ** 2))
        assert iso.n_absolute(p, T, h2_notebook) == pytest.approx(
            expected, rel=1e-14
        )

    def test_excess_equals_absolute_minus_Va_rho(self, h2_notebook):
        iso = self._make(Va=5e-4)
        T, p = 77.0, 1e5
        rho = h2_notebook.rho_molar(p, T)
        expected = iso.n_absolute(p, T, h2_notebook) - 5e-4 * rho
        assert iso.n_excess(p, T, h2_notebook) == pytest.approx(
            expected, rel=1e-14
        )

    def test_excess_equals_absolute_when_Va_zero(self, h2_notebook):
        iso = self._make(Va=0.0)
        T, p = 77.0, 1e5
        assert iso.n_excess(p, T, h2_notebook) == pytest.approx(
            iso.n_absolute(p, T, h2_notebook), rel=1e-14
        )

    def test_signed_energy_irrelevant_for_even_exponent(self, h2_notebook):
        """D-A with m=2 depends on E^2 only; flipping the sign must not
        change the answer."""
        plus = self._make(E=+4000.0).n_absolute(1e5, 77.0, h2_notebook)
        minus = self._make(E=-4000.0).n_absolute(1e5, 77.0, h2_notebook)
        assert plus == pytest.approx(minus, rel=1e-14)


class TestDubininAstakhovValidation:
    def test_rejects_nonpositive_n_max(self):
        with pytest.raises(ValueError):
            DubininAstakhov(
                n_max=0.0,
                micropore_volume=0.0,
                characteristic_energy=lambda T: 1.0,
                pseudo_saturation_pressure=lambda T, f: 1.0,
            )

    def test_rejects_negative_micropore_volume(self):
        with pytest.raises(ValueError):
            DubininAstakhov(
                n_max=1.0,
                micropore_volume=-1e-4,
                characteristic_energy=lambda T: 1.0,
                pseudo_saturation_pressure=lambda T, f: 1.0,
            )

    def test_rejects_nonpositive_exponent(self):
        with pytest.raises(ValueError):
            DubininAstakhov(
                n_max=1.0,
                micropore_volume=0.0,
                characteristic_energy=lambda T: 1.0,
                pseudo_saturation_pressure=lambda T, f: 1.0,
                exponent=0.0,
            )

    def test_zero_characteristic_energy_raises(self, h2_notebook):
        iso = DubininAstakhov(
            n_max=1.0,
            micropore_volume=0.0,
            characteristic_energy=lambda T: 0.0,
            pseudo_saturation_pressure=lambda T, f: 1e8,
        )
        with pytest.raises(ValueError, match="vanishes"):
            iso.n_absolute(1e5, 77.0, h2_notebook)

    def test_nonpositive_p0_raises(self, h2_notebook):
        iso = DubininAstakhov(
            n_max=1.0,
            micropore_volume=0.0,
            characteristic_energy=lambda T: 1.0,
            pseudo_saturation_pressure=lambda T, f: -1.0,
        )
        with pytest.raises(ValueError, match="Non-positive pseudo"):
            iso.n_absolute(1e5, 77.0, h2_notebook)


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------


class _DummyBranch(IsothermModel):
    """Deterministic branch used to test HybridDA routing."""

    def __init__(self, tag: float) -> None:
        self.tag = tag

    def n_absolute(self, p, T, fluid):
        return self.tag

    def n_excess(self, p, T, fluid):
        return self.tag


class TestHybridDA:
    def test_routes_sub_below_switch(self, h2_notebook):
        hybrid = HybridDA(_DummyBranch(42.0), _DummyBranch(0.7), T_switch=32.0)
        assert hybrid.n_absolute(1e5, 31.999, h2_notebook) == 42.0

    def test_routes_super_at_and_above_switch(self, h2_notebook):
        hybrid = HybridDA(_DummyBranch(42.0), _DummyBranch(0.7), T_switch=32.0)
        assert hybrid.n_absolute(1e5, 32.000, h2_notebook) == 0.7
        assert hybrid.n_absolute(1e5, 32.001, h2_notebook) == 0.7

    def test_excess_uses_same_branch_as_absolute(self, h2_notebook):
        hybrid = HybridDA(_DummyBranch(42.0), _DummyBranch(0.7), T_switch=32.0)
        assert hybrid.n_excess(1e5, 20.0, h2_notebook) == 42.0
        assert hybrid.n_excess(1e5, 50.0, h2_notebook) == 0.7


# ---------------------------------------------------------------------------
# Callable helpers
# ---------------------------------------------------------------------------


class TestCallableHelpers:
    def test_linear_energy(self):
        E = linear_energy(1000.0, 2.0)
        assert E(0.0) == 1000.0
        assert E(100.0) == 1200.0

    def test_constant_pressure_Pa(self, h2_notebook):
        g = constant_pressure_Pa(5e5)
        assert g(77.0, h2_notebook) == 5e5
        assert g(20.0, h2_notebook) == 5e5

    def test_constant_pressure_torr(self, h2_notebook):
        g = constant_pressure_torr(1000.0)
        assert g(77.0, h2_notebook) == pytest.approx(
            1000.0 * TORR_TO_PA, rel=1e-14
        )

    def test_exponential_pressure(self, h2_notebook):
        g = exponential_pressure(1e3, 0.1)
        assert g(10.0, h2_notebook) == pytest.approx(
            1e3 * math.exp(1.0), rel=1e-14
        )

    def test_exponential_pressure_torr(self, h2_notebook):
        g = exponential_pressure_torr(1.0, 0.1)
        assert g(10.0, h2_notebook) == pytest.approx(
            1.0 * TORR_TO_PA * math.exp(1.0), rel=1e-14
        )

    def test_saturation_pressure_delegates(self, h2_notebook):
        g = saturation_pressure()
        assert g(20.0, h2_notebook) == pytest.approx(
            h2_notebook.p_saturation(20.0), rel=1e-14
        )

    def test_amankwah_pressure(self, para_h2):
        g = amankwah_pressure(k=2.0)
        assert g(para_h2.T_critical, para_h2) == pytest.approx(
            para_h2.p_critical, rel=1e-12
        )
