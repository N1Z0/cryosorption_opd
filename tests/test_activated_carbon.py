"""Regression tests for :mod:`opd.adsorbents.activated_carbon`.

Reference isotherm formulas are re-implemented in this file so that library
refactors cannot silently change the fitted coefficients.
"""

from __future__ import annotations

import math

import pytest

from opd.adsorbents.activated_carbon import ActivatedCarbon208C
from opd.constants import R_UNIVERSAL, TORR_TO_PA

_N_MAX = 552.71294209 * (1.0 / 22400.0) * 1000.0
_V_A = 0.0004838173512692576


def _ref_super_n_abs(p: float, T: float) -> float:
    C = -4776.95770 + (-4.64880774) * T
    p0 = 3050558.0367990825 * TORR_TO_PA
    return _N_MAX * math.exp(
        -((R_UNIVERSAL * T / C) ** 2) * math.log(p0 / p) ** 2
    )


def _ref_sub_n_abs(p: float, T: float) -> float:
    C = -1384.82301956 + 181.31452794 * T
    p0 = 0.0074801 * TORR_TO_PA * math.exp(0.59807858 * T)
    return _N_MAX * math.exp(
        -((R_UNIVERSAL * T / C) ** 2) * math.log(p0 / p) ** 2
    )


@pytest.fixture(scope="module")
def ac():
    return ActivatedCarbon208C()


class TestSupercriticalBranch:
    @pytest.mark.parametrize(
        "p, T",
        [
            (1e3, 40.0),
            (5e4, 77.35),
            (1e5, 100.0),
            (5e5, 125.0),
            (1.3e6, 150.0),
        ],
    )
    def test_absolute_matches_reference(self, ac, h2_notebook, p, T):
        assert T >= 32.0
        got = ac.n_absolute(p, T, h2_notebook)
        expected = _ref_super_n_abs(p, T)
        assert got == pytest.approx(expected, rel=1e-10)


class TestSubcriticalBranch:
    @pytest.mark.parametrize(
        "p, T",
        [
            (1e3, 20.0),
            (5e4, 20.37),
            (1e5, 23.0),
            (5e4, 27.0),
            (1e5, 30.0),
        ],
    )
    def test_absolute_matches_reference(self, ac, h2_notebook, p, T):
        assert T < 32.0
        got = ac.n_absolute(p, T, h2_notebook)
        expected = _ref_sub_n_abs(p, T)
        assert got == pytest.approx(expected, rel=1e-10)


class TestBranchSelection:
    def test_sub_selected_just_below_32(self, ac, h2_notebook):
        T = 31.999
        p = 1e5
        assert ac.n_absolute(p, T, h2_notebook) == pytest.approx(
            _ref_sub_n_abs(p, T), rel=1e-10
        )

    def test_super_selected_at_32(self, ac, h2_notebook):
        T = 32.0
        p = 1e5
        assert ac.n_absolute(p, T, h2_notebook) == pytest.approx(
            _ref_super_n_abs(p, T), rel=1e-10
        )

    def test_super_selected_just_above_32(self, ac, h2_notebook):
        T = 32.001
        p = 1e5
        assert ac.n_absolute(p, T, h2_notebook) == pytest.approx(
            _ref_super_n_abs(p, T), rel=1e-10
        )


class TestExcess:
    def test_excess_convention(self, ac, h2_notebook):
        for p, T in ((5e4, 30.0), (1e5, 77.35), (1e5, 100.0)):
            abs_ref = (
                _ref_sub_n_abs(p, T) if T < 32.0 else _ref_super_n_abs(p, T)
            )
            expected = abs_ref - _V_A * h2_notebook.rho_molar(p, T)
            assert ac.n_excess(p, T, h2_notebook) == pytest.approx(
                expected, rel=1e-10
            )


class TestMaterialMetadata:
    def test_name(self, ac):
        assert "208C" in ac.name

    def test_skeletal_density(self, ac):
        assert ac.skeletal_density == pytest.approx(2150.0, rel=1e-12)

    def test_micropore_volume(self, ac):
        assert ac.micropore_volume == pytest.approx(_V_A, rel=1e-12)

    def test_cp_skeleton_is_callable_of_T(self, ac):
        assert callable(ac.cp_skeleton)
        for T in (20.0, 77.0, 300.0):
            assert ac.cp_skeleton(T) > 0.0

    def test_isosteric_heat_positive(self, ac):
        q = ac.isosteric_heat(1e5, 77.0)
        assert q == pytest.approx(1929.5548734380006, rel=1e-12)

    def test_isosteric_heat_constant_over_pT(self, ac):
        q0 = ac.isosteric_heat(1e4, 20.0)
        q1 = ac.isosteric_heat(1e6, 150.0)
        assert q0 == q1
