"""Tests for M5 Para-Ortho catalyst module.

Covers:
- ortho_eq_fraction: limits, monotonicity, high-T plateau
- ortho_eq_enthalpy_difference: sign, limits
- ParaOrthoCatalyst.k: Arrhenius law
- ParaOrthoCatalyst.dX_ortho_dt: drives X toward equilibrium
- ParaOrthoCatalyst.Q_conversion: sign and scaling
- cp_cryo_j_kg_k: cryogenic heat capacity polynomial
"""

from __future__ import annotations

import math

import pytest

from opd.adsorbents.activated_carbon import (
    _CP_CRYO_A,
    _CP_CRYO_B,
    cp_cryo_j_kg_k,
)
from opd.catalysts import (
    ParaOrthoCatalyst,
    ortho_eq_enthalpy_difference,
    ortho_eq_fraction,
)


# ---------------------------------------------------------------------------
# ortho_eq_fraction
# ---------------------------------------------------------------------------

class TestOrthoEqFraction:
    def test_low_temperature_is_nearly_pure_para(self):
        """At 20 K, equilibrium H₂ is >95% para (ortho fraction < 0.05)."""
        x = ortho_eq_fraction(20.0)
        assert 0.0 < x < 0.05, f"ortho fraction at 20 K = {x:.4f}"

    def test_high_temperature_approaches_0_75(self):
        """At T → ∞, ortho fraction → 3/4 (statistical weight)."""
        x = ortho_eq_fraction(3000.0)
        assert abs(x - 0.75) < 0.02, f"ortho fraction at 3000 K = {x:.4f}"

    def test_monotonically_increasing(self):
        temps = [20, 30, 50, 77, 150, 300]
        fracs = [ortho_eq_fraction(T) for T in temps]
        assert fracs == sorted(fracs), f"Not monotone: {fracs}"

    def test_fraction_in_unit_interval(self):
        for T in [10, 20, 50, 100, 300]:
            x = ortho_eq_fraction(T)
            assert 0.0 <= x <= 1.0

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            ortho_eq_fraction(-1.0)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            ortho_eq_fraction(0.0)


# ---------------------------------------------------------------------------
# ortho_eq_enthalpy_difference
# ---------------------------------------------------------------------------

class TestOrthoEqEnthalpyDifference:
    def test_positive_at_cryogenic(self):
        """Para→Ortho is endothermic below ~180 K → Δh > 0."""
        assert ortho_eq_enthalpy_difference(20.0) > 0.0

    def test_zero_above_200K(self):
        assert ortho_eq_enthalpy_difference(200.0) == pytest.approx(0.0)
        assert ortho_eq_enthalpy_difference(300.0) == pytest.approx(0.0)

    def test_decreases_with_temperature(self):
        dh20 = ortho_eq_enthalpy_difference(20.0)
        dh77 = ortho_eq_enthalpy_difference(77.0)
        assert dh20 > dh77 > 0.0

    def test_value_at_20K(self):
        """Known cryogenic value ≈ 527 J/mol at 20 K."""
        dh = ortho_eq_enthalpy_difference(20.0)
        assert dh == pytest.approx(527.0 * (1.0 - 20.0 / 200.0), rel=1e-9)


# ---------------------------------------------------------------------------
# ParaOrthoCatalyst
# ---------------------------------------------------------------------------

class TestParaOrthoCatalyst:
    def test_k_constant_zero_activation(self):
        cat = ParaOrthoCatalyst(k0=0.001, E_activation=0.0)
        assert cat.k(20.0) == pytest.approx(0.001)
        assert cat.k(300.0) == pytest.approx(0.001)

    def test_k_arrhenius(self):
        from opd.constants import R_UNIVERSAL
        Ea  = 1000.0   # J/mol
        k0  = 0.01
        T   = 50.0
        cat = ParaOrthoCatalyst(k0=k0, E_activation=Ea)
        expected = k0 * math.exp(-Ea / (R_UNIVERSAL * T))
        assert cat.k(T) == pytest.approx(expected, rel=1e-9)

    def test_dX_drives_toward_equilibrium_below_eq(self):
        """If X < X_eq, dX/dt must be positive."""
        cat = ParaOrthoCatalyst(k0=0.01)
        T   = 20.0
        X_eq = ortho_eq_fraction(T)
        # Start at pure para (X = 0), well below equilibrium
        dX  = cat.dX_ortho_dt(T, X_ortho=0.0)
        assert dX > 0.0

    def test_dX_zero_at_equilibrium(self):
        cat = ParaOrthoCatalyst(k0=0.01)
        T   = 20.0
        X_eq = ortho_eq_fraction(T)
        dX  = cat.dX_ortho_dt(T, X_ortho=X_eq)
        assert abs(dX) < 1e-15

    def test_dX_negative_above_equilibrium(self):
        """If X > X_eq, dX/dt must be negative (back-conversion)."""
        cat = ParaOrthoCatalyst(k0=0.01)
        T   = 20.0
        X_eq = ortho_eq_fraction(T)
        dX  = cat.dX_ortho_dt(T, X_ortho=min(X_eq + 0.01, 0.99))
        assert dX < 0.0

    def test_Q_conversion_endothermic_at_cryogenic(self):
        """At 20 K with X < X_eq, conversion absorbs heat → Q < 0."""
        cat     = ParaOrthoCatalyst(k0=0.01)
        Q       = cat.Q_conversion(T=20.0, X_ortho=0.0, n_total=1000.0)
        assert Q < 0.0, f"Expected endothermic (Q < 0), got {Q:.4f} W"

    def test_Q_scales_with_n_total(self):
        cat = ParaOrthoCatalyst(k0=0.01)
        Q1  = cat.Q_conversion(T=20.0, X_ortho=0.0, n_total=1000.0)
        Q2  = cat.Q_conversion(T=20.0, X_ortho=0.0, n_total=2000.0)
        assert Q2 == pytest.approx(Q1 * 2.0, rel=1e-9)

    def test_Q_zero_at_equilibrium(self):
        cat  = ParaOrthoCatalyst(k0=0.01)
        T    = 20.0
        X_eq = ortho_eq_fraction(T)
        Q    = cat.Q_conversion(T, X_ortho=X_eq, n_total=1000.0)
        assert abs(Q) < 1e-12


# ---------------------------------------------------------------------------
# cp_cryo_j_kg_k  (activated carbon, T-dependent)
# ---------------------------------------------------------------------------

class TestCpCryoJKgK:
    def test_correct_at_20K(self):
        """cp(20 K) = A*20 + B*20³."""
        T = 20.0
        expected = _CP_CRYO_A * T + _CP_CRYO_B * T ** 3
        assert cp_cryo_j_kg_k(T) == pytest.approx(expected, rel=1e-12)

    def test_room_temperature_matches_plateau(self):
        """cp(300 K) ≈ 850 J/(kg K) (within 10%)."""
        cp300 = cp_cryo_j_kg_k(300.0)
        assert 700.0 < cp300 < 950.0, f"cp(300K) = {cp300:.1f} J/(kg K)"

    def test_cryogenic_value_much_less_than_room_temp(self):
        """cp(20 K) << cp(300 K)."""
        assert cp_cryo_j_kg_k(20.0) < cp_cryo_j_kg_k(300.0) / 10.0

    def test_positive_everywhere(self):
        for T in [1.0, 5.0, 10.0, 20.0, 77.0, 200.0, 300.0]:
            assert cp_cryo_j_kg_k(T) > 0.0

    def test_monotonically_increasing(self):
        temps  = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0]
        values = [cp_cryo_j_kg_k(T) for T in temps]
        assert values == sorted(values)

    def test_floor_at_T_below_one(self):
        """Negative T is clamped to T=1 internally; must return positive cp."""
        assert cp_cryo_j_kg_k(-5.0) > 0.0
        assert cp_cryo_j_kg_k(0.0) > 0.0
