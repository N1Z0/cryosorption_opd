"""Tests for :mod:`opd.fluids.fluid_properties`."""

from __future__ import annotations

import pytest

from opd.fluids.fluid_properties import FluidProperties
from opd.fluids.hydrogen import normal_hydrogen, orthohydrogen, parahydrogen


class TestConstruction:
    def test_default_is_parahydrogen(self):
        assert FluidProperties().name == "ParaHydrogen"

    def test_explicit_name(self):
        assert FluidProperties("Nitrogen").name == "Nitrogen"

    def test_rejects_non_string_name(self):
        with pytest.raises(TypeError):
            FluidProperties(42)  # type: ignore[arg-type]

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            FluidProperties("")

    def test_presets(self):
        assert parahydrogen().name == "ParaHydrogen"
        assert orthohydrogen().name == "OrthoHydrogen"
        assert normal_hydrogen().name == "Hydrogen"


class TestParahydrogenConstants:
    """CoolProp identifier 'ParaHydrogen' should expose consistent SI values."""

    def test_molar_mass(self, para_h2):
        assert para_h2.molar_mass == pytest.approx(2.01588e-3, rel=1e-4)

    def test_critical_point(self, para_h2):
        assert para_h2.T_critical == pytest.approx(32.938, rel=1e-4)
        assert para_h2.p_critical == pytest.approx(1.2858e6, rel=1e-3)

    def test_triple_point(self, para_h2):
        assert para_h2.T_triple == pytest.approx(13.803, rel=1e-3)

    def test_normal_boiling_point(self, para_h2):
        assert para_h2.T_normal_boiling == pytest.approx(20.2713, rel=1e-4)

    def test_cached_properties_return_same_object_twice(self, para_h2):
        # cached_property semantics
        assert para_h2.T_critical is para_h2.T_critical


class TestSaturationBranch:
    def test_roundtrip_p_to_T_to_p(self, para_h2):
        p = 2.5e5
        T = para_h2.T_saturation(p)
        assert para_h2.p_saturation(T) == pytest.approx(p, rel=1e-6)

    def test_rho_L_greater_than_rho_V(self, para_h2):
        T = 20.0
        rho_L = para_h2.rho_molar_saturated_liquid(T)
        rho_V = para_h2.rho_molar_saturated_vapor(T)
        assert rho_L > rho_V > 0.0

    def test_h_vaporization_positive(self, para_h2):
        for T in (15.0, 20.0, 25.0, 30.0):
            assert para_h2.h_vaporization(T) > 0.0

    def test_h_vaporization_vanishes_at_critical(self, para_h2):
        # Approach the critical point — heat of vaporization must fall
        # toward zero.
        h_near = para_h2.h_vaporization(32.9)
        h_low = para_h2.h_vaporization(20.0)
        assert h_near < h_low


class TestSinglePhase:
    def test_density_monotone_in_pressure(self, para_h2):
        T = 100.0  # Well above T_c, guaranteed single phase
        rho_low = para_h2.rho_molar(1e5, T)
        rho_high = para_h2.rho_molar(1e6, T)
        assert rho_high > rho_low

    def test_cp_positive(self, para_h2):
        for p, T in ((1e5, 50.0), (1e5, 100.0), (1e6, 300.0)):
            assert para_h2.cp_molar(p, T) > 0.0

    def test_cv_less_than_cp(self, para_h2):
        p, T = 1e5, 100.0
        assert para_h2.cv_molar(p, T) < para_h2.cp_molar(p, T)


class TestTwoPhase:
    def test_from_TQ_returns_saturated_liquid_for_Q_zero(self, para_h2):
        T = 20.0
        p, rho, u, h = para_h2.from_TQ(T, 0.0)
        assert p == pytest.approx(para_h2.p_saturation(T), rel=1e-9)
        assert rho == pytest.approx(
            para_h2.rho_molar_saturated_liquid(T), rel=1e-9
        )
        assert u == pytest.approx(
            para_h2.u_molar_saturated_liquid(T), rel=1e-9
        )
        assert h == pytest.approx(
            para_h2.h_molar_saturated_liquid(T), rel=1e-9
        )

    def test_from_TQ_rejects_out_of_range(self, para_h2):
        with pytest.raises(ValueError):
            para_h2.from_TQ(20.0, -0.1)
        with pytest.raises(ValueError):
            para_h2.from_TQ(20.0, 1.5)


class TestAmankwahPseudoSat:
    def test_matches_p_c_at_T_c(self, para_h2):
        assert para_h2.amankwah_pseudo_saturation(
            para_h2.T_critical
        ) == pytest.approx(para_h2.p_critical, rel=1e-12)

    def test_k_equal_2_scaling(self, para_h2):
        p = para_h2.amankwah_pseudo_saturation(2 * para_h2.T_critical, k=2.0)
        assert p == pytest.approx(4 * para_h2.p_critical, rel=1e-12)

    def test_monotone_in_T(self, para_h2):
        Tc = para_h2.T_critical
        low = para_h2.amankwah_pseudo_saturation(1.1 * Tc)
        high = para_h2.amankwah_pseudo_saturation(2.0 * Tc)
        assert high > low


class TestRepr:
    def test_repr_roundtrips_through_eval(self):
        f = FluidProperties("Nitrogen")
        assert repr(f) == "FluidProperties('Nitrogen')"
