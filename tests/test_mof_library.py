"""Tests for M6: MOF library, adsorbent factory, and orbital heat models.

Covers:
- MIL-101(Cr) and IRMOF-20 material properties (density, V_a, cp, q_st)
- Isotherm sanity checks (positive, monotone in p, decreasing in T)
- AdsorbentFactory: get_adsorbent, list_adsorbents, register_adsorbent
- LEOHeatFlux: correct average, sun/eclipse cycling, sphere_area
- LunarHeatFlux: day/night cycling
- GatewayHeatFlux: sinusoidal profile
- MLIHeatFlux: constant output
- cp_cryo round-trips for MIL-101 and IRMOF-20
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opd.adsorbents import (
    ActivatedCarbon208C,
    IRMOF20,
    MIL101,
    get_adsorbent,
    list_adsorbents,
    register_adsorbent,
)
from opd.adsorbents.base import AdsorbentMaterial
from opd.adsorbents.mof_irmof20 import cp_cryo_irmof20
from opd.adsorbents.mof_mil101 import cp_cryo_mil101
from opd.environment import GatewayHeatFlux, LEOHeatFlux, LunarHeatFlux, MLIHeatFlux
from opd.fluids import parahydrogen


# ---------------------------------------------------------------------------
# MOF material properties
# ---------------------------------------------------------------------------

class TestMIL101Properties:
    def test_name(self):
        ads = MIL101()
        assert "MIL-101" in ads.name

    def test_skeletal_density_positive(self):
        ads = MIL101()
        assert ads.skeletal_density > 0.0

    def test_micropore_volume_large(self):
        """MIL-101 has ~1.9 cm³/g micropore volume."""
        ads = MIL101()
        assert ads.micropore_volume == pytest.approx(1.9e-3, rel=0.01)

    def test_isosteric_heat_positive(self):
        ads = MIL101()
        q = ads.isosteric_heat(1e5, 77.0)
        assert q > 0.0

    def test_isosteric_heat_larger_than_AC(self):
        """MIL-101 has q_st ~5800 J/mol > AC 208C ~1930 J/mol."""
        mil = MIL101()
        ac  = ActivatedCarbon208C()
        assert mil.isosteric_heat(1e5, 77.0) > ac.isosteric_heat(1e5, 77.0)

    def test_cp_positive_at_20K(self):
        assert cp_cryo_mil101(20.0) > 0.0

    def test_cp_positive_at_300K(self):
        assert cp_cryo_mil101(300.0) > 0.0

    def test_cp_cryogenic_much_less_than_room_temp(self):
        assert cp_cryo_mil101(20.0) < cp_cryo_mil101(300.0) / 5.0

    def test_cp_floor_negative_T(self):
        assert cp_cryo_mil101(-1.0) > 0.0


class TestIRMOF20Properties:
    def test_name(self):
        assert "IRMOF-20" in IRMOF20().name

    def test_skeletal_density_light(self):
        """IRMOF-20 is a very open Zn-MOF; density ~600 kg/m³."""
        ads = IRMOF20()
        assert 400.0 < ads.skeletal_density < 900.0

    def test_micropore_volume_large(self):
        """IRMOF-20 V_a ~ 1.53 cm³/g."""
        ads = IRMOF20()
        assert ads.micropore_volume == pytest.approx(1.53e-3, rel=0.01)

    def test_isosteric_heat_positive(self):
        assert IRMOF20().isosteric_heat(1e5, 77.0) > 0.0

    def test_cp_monotone(self):
        temps  = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0]
        values = [cp_cryo_irmof20(T) for T in temps]
        assert values == sorted(values)

    def test_cp_floor_at_zero_T(self):
        assert cp_cryo_irmof20(0.0) > 0.0


# ---------------------------------------------------------------------------
# Isotherm sanity checks
# ---------------------------------------------------------------------------

class TestIsothermSanity:
    @pytest.fixture(scope="class")
    def fluid(self):
        return parahydrogen()

    @pytest.mark.parametrize("ads_name", ["MIL-101", "IRMOF-20"])
    def test_n_abs_positive(self, ads_name, fluid):
        ads = get_adsorbent(ads_name)
        n = ads.n_absolute(1e5, 77.0, fluid)
        assert n > 0.0

    @pytest.mark.parametrize("ads_name", ["MIL-101", "IRMOF-20"])
    def test_n_abs_monotone_in_p(self, ads_name, fluid):
        """n_abs should increase with pressure at constant T."""
        ads = get_adsorbent(ads_name)
        pressures = [0.1e5, 0.5e5, 1.0e5, 5.0e5, 10.0e5]
        n_vals = [ads.n_absolute(p, 77.0, fluid) for p in pressures]
        assert n_vals == sorted(n_vals), f"Not monotone in p: {n_vals}"

    @pytest.mark.parametrize("ads_name", ["MIL-101", "IRMOF-20"])
    def test_n_abs_decreasing_in_T(self, ads_name, fluid):
        """At fixed pressure, adsorption decreases with increasing T."""
        ads = get_adsorbent(ads_name)
        p = 1.0e5
        n_20  = ads.n_absolute(p, 20.0, fluid)
        n_77  = ads.n_absolute(p, 77.0, fluid)
        n_200 = ads.n_absolute(p, 200.0, fluid)
        assert n_20 >= n_77 >= n_200, (
            f"n_abs should decrease with T: {n_20:.3f}≥{n_77:.3f}≥{n_200:.3f}"
        )

    @pytest.mark.parametrize("ads_name", ["MIL-101", "IRMOF-20"])
    def test_n_abs_zero_at_zero_pressure(self, ads_name, fluid):
        ads = get_adsorbent(ads_name)
        assert ads.n_absolute(0.0, 77.0, fluid) == pytest.approx(0.0)

    @pytest.mark.parametrize("ads_name", ["MIL-101", "IRMOF-20"])
    def test_n_abs_bounded_above_by_n_max(self, ads_name, fluid):
        """n_abs ≤ n_max by construction of the D-A model."""
        from opd.adsorbents.mof_mil101 import _N_MAX_MOL_PER_KG as N_MAX_MIL
        from opd.adsorbents.mof_irmof20 import _N_MAX_MOL_PER_KG as N_MAX_IRR
        n_max = N_MAX_MIL if ads_name == "MIL-101" else N_MAX_IRR
        ads = get_adsorbent(ads_name)
        for p in [1e4, 1e5, 1e6, 1e7]:
            n = ads.n_absolute(p, 77.0, fluid)
            assert n <= n_max * 1.001, f"n_abs={n} > n_max={n_max} at p={p}"

    def test_mil101_has_higher_isosteric_heat_than_ac(self, fluid):
        """MIL-101 q_st ~5800 J/mol > AC 208C ~1930 J/mol (literature values)."""
        mil = get_adsorbent("MIL-101")
        ac  = get_adsorbent("activated_carbon")
        assert mil.isosteric_heat(1e5, 77.0) > ac.isosteric_heat(1e5, 77.0)

    def test_mil101_n_abs_77K_in_physical_range(self, fluid):
        """MIL-101 n_abs at 77 K / 1 bar should be in 2–9 mol/kg (Latroche 2006)."""
        mil = get_adsorbent("MIL-101")
        n = mil.n_absolute(1e5, 77.0, fluid)
        assert 2.0 <= n <= 9.0, f"MIL-101 n_abs(77K, 1bar) = {n:.3f} mol/kg — out of expected range"


# ---------------------------------------------------------------------------
# AdsorbentFactory
# ---------------------------------------------------------------------------

class TestAdsorbentFactory:
    def test_list_returns_canonical_names(self):
        names = list_adsorbents()
        assert "activated_carbon" in names
        assert "MIL-101" in names
        assert "IRMOF-20" in names

    def test_get_by_canonical_name(self):
        ads = get_adsorbent("MIL-101")
        assert isinstance(ads, AdsorbentMaterial)

    def test_get_case_insensitive(self):
        ads1 = get_adsorbent("MIL-101")
        ads2 = get_adsorbent("mil-101")
        ads3 = get_adsorbent("mil101")
        assert ads1.name == ads2.name == ads3.name

    def test_get_alias_ac(self):
        ads = get_adsorbent("ac")
        assert "Activated Carbon" in ads.name

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_adsorbent("NoSuchMaterial_XYZ")

    def test_register_custom(self):
        def _my_ads():
            from opd.adsorbents.base import constant_isosteric_heat
            return AdsorbentMaterial(
                name="Test-MOF",
                skeletal_density=1000.0,
                micropore_volume=1.0e-3,
                isotherm=get_adsorbent("MIL-101").isotherm,
                cp_skeleton=lambda T: 500.0,
                isosteric_heat_fn=constant_isosteric_heat(3000.0),
            )
        register_adsorbent("Test-MOF", _my_ads, aliases=["test_mof"])
        ads = get_adsorbent("Test-MOF")
        assert ads.name == "Test-MOF"
        ads2 = get_adsorbent("test_mof")
        assert ads2.name == "Test-MOF"

    def test_each_call_returns_fresh_instance(self):
        a1 = get_adsorbent("MIL-101")
        a2 = get_adsorbent("MIL-101")
        assert a1 is not a2   # different objects


# ---------------------------------------------------------------------------
# Orbital heat flux models
# ---------------------------------------------------------------------------

class TestMLIHeatFlux:
    def test_constant_output(self):
        hl = MLIHeatFlux(q_eff=0.5, area=4.8)
        assert hl.Q_dot(0.0, 20.0) == pytest.approx(0.5 * 4.8)

    def test_sphere_area(self):
        V = 1.0
        A = MLIHeatFlux.sphere_area(V)
        r = (3 * V / (4 * math.pi)) ** (1/3)
        assert A == pytest.approx(4 * math.pi * r ** 2, rel=1e-9)

    def test_negative_q_raises(self):
        with pytest.raises(ValueError):
            MLIHeatFlux(q_eff=-0.1, area=4.0)


class TestLEOHeatFlux:
    _HL = LEOHeatFlux(area=4.84, q_sun=0.5, q_eclipse=0.15)

    def test_Q_dot_in_sun_near_q_sun(self):
        """At t=0 (start of sunlit period), Q_dot ≈ q_sun × area."""
        q = self._HL.Q_dot(0.0, 20.0) / 4.84
        assert pytest.approx(0.5, abs=0.05) == q

    def test_Q_dot_in_eclipse_near_q_eclipse(self):
        """At mid-eclipse, Q_dot ≈ q_eclipse × area."""
        T = 5400.0
        t_mid_eclipse = T * (0.62 + (1 - 0.62) / 2)
        q = self._HL.Q_dot(t_mid_eclipse, 20.0) / 4.84
        assert pytest.approx(0.15, abs=0.05) == q

    def test_Q_average(self):
        expected = (0.62 * 0.5 + 0.38 * 0.15) * 4.84
        assert self._HL.Q_average == pytest.approx(expected, rel=1e-9)

    def test_orbit_periodicity(self):
        """Q_dot should be the same at t and t + orbit_period."""
        T = 5400.0
        t = 1000.0
        assert self._HL.Q_dot(t, 20.0) == pytest.approx(
            self._HL.Q_dot(t + T, 20.0), rel=1e-6
        )

    def test_Q_dot_bounded_between_eclipse_and_sun(self):
        """Q_dot always between q_eclipse×A and q_sun×A."""
        area = 4.84
        t_samples = np.linspace(0, 5400.0, 100)
        for t in t_samples:
            q = self._HL.Q_dot(float(t), 20.0)
            assert 0.15 * area * 0.99 <= q <= 0.5 * area * 1.01


class TestLunarHeatFlux:
    _HL = LunarHeatFlux(area=4.84, q_day=0.8, q_night=0.1)

    def test_Q_average(self):
        expected = (0.5 * 0.8 + 0.5 * 0.1) * 4.84
        assert self._HL.Q_average == pytest.approx(expected, rel=1e-9)

    def test_q_night_in_night_half(self):
        T = self._HL.period
        t_night = T * 0.75   # 75% through = middle of night
        q = self._HL.Q_dot(t_night, 20.0)
        assert q == pytest.approx(0.1 * 4.84, rel=1e-9)


class TestGatewayHeatFlux:
    _HL = GatewayHeatFlux(area=4.84, q_mean=0.4, q_amplitude=0.3)

    def test_Q_average(self):
        """Time-average of sinusoid = mean."""
        T = self._HL.period
        t_samples = np.linspace(0, T, 10000)
        Q_vals = [self._HL.Q_dot(float(t), 20.0) for t in t_samples]
        assert np.mean(Q_vals) == pytest.approx(self._HL.Q_average, rel=0.01)

    def test_bounded(self):
        """Q_dot ∈ [(1-ampl)×q_mean×A, (1+ampl)×q_mean×A]."""
        area = 4.84
        q_lo = (1.0 - 0.3) * 0.4 * area
        q_hi = (1.0 + 0.3) * 0.4 * area
        for t in np.linspace(0, self._HL.period, 200):
            q = self._HL.Q_dot(float(t), 20.0)
            assert q_lo * 0.999 <= q <= q_hi * 1.001
