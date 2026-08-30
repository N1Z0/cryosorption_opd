"""Tests for the AX-21 adsorbent and the capacity criterion it cross-checks.

The reference values come from Richard, Bénard and Chahine, Adsorption 15
(2009) 43-51, Table 3 and Fig. 2a.  Since the source figure can only be
read to a few mol/kg, the isotherm assertions are deliberately loose
bands that would still catch a transcription error in any parameter.
"""

from __future__ import annotations

import numpy as np
import pytest

from opd.adsorbents import get_adsorbent, list_adsorbents
from opd.fluids import parahydrogen

FL = parahydrogen()
M = FL.molar_mass


@pytest.fixture(scope="module")
def ax21():
    return get_adsorbent("AX-21")


class TestRegistry:
    def test_listed(self):
        assert "AX-21" in list_adsorbents()

    @pytest.mark.parametrize("alias", ["ax21", "AX-21", "ax_21", "MAXSORB"])
    def test_aliases_resolve(self, alias):
        assert get_adsorbent(alias).name == "AX-21"


class TestPublishedParameters:
    def test_saturation_loading(self, ax21):
        """n_max is approached as p -> p_0 (Table 3: 71.6 mol/kg)."""
        assert ax21.isotherm.n_max == pytest.approx(71.6)

    def test_micropore_volume(self, ax21):
        assert ax21.micropore_volume == pytest.approx(1.43e-3)

    def test_adsorbed_phase_density_matches_carbon_class(self, ax21):
        """rho_ads = n_max*M/V_a should land near the 208C value.

        This is the paper's cross-check: two carbons whose n_max and V_a
        differ threefold imply the same adsorbed-phase density, which is
        what makes the supra-liquid figure credible rather than an
        artefact of one fit.
        """
        rho_ax = ax21.isotherm.n_max * M / ax21.micropore_volume
        ac = get_adsorbent("activated_carbon")
        rho_ac = (ac.isotherm.supercritical.n_max * M / ac.micropore_volume)
        assert rho_ax == pytest.approx(100.9, abs=0.5)
        assert rho_ax == pytest.approx(rho_ac, rel=0.05)


class TestIsothermAgainstSource:
    @pytest.mark.parametrize(
        "T, p_MPa, n_exc_lo, n_exc_hi",
        [
            # Peaks and tails read off Ref. [1], Fig. 2a.
            (45.0, 1.0, 35.0, 45.0),
            (60.0, 2.0, 28.0, 37.0),
            (77.0, 3.5, 22.0, 30.0),
            (298.0, 6.0, 1.5, 4.5),
        ],
    )
    def test_excess_within_published_band(self, ax21, T, p_MPa,
                                          n_exc_lo, n_exc_hi):
        n = ax21.n_excess(p_MPa * 1e6, T, FL)
        assert n_exc_lo <= n <= n_exc_hi

    def test_peak_shifts_to_higher_pressure_with_temperature(self, ax21):
        """The excess maximum moves right as T rises (Ref. [1], Fig. 2a)."""
        p_grid = np.linspace(0.05e6, 6.0e6, 400)
        peaks = []
        for T in (45.0, 60.0, 77.0, 93.0):
            n = np.array([ax21.n_excess(float(p), T, FL) for p in p_grid])
            peaks.append(p_grid[int(np.nanargmax(n))])
        assert np.all(np.diff(peaks) > 0.0)

    def test_loading_decreases_with_temperature(self, ax21):
        p = 13e5
        loads = [ax21.n_absolute(p, T, FL) for T in (40.0, 60.0, 80.0, 120.0)]
        assert np.all(np.diff(loads) < 0.0)

    def test_no_subcritical_branch_switch(self, ax21):
        """A single branch spans the critical point without a jump."""
        p = 13e5
        lo = ax21.n_absolute(p, 31.99, FL)
        hi = ax21.n_absolute(p, 32.01, FL)
        assert lo == pytest.approx(hi, rel=1e-3)


class TestCapacityCriterion:
    @staticmethod
    def _rho_eff(ads, p, T):
        return (ads.n_absolute(p, T, FL) * M
                / (1.0 / ads.skeletal_density + ads.micropore_volume))

    def test_criterion_is_independent_of_installed_mass(self, ax21):
        """rho_eff > rho_bulk must predict the sign of dm/dm_s for any m_s."""
        p, T, V = 13e5, 60.0, 1.0
        beneficial = self._rho_eff(ax21, p, T) > FL.rho_molar(p, T) * M

        def inventory(m_s):
            V_gas = V - m_s / ax21.skeletal_density - m_s * ax21.micropore_volume
            return (FL.rho_molar(p, T) * V_gas
                    + m_s * ax21.n_absolute(p, T, FL)) * M

        for m_s in (120.0, 300.0, 480.0, 700.0):
            grows = inventory(m_s + 1.0) > inventory(m_s)
            assert grows is beneficial

    def test_favourable_window_wider_than_208c(self, ax21):
        """AX-21's crossover sits below 208C's at elevated pressure.

        This is the result reported in the paper: the extrapolated 208C
        fit is conservative, not optimistic.
        """
        ac = get_adsorbent("activated_carbon")

        def crossover(ads, p):
            for T in np.linspace(20.5, 200.0, 1200):
                if self._rho_eff(ads, p, float(T)) > FL.rho_molar(p, float(T)) * M:
                    return float(T)
            return float("nan")

        for p_bar in (30.0, 50.0, 80.0):
            assert crossover(ax21, p_bar * 1e5) < crossover(ac, p_bar * 1e5)

    def test_beneficial_at_depot_operating_point(self, ax21):
        p, T = 13e5, 40.0
        assert self._rho_eff(ax21, p, T) > FL.rho_molar(p, T) * M

    def test_detrimental_in_compressed_liquid(self, ax21):
        """Below T* the bulk liquid is denser than the pores it displaces."""
        T = 24.0
        p = FL.p_saturation(T) * 1.5          # compressed liquid, not vapour
        rho_b = FL.rho_molar(p, T) * M
        assert rho_b > 60.0, "expected a dense liquid state, got {rho_b}"
        assert self._rho_eff(ax21, p, T) < rho_b


class TestSkeletonHeatCapacity:
    def test_cryogenic_and_ambient_bounds(self, ax21):
        cp = ax21.cp_skeleton
        assert 1.0 < cp(20.0) < 5.0
        assert 700.0 < cp(300.0) < 1000.0

    def test_monotone(self, ax21):
        cp = ax21.cp_skeleton
        vals = [cp(T) for T in (10.0, 40.0, 100.0, 300.0)]
        assert np.all(np.diff(vals) > 0.0)
