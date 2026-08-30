"""Mass conservation tests.

In a closed tank (no venting, no inflow) the total number of moles must
remain constant throughout the entire simulation to machine precision.

Both the bare tank and the AC-loaded tank are tested.  The AC case is
important because the adsorption model redistributes molecules between
the bulk and adsorbed phase at each ODE step; the *total* (bulk + adsorbed)
must remain invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from opd.adsorbents.activated_carbon import ActivatedCarbon208C
from opd.fluids.hydrogen import normal_hydrogen, parahydrogen
from opd.simulation import TransientSimulator
from opd.tank import ConstantHeatFlux, Tank, TankGeometry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_closed_tank(
    fluid,
    T0,
    Q0,
    V_free,
    Q_dot,
    t_end,
    adsorbent=None,
    m_sorb=0.0,
    n_points=300,
):
    geom = TankGeometry(volume=V_free)
    tank = Tank(
        fluid=fluid,
        geometry=geom,
        heat_leak=ConstantHeatFlux(Q_dot),
        adsorbent=adsorbent,
        m_sorb=m_sorb,
    )
    y0 = tank.initial_state_two_phase(T0, Q0)
    sim = TransientSimulator(tank, solver_kwargs={"rtol": 1e-10, "atol": 1e-8})
    return sim.run(y0=y0, t_span=(0.0, t_end), n_points=n_points)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBareTankMassConservation:
    """Bare-tank scenarios with various initial conditions."""

    @pytest.mark.parametrize(
        "T0, Q0, t_end",
        [
            (20.0, 0.3, 2000.0),
            (25.0, 0.5, 2000.0),
        ],
    )
    def test_n_total_constant(self, T0, Q0, t_end):
        result = run_closed_tank(parahydrogen(), T0, Q0, 1.0, 200.0, t_end)
        n = result.n_total
        rel_err = np.max(np.abs(n - n[0])) / abs(n[0])
        assert rel_err < 1e-8, f"Mass drift = {rel_err:.2e}"

    def test_conservation_report_mass_error(self):
        result = run_closed_tank(parahydrogen(), 20.0, 0.3, 1.0, 200.0, 2000.0)
        assert result.conservation.max_mass_error_rel < 1e-8


class TestACTankMassConservation:
    """AC-loaded tank: n_bulk + n_ads_abs must sum to n_total at every step."""

    def test_bulk_plus_ads_equals_total(self):
        ac = ActivatedCarbon208C()
        m_sorb = 50.0
        V_free = 0.9 - m_sorb / ac.skeletal_density
        result = run_closed_tank(
            normal_hydrogen(), 20.0, 0.3, V_free, 100.0, 3000.0,
            adsorbent=ac, m_sorb=m_sorb,
        )
        n_check = result.n_bulk + result.n_ads_abs
        rel_err = np.max(np.abs(n_check - result.n_total)) / result.n_total[0]
        assert rel_err < 1e-7

    def test_n_total_constant_with_ac(self):
        ac = ActivatedCarbon208C()
        m_sorb = 50.0
        V_free = 0.9 - m_sorb / ac.skeletal_density
        result = run_closed_tank(
            normal_hydrogen(), 20.0, 0.3, V_free, 100.0, 3000.0,
            adsorbent=ac, m_sorb=m_sorb,
        )
        n = result.n_total
        rel_err = np.max(np.abs(n - n[0])) / abs(n[0])
        assert rel_err < 1e-8
