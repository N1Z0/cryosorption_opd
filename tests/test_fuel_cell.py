"""Tests for the fuel-cell mass sink, subcooling boost, and Mars environment."""

from __future__ import annotations

import numpy as np
import pytest

from opd.constants import LHV_H2_MOLAR, M_H2
from opd.control import AlwaysOnController
from opd.cryocooler import CarnotCryocooler
from opd.environment.orbital import MarsOrbitHeatFlux
from opd.fluids import parahydrogen
from opd.power import FuelCell
from opd.simulation.simulator import TransientSimulator
from opd.tank.geometry import TankGeometry
from opd.tank.heat_loads import ConstantHeatFlux
from opd.tank.tank import Tank


# ---------------------------------------------------------------------------
# FuelCell unit behaviour
# ---------------------------------------------------------------------------

class TestFuelCellRates:
    def test_h2_rate_matches_lhv(self):
        fc = FuelCell(P_electrical=6000.0, t_start=0.0, t_end=3600.0,
                      eta_fc=0.55, ramp_s=0.0)
        expected = 6000.0 / (0.55 * LHV_H2_MOLAR)
        assert fc.n_dot_H2(1800.0) == pytest.approx(expected, rel=1e-12)

    def test_inactive_outside_window(self):
        fc = FuelCell(P_electrical=1000.0, t_start=100.0, t_end=200.0)
        assert fc.P_out(0.0) == 0.0
        assert fc.P_out(250.0) == 0.0
        assert fc.n_dot_H2(50.0) == 0.0

    def test_ramp_is_smooth_and_bounded(self):
        fc = FuelCell(P_electrical=1000.0, t_start=0.0, t_end=1000.0,
                      ramp_s=100.0)
        acts = [fc.activation(t) for t in np.linspace(-10, 1010, 500)]
        assert all(0.0 <= a <= 1.0 for a in acts)
        assert fc.activation(500.0) == 1.0
        assert fc.activation(50.0) == pytest.approx(0.5, abs=1e-12)

    def test_burst_mass_accounts_for_ramps(self):
        fc = FuelCell(P_electrical=6000.0, t_start=0.0, t_end=3600.0,
                      eta_fc=0.55, ramp_s=120.0)
        n_dot = 6000.0 / (0.55 * LHV_H2_MOLAR)
        expected_kg = n_dot * (3600.0 - 120.0) * M_H2
        assert fc.m_H2_per_burst_kg == pytest.approx(expected_kg, rel=1e-12)

    def test_validation(self):
        with pytest.raises(ValueError):
            FuelCell(P_electrical=-1.0, t_start=0.0, t_end=1.0)
        with pytest.raises(ValueError):
            FuelCell(P_electrical=1.0, t_start=1.0, t_end=1.0)
        with pytest.raises(ValueError):
            FuelCell(P_electrical=1.0, t_start=0.0, t_end=1.0, eta_fc=1.5)


# ---------------------------------------------------------------------------
# Tank integration: mass sink + energy accounting
# ---------------------------------------------------------------------------

class TestFuelCellMassSink:
    def _bare_tank_with_fc(self, P_fc=2000.0, t_end=1200.0):
        fl   = parahydrogen()
        geom = TankGeometry(volume=0.2)
        fc   = FuelCell(P_electrical=P_fc, t_start=0.0, t_end=t_end,
                        eta_fc=0.55, ramp_s=0.0)
        tank = Tank(
            fluid=fl, geometry=geom,
            heat_leak=ConstantHeatFlux(Q_leak=0.0),
            fuel_cell=fc,
        )
        return tank, fc

    def test_mass_drawdown_matches_schedule(self):
        tank, fc = self._bare_tank_with_fc()
        y0  = tank.initial_state_two_phase(21.0, 0.5)
        sim = TransientSimulator(tank=tank)
        r   = sim.run(y0, (0.0, 1200.0), n_points=30)

        expected_dn = fc.n_dot_H2(600.0) * 1200.0
        actual_dn   = float(r.n_total[0] - r.n_total[-1])
        assert actual_dn == pytest.approx(expected_dn, rel=1e-6)
        # Conservation audit must recognise the scheduled draw
        assert r.conservation.max_mass_error_rel < 1e-6

    def test_energy_decreases_with_gas_withdrawal(self):
        tank, _ = self._bare_tank_with_fc()
        y0  = tank.initial_state_two_phase(21.0, 0.5)
        sim = TransientSimulator(tank=tank)
        r   = sim.run(y0, (0.0, 1200.0), n_points=30)
        # Withdrawn saturated vapour carries positive enthalpy out
        assert r.U_total[-1] < r.U_total[0]
        assert r.conservation.max_energy_error_rel < 1e-3

    def test_no_fuel_cell_is_unaffected(self):
        fl   = parahydrogen()
        geom = TankGeometry(volume=0.2)
        tank = Tank(fluid=fl, geometry=geom,
                    heat_leak=ConstantHeatFlux(Q_leak=0.0))
        y0  = tank.initial_state_two_phase(21.0, 0.5)
        sim = TransientSimulator(tank=tank)
        r   = sim.run(y0, (0.0, 600.0), n_points=10)
        assert r.n_total[-1] == pytest.approx(r.n_total[0], rel=1e-9)


class TestSubcoolingBoost:
    def test_fuel_cell_boost_cools_faster(self):
        """With a Carnot cryocooler, fuel-cell power must accelerate cooling."""
        fl   = parahydrogen()
        geom = TankGeometry(volume=0.2)
        t_end = 1800.0

        def _run(with_fc: bool):
            fc = (FuelCell(P_electrical=4000.0, t_start=0.0, t_end=t_end,
                           ramp_s=0.0)
                  if with_fc else None)
            tank = Tank(
                fluid=fl, geometry=geom,
                heat_leak=ConstantHeatFlux(Q_leak=0.5),
                cryocooler=CarnotCryocooler(P_input=50.0, T_hot=300.0,
                                            eta_fraction=0.12),
                controller=AlwaysOnController(),
                fuel_cell=fc,
            )
            y0 = tank.initial_state_two_phase(24.0, 0.3)
            return TransientSimulator(tank=tank).run(
                y0, (0.0, t_end), n_points=20
            )

        r_fc   = _run(True)
        r_base = _run(False)
        assert r_fc.T[-1] < r_base.T[-1]
        # The boost must appear in the recorded cryocooler power
        assert r_fc.Q_cryo.max() > r_base.Q_cryo.max()
        assert r_fc.P_fc.max() == pytest.approx(4000.0, rel=1e-9)

    def test_boost_magnitude_is_cop_scaled(self):
        fl   = parahydrogen()
        geom = TankGeometry(volume=0.2)
        cryo = CarnotCryocooler(P_input=50.0, T_hot=300.0, eta_fraction=0.12)
        fc   = FuelCell(P_electrical=4000.0, t_start=0.0, t_end=100.0,
                        ramp_s=0.0)
        tank = Tank(fluid=fl, geometry=geom,
                    heat_leak=ConstantHeatFlux(Q_leak=0.0),
                    cryocooler=cryo, fuel_cell=fc)
        T = 24.0
        assert tank.Q_boost(4000.0, T) == pytest.approx(
            4000.0 * 0.12 * T / (300.0 - T), rel=1e-12
        )


# ---------------------------------------------------------------------------
# Mars orbital environment
# ---------------------------------------------------------------------------

class TestMarsOrbitHeatFlux:
    def test_average_flux(self):
        hl = MarsOrbitHeatFlux(area=2.0)
        expected = (0.62 * 0.25 + 0.38 * 0.08) * 2.0
        assert hl.Q_average == pytest.approx(expected, rel=1e-12)

    def test_flux_bounded_by_sun_and_eclipse(self):
        hl = MarsOrbitHeatFlux(area=2.0)
        for t in np.linspace(0.0, 2 * 6810.0, 200):
            q = hl.Q_dot(float(t), 20.0)
            assert 0.08 * 2.0 - 1e-9 <= q <= 0.25 * 2.0 + 1e-9

    def test_colder_than_leo(self):
        from opd.environment.orbital import LEOHeatFlux
        leo  = LEOHeatFlux(area=2.0)
        mars = MarsOrbitHeatFlux(area=2.0)
        assert mars.Q_average < leo.Q_average
