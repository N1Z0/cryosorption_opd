"""Tank — the central ODE assembler for the lumped-parameter model.

:class:`Tank` owns:

* a :class:`~opd.simulation.thermal_state.ThermalStateResolver` (the EOS),
* a :class:`~opd.tank.heat_loads.HeatLeakModel`,
* optionally an :class:`~opd.adsorbents.base.AdsorbentMaterial` with mass,
* optionally a :class:`~opd.cryocooler.base.CryocoolerModel` + a
  :class:`~opd.control.pressure_control.PressureController`,
* optionally a :class:`~opd.catalysts.base.ParaOrthoCatalyst`,
* an optional vent pressure ``p_vent``.

It provides :meth:`ode_rhs` (the callable passed to ``solve_ivp``),
:meth:`initial_state` (encodes an (T, Q) or (T, p) start state into the
``[n, U]`` or ``[n, U, X_ortho]`` vector), and the event-function helpers
used by :class:`~opd.simulation.simulator.TransientSimulator`.

ODE state vector
----------------
**Without catalyst (default):** ``y = [n_H2_total (mol), U_total (J)]``

**With catalyst:** ``y = [n_H2_total (mol), U_total (J), X_ortho (–)]``

where ``X_ortho`` is the ortho-H₂ mole fraction.

Right-hand side
---------------
General form (vent + fuel cell both optional):

    dn/dt  = −ṅ_vent − ṅ_fc
    dU/dt  = Q̇_leak − d·Q̇_cryo − Q̇_boost + Q̇_cat − (ṅ_vent + ṅ_fc)·h_out
    dX/dt  = k(T)·(X_eq(T) − X)     (catalyst present only)

where ``ṅ_fc`` is the fuel-cell H₂ draw, ``Q̇_boost = P_fc·COP(T)`` is
the extra cryocooler extraction funded by fuel-cell electricity, and
``h_out`` is the enthalpy of the withdrawn ullage gas (saturated vapour
inside the dome).

The venting rate is computed from the isobaric constraint ``dp/dt = 0``
via numerical partial derivatives of the tank EOS.

The catalyst heat ``Q̇_cat`` is endothermic (negative) when para→ortho
conversion occurs during warming, and exothermic (positive) when the
ortho fraction relaxes back down during a transient cooldown.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ..adsorbents.base import AdsorbentMaterial
from ..fluids.fluid_properties import FluidProperties
from ..simulation.thermal_state import ResolvedState, ThermalStateResolver
from .geometry import TankGeometry
from .heat_loads import HeatLeakModel

if TYPE_CHECKING:
    from ..cryocooler.base import CryocoolerModel
    from ..control.pressure_control import PressureController
    from ..catalysts.base import ParaOrthoCatalyst
    from ..power.fuel_cell import FuelCell

__all__ = ["Tank"]


class Tank:
    """Lumped-parameter hydrogen tank ODE model (1-Temp, equilibrium adsorption).

    Parameters
    ----------
    fluid
        Fluid property provider.
    geometry
        Tank geometry (volume, surface area).
    heat_leak
        Model for the external thermal power into the tank contents.
    adsorbent
        Sorbent material.  ``None`` for a bare tank.
    m_sorb
        Sorbent mass, kg.
    p_vent
        Relief-valve set pressure, Pa.  When the resolved pressure exceeds
        this value the vent opens and gas is expelled at the current
        specific enthalpy.  ``None`` disables venting entirely.
    T_skel_ref
        Reference temperature for the skeleton enthalpy integral, K.
    cryocooler
        Heat-extraction device.  ``None`` = no active cooling.
    controller
        Pressure controller that maps ``(t, p)`` → duty ``[0, 1]``.
        Ignored when ``cryocooler`` is ``None``.
    catalyst
        Para-ortho conversion catalyst.  When set, the state vector gains
        a third element ``X_ortho``.
    fuel_cell
        Optional :class:`~opd.power.fuel_cell.FuelCell`.  While active it
        (a) withdraws ullage gas at rate ``n_dot_H2(t)`` (mass sink with
        enthalpy outflow) and (b) feeds its electrical output ``P_out(t)``
        into the cryocooler as *additional* input power (subcooling
        boost).  The boost uses the cryocooler's ``COP(T)`` when
        available (e.g. :class:`~opd.cryocooler.base.CarnotCryocooler`)
        and is not modulated by the pressure controller — the burst is
        commanded by the refuelling schedule, not by tank pressure.
    """

    def __init__(
        self,
        fluid: FluidProperties,
        geometry: TankGeometry,
        heat_leak: HeatLeakModel,
        adsorbent: Optional[AdsorbentMaterial] = None,
        m_sorb: float = 0.0,
        p_vent: Optional[float] = None,
        T_skel_ref: float = 0.0,
        cryocooler: Optional["CryocoolerModel"] = None,
        controller: Optional["PressureController"] = None,
        catalyst: Optional["ParaOrthoCatalyst"] = None,
        fuel_cell: Optional["FuelCell"] = None,
    ) -> None:
        if m_sorb > 0.0 and adsorbent is None:
            raise ValueError("m_sorb > 0 requires an adsorbent")
        if cryocooler is not None and controller is None:
            from ..control.pressure_control import AlwaysOnController
            controller = AlwaysOnController()

        self._fluid    = fluid
        self._geometry = geometry
        self._heat_leak = heat_leak
        self._ads      = adsorbent
        self._m_sorb   = m_sorb
        self._p_vent   = p_vent
        self._cryo     = cryocooler
        self._ctrl     = controller
        self._catalyst = catalyst
        self._fc       = fuel_cell

        V_free = geometry.free_volume(m_sorb=m_sorb, adsorbent=adsorbent)
        self._resolver = ThermalStateResolver(
            fluid=fluid,
            V_free=V_free,
            adsorbent=adsorbent,
            m_sorb=m_sorb,
            T_skel_ref=T_skel_ref,
        )

        # (T_prev, p_prev) cache the last resolved state for warm-starting
        # the single-phase Newton solver.  Without the pressure warm start
        # the ideal-gas initial guess is poor in the dense supercritical
        # region and fsolve can hop between roots, making the ODE
        # right-hand side non-smooth.
        self._T_prev: Optional[float] = None
        self._p_prev: Optional[float] = None

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def resolver(self) -> ThermalStateResolver:
        return self._resolver

    @property
    def fluid(self) -> FluidProperties:
        return self._fluid

    @property
    def p_vent(self) -> Optional[float]:
        return self._p_vent

    @property
    def has_catalyst(self) -> bool:
        """True when a para-ortho catalyst is attached."""
        return self._catalyst is not None

    @property
    def fuel_cell(self) -> Optional["FuelCell"]:
        """The attached fuel cell, or ``None``."""
        return self._fc

    @property
    def state_size(self) -> int:
        """Length of the ODE state vector."""
        return 3 if self.has_catalyst else 2

    # ------------------------------------------------------------------
    # Initial-state helpers
    # ------------------------------------------------------------------

    def initial_state_two_phase(
        self, T: float, Q: float, X_ortho: float = 0.0
    ) -> np.ndarray:
        """Encode a two-phase start state.

        Parameters
        ----------
        T
            Initial temperature, K.
        Q
            Initial vapour quality (CoolProp molar convention).
        X_ortho
            Initial ortho fraction (used only when a catalyst is present).
        """
        n, U = self._resolver.encode_two_phase(T, Q)
        if self.has_catalyst:
            return np.array([n, U, X_ortho])
        return np.array([n, U])

    def initial_state_single_phase(
        self, T: float, p: float, X_ortho: float = 0.0
    ) -> np.ndarray:
        """Encode a single-phase start state.

        Parameters
        ----------
        T
            Initial temperature, K.
        p
            Initial pressure, Pa.
        X_ortho
            Initial ortho fraction (used only when a catalyst is present).
        """
        n, U = self._resolver.encode_single_phase(T, p)
        if self.has_catalyst:
            return np.array([n, U, X_ortho])
        return np.array([n, U])

    def resolve_state(self, y: np.ndarray) -> ResolvedState:
        """Resolve the full thermodynamic state from a state-vector slice."""
        return self._resolver.resolve(
            float(y[0]), float(y[1]),
            T_guess=self._T_prev, p_guess=self._p_prev,
        )

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE right-hand side ``dy/dt = f(t, y)``.

        Parameters
        ----------
        t
            Current time, s.
        y
            State vector.  Length 2 without catalyst, length 3 with.
        """
        n_total = float(y[0])
        U_total = float(y[1])
        X_ortho = float(y[2]) if self.has_catalyst else 0.0

        state = self._resolver.resolve(
            n_total, U_total, T_guess=self._T_prev, p_guess=self._p_prev
        )
        self._T_prev = state.T
        self._p_prev = state.p

        # --- external heat inputs ---
        Q_dot_net = self._heat_leak.Q_dot(t, state.T)

        # cryocooler: subtracts from energy
        if self._cryo is not None:
            duty   = self._ctrl.duty(t, state.p)
            Q_cryo = self._cryo.Q_cryo(t, state.T) * duty
            Q_dot_net -= Q_cryo

        # para-ortho catalyst heat: endothermic (para→ortho) when heating,
        # exothermic (ortho→para relaxation) during transient cooldown
        if self._catalyst is not None:
            Q_cat = self._catalyst.Q_conversion(state.T, X_ortho, n_total)
            Q_dot_net += Q_cat

        # --- fuel cell: gas withdrawal + cryocooler subcooling boost ---
        n_dot_fc = 0.0
        h_out    = 0.0
        if self._fc is not None:
            P_fc = self._fc.P_out(t)
            if P_fc > 0.0:
                n_dot_fc = self._fc.n_dot_H2(t)
                h_out    = self._h_outflow(state)
                Q_dot_net -= self.Q_boost(P_fc, state.T)

        # --- venting ---
        if self._p_vent is not None and state.p >= self._p_vent:
            n_dot_vent = self._vent_rate(
                n_total, U_total, state, Q_dot_net, n_dot_fc=n_dot_fc
            )
        else:
            n_dot_vent = 0.0

        if n_dot_vent > 0.0 and h_out == 0.0:
            h_out = self._h_outflow(state)

        dy = np.empty(self.state_size)
        dy[0] = -(n_dot_vent + n_dot_fc)
        dy[1] = Q_dot_net - (n_dot_vent + n_dot_fc) * h_out

        if self.has_catalyst:
            dy[2] = self._catalyst.dX_ortho_dt(state.T, X_ortho)

        return dy

    # ------------------------------------------------------------------
    # Outflow helpers
    # ------------------------------------------------------------------

    def _h_outflow(self, state: ResolvedState) -> float:
        """Molar enthalpy of gas leaving the tank, J/mol.

        Vent and fuel-cell feed lines draw from the ullage, so inside the
        vapour dome the withdrawn gas is *saturated vapour*; outside the
        dome the single-phase enthalpy at ``(p, T)`` applies.
        """
        if state.phase == "two_phase":
            return self._fluid.h_molar_saturated_vapor(state.T)
        return self._fluid.h_molar(state.p, state.T)

    def Q_boost(self, P_fc: float, T_cold: float) -> float:
        """Extra cold-side heat extraction from fuel-cell power, W.

        Uses the cryocooler's ``COP(T)`` when available.  Without a
        cryocooler (or with a COP-less model) the boost is zero — the
        fuel cell then only acts as a mass sink.
        """
        if self._cryo is None or P_fc <= 0.0:
            return 0.0
        cop = getattr(self._cryo, "COP", None)
        if cop is None:
            return 0.0
        return P_fc * cop(T_cold)

    # ------------------------------------------------------------------
    # Venting helpers
    # ------------------------------------------------------------------

    def _vent_rate(
        self,
        n: float,
        U: float,
        state: ResolvedState,
        Q_dot: float,
        n_dot_fc: float = 0.0,
    ) -> float:
        """Compute ṅ_vent [mol/s] from the isobaric constraint dp/dt = 0.

        From ``dp/dt = (∂p/∂n)_U · dn/dt + (∂p/∂U)_n · dU/dt = 0``
        and the ODE:  ``dn/dt = −(ṅ_v + ṅ_fc)``,
        ``dU/dt = Q̇ − (ṅ_v + ṅ_fc) · h_out``:

        .. math::

            \\dot{n}_{\\text{vent}} =
                \\frac{\\dot{Q} \\cdot (\\partial p / \\partial U)_n}
                     {(\\partial p / \\partial n)_U
                      + h_{\\text{out}} \\cdot (\\partial p / \\partial U)_n}
                - \\dot{n}_{\\text{fc}}

        i.e. any fuel-cell withdrawal directly reduces the vent rate
        required to hold the pressure.  Partial derivatives are computed
        with centred finite differences.  ``Q_dot`` here is the *net*
        power (heat_leak − cryocooler − catalyst − boost).
        """
        r = self._resolver

        eps_n = max(abs(n) * 1e-5, 1e-2)
        eps_U = max(abs(U) * 1e-5, 1.0)

        try:
            dp_dn = (
                r.resolve(n + eps_n, U, T_guess=state.T).p
                - r.resolve(n - eps_n, U, T_guess=state.T).p
            ) / (2.0 * eps_n)

            dp_dU = (
                r.resolve(n, U + eps_U, T_guess=state.T).p
                - r.resolve(n, U - eps_U, T_guess=state.T).p
            ) / (2.0 * eps_U)
        except Exception:
            return 0.0

        h_out = self._h_outflow(state)
        denom = dp_dn + h_out * dp_dU

        if denom <= 0.0:
            return 0.0
        return max(0.0, Q_dot * dp_dU / denom - n_dot_fc)

    # ------------------------------------------------------------------
    # Event functions for solve_ivp
    # ------------------------------------------------------------------

    def event_pressure_target(self, p_target: float):
        """Return a terminal event function that fires when ``p == p_target``."""
        resolver = self._resolver

        def _event(t: float, y: np.ndarray) -> float:
            try:
                state = resolver.resolve(float(y[0]), float(y[1]))
                return state.p - p_target
            except Exception:
                # Keep the event on the "not yet reached" side so a
                # transient resolve failure cannot spuriously terminate
                # the integration (returning +1 would cross zero).
                return -1.0

        _event.terminal  = True
        _event.direction = 1.0
        return _event

    def event_n_below(self, n_target: float):
        """Return a terminal event function that fires when ``n_total == n_target``."""

        def _event(t: float, y: np.ndarray) -> float:
            return float(y[0]) - n_target

        _event.terminal  = True
        _event.direction = -1.0
        return _event

    def event_pressure_drop(self, p_target: float):
        """Return a terminal event function that fires when pressure drops to ``p_target``.

        Useful for stopping a cryocooler rundown once the tank has cooled
        to the setpoint.
        """
        resolver = self._resolver

        def _event(t: float, y: np.ndarray) -> float:
            try:
                state = resolver.resolve(float(y[0]), float(y[1]))
                return state.p - p_target
            except Exception:
                return -1.0

        _event.terminal  = True
        _event.direction = -1.0
        return _event
