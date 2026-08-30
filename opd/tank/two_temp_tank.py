"""Two-Temperature lumped-parameter tank.

Extends the 1-Temp :class:`~opd.tank.tank.Tank` to track the sorbent
energy as an independent ODE state variable.

ODE state vector
----------------
``y = [n_H2_total (mol),  U_total (J),  H_skel (J)]``

where :math:`H_{\\mathrm{skel}} = m_{\\mathrm{sorb}}\\,\\int_0^{T_{\\mathrm{sorb}}} c_p(T)\\,\\mathrm{d}T`
is the skeleton enthalpy measured from 0 K.  Using enthalpy rather than
temperature as the state variable eliminates the stiffness singularity that
arises from dividing by the cryogenic heat capacity
:math:`C_{\\mathrm{sorb}} = m\\,c_p(T) \\to 0` as :math:`T \\to 0`.

Right-hand side
---------------

.. math::

    \\frac{\\mathrm{d}n}{\\mathrm{d}t}
        &= -\\dot{n}_{\\mathrm{vent}}                                        \\\\

    \\frac{\\mathrm{d}U_{\\mathrm{total}}}{\\mathrm{d}t}
        &= \\dot{Q}_{\\mathrm{leak}}
          - \\dot{Q}_{\\mathrm{cryo}}
          - \\dot{n}_{\\mathrm{vent}} h_{\\mathrm{vent}}                      \\\\

    \\frac{\\mathrm{d}H_{\\mathrm{skel}}}{\\mathrm{d}t}
        &= -Q_{\\mathrm{HX}} - \\dot{Q}_{\\mathrm{cryo}}

where :math:`Q_{\\mathrm{HX}} = (UA)_{\\mathrm{sf}}(T_{\\mathrm{sorb}} - T_{\\mathrm{fluid}})`
and :math:`T_{\\mathrm{sorb}}` is recovered from :math:`H_{\\mathrm{skel}}`
by numerically inverting the enthalpy integral.

The heat leak :math:`\\dot{Q}_{\\mathrm{leak}}` is assumed to enter the bulk
fluid directly.

Limiting behaviour
------------------
* ``UA_sf → ∞``: :math:`T_{\\mathrm{sorb}} → T_{\\mathrm{fluid}}` → 1-Temp
  equilibrium.
* ``UA_sf = 0``: sorbent and fluid are thermally isolated.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..adsorbents.activated_carbon import cp_cryo_j_kg_k
from ..adsorbents.base import AdsorbentMaterial
from ..fluids.fluid_properties import FluidProperties
from ..simulation.two_temp_resolver import TwoTempResolver, TwoTempState
from .geometry import TankGeometry
from .heat_loads import HeatLeakModel

if TYPE_CHECKING:
    from typing import Callable
    from ..catalysts.base import ParaOrthoCatalyst
    from ..control.pressure_control import PressureController
    from ..cryocooler.base import CryocoolerModel

__all__ = ["TwoTempTank"]


class TwoTempTank:
    """Lumped-parameter hydrogen tank with separate sorbent temperature.

    Parameters
    ----------
    fluid
        Fluid property provider.
    geometry
        Tank geometry.
    heat_leak
        External thermal power into the *bulk fluid*, W.
    adsorbent
        Sorbent material (required — 2-Temp model only makes sense with a
        sorbent bed).
    m_sorb
        Sorbent mass, kg.
    UA_sf
        Sorbent-to-fluid heat-transfer coefficient × area product, W/K.
        Large values approach the 1-Temp limit.
    cryocooler
        Heat-extraction device (cools the sorbent).  ``None`` = no active
        cooling.
    controller
        Pressure controller.  Defaults to ``AlwaysOnController`` when a
        cryocooler is provided without an explicit controller.
    p_vent
        Relief-valve set pressure, Pa.  ``None`` disables venting.
    T_skel_ref
        Reference temperature for the skeleton enthalpy, K.
    """

    def __init__(
        self,
        fluid: FluidProperties,
        geometry: TankGeometry,
        heat_leak: HeatLeakModel,
        adsorbent: AdsorbentMaterial,
        m_sorb: float,
        UA_sf: float = 500.0,
        cryocooler: Optional["CryocoolerModel"] = None,
        controller: Optional["PressureController"] = None,
        p_vent: Optional[float] = None,
        T_skel_ref: float = 0.0,
        cp_sorb_fn: Optional["Callable[[float], float]"] = None,
    ) -> None:
        if m_sorb <= 0.0:
            raise ValueError("TwoTempTank requires m_sorb > 0")
        if cryocooler is not None and controller is None:
            from ..control.pressure_control import AlwaysOnController
            controller = AlwaysOnController()

        self._fluid     = fluid
        self._geometry  = geometry
        self._heat_leak = heat_leak
        self._ads       = adsorbent
        self._m_sorb    = m_sorb
        self._UA_sf     = UA_sf
        self._cryo      = cryocooler
        self._ctrl      = controller
        self._p_vent    = p_vent

        V_free = geometry.free_volume(m_sorb=m_sorb, adsorbent=adsorbent)
        _cp_fn = cp_sorb_fn if cp_sorb_fn is not None else cp_cryo_j_kg_k
        self._resolver = TwoTempResolver(
            fluid=fluid,
            V_free=V_free,
            adsorbent=adsorbent,
            m_sorb=m_sorb,
            T_skel_ref=T_skel_ref,
            cp_sorb_fn=_cp_fn,
        )
        self._T_fl_prev: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def resolver(self) -> TwoTempResolver:
        return self._resolver

    @property
    def fluid(self) -> FluidProperties:
        return self._fluid

    @property
    def p_vent(self) -> Optional[float]:
        return self._p_vent

    @property
    def state_size(self) -> int:
        return 3

    # ------------------------------------------------------------------
    # Initial-state helpers
    # ------------------------------------------------------------------

    def _T_sorb_from_y(self, y: np.ndarray) -> float:
        """Recover sorbent temperature from the H_skel state variable."""
        return self._resolver.T_sorb_from_H_skel(float(y[2]))

    def initial_state_two_phase(
        self,
        T_fluid: float,
        Q_bulk: float,
        T_sorb: Optional[float] = None,
    ) -> np.ndarray:
        """Encode a two-phase start state.

        Parameters
        ----------
        T_fluid
            Initial fluid temperature (= saturation temperature), K.
        Q_bulk
            Initial bulk vapour quality.
        T_sorb
            Initial sorbent temperature, K.  Defaults to ``T_fluid``
            (1-Temp start).

        Returns
        -------
        numpy.ndarray
            State vector ``[n_total, U_total, H_skel]``.
        """
        if T_sorb is None:
            T_sorb = T_fluid
        n, U  = self._resolver.encode_two_phase(T_fluid, Q_bulk, T_sorb)
        H_skel = self._resolver._H_skel(T_sorb)
        return np.array([n, U, H_skel])

    def initial_state_single_phase(
        self,
        T_fluid: float,
        p: float,
        T_sorb: Optional[float] = None,
    ) -> np.ndarray:
        """Encode a single-phase start state.

        Returns ``[n_total, U_total, H_skel]``.
        """
        if T_sorb is None:
            T_sorb = T_fluid
        n, U   = self._resolver.encode_single_phase(T_fluid, p, T_sorb)
        H_skel = self._resolver._H_skel(T_sorb)
        return np.array([n, U, H_skel])

    def resolve_state(self, y: np.ndarray) -> TwoTempState:
        """Resolve the full thermodynamic state from a state-vector."""
        T_sorb = self._T_sorb_from_y(y)
        return self._resolver.resolve(
            float(y[0]), float(y[1]), T_sorb,
            T_fl_guess=self._T_fl_prev,
        )

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE right-hand side ``dy/dt = f(t, y)``.

        State: ``[n_total, U_total, H_skel]``.
        H_skel = m_sorb * ∫₀^{T_sorb} cp(T) dT  (enthalpy of sorbent skeleton)
        """
        n_total = float(y[0])
        U_total = float(y[1])
        H_skel  = float(y[2])

        # Recover T_sorb from enthalpy (monotone inversion, no stiffness issue)
        T_sorb = self._resolver.T_sorb_from_H_skel(H_skel)

        state = self._resolver.resolve(
            n_total, U_total, T_sorb, T_fl_guess=self._T_fl_prev
        )
        self._T_fl_prev = state.T_fluid

        T_fluid = state.T_fluid

        # --- sorbent–fluid heat exchange ---
        Q_HX = self._UA_sf * (T_sorb - T_fluid)  # + if sorb hotter than fluid

        # --- heat leak into bulk fluid ---
        Q_leak = self._heat_leak.Q_dot(t, T_fluid)

        # --- cryocooler (cools sorbent) ---
        Q_cryo = 0.0
        if self._cryo is not None:
            duty   = self._ctrl.duty(t, state.p)
            Q_cryo = self._cryo.Q_cryo(t, T_sorb) * duty

        # --- sorbent enthalpy rate (no division by C_sorb → no stiffness) ---
        dH_skel = -Q_HX - Q_cryo

        # --- venting ---
        Q_dot_net = Q_leak - Q_cryo
        if self._p_vent is not None and state.p >= self._p_vent:
            n_dot_vent = self._vent_rate(
                n_total, U_total, T_sorb, state, Q_dot_net
            )
        else:
            n_dot_vent = 0.0

        h_vent = (
            self._fluid.h_molar(state.p, T_fluid) if n_dot_vent > 0.0 else 0.0
        )

        return np.array([
            -n_dot_vent,
            Q_leak - Q_cryo - n_dot_vent * h_vent,
            dH_skel,
        ])

    # ------------------------------------------------------------------
    # Venting helpers
    # ------------------------------------------------------------------

    def _vent_rate(
        self,
        n: float,
        U: float,
        T_sorb: float,
        state: TwoTempState,
        Q_dot: float,
    ) -> float:
        """Algebraic isobaric venting rate from dp/dt = 0 (2-Temp version).

        T_sorb is the *resolved* sorbent temperature (already inverted from H_skel).
        """
        r = self._resolver

        eps_n = max(abs(n) * 1e-5, 1e-2)
        eps_U = max(abs(U) * 1e-5, 1.0)

        try:
            dp_dn = (
                r.resolve(n + eps_n, U, T_sorb).p
                - r.resolve(n - eps_n, U, T_sorb).p
            ) / (2.0 * eps_n)

            dp_dU = (
                r.resolve(n, U + eps_U, T_sorb).p
                - r.resolve(n, U - eps_U, T_sorb).p
            ) / (2.0 * eps_U)
        except Exception:
            return 0.0

        h_vent = self._fluid.h_molar(state.p, state.T_fluid)
        denom  = dp_dn + h_vent * dp_dU
        if denom <= 0.0:
            return 0.0
        return max(0.0, Q_dot * dp_dU / denom)

    # ------------------------------------------------------------------
    # Event functions for solve_ivp
    # ------------------------------------------------------------------

    def event_pressure_target(self, p_target: float):
        """Terminal event: fire when pressure rises to ``p_target``.

        Uses a *cache-free* resolve so that repeated event-function calls
        during scipy's root-finding step do not corrupt the ODE warm-start.
        """
        resolver = self._resolver

        def _resolve_nocache(y):
            """Resolve without updating the resolver's warm-start cache."""
            saved_T  = resolver._cache_T_fl
            saved_lp = resolver._cache_lnp
            try:
                T_s = resolver.T_sorb_from_H_skel(float(y[2]))
                s   = resolver.resolve(float(y[0]), float(y[1]), T_s)
            finally:
                resolver._cache_T_fl = saved_T
                resolver._cache_lnp  = saved_lp
            return s

        def _event(t: float, y: np.ndarray) -> float:
            try:
                return _resolve_nocache(y).p - p_target
            except Exception:
                return -abs(p_target) * 0.1

        _event.terminal  = True
        _event.direction = 1.0
        return _event

    def event_pressure_drop(self, p_target: float):
        """Terminal event: fire when pressure drops to ``p_target``."""
        resolver = self._resolver

        def _resolve_nocache(y):
            saved_T  = resolver._cache_T_fl
            saved_lp = resolver._cache_lnp
            try:
                T_s = resolver.T_sorb_from_H_skel(float(y[2]))
                s   = resolver.resolve(float(y[0]), float(y[1]), T_s)
            finally:
                resolver._cache_T_fl = saved_T
                resolver._cache_lnp  = saved_lp
            return s

        def _event(t: float, y: np.ndarray) -> float:
            try:
                return _resolve_nocache(y).p - p_target
            except Exception:
                return abs(p_target) * 0.1

        _event.terminal  = True
        _event.direction = -1.0
        return _event

    def event_n_below(self, n_target: float):
        """Terminal event: fire when n_total drops to ``n_target``."""
        def _event(t: float, y: np.ndarray) -> float:
            return float(y[0]) - n_target

        _event.terminal  = True
        _event.direction = -1.0
        return _event
