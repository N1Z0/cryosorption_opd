"""Transient simulator: wraps ``scipy.integrate.solve_ivp``.

:class:`TransientSimulator` integrates the tank ODE, post-processes the
raw output into a :class:`~opd.simulation.results.SimulationResult`, and
computes the conservation report automatically.

It handles both the **1-Temp** state vector ``[n, U]`` (or ``[n, U, X_ortho]``
when a para-ortho catalyst is attached) and the **2-Temp** state vector
``[n, U, T_sorb]`` produced by :class:`~opd.tank.two_temp_tank.TwoTempTank`.
The tank's ``resolve_state(y)`` method is used for post-processing so the
simulator does not need to know which model is in use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from ..tank.tank import Tank
    from ..tank.two_temp_tank import TwoTempTank

from .results import ConservationReport, SimulationResult

__all__ = ["TransientSimulator"]

_DEFAULT_SOLVER_KWARGS: dict[str, Any] = {
    "method": "BDF",
    "rtol": 1e-8,
    "atol": 1e-6,
    "dense_output": False,
}

AnyTank = Union["Tank", "TwoTempTank"]


class TransientSimulator:
    """Integrate the tank ODE and produce a :class:`~opd.simulation.results.SimulationResult`.

    Parameters
    ----------
    tank
        A :class:`~opd.tank.tank.Tank` or
        :class:`~opd.tank.two_temp_tank.TwoTempTank` instance.
    solver_kwargs
        Extra keyword arguments merged into the ``solve_ivp`` call.
        Defaults: ``method='BDF'``, ``rtol=1e-8``, ``atol=1e-6``.
    """

    def __init__(
        self,
        tank: AnyTank,
        solver_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._tank = tank
        self._solver_kwargs = dict(_DEFAULT_SOLVER_KWARGS)
        if solver_kwargs:
            self._solver_kwargs.update(solver_kwargs)

    def run(
        self,
        y0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        n_points: int = 500,
        events: Optional[list] = None,
    ) -> SimulationResult:
        """Run the transient simulation.

        Parameters
        ----------
        y0
            Initial state vector.
        t_span
            ``(t_start, t_end)`` in seconds.
        t_eval
            Explicit time array for output.  If ``None``, ``n_points``
            evenly-spaced points over ``t_span`` are used.
        n_points
            Number of output points when ``t_eval`` is not supplied.
        events
            List of event functions for ``solve_ivp``.

        Returns
        -------
        SimulationResult
        """
        tank  = self._tank
        t0, t1 = float(t_span[0]), float(t_span[1])

        if t_eval is None:
            t_eval = np.linspace(t0, t1, n_points)

        kwargs = dict(self._solver_kwargs)
        if events is not None:
            kwargs["events"] = events

        # Reset warm-start caches
        for attr in ("_T_prev", "_p_prev", "_T_fl_prev"):
            if hasattr(tank, attr):
                setattr(tank, attr, None)

        sol = solve_ivp(
            fun=tank.ode_rhs,
            t_span=(t0, t1),
            y0=np.asarray(y0, dtype=float),
            t_eval=t_eval,
            **kwargs,
        )

        if not sol.success:
            raise RuntimeError(
                f"TransientSimulator: solve_ivp failed — {sol.message}"
            )

        # Append the terminal event point (see M3 errata in theory manual)
        t_raw = sol.t
        y_raw = sol.y
        if sol.status == 1:
            for t_ev_arr, y_ev_arr in zip(sol.t_events, sol.y_events):
                if len(t_ev_arr) > 0:
                    t_raw = np.append(t_raw, t_ev_arr[-1])
                    y_raw = np.hstack([y_raw, y_ev_arr[-1:].T])

        t_arr = t_raw
        n_arr = y_raw[0]
        U_arr = y_raw[1]
        N     = len(t_arr)

        # 3rd state slot: T_sorb (2-Temp) or X_ortho (1-Temp + catalyst), or absent
        y3_arr = y_raw[2] if y_raw.shape[0] > 2 else None

        # ------------------------------------------------------------------
        # Determine whether this is a 2-Temp run
        # ------------------------------------------------------------------
        from ..tank.two_temp_tank import TwoTempTank as _TT

        is_two_temp = isinstance(tank, _TT)

        # ------------------------------------------------------------------
        # Post-process: resolve thermodynamic state at each step
        # ------------------------------------------------------------------
        T_arr       = np.empty(N)
        T_sorb_arr  = np.full(N, float("nan"))
        p_arr       = np.empty(N)
        Q_vapor_arr = np.empty(N)
        phase_arr   = np.empty(N, dtype=object)
        n_bulk_arr  = np.empty(N)
        n_ads_arr   = np.empty(N)
        Q_dot_arr   = np.empty(N)
        Q_cryo_arr  = np.zeros(N)
        X_ortho_arr = np.full(N, float("nan"))
        n_dot_vent_arr = np.zeros(N)
        n_dot_fc_arr   = np.zeros(N)
        P_fc_arr       = np.zeros(N)
        H_out_fc_arr   = np.zeros(N)   # ṅ_fc · h_out, W (energy audit)

        fc = getattr(tank, "_fc", None)

        T_hint: Optional[float] = None

        for i in range(N):
            n_i = float(n_arr[i])
            U_i = float(U_arr[i])
            t_i = float(t_arr[i])

            if is_two_temp:
                H_skel_i = float(y3_arr[i]) if y3_arr is not None else 0.0
                T_sorb_i = tank.resolver.T_sorb_from_H_skel(H_skel_i)
                state = tank.resolver.resolve(
                    n_i, U_i, T_sorb_i, T_fl_guess=T_hint
                )
                T_hint        = state.T_fluid
                T_arr[i]      = state.T_fluid
                T_sorb_arr[i] = T_sorb_i
            else:
                state = tank.resolver.resolve(n_i, U_i, T_guess=T_hint)
                T_hint   = state.T
                T_arr[i] = state.T
                if y3_arr is not None:
                    X_ortho_arr[i] = float(y3_arr[i])

            p_arr[i]       = state.p
            Q_vapor_arr[i] = state.Q_vapor
            phase_arr[i]   = state.phase
            n_bulk_arr[i]  = state.n_bulk
            n_ads_arr[i]   = state.n_ads_abs

            # Net heat-leak power (before cryocooler, as recorded)
            Q_dot_arr[i] = tank._heat_leak.Q_dot(t_i, float(T_arr[i]))

            # Cryocooler instantaneous power
            cryo = getattr(tank, "_cryo", None)
            ctrl = getattr(tank, "_ctrl", None)
            if cryo is not None and ctrl is not None:
                T_cold = T_sorb_arr[i] if is_two_temp else T_arr[i]
                duty   = ctrl.duty(t_i, p_arr[i])
                Q_cryo_arr[i] = cryo.Q_cryo(t_i, T_cold) * duty

            # Fuel cell: gas draw + cryocooler subcooling boost
            if fc is not None:
                P_fc_i = fc.P_out(t_i)
                if P_fc_i > 0.0:
                    P_fc_arr[i]     = P_fc_i
                    n_dot_fc_arr[i] = fc.n_dot_H2(t_i)
                    boost_fn = getattr(tank, "Q_boost", None)
                    if boost_fn is not None:
                        Q_cryo_arr[i] += boost_fn(P_fc_i, float(T_arr[i]))
                    h_out_fn = getattr(tank, "_h_outflow", None)
                    if h_out_fn is not None:
                        H_out_fc_arr[i] = n_dot_fc_arr[i] * h_out_fn(state)

        # Cumulative net heat (heat_leak - cryocooler - fuel-cell enthalpy draw)
        Q_net_arr = Q_dot_arr - Q_cryo_arr - H_out_fc_arr
        Q_acc_arr = np.zeros(N)
        if N > 1:
            dt    = np.diff(t_arr)
            Q_mid = 0.5 * (Q_net_arr[:-1] + Q_net_arr[1:])
            Q_acc_arr[1:] = np.cumsum(dt * Q_mid)

        # ------------------------------------------------------------------
        # Conservation report
        # ------------------------------------------------------------------
        n0 = n_arr[0]
        # Expected inventory: n0 minus the integrated fuel-cell draw
        # (venting is intentionally excluded — vented runs report the
        # apparent drift, exactly as before).
        n_expected = np.full(N, n0)
        if N > 1 and fc is not None:
            dt_fc  = np.diff(t_arr)
            fc_mid = 0.5 * (n_dot_fc_arr[:-1] + n_dot_fc_arr[1:])
            n_expected[1:] = n0 - np.cumsum(dt_fc * fc_mid)
        if n0 != 0.0:
            mass_err = float(np.max(np.abs(n_arr - n_expected)) / abs(n0))
        else:
            mass_err = 0.0

        dU   = U_arr - U_arr[0]
        mask = np.abs(Q_acc_arr) > 1.0
        if mask.any():
            energy_err = float(
                np.max(np.abs((dU - Q_acc_arr)[mask]) / np.abs(Q_acc_arr[mask]))
            )
        else:
            energy_err = 0.0

        conservation = ConservationReport(
            max_mass_error_rel=float(mass_err),
            max_energy_error_rel=float(energy_err),
        )

        return SimulationResult(
            t=t_arr,
            n_total=n_arr,
            U_total=U_arr,
            T=T_arr,
            p=p_arr,
            Q_vapor=Q_vapor_arr,
            phase=phase_arr,
            n_bulk=n_bulk_arr,
            n_ads_abs=n_ads_arr,
            Q_accumulated=Q_acc_arr,
            Q_dot=Q_dot_arr,
            n_dot_vent=n_dot_vent_arr,
            conservation=conservation,
            T_sorb=T_sorb_arr,
            Q_cryo=Q_cryo_arr,
            X_ortho=X_ortho_arr,
            n_dot_fc=n_dot_fc_arr,
            P_fc=P_fc_arr,
        )
