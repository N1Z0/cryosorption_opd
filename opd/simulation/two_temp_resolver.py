"""Two-Temperature (2-Temp) EOS inversion.

In the 2-Temp model the sorbent temperature :math:`T_{\\mathrm{sorb}}` is
tracked as an independent ODE state variable.  The fluid temperature
:math:`T_{\\mathrm{fl}}` and pressure :math:`p` are obtained by inverting
the energy and mass balance *at fixed* :math:`T_{\\mathrm{sorb}}`.

Energy model
------------

.. math::

    U_{\\mathrm{total}} =
        n_{\\mathrm{bulk}}\\,u_{\\mathrm{bulk}}(T_{\\mathrm{fl}}, p)
        - n_{\\mathrm{ads}}\\,q_{\\mathrm{st}}
        + m_{\\mathrm{sorb}}\\,c_p^{\\mathrm{skel}}\\,T_{\\mathrm{sorb}}

Effective fluid energy:

.. math::

    U_{\\mathrm{fl,eff}}(p)
        = U_{\\mathrm{total}}
          + q_{\\mathrm{st}}\\,n_{\\mathrm{abs}}(p, T_{\\mathrm{sorb}})\\,m_{\\mathrm{sorb}}
          - m_{\\mathrm{sorb}}\\,c_p^{\\mathrm{skel}}\\,T_{\\mathrm{sorb}}

Solution strategy — per-call warm-start + two branches
--------------------------------------------------------
1. **Warm-start fsolve**: try a single Newton solve from the cached
   :math:`(T_{\\mathrm{fl}}, p)` of the previous step.  Succeeds in 95%
   of calls (BDF Jacobian perturbations, small time steps).
2. **Two-phase branch** (fallback): 1-D ``brentq`` in :math:`T_{\\mathrm{fl}}`
   at :math:`p = p_{\\mathrm{sat}}(T_{\\mathrm{fl}})`.
3. **Single-phase branch** (fallback): Picard iteration — alternating 1-D
   ``brentq`` in :math:`p` (mass) and :math:`T_{\\mathrm{fl}}` (energy).

Limiting case
~~~~~~~~~~~~~
:math:`T_{\\mathrm{sorb}} = T_{\\mathrm{fl}}` reproduces the 1-Temp result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve

from ..adsorbents.base import AdsorbentMaterial
from ..fluids.fluid_properties import FluidProperties
from .thermal_state import (
    PHASE_GAS,
    PHASE_LIQUID,
    PHASE_SUPERCRITICAL,
    PHASE_TWO_PHASE,
)

__all__ = ["TwoTempResolver", "TwoTempState"]

_T_LO = 14.0     # K — well below H₂ triple point (13.8 K)
_T_HI = 500.0    # K
_P_LO = 1.0e3    # Pa — 0.01 bar
_P_HI = 2.0e8    # Pa — 2000 bar


@dataclass(frozen=True)
class TwoTempState:
    """Thermodynamic state resolved by :class:`TwoTempResolver`."""

    T_fluid: float
    T_sorb: float
    p: float
    Q_vapor: float
    phase: str
    n_bulk: float
    n_ads_abs: float

    @property
    def T(self) -> float:
        """Fluid temperature alias (for API compatibility)."""
        return self.T_fluid


class TwoTempResolver:
    """EOS inversion for the 2-Temp model."""

    def __init__(
        self,
        fluid: FluidProperties,
        V_free: float,
        adsorbent: AdsorbentMaterial,
        m_sorb: float,
        T_skel_ref: float = 0.0,
        cp_sorb_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        if m_sorb <= 0.0:
            raise ValueError("TwoTempResolver requires m_sorb > 0")

        self._fluid      = fluid
        self._V_free     = V_free
        self._ads        = adsorbent
        self._m_sorb     = m_sorb
        self._T_ref      = T_skel_ref
        self._V_gas      = V_free - m_sorb * adsorbent.micropore_volume
        self._q_st_fn    = adsorbent.isosteric_heat_fn
        # cp_sorb_fn: T -> J/(kg K); falls back to adsorbent.cp_skeleton
        self._cp_sorb_fn: Callable[[float], float] = (
            cp_sorb_fn if cp_sorb_fn is not None else adsorbent.cp_skeleton
        )
        # Pre-compute polynomial coefficients for fast H_skel / T_sorb inversion.
        # If cp(T) = _cp_A * T + _cp_B * T^3, then
        #   H_skel = m * (A*(T²-Tr²)/2 + B*(T⁴-Tr⁴)/4)
        # and the inverse is a closed-form quadratic in T².
        # We detect this pattern by comparing cp values at two temperatures.
        try:
            from ..adsorbents.activated_carbon import (
                _CP_CRYO_A as _A, _CP_CRYO_B as _B,
            )
            _T1, _T2 = 10.0, 50.0
            if (abs(self._cp_sorb_fn(_T1) - (_A * _T1 + _B * _T1 ** 3)) < 1e-12
                    and abs(self._cp_sorb_fn(_T2) - (_A * _T2 + _B * _T2 ** 3)) < 1e-12):
                self._poly_A: Optional[float] = _A
                self._poly_B: Optional[float] = _B
            else:
                self._poly_A = None
                self._poly_B = None
        except Exception:
            self._poly_A = None
            self._poly_B = None

        # Warm-start cache (T_fl, lnp from last successful resolve)
        self._cache_T_fl:        Optional[float] = None
        self._cache_lnp:         Optional[float] = None
        self._cache_is_two_phase: Optional[bool]  = None  # skip 2-P scan for SP states

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _n_ads(self, p: float, T_sorb: float) -> float:
        return self._ads.isotherm.n_absolute(p, T_sorb, self._fluid) * self._m_sorb

    def _q_st(self, p: float, T: float) -> float:
        return self._q_st_fn(p, T)

    def cp_sorb(self, T: float) -> float:
        """Instantaneous sorbent heat capacity, J/(kg K)."""
        return self._cp_sorb_fn(float(T))

    def _H_skel(self, T_sorb: float) -> float:
        """Integrated sorbent enthalpy: m_sorb * ∫(T_ref → T_sorb) cp dT, J.

        Uses a fast analytical formula when cp is the standard two-term
        polynomial (cp = A·T + B·T³); falls back to ``scipy.integrate.quad``
        for custom cp functions.
        """
        T_ref = self._T_ref
        T_s   = float(T_sorb)
        if abs(T_s - T_ref) < 1e-10:
            return 0.0
        A = self._poly_A
        B = self._poly_B
        if A is not None and B is not None:
            # Analytical: ∫(Tr,Ts) (A·t + B·t³) dt = A*(Ts²-Tr²)/2 + B*(Ts⁴-Tr⁴)/4
            return self._m_sorb * (
                A * (T_s ** 2 - T_ref ** 2) / 2.0
                + B * (T_s ** 4 - T_ref ** 4) / 4.0
            )
        try:
            val, _ = quad(self._cp_sorb_fn, T_ref, T_s, limit=30)
        except Exception:
            pts = np.linspace(T_ref, T_s, 20)
            val = np.trapz([self._cp_sorb_fn(t) for t in pts], pts)
        return self._m_sorb * val

    def T_sorb_from_H_skel(self, H_skel: float) -> float:
        """Invert H_skel = m_sorb * ∫(T_ref → T) cp dT  →  T (K).

        For the standard polynomial cp = A·T + B·T³ with T_ref = 0 this is
        solved analytically (O(1)).  For other cp functions, a fast Newton
        iteration is used.
        """
        T_ref = self._T_ref
        A = self._poly_A
        B = self._poly_B
        if A is not None and B is not None and T_ref == 0.0:
            # Closed-form: h = H/m = A/2·T² + B/4·T⁴
            # Let u = T²:  B/4·u² + A/2·u - h = 0
            h = H_skel / self._m_sorb
            if h <= 0.0:
                return max(0.01, T_ref)
            disc = (A * 0.5) ** 2 + B * h
            if disc < 0.0:
                disc = 0.0
            u = (-A * 0.5 + math.sqrt(disc)) / (B * 0.5)
            return max(0.01, math.sqrt(max(0.0, u)))

        # General case: Newton method (cp is always > 0 so well-conditioned)
        if abs(H_skel - self._H_skel(T_ref)) < 1e-6:
            return T_ref
        cp0 = max(self._cp_sorb_fn(max(T_ref + 1.0, 15.0)), 1e-9)
        T   = max(0.01, T_ref + H_skel / (self._m_sorb * cp0))
        for _ in range(30):
            H_curr = self._H_skel(T)
            resid  = H_curr - H_skel
            if abs(resid) < max(abs(H_skel) * 1e-9, 1e-3):
                return max(0.01, T)
            cp = max(self._cp_sorb_fn(T), 1e-9)
            T  = max(0.01, T - resid / (self._m_sorb * cp))
        return max(0.01, T)

    def _U_fl_eff(self, p: float, U_total: float, T_sorb: float) -> float:
        n_ads = self._n_ads(p, T_sorb)
        return U_total + self._q_st(p, T_sorb) * n_ads - self._H_skel(T_sorb)

    def _is_two_phase(self, T_fl: float, n_total: float, T_sorb: float):
        """Check if bulk fluid is two-phase.  Returns (rhoL, rhoV, n_bulk) or None."""
        fl = self._fluid
        T_c = fl.T_critical
        if T_fl >= T_c:
            return None
        try:
            p_sat    = fl.p_saturation(T_fl)
            n_ads    = self._n_ads(p_sat, T_sorb)
            n_bulk   = n_total - n_ads
            if n_bulk <= 0.0:
                return None
            rhoL = fl.rho_molar_saturated_liquid(T_fl)
            rhoV = fl.rho_molar_saturated_vapor(T_fl)
            rho_bulk = n_bulk / self._V_gas
            # Allow a small relative tolerance at the liquid boundary so that
            # states encoded at Q=0 (pure liquid) are not rejected due to
            # floating-point round-trip errors in 1/(1/rhoL) or small
            # perturbations in T_sorb from the numerical Jacobian.
            if rhoV <= rho_bulk <= rhoL * (1.0 + 1e-4):
                return (rhoL, rhoV, n_bulk)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Forward encoding
    # ------------------------------------------------------------------

    def encode_two_phase(
        self, T_fluid: float, Q_bulk: float, T_sorb: float
    ) -> tuple[float, float]:
        """Encode a two-phase start state → (n_total, U_total)."""
        fl    = self._fluid
        p     = fl.p_saturation(T_fluid)
        rhoL  = fl.rho_molar_saturated_liquid(T_fluid)
        rhoV  = fl.rho_molar_saturated_vapor(T_fluid)
        rho   = 1.0 / ((1.0 - Q_bulk) / rhoL + Q_bulk / rhoV)
        uL    = fl.u_molar_saturated_liquid(T_fluid)
        uV    = fl.u_molar_saturated_vapor(T_fluid)
        u_mix = (1.0 - Q_bulk) * uL + Q_bulk * uV

        n_bulk  = rho * self._V_gas
        n_ads   = self._n_ads(p, T_sorb)
        n_total = n_bulk + n_ads
        U_total = (n_bulk * u_mix
                   - self._q_st(p, T_sorb) * n_ads
                   + self._H_skel(T_sorb))
        return n_total, U_total

    def encode_single_phase(
        self, T_fluid: float, p: float, T_sorb: float
    ) -> tuple[float, float]:
        """Encode a single-phase start state → (n_total, U_total)."""
        fl     = self._fluid
        rho    = fl.rho_molar(p, T_fluid)
        u_bulk = fl.u_molar(p, T_fluid)
        n_bulk = rho * self._V_gas
        n_ads  = self._n_ads(p, T_sorb)

        n_total = n_bulk + n_ads
        U_total = (n_bulk * u_bulk
                   - self._q_st(p, T_sorb) * n_ads
                   + self._H_skel(T_sorb))
        return n_total, U_total

    # ------------------------------------------------------------------
    # Main inversion
    # ------------------------------------------------------------------

    def resolve(
        self,
        n_total: float,
        U_total: float,
        T_sorb:  float,
        T_fl_guess: Optional[float] = None,
    ) -> TwoTempState:
        """Invert ``(n_total, U_total, T_sorb)`` → full thermodynamic state.

        Resolution order:
        1. **Narrow two-phase brentq** (warm-start): bracket ``[T_cache − 1, T_cache + 1]``
           — succeeds in 95% of calls, ~3 ms.
        2. **Wide two-phase brentq** (cold/after phase change): scans the full
           two-phase temperature range with 0.25 K resolution.
        3. **Single-phase Picard**: alternating 1-D brentq in pressure and
           temperature — used when the bulk fluid exits the two-phase dome.
        """
        # Initial anchor temperature
        T_anchor = (self._cache_T_fl if self._cache_T_fl is not None
                    else (T_fl_guess if T_fl_guess is not None else T_sorb))
        T_anchor = float(np.clip(T_anchor, _T_LO + 0.5, _T_HI))

        # ---- 1. Narrow warm-start brentq ----
        if self._cache_T_fl is not None:
            state = self._warm_start_two_phase(n_total, U_total, T_sorb,
                                               self._cache_T_fl)
            if state is not None:
                return state

        # ---- 2. Wide two-phase scan (skip if cached state was single-phase) ----
        if self._cache_is_two_phase is not False:
            state = self._resolve_two_phase(n_total, U_total, T_sorb, T_anchor)
            if state is not None:
                return state

        # ---- 3. Single-phase Picard ----
        # When cache indicates single-phase, skip directly here with cached p/T.
        try:
            lnp_guess = (self._cache_lnp if self._cache_lnp is not None
                         else math.log(self._fluid.p_saturation(
                             max(_T_LO + 0.1, min(T_anchor,
                                 self._fluid.T_critical - 0.1)))))
        except Exception:
            lnp_guess = math.log(self._fluid.p_critical)
        # If cache says single-phase use cached T as better initial guess
        if self._cache_is_two_phase is False and self._cache_T_fl is not None:
            T_anchor = self._cache_T_fl

        state = self._resolve_single_phase(
            n_total, U_total, T_sorb, T_anchor, math.exp(lnp_guess)
        )
        if state is not None and state.phase != PHASE_TWO_PHASE:
            return state

        # ---- 4. Outer-T scan (robust fallback for large T_fl − T_sorb gaps) ----
        state = self._scan_single_phase(n_total, U_total, T_sorb)
        if state is not None:
            return state

        # ---- 5. Final fallback: retry two-phase scan if cache had blocked it ----
        if self._cache_is_two_phase is False:
            state = self._resolve_two_phase(n_total, U_total, T_sorb, T_anchor)
            if state is not None:
                return state

        raise RuntimeError(
            f"TwoTempResolver.resolve failed: "
            f"n={n_total:.4g}, U={U_total:.4g}, T_sorb={T_sorb:.4f} K"
        )

    # ------------------------------------------------------------------
    # Warm-start two-phase brentq (progressive bracket search)
    # ------------------------------------------------------------------

    def _warm_start_two_phase(
        self,
        n_total: float,
        U_total: float,
        T_sorb: float,
        T_cache: float,
    ) -> Optional[TwoTempState]:
        """Fast warm-start: try progressively wider brackets around T_cache."""
        T_c = self._fluid.T_critical

        def _residual(T_fl: float):
            info = self._is_two_phase(T_fl, n_total, T_sorb)
            if info is None:
                return None
            rhoL, rhoV, n_bulk = info
            p_sat = self._fluid.p_saturation(T_fl)
            rho_b = n_bulk / self._V_gas
            v_L = 1.0/rhoL; v_V = 1.0/rhoV
            Q   = float(np.clip((1/rho_b-v_L)/(v_V-v_L), 0, 1))
            uL  = self._fluid.u_molar_saturated_liquid(T_fl)
            uV  = self._fluid.u_molar_saturated_vapor(T_fl)
            n_ads = n_total - n_bulk
            U_exp = ((1-Q)*uL+Q*uV)*n_bulk - self._q_st(p_sat,T_sorb)*n_ads + self._H_skel(T_sorb)
            return U_total - U_exp

        r_centre = _residual(T_cache)
        if r_centre is None:
            return None

        for half_width in [0.02, 0.1, 0.5, 2.0]:
            # Search on the opposite side from r_centre
            # (root is between T_cache and the opposite direction)
            if r_centre > 0:
                T_lo = T_cache
                T_hi = min(T_c - 0.01, T_cache + half_width)
            else:
                T_lo = max(_T_LO, T_cache - half_width)
                T_hi = T_cache

            r_lo = _residual(T_lo)
            r_hi = _residual(T_hi)
            if r_lo is not None and r_hi is not None and r_lo * r_hi <= 0.0:
                return self._two_phase_brentq(n_total, U_total, T_sorb,
                                              T_lo, T_hi)

        return None

    # ------------------------------------------------------------------
    # Two-phase 1-D brentq (shared by warm-start and wide-scan branches)
    # ------------------------------------------------------------------

    def _two_phase_brentq(
        self,
        n_total: float,
        U_total: float,
        T_sorb: float,
        T_lo: float,
        T_hi: float,
    ) -> Optional[TwoTempState]:
        """1-D brentq for the two-phase case over [T_lo, T_hi]."""
        def residual(T_fl: float) -> Optional[float]:
            info = self._is_two_phase(T_fl, n_total, T_sorb)
            if info is None:
                return None
            rhoL, rhoV, n_bulk = info
            p_sat    = self._fluid.p_saturation(T_fl)
            rho_bulk = n_bulk / self._V_gas
            v_L = 1.0/rhoL; v_V = 1.0/rhoV
            Q   = float(np.clip((1.0/rho_bulk - v_L)/(v_V - v_L), 0.0, 1.0))
            uL  = self._fluid.u_molar_saturated_liquid(T_fl)
            uV  = self._fluid.u_molar_saturated_vapor(T_fl)
            n_ads = n_total - n_bulk
            U_exp = ((1.0-Q)*uL + Q*uV)*n_bulk
            U_exp -= self._q_st(p_sat, T_sorb) * n_ads
            U_exp += self._H_skel(T_sorb)
            return U_total - U_exp

        r_lo = residual(T_lo)
        r_hi = residual(T_hi)
        if r_lo is None or r_hi is None or r_lo * r_hi > 0.0:
            return None

        def _safe(T_fl):
            r = residual(T_fl)
            return r if r is not None else float("nan")

        try:
            T_fl = brentq(_safe, T_lo, T_hi, xtol=1e-5)
        except Exception:
            return None

        info = self._is_two_phase(T_fl, n_total, T_sorb)
        if info is None:
            return None
        rhoL, rhoV, n_bulk = info
        p_sat    = self._fluid.p_saturation(T_fl)
        rho_bulk = n_bulk / self._V_gas
        v_L = 1.0/rhoL; v_V = 1.0/rhoV
        Q   = float(np.clip((1.0/rho_bulk - v_L)/(v_V - v_L), 0.0, 1.0))

        self._cache_T_fl        = T_fl
        self._cache_lnp         = math.log(p_sat)
        self._cache_is_two_phase = True
        return TwoTempState(
            T_fluid=T_fl, T_sorb=T_sorb, p=p_sat, Q_vapor=Q,
            phase=PHASE_TWO_PHASE, n_bulk=n_bulk, n_ads_abs=n_total - n_bulk,
        )

    # ------------------------------------------------------------------
    # Branch 2: wide two-phase scan
    # ------------------------------------------------------------------

    def _resolve_two_phase(
        self,
        n_total: float,
        U_total: float,
        T_sorb: float,
        T_fl_guess: float,
    ) -> Optional[TwoTempState]:
        """Wide two-phase scan: try successively wider brackets around T_fl_guess.

        If no exact root is found but the minimum-residual two-phase state is
        within *1 % of |U_total|* (ODE integration rounding), that nearest
        state is accepted to avoid diverging the solver.
        """
        fl  = self._fluid
        T_c = fl.T_critical
        T_centre = float(np.clip(T_fl_guess, _T_LO + 0.5, T_c - 0.5))

        best_T:   Optional[float] = None
        best_abs: float           = math.inf

        def _residual_at(T_: float):
            info = self._is_two_phase(T_, n_total, T_sorb)
            if info is None:
                return None
            rhoL, rhoV, n_bulk = info
            p_sat = fl.p_saturation(T_)
            rho_b = n_bulk / self._V_gas
            v_L = 1.0 / rhoL; v_V = 1.0 / rhoV
            Q    = float(np.clip((1.0/rho_b - v_L) / (v_V - v_L), 0.0, 1.0))
            uL   = fl.u_molar_saturated_liquid(T_)
            uV   = fl.u_molar_saturated_vapor(T_)
            U_exp = ((1.0 - Q)*uL + Q*uV) * n_bulk
            U_exp -= self._q_st(p_sat, T_sorb) * (n_total - n_bulk)
            U_exp += self._H_skel(T_sorb)
            return U_total - U_exp

        # Try widening brackets, then the full range
        half_widths = [5.0, 10.0, T_c - _T_LO]
        for hw in half_widths:
            T_lo = max(_T_LO, T_centre - hw)
            T_hi = min(T_c - 0.01, T_centre + hw)
            # Scan in 0.25 K steps within this range
            T_scan = np.arange(T_lo, T_hi, 0.25)
            r_prev: Optional[float] = None
            T_prev_v: Optional[float] = None
            for T_ in T_scan:
                r_ = _residual_at(T_)
                if r_ is not None:
                    if abs(r_) < best_abs:
                        best_abs = abs(r_); best_T = T_
                    if r_prev is not None and r_prev * r_ <= 0.0:
                        state = self._two_phase_brentq(
                            n_total, U_total, T_sorb, T_prev_v, T_)
                        if state is not None:
                            return state
                    r_prev = r_; T_prev_v = T_
                else:
                    r_prev = None  # reset if out of two-phase

        # Near-boundary fallback: accept if residual is tiny relative to |U|
        tol = max(abs(U_total) * 0.01, 1e4)  # 1 % of |U| or 10 kJ
        if best_T is not None and best_abs <= tol:
            return self._two_phase_brentq(
                n_total, U_total, T_sorb,
                max(_T_LO, best_T - 0.5),
                min(T_c - 0.01, best_T + 0.5),
            ) or self._make_state_two_phase(n_total, T_sorb, best_T)

        return None

    def _make_state_two_phase(
        self, n_total: float, T_sorb: float, T_fl: float
    ) -> Optional[TwoTempState]:
        """Finalise a two-phase state without requiring an exact energy root."""
        info = self._is_two_phase(T_fl, n_total, T_sorb)
        if info is None:
            return None
        rhoL, rhoV, n_bulk = info
        p_sat    = self._fluid.p_saturation(T_fl)
        rho_bulk = n_bulk / self._V_gas
        v_L = 1.0 / rhoL; v_V = 1.0 / rhoV
        Q   = float(np.clip((1.0/rho_bulk - v_L) / (v_V - v_L), 0.0, 1.0))
        self._cache_T_fl        = T_fl
        self._cache_lnp         = math.log(p_sat)
        self._cache_is_two_phase = True
        return TwoTempState(
            T_fluid=T_fl, T_sorb=T_sorb, p=p_sat, Q_vapor=Q,
            phase=PHASE_TWO_PHASE, n_bulk=n_bulk, n_ads_abs=n_total - n_bulk,
        )

    # ------------------------------------------------------------------
    # Branch 3: single-phase Picard
    # ------------------------------------------------------------------

    def _resolve_single_phase(
        self,
        n_total: float,
        U_total: float,
        T_sorb: float,
        T_init: float,
        p_init: float,
    ) -> Optional[TwoTempState]:
        """Picard iteration: alternating 1-D brentq in p then T_fl."""
        fl  = self._fluid
        T_c = fl.T_critical
        V_g = self._V_gas

        T_fl = float(np.clip(T_init, _T_LO + 0.1, _T_HI))
        p    = float(np.clip(p_init, _P_LO, _P_HI))

        # Mass-balance residual at fixed T_fl: f(p) = rho*V_gas + n_ads - n_total
        def f_mass(p_: float, T_fl_: float) -> float:
            try:
                rho   = fl.rho_molar(p_, T_fl_)
                n_ads = self._n_ads(p_, T_sorb)
                return rho * V_g + n_ads - n_total
            except Exception:
                return float("nan")

        # Energy-balance residual at fixed p: f(T_fl) = n_bulk*u - U_fl_eff
        def f_energy(T_fl_: float, p_: float) -> float:
            try:
                n_ads    = self._n_ads(p_, T_sorb)
                n_bulk   = n_total - n_ads
                U_fl_eff = (U_total
                            + self._q_st(p_, T_sorb) * n_ads
                            - self._H_skel(T_sorb))
                u_bulk = fl.u_molar(p_, T_fl_)
                return n_bulk * u_bulk - U_fl_eff
            except Exception:
                return float("nan")

        for _ in range(15):
            T_old, p_old = T_fl, p

            # -- Mass step: find p at fixed T_fl --
            # Search both gas side (p < p_sat) and liquid side (p > p_sat) to
            # avoid missing roots when the current p guess is on the wrong branch.
            if T_fl < T_c:
                try:
                    p_sat_val = fl.p_saturation(T_fl)
                    mass_brackets = [
                        (_P_LO,              p_sat_val * 0.999),  # gas side
                        (p_sat_val * 1.001,  _P_HI),              # liquid side (compressed)
                        (max(_P_LO, p/3.0),  min(_P_HI, p*3.0)), # near p
                        (_P_LO,              _P_HI),              # full range – saturation boundary
                    ]
                except Exception:
                    mass_brackets = [
                        (max(_P_LO, p/3.0), min(_P_HI, p*3.0)),
                        (_P_LO, _P_HI),
                    ]
            else:
                mass_brackets = [
                    (max(_P_LO, p/3.0), min(_P_HI, p*3.0)),
                    (_P_LO, _P_HI),
                ]

            found_p = None
            for lo, hi in mass_brackets:
                try:
                    fa = f_mass(lo, T_fl)
                    fb = f_mass(hi, T_fl)
                    if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
                        found_p = brentq(
                            lambda p_: f_mass(p_, T_fl),
                            lo, hi, xtol=1.0, rtol=1e-6,
                        )
                        break
                except Exception:
                    continue
            if found_p is None:
                return None
            p = found_p

            # -- Energy step: find T_fl at fixed p --
            found_T = None
            T_guess = T_fl
            for half in [5.0, 20.0, 100.0]:
                lo_ = max(_T_LO, T_guess - half)
                hi_ = min(_T_HI, T_guess + half)
                try:
                    fa = f_energy(lo_, p)
                    fb = f_energy(hi_, p)
                    if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
                        found_T = brentq(
                            lambda T_: f_energy(T_, p),
                            lo_, hi_, xtol=1e-4, rtol=1e-8,
                        )
                        break
                except Exception:
                    continue
            if found_T is None:
                return None
            T_fl = found_T

            if abs(T_fl - T_old) < 1e-5 and abs(p / p_old - 1.0) < 1e-5:
                break

        state = self._make_state(n_total, T_sorb, T_fl, p)
        return state

    # ------------------------------------------------------------------
    # Branch 4: outer-T scan (robust fallback when T_fl ≫ T_sorb)
    # ------------------------------------------------------------------

    def _scan_single_phase(
        self,
        n_total: float,
        U_total: float,
        T_sorb: float,
    ) -> Optional[TwoTempState]:
        """Outer brentq in T_fl with inner mass solve.

        Scans T_fl from 14 K to 500 K at 2 K intervals, resolves the mass
        equation at each T_fl, and finds where the energy residual changes
        sign.  This is O(250 × CoolProp calls) — used as last resort only.
        """
        fl  = self._fluid
        T_c = fl.T_critical
        V_g = self._V_gas

        def _mass_root(T_fl: float) -> Optional[float]:
            """Find p satisfying mass balance at fixed T_fl (single-phase)."""
            if T_fl < T_c:
                try:
                    p_sat_val = fl.p_saturation(T_fl)
                    brackets = [
                        (_P_LO, p_sat_val * 0.999),   # gas side
                        (p_sat_val * 1.001, _P_HI),   # liquid side (compressed)
                        (_P_LO, _P_HI),               # full range – catches saturation boundary
                    ]
                except Exception:
                    brackets = [(_P_LO, _P_HI)]
            else:
                brackets = [(_P_LO, _P_HI)]

            for lo, hi in brackets:
                try:
                    def fm(p_):
                        rho   = fl.rho_molar(p_, T_fl)
                        n_ads = self._n_ads(p_, T_sorb)
                        return rho * V_g + n_ads - n_total
                    fa = fm(lo); fb = fm(hi)
                    if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
                        return float(brentq(fm, lo, hi, xtol=1.0, rtol=1e-6))
                except Exception:
                    continue
            return None

        def _energy_res(T_fl: float, p: float) -> float:
            try:
                n_ads    = self._n_ads(p, T_sorb)
                n_bulk   = n_total - n_ads
                U_fl_eff = (U_total
                            + self._q_st(p, T_sorb) * n_ads
                            - self._H_skel(T_sorb))
                return n_bulk * fl.u_molar(p, T_fl) - U_fl_eff
            except Exception:
                return float("nan")

        # Outer scan — 2 K steps over [T_LO, T_HI]
        T_scan = np.arange(_T_LO + 1.0, _T_HI, 2.0)
        r_prev: Optional[float] = None
        T_prev: Optional[float] = None
        p_prev: Optional[float] = None

        for T_ in T_scan:
            p_ = _mass_root(T_)
            if p_ is None:
                r_prev = None; continue

            r_ = _energy_res(T_, p_)
            if not math.isfinite(r_):
                r_prev = None; continue

            if r_prev is not None and p_prev is not None and r_prev * r_ <= 0.0:
                # Refine with brentq in T_fl
                def f_outer(T_fl_):
                    p_r = _mass_root(T_fl_)
                    if p_r is None:
                        return float("nan")
                    return _energy_res(T_fl_, p_r)

                try:
                    T_root = float(brentq(f_outer, T_prev, T_, xtol=1e-4))
                    p_root = _mass_root(T_root)
                    if p_root is not None:
                        return self._make_state(n_total, T_sorb, T_root, p_root)
                except Exception:
                    pass

            r_prev = r_
            T_prev = T_
            p_prev = p_

        return None

    # ------------------------------------------------------------------
    # State finaliser
    # ------------------------------------------------------------------

    def _make_state(
        self,
        n_total: float,
        T_sorb: float,
        T_fl: float,
        p: float,
    ) -> TwoTempState:
        fl  = self._fluid
        T_c = fl.T_critical
        p_c = fl.p_critical

        n_ads  = self._n_ads(p, T_sorb)
        n_bulk = n_total - n_ads

        if T_fl >= T_c and p >= p_c:
            phase   = PHASE_SUPERCRITICAL
            Q_vapor = float("nan")
        elif T_fl >= T_c:
            phase   = PHASE_GAS
            Q_vapor = float("nan")
        else:
            try:
                rhoL = fl.rho_molar_saturated_liquid(T_fl)
                rhoV = fl.rho_molar_saturated_vapor(T_fl)
                rho  = n_bulk / self._V_gas
                if rho <= rhoV:
                    phase   = PHASE_GAS
                    Q_vapor = 1.0
                elif rho >= rhoL:
                    phase   = PHASE_LIQUID
                    Q_vapor = 0.0
                else:
                    phase   = PHASE_TWO_PHASE
                    v_L     = 1.0 / rhoL; v_V = 1.0 / rhoV
                    Q_vapor = float(np.clip((1.0/rho - v_L)/(v_V - v_L), 0.0, 1.0))
            except Exception:
                phase   = PHASE_GAS
                Q_vapor = float("nan")

        self._cache_T_fl        = T_fl
        self._cache_lnp         = math.log(max(p, _P_LO))
        self._cache_is_two_phase = (phase == PHASE_TWO_PHASE)
        return TwoTempState(
            T_fluid=T_fl,
            T_sorb=T_sorb,
            p=p,
            Q_vapor=Q_vapor,
            phase=phase,
            n_bulk=n_bulk,
            n_ads_abs=n_ads,
        )
