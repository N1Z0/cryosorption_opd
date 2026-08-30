"""Thermal State Resolver — the tank EOS inversion.

The lumped-parameter tank tracks two conserved scalars per time-step:

.. math::

    \\frac{\\mathrm{d}n_{\\text{total}}}{\\mathrm{d}t}
        &= \\dot{n}_{\\text{in}} - \\dot{n}_{\\text{vent}}

    \\frac{\\mathrm{d}U_{\\text{total}}}{\\mathrm{d}t}
        &= \\dot{Q}_{\\text{wall}} - \\dot{Q}_{\\text{cryo}}
         + \\dot{n}_{\\text{in}}\\,h_{\\text{in}}
         - \\dot{n}_{\\text{vent}}\\,h_{\\text{vent}}

At every ODE right-hand side evaluation the solver needs to recover the
full thermodynamic state ``(T, p, Q, ...)`` from the pair
``(n_total, U_total)``.  :class:`ThermalStateResolver` performs this
inversion.

Energy model (1-Temp, equilibrium adsorption)
---------------------------------------------
The total internal energy is

.. math::

    U_{\\text{total}} = n_{\\text{total}}\\,u_{\\text{bulk}}(p,T)
        - n_{\\text{ads}}\\,q_{\\text{st}}(p,T)
        + m_{\\text{sorb}}\\,
          \\int_{T_{\\text{ref}}}^{T} c_p^{\\text{skel}}(T')\\,\\mathrm{d}T'

where :math:`n_{\\text{ads}} = m_{\\text{sorb}}\\,n_{\\text{abs}}(p,T)` is
the absolute adsorbed amount (mol) and :math:`q_{\\text{st}} > 0` is the
exothermic isosteric heat (energy released per mole adsorbed, so the
adsorbed phase has *lower* internal energy :math:`u_{\\text{bulk}} - q_{\\text{st}}`
than the free bulk molecule at the same state point).

For two-phase bulk states :math:`u_{\\text{bulk}}` is replaced by the
quality-weighted mixture :math:`u_{\\text{mix}} = (1-Q)\\,u_L + Q\\,u_V`.

2-Temp extension (M5)
---------------------
In the present 1-Temp formulation :math:`T_{\\text{sorb}} = T_{\\text{fluid}} = T`.
When M5 adds a separate sorbent temperature state, the only change here
is to accept ``U_skel`` (tracked ODE state) in place of
``m_sorb * H_skel(T)`` and to thread :math:`T_{\\text{sorb}}` into
``cp_skeleton(T_sorb)`` — the rest of the interface stays identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve

from ..adsorbents.base import AdsorbentMaterial
from ..constants import R_UNIVERSAL
from ..fluids.fluid_properties import FluidProperties

__all__ = [
    "ResolvedState",
    "ThermalStateResolver",
    "PHASE_TWO_PHASE",
    "PHASE_LIQUID",
    "PHASE_GAS",
    "PHASE_SUPERCRITICAL",
]

PHASE_TWO_PHASE: str = "two_phase"
PHASE_LIQUID: str = "liquid"
PHASE_GAS: str = "gas"
PHASE_SUPERCRITICAL: str = "supercritical"


@dataclass(frozen=True)
class ResolvedState:
    """Full thermodynamic state recovered by :class:`ThermalStateResolver`.

    Attributes
    ----------
    T
        Temperature, K (equal to ``T_sorb`` in 1-Temp mode).
    p
        Pressure, Pa.
    phase
        One of :data:`PHASE_TWO_PHASE`, :data:`PHASE_LIQUID`,
        :data:`PHASE_GAS`, :data:`PHASE_SUPERCRITICAL`.
    Q_vapor
        Vapour quality in ``[0, 1]``; :data:`math.nan` for single-phase.
    n_bulk
        Bulk-fluid moles (free gas + liquid outside micropores), mol.
    n_ads_abs
        Absolute adsorbed moles (inside micropores), mol.
    rho_molar_bulk
        Bulk-fluid molar density, mol m⁻³.
    u_molar_bulk
        Bulk-fluid molar internal energy, J mol⁻¹.
    """

    T: float
    p: float
    phase: str
    Q_vapor: float
    n_bulk: float
    n_ads_abs: float
    rho_molar_bulk: float
    u_molar_bulk: float


class ThermalStateResolver:
    """Inverts the lumped-parameter tank EOS.

    Given the conserved pair ``(n_total, U_total)`` at a single instant,
    returns the full :class:`ResolvedState`.

    Parameters
    ----------
    fluid
        Fluid property provider.  The default :class:`~opd.fluids.FluidProperties`
        uses ParaHydrogen.
    V_free
        Geometric free volume available to fluids, m³.
        ``V_free = V_tank − m_sorb / rho_skeletal`` (sorbent skeleton
        volume is excluded *before* passing to the resolver).
    adsorbent
        Adsorbent material.  ``None`` → bare tank (no adsorption).
    m_sorb
        Sorbent mass, kg.  Must be ``0.0`` when ``adsorbent is None``.
    T_skel_ref
        Reference temperature for the skeleton enthalpy integral, K.
        Default 0 K.  Must be the same value used when computing the
        initial ``U_total`` via :meth:`encode_two_phase` /
        :meth:`encode_single_phase`.
    """

    def __init__(
        self,
        fluid: FluidProperties,
        V_free: float,
        adsorbent: Optional[AdsorbentMaterial] = None,
        m_sorb: float = 0.0,
        T_skel_ref: float = 0.0,
    ) -> None:
        if V_free <= 0.0:
            raise ValueError(f"V_free must be positive, got {V_free}")
        if m_sorb < 0.0:
            raise ValueError("m_sorb must be non-negative")
        if m_sorb > 0.0 and adsorbent is None:
            raise ValueError("m_sorb > 0 requires an adsorbent to be supplied")

        self._fluid = fluid
        self._ads = adsorbent
        self._m_sorb = m_sorb
        self._T_ref = T_skel_ref

        Va = adsorbent.micropore_volume if adsorbent is not None else 0.0
        self._V_gas: float = V_free - m_sorb * Va
        if self._V_gas <= 0.0:
            raise ValueError(
                f"Effective gas volume V_gas = V_free − m_sorb·Va = "
                f"{self._V_gas:.4g} m³ is non-positive."
            )

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def V_gas(self) -> float:
        """Effective gas-accessible volume, m³.

        ``V_gas = V_free − m_sorb · Va`` where ``Va`` is the adsorbent's
        micropore volume (m³ kg⁻¹).
        """
        return self._V_gas

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _n_ads(self, p: float, T: float) -> float:
        """Total absolute adsorbed amount, mol."""
        if self._ads is None or self._m_sorb == 0.0:
            return 0.0
        return self._m_sorb * self._ads.n_absolute(p, T, self._fluid)

    def _q_st(self, p: float, T: float) -> float:
        """Isosteric heat, J mol⁻¹ (positive = exothermic).  0 for bare tank."""
        if self._ads is None or self._m_sorb == 0.0:
            return 0.0
        return self._ads.isosteric_heat(p, T)

    def _H_skel(self, T: float) -> float:
        """Skeleton enthalpy, J.  0 for bare tank.

        .. math:: H_{\\text{skel}}(T) = m_{\\text{sorb}}
                  \\int_{T_{\\text{ref}}}^{T} c_p^{\\text{skel}}(T')\\,\\mathrm{d}T'
        """
        if self._ads is None or self._m_sorb == 0.0:
            return 0.0
        val, _ = quad(self._ads.cp_skeleton, self._T_ref, T, limit=50)
        return self._m_sorb * val

    # ------------------------------------------------------------------
    # Forward EOS  (initialisation helpers)
    # ------------------------------------------------------------------

    def encode_two_phase(self, T: float, Q: float) -> tuple[float, float]:
        """Forward EOS for a two-phase state.

        Determines ``n_total`` from the phase constraints (the bulk
        density is fully specified by ``T`` and ``Q``), then computes
        ``U_total``.

        Parameters
        ----------
        T
            Temperature, K.  Must be in ``(T_triple, T_critical)``.
        Q
            Vapour quality in ``[0, 1]``.

        Returns
        -------
        n_total, U_total : float, float
        """
        if not 0.0 <= Q <= 1.0:
            raise ValueError(f"Vapour quality must be in [0, 1], got Q={Q}")

        fluid = self._fluid
        p = fluid.p_saturation(T)

        rho_L = fluid.rho_molar_saturated_liquid(T)
        rho_V = fluid.rho_molar_saturated_vapor(T)
        # CoolProp defines Q as the molar vapour fraction (n_V / n_total).
        # The bulk molar density follows the harmonic-mean rule:
        #   ρ = 1 / [(1−Q)/ρ_L + Q/ρ_V]
        # NOT the arithmetic mean (1−Q)ρ_L + Q·ρ_V.
        rho_mix = 1.0 / ((1.0 - Q) / rho_L + Q / rho_V)

        n_bulk = rho_mix * self._V_gas
        n_ads = self._n_ads(p, T)
        n_total = n_bulk + n_ads

        u_L = fluid.u_molar_saturated_liquid(T)
        u_V = fluid.u_molar_saturated_vapor(T)
        # u_mix is the mole-fraction-weighted average (correct for Q = n_V/n_total)
        u_mix = (1.0 - Q) * u_L + Q * u_V

        q_st = self._q_st(p, T)
        H_skel = self._H_skel(T)

        U_total = n_total * u_mix - n_ads * q_st + H_skel
        return n_total, U_total

    def encode_single_phase(self, T: float, p: float) -> tuple[float, float]:
        """Forward EOS for a single-phase state.

        Parameters
        ----------
        T
            Temperature, K.
        p
            Pressure, Pa.  Must be outside the vapour dome (either
            supercritical / superheated gas, or compressed liquid).

        Returns
        -------
        n_total, U_total : float, float
        """
        fluid = self._fluid
        rho = fluid.rho_molar(p, T)
        n_bulk = rho * self._V_gas
        n_ads = self._n_ads(p, T)
        n_total = n_bulk + n_ads

        u = fluid.u_molar(p, T)
        q_st = self._q_st(p, T)
        H_skel = self._H_skel(T)

        U_total = n_total * u - n_ads * q_st + H_skel
        return n_total, U_total

    # ------------------------------------------------------------------
    # Residual functions
    # ------------------------------------------------------------------

    def _res_two_phase(self, T: float, n_total: float, U_total: float) -> float:
        """Signed energy residual for the two-phase branch.

        Positive when the model undershoots ``U_total``, negative when it
        overshoots, so the root (``= 0``) is the physically correct T.

        Notes
        -----
        The vapour quality ``Q`` is *derived* from the mass constraint,
        not solved independently.  This works because, inside the dome,
        specifying ``(T, n_total)`` uniquely determines ``Q`` once
        ``n_ads(p_sat(T), T)`` is known.
        """
        fluid = self._fluid
        p = fluid.p_saturation(T)

        n_ads = self._n_ads(p, T)
        n_bulk = n_total - n_ads

        rho_L = fluid.rho_molar_saturated_liquid(T)
        rho_V = fluid.rho_molar_saturated_vapor(T)
        rho_bulk = n_bulk / self._V_gas
        # Invert the harmonic-mean rule for Q (see encode_two_phase note):
        #   ρ_bulk = 1/[(1−Q)/ρ_L + Q/ρ_V]
        #   → Q = (1/ρ_bulk − 1/ρ_L) / (1/ρ_V − 1/ρ_L)
        v_L = 1.0 / rho_L
        v_V = 1.0 / rho_V
        Q = (1.0 / rho_bulk - v_L) / (v_V - v_L)

        u_L = fluid.u_molar_saturated_liquid(T)
        u_V = fluid.u_molar_saturated_vapor(T)
        u_mix = (1.0 - Q) * u_L + Q * u_V

        q_st = self._q_st(p, T)
        H_skel = self._H_skel(T)

        U_model = n_total * u_mix - n_ads * q_st + H_skel
        return U_total - U_model

    def _res_single_phase(
        self, x: np.ndarray, n_total: float, U_total: float
    ) -> np.ndarray:
        """Normalised 2-D residual vector for the single-phase branch.

        Solver variables are ``x = [T, ln(p)]``. Both residuals are
        normalised to O(1) so that the Jacobian estimated by ``fsolve``
        is well-conditioned regardless of the units of ``n`` and ``U``.
        """
        T = float(x[0])
        p = math.exp(float(x[1]))

        try:
            rho = self._fluid.rho_molar(p, T)
            u = self._fluid.u_molar(p, T)
        except Exception:
            return np.array([1e8, 1e8])

        n_ads = self._n_ads(p, T)
        q_st = self._q_st(p, T)
        n_bulk = n_total - n_ads
        H_skel = self._H_skel(T)

        # mass residual: mol, normalised by n_total
        r1 = (n_bulk - rho * self._V_gas) / max(abs(n_total), 1.0)
        # energy residual: J, normalised by |U_total|
        r2 = (U_total - n_total * u + n_ads * q_st - H_skel) / max(
            abs(U_total), 1.0
        )
        return np.array([r1, r2])

    # ------------------------------------------------------------------
    # Main inversion
    # ------------------------------------------------------------------

    def resolve(
        self,
        n_total: float,
        U_total: float,
        T_guess: Optional[float] = None,
        p_guess: Optional[float] = None,
    ) -> ResolvedState:
        """Invert the tank EOS: ``(n_total, U_total) → ResolvedState``.

        Parameters
        ----------
        n_total
            Total hydrogen moles, mol.
        U_total
            Total internal energy (fluid + adsorbed phase + skeleton), J.
        T_guess
            Temperature initial guess for the single-phase fallback, K.
        p_guess
            Pressure initial guess for the single-phase fallback, Pa.

        Returns
        -------
        ResolvedState

        Algorithm
        ---------
        1. Evaluate :meth:`_res_two_phase` at both ends of the saturation
           line.  If a sign change exists, use Brent's method (1-D, very
           robust).  Accept only when the derived vapour quality
           ``Q ∈ [0, 1]``.
        2. Fall back to a 2-D single-phase solve via ``fsolve`` in
           ``(T, ln p)`` with multiple initial guesses; keep the attempt
           with the smallest residual norm.
        """
        fluid = self._fluid
        T_triple = fluid.T_triple
        T_crit = fluid.T_critical
        p_crit = fluid.p_critical

        # ---- attempt two-phase -------------------------------------------
        T_lo = T_triple + 0.02
        T_hi = T_crit - 0.02
        try:
            f_lo = self._res_two_phase(T_lo, n_total, U_total)
            f_hi = self._res_two_phase(T_hi, n_total, U_total)
        except Exception:
            f_lo, f_hi = 1.0, 1.0  # force single-phase branch

        if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo * f_hi < 0.0:
            T_sol = brentq(
                self._res_two_phase,
                T_lo,
                T_hi,
                args=(n_total, U_total),
                xtol=1e-9,
                rtol=1e-12,
                maxiter=200,
            )
            p_sol = fluid.p_saturation(T_sol)
            n_ads = self._n_ads(p_sol, T_sol)
            n_bulk = n_total - n_ads

            rho_L = fluid.rho_molar_saturated_liquid(T_sol)
            rho_V = fluid.rho_molar_saturated_vapor(T_sol)
            rho_bulk = n_bulk / self._V_gas
            v_L = 1.0 / rho_L
            v_V = 1.0 / rho_V
            Q = (1.0 / rho_bulk - v_L) / (v_V - v_L)

            if -1e-6 <= Q <= 1.0 + 1e-6:
                Q_c = max(0.0, min(1.0, Q))
                u_L = fluid.u_molar_saturated_liquid(T_sol)
                u_V = fluid.u_molar_saturated_vapor(T_sol)
                u_mix = (1.0 - Q_c) * u_L + Q_c * u_V
                return ResolvedState(
                    T=T_sol,
                    p=p_sol,
                    phase=PHASE_TWO_PHASE,
                    Q_vapor=Q_c,
                    n_bulk=n_bulk,
                    n_ads_abs=n_ads,
                    rho_molar_bulk=rho_bulk,
                    u_molar_bulk=u_mix,
                )

        # ---- single-phase fallback ---------------------------------------
        if T_guess is None:
            T_guess = T_crit * 1.5
        if p_guess is None:
            p_guess = max(
                n_total * R_UNIVERSAL * T_guess / self._V_gas, 1e3
            )

        candidates: list[np.ndarray] = [
            np.array([T_guess, math.log(p_guess)]),
            np.array([T_crit * 1.2, math.log(p_crit * 0.8)]),
            np.array([T_crit * 2.5, math.log(p_crit * 4.0)]),
            np.array([T_crit * 0.8, math.log(p_crit * 1.5)]),
            np.array([T_triple * 3.0, math.log(1e5)]),
        ]

        best_x: Optional[np.ndarray] = None
        best_norm = math.inf
        for x_init in candidates:
            try:
                x_sol, info, ier, _ = fsolve(
                    self._res_single_phase,
                    x_init,
                    args=(n_total, U_total),
                    full_output=True,
                )
            except Exception:
                continue
            norm = float(np.linalg.norm(info["fvec"]))
            if norm < best_norm:
                best_norm = norm
                best_x = x_sol

        if best_x is None:
            raise RuntimeError(
                f"ThermalStateResolver failed for "
                f"n_total={n_total:.6g} mol, U_total={U_total:.6g} J"
            )

        T_sol = float(best_x[0])
        p_sol = math.exp(float(best_x[1]))

        n_ads = self._n_ads(p_sol, T_sol)
        n_bulk = n_total - n_ads
        rho = fluid.rho_molar(p_sol, T_sol)
        u = fluid.u_molar(p_sol, T_sol)

        # Phase label.
        # "Supercritical fluid" requires T > T_c AND p > p_c.
        # T > T_c but p < p_c is superheated gas (continuous with the gas phase
        # approaching from below T_c).
        if T_sol > T_crit and p_sol > p_crit:
            phase = PHASE_SUPERCRITICAL
        elif T_sol <= T_crit:
            try:
                if p_sol > fluid.p_saturation(T_sol) * (1.0 + 1e-6):
                    phase = PHASE_LIQUID
                else:
                    phase = PHASE_GAS
            except Exception:
                phase = PHASE_GAS
        else:
            phase = PHASE_GAS

        return ResolvedState(
            T=T_sol,
            p=p_sol,
            phase=phase,
            Q_vapor=math.nan,
            n_bulk=n_bulk,
            n_ads_abs=n_ads,
            rho_molar_bulk=rho,
            u_molar_bulk=u,
        )
