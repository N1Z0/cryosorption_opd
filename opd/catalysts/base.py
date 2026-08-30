"""Para-ortho hydrogen conversion catalyst models.

Background
----------
Dihydrogen (H₂) exists as two nuclear spin isomers:

* **Para-H₂** (antiparallel nuclear spins, even rotational quantum numbers
  J = 0, 2, 4, …).  At LH₂ temperatures (~20 K) this is the equilibrium
  ground state; commercial LH₂ is converted to >99% para before storage.

* **Ortho-H₂** (parallel nuclear spins, odd J = 1, 3, 5, …).

The interconversion is strongly *endothermic* when the system is colder
than the ortho–para inversion temperature (~180 K):

    para-H₂  →  ortho-H₂    requires  :math:`\\Delta H_{po}(T) > 0`

A catalyst (e.g. Fe₂O₃ or APACHI) in the adsorbent bed can accelerate
this conversion.  Because the heat of conversion is endothermic at
orbital storage temperatures (20–80 K), it provides a *natural heat sink*
that partially offsets the exothermic heat of adsorption during pressure-
knockdown strokes of the Smart Ullage cycle.

At cryogenic temperatures the equilibrium ortho fraction is very small
(< 5% at 20 K, < 25% at 30 K), so the maximum possible conversion heat
per orbit cycle is bounded by the initial para fraction.

References
----------
* Leachman, J.W. et al. (2009), NIST properties of para/ortho-H₂.
* Barron, R.F. (1985), "Cryogenic Systems", 2nd ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["ParaOrthoCatalyst", "ortho_eq_fraction", "ortho_eq_enthalpy_difference"]

# Rotational temperature for H₂, K
_THETA_ROT: float = 85.4

# Enthalpy difference h_ortho − h_para at T → 0, J/mol (endothermic at low T)
# Literature value at 20 K ≈ 527 J/mol.  Approximated here as a polynomial
# fit (see Leachman 2009) evaluated over 15–300 K.
# We use a simple two-point approximation: varies from ~527 J/mol at 20 K
# to ~0 J/mol at ~200 K.
_DH_PO_20K: float = 527.0   # J/mol
_T_ZERO_DH: float = 200.0   # K (above this the enthalpy difference is ~0)


def ortho_eq_fraction(T: float, n_terms: int = 15) -> float:
    """Equilibrium mole fraction of ortho-H₂ at temperature ``T``.

    Computed from the rotational partition function:

    .. math::

        X_{\\mathrm{ortho}}^{\\mathrm{eq}}(T) =
            \\frac{Z_{\\mathrm{ortho}}}{Z_{\\mathrm{ortho}} + Z_{\\mathrm{para}}}

    where

    .. math::

        Z_{\\mathrm{ortho}} = \\sum_{J=1,3,5,\\ldots} 3\\,(2J+1)
            \\exp\\!\\left[-\\frac{J(J+1)\\,\\Theta_{\\mathrm{rot}}}{T}\\right]

    and

    .. math::

        Z_{\\mathrm{para}} = \\sum_{J=0,2,4,\\ldots} (2J+1)
            \\exp\\!\\left[-\\frac{J(J+1)\\,\\Theta_{\\mathrm{rot}}}{T}\\right]

    with :math:`\\Theta_{\\mathrm{rot}} = 85.4\\,\\mathrm{K}`.

    Parameters
    ----------
    T
        Temperature, K.  Must be > 0.
    n_terms
        Number of rotational levels to sum (default 15 is sufficient for
        T < 300 K).

    Returns
    -------
    float
        Equilibrium ortho fraction in ``[0, 0.75]``.
    """
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    Z_o = sum(
        3.0 * (2 * J + 1) * math.exp(-J * (J + 1) * _THETA_ROT / T)
        for J in range(1, 2 * n_terms, 2)
    )
    Z_p = sum(
        (2 * J + 1) * math.exp(-J * (J + 1) * _THETA_ROT / T)
        for J in range(0, 2 * n_terms, 2)
    )
    return Z_o / (Z_o + Z_p)


def ortho_eq_enthalpy_difference(T: float) -> float:
    """Approximate molar enthalpy difference :math:`h_{\\mathrm{ortho}} - h_{\\mathrm{para}}`, J/mol.

    Uses a simple linear interpolation between the known low-T value
    (~527 J/mol at 20 K) and zero at ~200 K.  Positive values indicate
    that ortho-H₂ has *higher* enthalpy (para → ortho is endothermic
    below ~180 K).

    Parameters
    ----------
    T
        Temperature, K.

    Returns
    -------
    float
        Enthalpy difference, J/mol.
    """
    if T >= _T_ZERO_DH:
        return 0.0
    return _DH_PO_20K * (1.0 - T / _T_ZERO_DH)


@dataclass(frozen=True)
class ParaOrthoCatalyst:
    """Lumped-parameter para-ortho conversion catalyst model.

    Assumes first-order kinetics with a rate constant that follows an
    Arrhenius law:

    .. math::

        \\frac{\\mathrm{d}X_{\\mathrm{ortho}}}{\\mathrm{d}t} =
            k(T) \\cdot (X_{\\mathrm{ortho}}^{\\mathrm{eq}}(T) - X_{\\mathrm{ortho}})

    with

    .. math::

        k(T) = k_0 \\exp\\!\\left(-\\frac{E_a}{R\\,T}\\right)

    The heat added to the gas (negative = endothermic sink) is

    .. math::

        \\dot{Q}_{\\mathrm{cat}} =
            -\\Delta H_{po}(T) \\cdot n_{\\mathrm{total}} \\cdot
            \\frac{\\mathrm{d}X_{\\mathrm{ortho}}}{\\mathrm{d}t}

    Parameters
    ----------
    k0
        Pre-exponential rate constant, s⁻¹.
    E_activation
        Activation energy, J/mol.  A typical Fe₂O₃ catalyst has
        ``E_a ≈ 0`` (barrierless at cryogenic conditions) for the
        *homogeneous* conversion; catalytic conversion is faster.
        Default 0.0 (temperature-independent rate).
    """

    k0: float = 1.0e-3    # s⁻¹  (slow conversion, ~15 min half-life at 20 K)
    E_activation: float = 0.0  # J/mol

    def k(self, T: float) -> float:
        """Rate constant at temperature ``T``, s⁻¹."""
        from ..constants import R_UNIVERSAL
        if self.E_activation == 0.0:
            return self.k0
        return self.k0 * math.exp(-self.E_activation / (R_UNIVERSAL * T))

    def dX_ortho_dt(self, T: float, X_ortho: float) -> float:
        """Rate of change of ortho fraction, s⁻¹.

        Parameters
        ----------
        T
            Temperature, K.
        X_ortho
            Current ortho mole fraction (0–1).

        Returns
        -------
        float
            :math:`\\mathrm{d}X_{\\mathrm{ortho}}/\\mathrm{d}t`, s⁻¹.
        """
        return self.k(T) * (ortho_eq_fraction(T) - X_ortho)

    def Q_conversion(self, T: float, X_ortho: float, n_total: float) -> float:
        """Instantaneous heat added to the gas from para-ortho conversion, W.

        Parameters
        ----------
        T
            Gas temperature, K.
        X_ortho
            Current ortho fraction.
        n_total
            Total H₂ inventory, mol.

        Returns
        -------
        float
            Heat added to the gas, W.  *Negative* = endothermic sink (heat
            absorbed from gas during para→ortho conversion at low T).
        """
        dXdt   = self.dX_ortho_dt(T, X_ortho)
        dH     = ortho_eq_enthalpy_difference(T)
        return -dH * n_total * dXdt
