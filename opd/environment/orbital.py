"""Orbital environment heat-flux models.

All models are :class:`~opd.tank.heat_loads.HeatLeakModel` subclasses
and therefore slot directly into the tank's ``heat_leak`` parameter.

Physical basis
--------------

The heat delivered to a cryogenic tank in orbit is dominated by:

1. **Solar direct flux** :math:`G_\\odot = 1361\\,\\mathrm{W/m^2}`
   (AM0 solar constant, 2019 NIST value).
2. **Albedo** (reflected sunlight from the planet): 30% of solar for Earth,
   12% for the Moon.
3. **Planetary infrared (OLR)**: Earth ~237 W/m², Moon varies widely.
4. **Multi-Layer Insulation (MLI)**: Reduces the effective heat flux by
   3–4 orders of magnitude. Typical performance: 0.2–1.0 W/m².

The models here use an *effective specific heat flux* ``q_eff``
(W/m²) through the MLI that already incorporates all radiative sources.
The actual power delivered is:

    :math:`\\dot{Q}(t) = q_{\\mathrm{eff}}(t) \\times A_{\\mathrm{tank}}`

where :math:`A_{\\mathrm{tank}}` is the outer tank surface area in m².

Sun/Eclipse profile
-------------------
A step-function profile is used:

* **Sunlit fraction** :math:`\\beta = T_{\\mathrm{sun}} / T_{\\mathrm{orbit}}`.
  For 400 km LEO: ~56 min sun, 34 min eclipse → β ≈ 0.62.
* In eclipse the only heat source is OLR and spacecraft conduction, so
  :math:`q_{\\mathrm{eclipse}} \\ll q_{\\mathrm{sun}}`.

The implementation uses a smooth cosine taper at the sun/eclipse
transitions (width ``transition_fraction=0.05``) to avoid step
discontinuities that can confuse the BDF solver's event detection.

References
----------
* Griffin & French (2004) "Space Vehicle Design", 2nd ed. AIAA.
* NASA/TP-2012-217281 "Cryogenic Fluid Management Technology Development".
* Plachta & Kittel (2003) JSC "An Updated Zero Boil-Off Cryogenic
  Propellant Storage Analysis Applied to Upper Stages or Depots in a LEO
  Environment". AIAA 2003-4898.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..tank.heat_loads import HeatLeakModel

__all__ = [
    "MLIHeatFlux",
    "LEOHeatFlux",
    "LunarHeatFlux",
    "GatewayHeatFlux",
    "MarsOrbitHeatFlux",
]


# ---------------------------------------------------------------------------
# MLI helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLIHeatFlux(HeatLeakModel):
    """Simple MLI-wall heat-leak model.

    The heat penetrating an MLI blanket is approximated as:

    .. math::

        \\dot{Q} = q_{\\mathrm{eff}} \\times A_{\\mathrm{tank}}

    where :math:`q_{\\mathrm{eff}}` is the effective specific heat flux
    (W/m²) through the MLI and :math:`A_{\\mathrm{tank}}` is the outer
    surface area of the tank (m²).

    This is the simplest possible model; for orbit-averaged performance
    use :class:`LEOHeatFlux` or :class:`LunarHeatFlux` which modulate
    ``q_eff`` with the Sun/eclipse cycle.

    Parameters
    ----------
    q_eff
        Effective heat flux through MLI, W/m².
        Typical range: 0.2–1.0 W/m² for 30–60 layer MLI blankets.
    area
        Outer tank surface area, m².
        For a sphere: A = 4 π r² = π^(1/3) (6 V)^(2/3).
    """

    q_eff: float    # W/m²
    area: float     # m²

    def __post_init__(self) -> None:
        if self.q_eff < 0.0:
            raise ValueError(f"q_eff must be ≥ 0, got {self.q_eff}")
        if self.area <= 0.0:
            raise ValueError(f"area must be > 0, got {self.area}")

    def Q_dot(self, t: float, T_fluid: float) -> float:
        return self.q_eff * self.area

    @staticmethod
    def sphere_area(volume_m3: float) -> float:
        """Surface area of a sphere with given volume, m²."""
        return math.pi ** (1 / 3) * (6.0 * volume_m3) ** (2 / 3)


# ---------------------------------------------------------------------------
# LEO cycling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LEOHeatFlux(HeatLeakModel):
    """Low-Earth-Orbit heat leak with 90-minute Sun/eclipse cycling.

    The heat flux through the MLI alternates between a higher value in
    sunlight and a lower value in eclipse.  The transition is smoothed
    with a cosine taper.

    Parameters
    ----------
    area
        Outer tank surface area, m².
    q_sun
        Effective specific heat flux through MLI during direct sunlight,
        W/m².  Typical: 0.4–0.8 W/m² (30-layer MLI at LEO).
    q_eclipse
        Effective specific heat flux during eclipse, W/m².
        Typical: 0.1–0.3 W/m² (planet OLR only).
    orbit_period
        Orbital period, s.  Default: 5400 s (90 min, ~400 km LEO).
    beta_sun
        Fraction of orbit spent in sunlight.  Default: 0.62 (400 km LEO).
    t0_offset
        Phase offset so the simulation starts at a convenient orbit phase
        (e.g. 0.0 = start at beginning of sunlit period).  Seconds.
    transition_fraction
        Fraction of half-period used for the cosine-tapered transition.
        Default: 0.05 (≈ 2.7 min).
    """

    area: float
    q_sun: float = 0.5
    q_eclipse: float = 0.15
    orbit_period: float = 5400.0
    beta_sun: float = 0.62
    t0_offset: float = 0.0
    transition_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.area <= 0:
            raise ValueError("area must be positive")
        if not 0 < self.beta_sun < 1:
            raise ValueError("beta_sun must be in (0, 1)")

    def _sun_fraction(self, t: float) -> float:
        """Returns fraction ∈ [0, 1] indicating sunlit (1) vs eclipse (0)."""
        T = self.orbit_period
        t_norm = ((t - self.t0_offset) % T) / T  # 0..1 within orbit

        t_sun = self.beta_sun
        t_eco = 1.0 - t_sun
        half_trans = self.transition_fraction / 2.0

        if t_norm < t_sun - half_trans:
            return 1.0
        elif t_norm < t_sun + half_trans:
            # Cosine taper: sun → eclipse
            x = (t_norm - (t_sun - half_trans)) / (2 * half_trans)
            return 0.5 * (1.0 + math.cos(math.pi * x))
        elif t_norm < 1.0 - half_trans:
            return 0.0
        else:
            # Cosine taper: eclipse → sun
            x = (t_norm - (1.0 - half_trans)) / (2 * half_trans)
            return 0.5 * (1.0 - math.cos(math.pi * x))

    def Q_dot(self, t: float, T_fluid: float) -> float:
        f = self._sun_fraction(t)
        q = self.q_eclipse + f * (self.q_sun - self.q_eclipse)
        return q * self.area

    @property
    def Q_average(self) -> float:
        """Time-averaged heat power, W."""
        return (self.beta_sun * self.q_sun
                + (1 - self.beta_sun) * self.q_eclipse) * self.area

    @staticmethod
    def sphere_area(volume_m3: float) -> float:
        """Surface area of a sphere with given volume, m²."""
        return math.pi ** (1 / 3) * (6.0 * volume_m3) ** (2 / 3)


# ---------------------------------------------------------------------------
# Lunar surface / Gateway cycling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LunarHeatFlux(HeatLeakModel):
    """Lunar surface 14-day day / 14-day night heat-leak model.

    On the Lunar surface (or in low Lunar orbit), the thermal environment
    is extreme: lunar day surface temperatures reach ~390 K; lunar night
    drops to ~100 K.  The MLI reduces the effective heat leak, but the
    ratio of day/night loading is still significant.

    For a depot in Lunar orbit (not on surface), the period is the
    *orbital* period (≈ 7 days for Gateway NRHO), not the Lunar day.

    Parameters
    ----------
    area
        Outer tank surface area, m².
    q_day
        Effective specific heat flux through MLI during Lunar day, W/m².
        Typical: 0.5–1.0 W/m² (Lunar day peak direct + albedo + OLR).
    q_night
        Effective specific heat flux during Lunar night, W/m².
        Typical: 0.05–0.2 W/m² (cold-sink; OLR dominated by tank itself).
    period
        Full Lunar day + night period, s.  Default: 2,419,200 s = 28 days.
    beta_day
        Fraction of period in Lunar day.  Default: 0.50 (half/half).
    t0_offset
        Phase offset, s.
    """

    area: float
    q_day: float = 0.8
    q_night: float = 0.1
    period: float = 28 * 24 * 3600.0   # 28 days
    beta_day: float = 0.5
    t0_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.area <= 0:
            raise ValueError("area must be positive")

    def _day_fraction(self, t: float) -> float:
        T = self.period
        t_norm = ((t - self.t0_offset) % T) / T
        return 1.0 if t_norm < self.beta_day else 0.0

    def Q_dot(self, t: float, T_fluid: float) -> float:
        f = self._day_fraction(t)
        q = self.q_night + f * (self.q_day - self.q_night)
        return q * self.area

    @property
    def Q_average(self) -> float:
        """Time-averaged heat power, W."""
        return (self.beta_day * self.q_day
                + (1 - self.beta_day) * self.q_night) * self.area


@dataclass(frozen=True)
class MarsOrbitHeatFlux(LEOHeatFlux):
    """Low Mars Orbit (LMO, ~300 km) heat leak with Sun/eclipse cycling.

    Inherits the smooth sun/eclipse machinery from :class:`LEOHeatFlux`
    with Mars-appropriate defaults:

    * Solar constant at Mars: :math:`G_\\odot \\approx 586\\,\\mathrm{W/m^2}`
      (43 % of Earth's), so the sunlit MLI flux scales down accordingly.
    * Mars OLR is weak (~110 W/m² average) and the albedo is low (0.25),
      giving a colder eclipse leg than LEO.
    * 300 km LMO orbital period ≈ 6 810 s (~113 min); sunlit fraction
      ≈ 0.62 for a near-equatorial orbit.

    References: Larson & Pranke (2000) App. C; NASA-TM-2010-216437
    (Mars propellant depot thermal environments).
    """

    q_sun: float = 0.25          # W/m² through MLI (scaled by G_Mars/G_Earth)
    q_eclipse: float = 0.08      # W/m² (weak Mars OLR)
    orbit_period: float = 6810.0  # s (~113 min, 300 km LMO)
    beta_sun: float = 0.62


@dataclass(frozen=True)
class GatewayHeatFlux(HeatLeakModel):
    """Near-Rectilinear Halo Orbit (NRHO) heat flux for Lunar Gateway.

    The Gateway NRHO has a ~6.5-day period with highly variable
    Earth/Sun/Moon geometry.  The effective heat load oscillates
    between a maximum near periselene and a minimum near aposelene.

    This model uses a sinusoidal approximation of the orbit-averaged
    heat flux for simplicity; the phase offsets and amplitudes are
    based on published NRHO thermal analyses.

    Parameters
    ----------
    area
        Outer tank surface area, m².
    q_mean
        Mean specific heat flux, W/m².  Typical: 0.3–0.6 W/m².
    q_amplitude
        Peak-to-mean amplitude fraction (0–1).  Default: 0.4
        (i.e., q varies ±40% about the mean).
    period
        NRHO orbital period, s.  Default: 561,600 s = 6.5 days.
    t0_offset
        Phase offset, s.
    """

    area: float
    q_mean: float = 0.4
    q_amplitude: float = 0.4
    period: float = 6.5 * 24 * 3600.0
    t0_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.area <= 0:
            raise ValueError("area must be positive")
        if not 0.0 <= self.q_amplitude <= 1.0:
            raise ValueError("q_amplitude must be in [0, 1]")

    def Q_dot(self, t: float, T_fluid: float) -> float:
        phase = 2.0 * math.pi * (t - self.t0_offset) / self.period
        q = self.q_mean * (1.0 + self.q_amplitude * math.sin(phase))
        return q * self.area

    @property
    def Q_average(self) -> float:
        """Time-averaged heat power (= q_mean × area), W."""
        return self.q_mean * self.area
