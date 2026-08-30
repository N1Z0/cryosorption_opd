"""Tank geometry descriptor.

For M3 the only required quantity is the internal volume.  Surface area and
shape parameters for MLI / radiation calculations will be added in M5 when
the wall heat-leak model is tightened.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..adsorbents.base import AdsorbentMaterial

__all__ = ["TankGeometry"]


@dataclass(frozen=True)
class TankGeometry:
    """Immutable geometric description of a cylindrical / spherical tank.

    Parameters
    ----------
    volume
        Internal volume, m³.
    surface_area
        Internal surface area, m².  Optional for M3 (used by
        :class:`~opd.tank.heat_loads.UAEnvironmentCoupling` and the MLI
        model planned for M5).  Defaults to the surface area of a sphere
        with the given volume.
    """

    volume: float
    surface_area: float = 0.0

    def __post_init__(self) -> None:
        if self.volume <= 0.0:
            raise ValueError(f"Tank volume must be positive, got {self.volume}")
        if self.surface_area < 0.0:
            raise ValueError("surface_area must be non-negative")

    def free_volume(
        self, m_sorb: float = 0.0, adsorbent: "AdsorbentMaterial | None" = None
    ) -> float:
        """Geometric free volume after subtracting the sorbent skeleton.

        Parameters
        ----------
        m_sorb
            Sorbent mass, kg.
        adsorbent
            Adsorbent material (needed for its :attr:`skeletal_density`).

        Returns
        -------
        float
            ``V_free = volume − m_sorb / rho_skeletal``, m³.
        """
        if m_sorb == 0.0 or adsorbent is None:
            return self.volume
        V_skel = m_sorb / adsorbent.skeletal_density
        V_free = self.volume - V_skel
        if V_free <= 0.0:
            raise ValueError(
                f"Sorbent skeleton occupies {V_skel:.4g} m³ "
                f"but the tank volume is only {self.volume:.4g} m³."
            )
        return V_free
