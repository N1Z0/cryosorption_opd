"""Reserved namespace for Para-Ortho hydrogen catalyst models (Milestone 5).

A thin bed of Fe/Ni oxide or APACHI catalyst placed inside the MOF module
catalyses the spin-isomer interconversion of hydrogen. In the orbital
depot context, the **endothermic** para → ortho conversion
(~\\ :math:`699\\,\\mathrm{J\\,mol^{-1}}` at 20 K) is a natural heat sink that
partially offsets the **exothermic** heat of adsorption released when H₂ is
taken up by the adsorbent during pressure-knockdown strokes of the
"Smart Ullage Control" cycle.

Planned contents (M5)
---------------------

``opd.catalysts.base.CatalystMaterial``
    ABC carrying bed composition, surface area, activation energy, and a
    conversion-kinetics model.

``opd.catalysts.ortho_para.FeOxideCatalyst``
    Concrete first-cut based on literature values for iron oxide at LH₂
    temperatures.

``opd.catalysts.bed.ParaOrthoCatalystBed``
    A node-mixin that exposes a conversion-rate term to the energy and
    species balances without needing to be a standalone
    :class:`~opd.nodes.base.Node`. Lets existing
    :class:`~opd.nodes.tank_contents.TankContentsNode` gain a catalyst
    sub-term additively.

Implemented in M5
-----------------
:class:`~opd.catalysts.base.ParaOrthoCatalyst`
    First-order rate model with Arrhenius kinetics.  The catalyst object
    is passed to :class:`~opd.tank.tank.Tank` or
    :class:`~opd.tank.two_temp_tank.TwoTempTank`; it adds an endothermic
    heat-sink term to the energy ODE and requires a new state variable
    ``X_ortho`` (ortho mole fraction) in the state vector.

:func:`~opd.catalysts.base.ortho_eq_fraction`
    Equilibrium ortho fraction from the rotational partition function.

:func:`~opd.catalysts.base.ortho_eq_enthalpy_difference`
    Molar enthalpy difference :math:`h_{\\mathrm{ortho}} - h_{\\mathrm{para}}`.
"""

from __future__ import annotations

from .base import (
    ParaOrthoCatalyst,
    ortho_eq_enthalpy_difference,
    ortho_eq_fraction,
)

__all__ = [
    "ParaOrthoCatalyst",
    "ortho_eq_fraction",
    "ortho_eq_enthalpy_difference",
]
