#!/usr/bin/env python3
"""Example: constant heat leak into a 1 m³ hydrogen tank.

Compares pressure rise in a bare tank vs. one loaded with 208C activated
carbon. Run from the repository root after ``pip install -e ".[dev]"``:

    python examples/bare_tank_transient.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from opd.adsorbents.activated_carbon import ActivatedCarbon208C
from opd.fluids.hydrogen import normal_hydrogen
from opd.simulation import TransientSimulator
from opd.tank import ConstantHeatFlux, Tank, TankGeometry

# Initial two-phase state (T, vapour quality)
T0 = 20.28
Q0 = 0.004562952718855845
V_TANK = 1.0
M_SORB = 480.0
Q_LEAK_W = 1000.0
T_END_S = 3600.0


def run_case(*, with_sorbent: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fluid = normal_hydrogen()
    geom = TankGeometry(volume=V_TANK)
    adsorbent = ActivatedCarbon208C() if with_sorbent else None
    m_sorb = M_SORB if with_sorbent else 0.0

    tank = Tank(
        fluid=fluid,
        geometry=geom,
        heat_leak=ConstantHeatFlux(Q_LEAK_W),
        adsorbent=adsorbent,
        m_sorb=m_sorb,
    )
    y0 = tank.initial_state_two_phase(T0, Q0)
    sim = TransientSimulator(tank)
    result = sim.run(y0, t_span=(0.0, T_END_S), n_points=400)
    return result.t, result.p, result.T


def main() -> None:
    t_bare, p_bare, T_bare = run_case(with_sorbent=False)
    t_ac, p_ac, T_ac = run_case(with_sorbent=True)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(t_bare / 3600, p_bare / 1e5, label="bare tank")
    axes[0].plot(t_ac / 3600, p_ac / 1e5, label="208C loaded")
    axes[0].set_ylabel("pressure / bar")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_bare / 3600, T_bare, label="bare tank")
    axes[1].plot(t_ac / 3600, T_ac, label="208C loaded")
    axes[1].set_xlabel("time / h")
    axes[1].set_ylabel("temperature / K")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("1 m³ tank, Q̇ = 1 kW")
    fig.tight_layout()
    out = "examples/bare_tank_transient.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
