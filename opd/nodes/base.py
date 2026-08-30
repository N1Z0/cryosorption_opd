"""Abstract base class for lumped-parameter nodes.

A :class:`Node` is any element of the tank system that contributes:

* a slice of the global ODE state vector,
* a right-hand-side contribution (the time derivatives of its slice), and
* scalar bookkeeping quantities (energy, mass) for post-run conservation
  audits.

The design is intentionally minimal for M3; it will be extended when the
tank-wall node (M3) and the cryocooler (M5) are wired in.

2-Temp note
-----------
A node that needs its own temperature (e.g. the sorbent skeleton in M5)
simply adds a temperature entry to its :meth:`state_names` tuple and
manages the corresponding ODE state.  No other change to the interface
is required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Node"]


class Node(ABC):
    """Abstract base for a single thermal / mass node."""

    @property
    @abstractmethod
    def state_names(self) -> tuple[str, ...]:
        """Names of the ODE state variables owned by this node.

        Used by :class:`~opd.simulation.results.SimulationResult` to label
        columns and by the simulator to slice the global state vector.
        """

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Return the initial values of the node's ODE state variables."""

    @abstractmethod
    def rhs(
        self,
        t: float,
        state_slice: np.ndarray,
        context: "NodeContext",  # noqa: F821
    ) -> np.ndarray:
        """Compute ``d(state_slice)/dt``.

        Parameters
        ----------
        t
            Current simulation time, s.
        state_slice
            The portion of the global state vector belonging to this node.
        context
            A :class:`NodeContext` populated by :class:`~opd.tank.tank.Tank`
            before any node's :meth:`rhs` is called.  It carries the resolved
            thermodynamic state, heat-flux terms, and any derived quantities
            that multiple nodes might need.
        """

    @abstractmethod
    def mass(self, state_slice: np.ndarray) -> float:
        """Return the hydrogen mass (mol) stored in this node."""

    @abstractmethod
    def energy(self, state_slice: np.ndarray) -> float:
        """Return the internal energy (J) stored in this node."""
