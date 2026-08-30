"""Adsorbent configuration factory — a registry for easy material swapping.

Usage
-----
>>> from opd.adsorbents import get_adsorbent, list_adsorbents
>>> ads = get_adsorbent("MIL-101")         # case-insensitive
>>> ads = get_adsorbent("activated_carbon")
>>> print(list_adsorbents())
['activated_carbon', 'MIL-101', 'IRMOF-20']

The factory decouples the rest of the simulation stack from concrete
import paths: the caller only needs the material's canonical name string.
Custom materials can be registered at runtime via
:func:`register_adsorbent`.

All factory calls return a **fresh** :class:`AdsorbentMaterial` instance,
so the returned objects are safe to mutate-free (they are frozen dataclasses
in any case).
"""

from __future__ import annotations

from typing import Callable

from .base import AdsorbentMaterial

__all__ = ["get_adsorbent", "list_adsorbents", "register_adsorbent"]

# Registry: canonical_name (lower) -> factory callable
_REGISTRY: dict[str, Callable[[], AdsorbentMaterial]] = {}
_CANONICAL_NAMES: dict[str, str] = {}   # alias → canonical (preserves case)


def _register_builtins() -> None:
    from .activated_carbon import ActivatedCarbon208C
    from .ax21 import AX21
    from .mof_mil101 import MIL101
    from .mof_irmof20 import IRMOF20

    _add("activated_carbon", ActivatedCarbon208C, aliases=["ac", "ac208c", "208c"])
    _add("AX-21", AX21, aliases=["ax21", "ax-21", "ax_21", "maxsorb"])
    _add("MIL-101", MIL101, aliases=["mil101", "mil-101", "mil_101"])
    _add("IRMOF-20", IRMOF20, aliases=["irmof20", "irmof-20", "irmof_20"])


def _add(
    canonical: str,
    factory: Callable[[], AdsorbentMaterial],
    aliases: list[str] | None = None,
) -> None:
    key = canonical.lower()
    _REGISTRY[key] = factory
    _CANONICAL_NAMES[key] = canonical
    for alias in (aliases or []):
        _REGISTRY[alias.lower()] = factory
        _CANONICAL_NAMES[alias.lower()] = canonical


def get_adsorbent(name: str) -> AdsorbentMaterial:
    """Return a fresh :class:`AdsorbentMaterial` by name (case-insensitive).

    Parameters
    ----------
    name
        Canonical name or alias. See :func:`list_adsorbents` for options.

    Returns
    -------
    AdsorbentMaterial

    Raises
    ------
    KeyError
        If ``name`` is not in the registry.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted({_CANONICAL_NAMES[k] for k in _REGISTRY}))
        raise KeyError(
            f"Unknown adsorbent {name!r}. "
            f"Available: {available}. "
            f"Register custom materials with register_adsorbent()."
        )
    return _REGISTRY[key]()


def list_adsorbents() -> list[str]:
    """Return a sorted list of canonical adsorbent names."""
    return sorted({_CANONICAL_NAMES[k] for k in _REGISTRY})


def register_adsorbent(
    canonical_name: str,
    factory: Callable[[], AdsorbentMaterial],
    aliases: list[str] | None = None,
) -> None:
    """Add a custom adsorbent to the registry.

    Parameters
    ----------
    canonical_name
        Display name used in plots and logs.
    factory
        Zero-argument callable returning a fresh :class:`AdsorbentMaterial`.
    aliases
        Optional list of alternative lookup strings.

    Example
    -------
    >>> register_adsorbent("My-MOF", lambda: AdsorbentMaterial(...))
    >>> ads = get_adsorbent("My-MOF")
    """
    _add(canonical_name, factory, aliases)


# Populate registry at import time
_register_builtins()
