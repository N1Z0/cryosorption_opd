"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from opd.fluids.hydrogen import normal_hydrogen, parahydrogen


@pytest.fixture(scope="session")
def para_h2():
    return parahydrogen()


@pytest.fixture(scope="session")
def h2_notebook():
    """Normal hydrogen (CoolProp ``Hydrogen``) for regression tests."""
    return normal_hydrogen()
