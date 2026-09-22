"""Shared fixtures. The package is installed (`pip install -e .`)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)
