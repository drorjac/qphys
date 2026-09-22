"""Leggett-Garg, implemented as a guardrail rather than as evidence.

The temptation: compute K3 = C12 + C23 - C13 on a recorded series, find it
above the macrorealist bound of 1, and announce quantum behaviour. Do not.

For a passively recorded series with all three correlators estimated from one
aligned window,

    K3 = mean_t [ q_t q_{t+T} + q_{t+T} q_{t+2T} - q_t q_{t+2T} ]

every bracket is <= 1 by exhaustive enumeration over the eight sign
assignments, so **K3 <= 1 is an algebraic identity here, not an empirical
test**. A recorded series satisfies macrorealism by construction: every q_t
has a definite stored value. There is no loophole to close and no experiment
being run.

Worse, a classical Gaussian process with r(tau) = cos(w tau), dichotomised by
sign, has correlator (2/pi) arcsin(r) by the Van Vleck law -- a triangle wave
that saturates K3 = 1 exactly at every lag. A classical oscillator imitates
the quantum shape maximally.

So `k3_aligned` exists to be asserted against: if it ever exceeds 1, there is
a windowing bug in the estimator. That is the only thing this module is for.
"""

from __future__ import annotations

import itertools

import numpy as np

CLASSICAL_BOUND = 1.0
QUANTUM_MAX = 1.5
QUANTUM_MAX_ANGLE = np.pi / 3.0


def k3_from_correlators(c12: float, c23: float, c13: float) -> float:
    return float(c12 + c23 - c13)


def classical_bound_exhaustive() -> float:
    """max K3 over every deterministic +-1 assignment. Exactly 1."""
    best = -np.inf
    for q1, q2, q3 in itertools.product((-1, 1), repeat=3):
        best = max(best, k3_from_correlators(q1 * q2, q2 * q3, q1 * q3))
    return float(best)


def k3_quantum(wt):
    """A two-level system under ideal projective measurement: 2cos - cos2.

    Accepts an array so the maximum can be found on a grid rather than
    asserted at the angle we expect to find it at.
    """
    w = np.asarray(wt, float)
    return 2.0 * np.cos(w) - np.cos(2.0 * w)


def quantum_maximum() -> tuple:
    """(max K3, the angle achieving it) = (1.5, pi/3), found numerically."""
    grid = np.linspace(0.0, np.pi, 200_001)
    vals = k3_quantum(grid)
    i = int(np.argmax(vals))
    return float(vals[i]), float(grid[i])


def k3_aligned(q, lag: int) -> float:
    """The estimator, on one aligned window. Provably <= 1 for any input.

    `q` must be +-1 valued. The alignment is the point: all three correlators
    come from the same t index, which is what makes the bound algebraic.
    """
    x = np.asarray(q, float)
    if not np.all(np.isin(np.unique(x), (-1.0, 1.0))):
        raise ValueError("k3_aligned expects a +-1 valued series")
    n = len(x) - 2 * lag
    if n <= 0:
        raise ValueError(f"series too short for lag {lag}")
    a, b, c = x[:n], x[lag : lag + n], x[2 * lag : 2 * lag + n]
    return float(np.mean(a * b + b * c - a * c))


def van_vleck_correlator(r) -> np.ndarray:
    """(2/pi) arcsin(r): the sign correlator of a Gaussian process."""
    return (2.0 / np.pi) * np.arcsin(np.clip(np.asarray(r, float), -1.0, 1.0))


def k3_van_vleck(omega: float, lag: float) -> float:
    """K3 for a sign-dichotomised Gaussian with r(tau) = cos(omega tau).

    Equals exactly 1 for every lag with 2*omega*lag <= pi, because
    arcsin(cos t) = pi/2 - t is linear there and the three terms cancel. The
    classical process saturates the macrorealist bound everywhere.
    """
    c1 = van_vleck_correlator(np.cos(omega * lag))
    c2 = van_vleck_correlator(np.cos(2.0 * omega * lag))
    return k3_from_correlators(float(c1), float(c1), float(c2))
