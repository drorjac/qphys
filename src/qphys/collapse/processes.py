"""Synthetic processes, including the negative controls.

A result that appears on one process and not on a control is a result. A
result that appears on Mersenne-Twister bits is a bug.
"""

from __future__ import annotations

import numpy as np


def iid_bits(n: int, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    return (rng.random(n) < p).astype(int)


def perturbed_coin(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Flip state with probability `p`; emit the current state."""
    out = np.empty(n, dtype=int)
    s = int(rng.integers(2))
    for i in range(n):
        out[i] = s
        if rng.random() < p:
            s = 1 - s
    return out


def periodic(n: int, period: int) -> np.ndarray:
    return (np.arange(n) % period < period // 2).astype(int)


def ar2_sign(n: int, rng: np.random.Generator, phi1=1.6, phi2=-0.9) -> np.ndarray:
    """Sign of an oscillatory AR(2): a classical oscillator, for the LG test."""
    x = np.zeros(n + 100)
    for t in range(2, n + 100):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + rng.normal()
    return (x[100:] > 0).astype(int)


def random_walk_sign(n: int, rng: np.random.Generator) -> np.ndarray:
    return (np.cumsum(rng.normal(size=n)) > 0).astype(int)


def to_pm1(bits) -> np.ndarray:
    return 2 * np.asarray(bits, int) - 1


def symbolise(x, method: str = "median") -> np.ndarray:
    """Real series -> symbols. A result surviving only one symbolisation is
    not a result, so the sweep is part of the library rather than a script."""
    a = np.asarray(x, float)
    if method == "median":
        return (a > np.median(a)).astype(int)
    if method == "mean":
        return (a > np.mean(a)).astype(int)
    if method == "positive":
        return (a > 0).astype(int)
    if method == "tercile":
        lo, hi = np.quantile(a, [1 / 3, 2 / 3])
        return np.digitize(a, [lo, hi])
    if method == "ordinal":
        return (np.diff(a, prepend=a[0]) > 0).astype(int)
    raise ValueError(f"unknown symbolisation {method!r}")


def transition_matrix(symbols, n_states: int | None = None) -> np.ndarray:
    """Maximum-likelihood first-order transition matrix."""
    s = np.asarray(symbols, int)
    k = int(n_states or s.max() + 1)
    counts = np.zeros((k, k))
    np.add.at(counts, (s[:-1], s[1:]), 1.0)
    rows = counts.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    return counts / rows
