"""What psi predicts: where the particle is found.

psi is not a position. It predicts a distribution of positions: a detector
placed across the whole line clicks at x with probability density |psi(x)|^2
(the Born rule). One measurement gives one point; only the histogram of many
identically prepared runs approaches |psi|^2.

This module draws those points and measures how fast the histogram
converges, which is the operational content of "psi predicts the location".
"""

from __future__ import annotations

import numpy as np

from qphys.schrodinger.grid import Grid


def sample_positions(grid: Grid, psi: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """n independent detections, each from a fresh copy of psi.

    Inverse-CDF sampling on the grid, jittered uniformly within a cell so the
    samples are continuous.
    """
    rng = np.random.default_rng(seed)
    p = np.abs(psi) ** 2
    p = p / p.sum()
    idx = rng.choice(grid.n, size=n, p=p)
    return grid.x[idx] + grid.dx * (rng.random(n) - 0.5)


def total_variation(
    grid: Grid, psi: np.ndarray, samples: np.ndarray, bins: int = 80
) -> float:
    """TV distance between the sample histogram and |psi|^2 on the same bins.

    0 means identical, 1 means disjoint. For a correct sampler it shrinks
    like 1/sqrt(n); the experiment fits that exponent rather than assuming it.
    """
    rho = np.abs(psi) ** 2
    lo, hi = _support(grid, rho)
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(samples, edges)
    emp = counts / counts.sum()
    cdf = np.concatenate([[0.0], np.cumsum(rho) * grid.dx])
    xs = np.concatenate([grid.x - grid.dx / 2, [grid.x[-1] + grid.dx / 2]])
    theo = np.diff(np.interp(edges, xs, cdf))
    theo = theo / theo.sum()
    return float(0.5 * np.abs(emp - theo).sum())


def _support(grid: Grid, rho: np.ndarray, tail: float = 1e-6) -> tuple[float, float]:
    cdf = np.cumsum(rho) / rho.sum()
    return float(grid.x[np.searchsorted(cdf, tail)]), float(
        grid.x[np.searchsorted(cdf, 1 - tail)]
    )


def collapse(
    grid: Grid, psi: np.ndarray, x_detected: float, resolution: float
) -> np.ndarray:
    """The state right after a detector of finite resolution fires at x_detected.

    A position measurement with Gaussian resolution multiplies psi by the
    detector's window and renormalises. The resulting narrow packet then
    spreads again under the Schrodinger equation; the tutorial shows both.
    """
    window = np.exp(-((grid.x - x_detected) ** 2) / (4 * resolution**2))
    return grid.normalise(window * psi)
