"""A wave packet hitting a rectangular barrier.

Classically a particle with E < V0 is always reflected. The Schrodinger
equation gives a non-zero transmission probability, decaying like
exp(-2 kappa a) with kappa = sqrt(2 (V0 - E)). Above the barrier it gives the
opposite surprise: reflection is non-zero, and transmission oscillates and
reaches exactly 1 at resonances where the barrier holds a whole number of
half wavelengths.

The plane-wave formula is for a state of one exact energy, which no particle
has. A packet carries a spread of momenta, so the fair prediction for a
packet is T(E) averaged over its momentum distribution. The simulation is
compared with both.
"""

from __future__ import annotations

import numpy as np

from qphys.schrodinger.evolve import gaussian_packet, momentum_density, split_operator
from qphys.schrodinger.grid import Grid, barrier


def transmission_plane_wave(E: np.ndarray | float, V0: float, a: float) -> np.ndarray:
    """T(E) for a barrier of height V0 and width a, hbar = m = 1."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    T = np.empty_like(E)
    below, above = E < V0, E > V0
    kappa = np.sqrt(2 * (V0 - E[below]))
    T[below] = 1 / (
        1 + V0**2 * np.sinh(kappa * a) ** 2 / (4 * E[below] * (V0 - E[below]))
    )
    q = np.sqrt(2 * (E[above] - V0))
    T[above] = 1 / (1 + V0**2 * np.sin(q * a) ** 2 / (4 * E[above] * (E[above] - V0)))
    at = ~(below | above)
    T[at] = 1 / (1 + V0 * a**2 / 2)
    return T


def tunnel(
    k0: float,
    V0: float = 1.0,
    a: float = 1.0,
    sigma: float = 4.0,
    x0: float = -30.0,
    grid: Grid | None = None,
    dt: float = 0.02,
) -> dict:
    """Send one packet at the barrier and count what comes out the far side."""
    grid = grid or Grid(-160.0, 160.0, 4096)
    V = barrier(grid.x, V0, a)
    psi0 = gaussian_packet(grid, x0, k0, sigma)
    t_end = 2 * abs(x0) / k0
    steps = int(np.ceil(t_end / dt))
    ev = split_operator(grid, V, psi0, dt, steps, every=steps)
    psi = ev.psi[-1]
    right = grid.x > a / 2
    T_sim = float((np.abs(psi[right]) ** 2).sum() * grid.dx)

    dk = 2 * np.pi / (grid.n * grid.dx)
    rho = momentum_density(grid, psi0)
    forward = grid.k > 0
    T_k = transmission_plane_wave(0.5 * grid.k[forward] ** 2, V0, a)
    T_packet = float((T_k * rho[forward]).sum() * dk / (rho[forward].sum() * dk))
    return {
        "k0": k0,
        "E0": 0.5 * k0**2,
        "V0": V0,
        "a": a,
        "sigma": sigma,
        "T_sim": T_sim,
        "T_packet": T_packet,
        "T_plane_wave": float(transmission_plane_wave(0.5 * k0**2, V0, a)[0]),
        "norm_end": float((np.abs(psi) ** 2).sum() * grid.dx),
    }
