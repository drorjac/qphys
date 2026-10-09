"""Time evolution: psi(t) = exp(-i H t) psi(0).

The split-operator method (Strang splitting) alternates half a potential
step in position space with a full kinetic step in momentum space:

    psi <- exp(-i V dt/2) F^-1 exp(-i k^2 dt/2) F exp(-i V dt/2) psi

Each factor is a pure phase, so every step is exactly unitary and the norm
is conserved to rounding error whatever dt is. Accuracy is a separate
question: the splitting error is O(dt^3) per step, O(dt^2) over a fixed
time, and is measured against the exact eigen-expansion in
`experiments.conservation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qphys.schrodinger.grid import Grid


def gaussian_packet(grid: Grid, x0: float, k0: float, sigma: float) -> np.ndarray:
    """A minimum-uncertainty packet centred at x0 with mean momentum k0.

    It has <x> = x0, <p> = k0, dx = sigma, dp = 1/(2 sigma): the product
    sits exactly on the Heisenberg bound 1/2.
    """
    x = grid.x
    psi = np.exp(-((x - x0) ** 2) / (4 * sigma**2) + 1j * k0 * x)
    return grid.normalise(psi)


@dataclass
class Evolution:
    t: np.ndarray  # (T,)
    psi: np.ndarray  # (T, N) complex
    grid: Grid
    V: np.ndarray


def split_operator(
    grid: Grid,
    V: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    steps: int,
    every: int = 1,
) -> Evolution:
    """Evolve psi0 for `steps` steps of dt, keeping every `every`-th frame."""
    half_v = np.exp(-0.5j * V * dt)
    kinetic = np.exp(-0.5j * grid.k**2 * dt)
    psi = psi0.astype(complex)
    frames, times = [psi.copy()], [0.0]
    for s in range(1, steps + 1):
        psi = half_v * np.fft.ifft(kinetic * np.fft.fft(half_v * psi))
        if s % every == 0:
            frames.append(psi.copy())
            times.append(s * dt)
    return Evolution(np.array(times), np.array(frames), grid, V)


def eigen_expansion(grid: Grid, energies, states, psi0, t: np.ndarray) -> np.ndarray:
    """The exact evolution within the span of the given eigenstates.

    psi(t) = sum_n c_n exp(-i E_n t) phi_n, with c_n = <phi_n | psi0>. Used as
    the reference the split-operator integrator is measured against.
    """
    c = states.conj() @ psi0 * grid.dx
    phases = np.exp(-1j * np.outer(t, energies))
    return (phases * c) @ states


# --- Observables ------------------------------------------------------------


def norm(grid: Grid, psi: np.ndarray) -> float:
    return float(grid.integrate(np.abs(psi) ** 2).real)


def expect_x(grid: Grid, psi: np.ndarray) -> float:
    return float(grid.integrate(grid.x * np.abs(psi) ** 2).real)


def momentum_density(grid: Grid, psi: np.ndarray) -> np.ndarray:
    """|psi(k)|^2 normalised as a density over k."""
    phi = np.fft.fft(psi)
    dens = np.abs(phi) ** 2
    dk = 2 * np.pi / (grid.n * grid.dx)
    return dens / (dens.sum() * dk)


def expect_p(grid: Grid, psi: np.ndarray) -> float:
    dk = 2 * np.pi / (grid.n * grid.dx)
    return float((grid.k * momentum_density(grid, psi)).sum() * dk)


def spread_x(grid: Grid, psi: np.ndarray) -> float:
    m = expect_x(grid, psi)
    return float(np.sqrt(grid.integrate((grid.x - m) ** 2 * np.abs(psi) ** 2).real))


def spread_p(grid: Grid, psi: np.ndarray) -> float:
    dk = 2 * np.pi / (grid.n * grid.dx)
    rho = momentum_density(grid, psi)
    m = (grid.k * rho).sum() * dk
    return float(np.sqrt(((grid.k - m) ** 2 * rho).sum() * dk))


def energy(grid: Grid, V: np.ndarray, psi: np.ndarray) -> float:
    """<H> with the kinetic part evaluated spectrally, as the integrator sees it."""
    dk = 2 * np.pi / (grid.n * grid.dx)
    kinetic = (0.5 * grid.k**2 * momentum_density(grid, psi)).sum() * dk
    potential = grid.integrate(V * np.abs(psi) ** 2).real
    return float(kinetic + potential)


def ehrenfest_residual(ev: Evolution) -> float:
    """max |d<p>/dt + <V'(x)>| over the run, relative to max |<V'(x)>|.

    Ehrenfest's theorem says the mean momentum obeys Newton's law with the
    force averaged over |psi|^2. This is the bridge from the wave equation
    back to classical mechanics, and a check that the integrator respects it.
    """
    g = ev.grid
    p = np.array([expect_p(g, f) for f in ev.psi])
    dVdx = np.gradient(ev.V, g.dx)
    force = np.array([-g.integrate(dVdx * np.abs(f) ** 2).real for f in ev.psi])
    dpdt = np.gradient(p, ev.t)
    inner = slice(1, -1)
    return float(np.abs(dpdt[inner] - force[inner]).max() / np.abs(force).max())
