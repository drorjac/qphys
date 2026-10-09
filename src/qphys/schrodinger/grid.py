"""The spatial grid and the discrete Hamiltonian.

A wavefunction on a computer is a vector of samples psi_j = psi(x_j). The
kinetic operator -1/2 d2/dx2 becomes a banded matrix, the potential a
diagonal one, and the Schrodinger equation becomes a linear ODE in C^N.
Everything else in this sub-project is built on these two objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Grid:
    """N equally spaced points on [x_min, x_max).

    The right end is excluded so that the same grid serves the FFT, which
    assumes periodicity. Physical results are only trusted while psi stays
    well away from both ends.
    """

    x_min: float = -20.0
    x_max: float = 20.0
    n: int = 1024

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.n

    @property
    def x(self) -> np.ndarray:
        return self.x_min + self.dx * np.arange(self.n)

    @property
    def k(self) -> np.ndarray:
        """Wavenumbers in FFT order. With hbar = 1 these are momenta."""
        return 2 * np.pi * np.fft.fftfreq(self.n, d=self.dx)

    def integrate(self, f: np.ndarray) -> float | complex:
        return f.sum() * self.dx

    def normalise(self, psi: np.ndarray) -> np.ndarray:
        return psi / np.sqrt(self.integrate(np.abs(psi) ** 2).real)


def laplacian(grid: Grid, order: int = 4) -> sp.csr_matrix:
    """Finite-difference d2/dx2 with Dirichlet ends.

    order=2 is the three-point stencil (error ~ dx^2), order=4 the five-point
    one (error ~ dx^4). The stationary-state convergence test measures both.
    """
    n, h2 = grid.n, grid.dx**2
    if order == 2:
        diags, offsets = [np.full(n, -2.0), np.ones(n - 1), np.ones(n - 1)], [0, -1, 1]
    elif order == 4:
        diags = [
            np.full(n, -30.0),
            np.full(n - 1, 16.0),
            np.full(n - 1, 16.0),
            np.full(n - 2, -1.0),
            np.full(n - 2, -1.0),
        ]
        offsets = [0, -1, 1, -2, 2]
        # The stencil reaches two points past each end. psi vanishes on the
        # wall one step out, and continues as an odd reflection, so the
        # point two steps out is -psi_0. Without this the error is O(dx).
        diags[0][[0, -1]] = -29.0
        diags = [d / 12.0 for d in diags]
    else:
        raise ValueError(f"order must be 2 or 4, not {order}")
    return sp.diags(diags, offsets, format="csr") / h2


def hamiltonian(grid: Grid, V: np.ndarray, order: int = 4) -> sp.csr_matrix:
    """H = -1/2 d2/dx2 + V(x), a real symmetric sparse matrix.

    Symmetric means Hermitian here, and that single property is what makes
    energies real and time evolution norm-preserving.
    """
    return (-0.5 * laplacian(grid, order) + sp.diags(V)).tocsr()


# --- Potentials -------------------------------------------------------------


def harmonic(x: np.ndarray, omega: float = 1.0) -> np.ndarray:
    return 0.5 * omega**2 * x**2


def quartic(x: np.ndarray, g: float = 1.0) -> np.ndarray:
    return g * x**4


def double_well(x: np.ndarray, a: float = 1.0, b: float = 0.25) -> np.ndarray:
    """V = b x^4 - a x^2: two minima, a barrier at x = 0."""
    return b * x**4 - a * x**2


def barrier(
    x: np.ndarray, height: float, width: float, center: float = 0.0
) -> np.ndarray:
    """A rectangular barrier, cell-averaged.

    Each grid cell gets the barrier's height times the fraction of the cell
    it covers. Sampling the step pointwise instead makes the barrier as wide
    as the number of grid points inside it, which on a dx = 0.078 grid was
    1.6% too wide and cut the tunnelling probability by up to 3.0%, because
    transmission falls exponentially with width.
    """
    dx = x[1] - x[0]
    lo, hi = center - width / 2, center + width / 2
    overlap = np.clip(np.minimum(x + dx / 2, hi) - np.maximum(x - dx / 2, lo), 0, None)
    return height * overlap / dx


def box_grid(length: float, n: int) -> Grid:
    """The interior points of an infinite well [0, length].

    The Dirichlet Laplacian assumes psi = 0 one step beyond each end, so
    placing the grid at x_j = j L / (n + 1), j = 1..n, puts those zeros
    exactly on the walls. This is exact where a large finite wall is not.
    """
    dx = length / (n + 1)
    return Grid(dx, length, n)


def spectral_hamiltonian(grid: Grid, V: np.ndarray) -> np.ndarray:
    """The dense Hamiltonian whose kinetic term is exact for the grid's Fourier modes.

    This is the operator the split-operator integrator actually exponentiates,
    so its eigenstates are the right reference for measuring that
    integrator's time-stepping error alone.
    """
    F = np.fft.fft(np.eye(grid.n), axis=0)
    T = np.fft.ifft(0.5 * grid.k[:, None] ** 2 * F, axis=0)
    H = T + np.diag(V)
    return 0.5 * (H + H.conj().T)
