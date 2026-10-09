"""Stationary states: H psi = E psi.

Separating psi(x, t) = phi(x) exp(-i E t) turns the time-dependent equation
into an eigenvalue problem. On the grid it is a matrix eigenvalue problem,
solved here for the lowest few states, and checked against the three closed
forms every textbook derives: the harmonic oscillator, the infinite well,
and (approximately) the WKB spectrum of the quartic oscillator.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse.linalg as spla
from scipy.integrate import quad
from scipy.special import eval_hermite

from qphys.schrodinger.grid import Grid, hamiltonian


def eigenstates(grid: Grid, V: np.ndarray, k: int = 6, order: int = 4):
    """The k lowest (E_n, phi_n), phi_n normalised and with phi_n > 0 at its first peak.

    Shift-invert around the bottom of the potential finds the low end of the
    spectrum quickly; the sign convention removes the arbitrary global sign
    so that states can be compared with closed forms pointwise.
    """
    H = hamiltonian(grid, V, order)
    # ARPACK starts from a random vector unless given one; a fixed one makes
    # results/ reproducible to the last digit. It is random rather than
    # constant because a constant vector is even, and nearly orthogonal to
    # every odd state of a symmetric potential.
    v0 = np.random.default_rng(0).standard_normal(grid.n)
    E, vecs = spla.eigsh(H, k=k, sigma=float(V.min()) - 1.0, which="LM", v0=v0)
    idx = np.argsort(E)
    E, vecs = E[idx], vecs[:, idx]
    states = []
    for j in range(k):
        phi = grid.normalise(vecs[:, j])
        if phi[np.argmax(np.abs(phi) > 1e-3 * np.abs(phi).max())] < 0:
            phi = -phi
        states.append(phi)
    return E, np.array(states)


# --- Closed forms -----------------------------------------------------------


def harmonic_energy(n: int | np.ndarray, omega: float = 1.0):
    return omega * (np.asarray(n) + 0.5)


def harmonic_state(x: np.ndarray, n: int, omega: float = 1.0) -> np.ndarray:
    xi = np.sqrt(omega) * x
    norm = (omega / np.pi) ** 0.25 / math.sqrt(2.0**n * math.factorial(n))
    return norm * eval_hermite(n, xi) * np.exp(-(xi**2) / 2)


def box_energy(n: int | np.ndarray, length: float):
    """E_n = n^2 pi^2 / (2 L^2), n = 1, 2, ..."""
    return (np.asarray(n) * np.pi / length) ** 2 / 2


def quartic_wkb_energy(n: int | np.ndarray, g: float = 1.0):
    """Bohr-Sommerfeld for V = g x^4: oint p dx = 2 pi (n + 1/2).

    With turning points at a = (E/g)^(1/4) the action is
    2 sqrt(2) E^(3/4) g^(-1/4) I, where I = int_{-1}^{1} sqrt(1 - u^4) du,
    so E_n = [pi (n + 1/2) g^(1/4) / (sqrt(2) I)]^(4/3). The exponent 4/3 is
    what symbolic regression is asked to find from numerical eigenvalues.
    """
    I = quad(lambda u: np.sqrt(1 - u**4), -1, 1)[0]
    return (np.pi * (np.asarray(n) + 0.5) * g**0.25 / (np.sqrt(2) * I)) ** (4 / 3)
