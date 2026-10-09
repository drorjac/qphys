"""Reading the Schrodinger equation back out of psi(x, t).

The forward problem is: given H, compute psi(t). The inverse problem asked
here is: given only samples of psi(x, t), which linear equation
i d(psi)/dt = sum_j xi_j theta_j[psi] generated them? The candidate terms
theta_j are a library of plausible right-hand sides (derivatives up to
fourth order, powers of x times psi, a cubic nonlinearity), and sparse
regression (sequentially thresholded least squares, as in SINDy and PDE-FIND)
keeps the few that are needed.

The library is deliberately larger than the answer. Recovering
-1/2 psi_xx + 1/2 x^2 psi from it, and nothing else, is the test. The
coefficient of psi_xx is -hbar^2/(2m) in physical units, so its value is a
measurement of hbar/m from data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from qphys.schrodinger.evolve import gaussian_packet, split_operator
from qphys.schrodinger.grid import Grid, harmonic

TERMS = (
    "psi",
    "x psi",
    "x^2 psi",
    "x^3 psi",
    "x^4 psi",
    "psi_x",
    "psi_xx",
    "psi_xxx",
    "psi_xxxx",
    "|psi|^2 psi",
)
TRUTH = {"psi_xx": -0.5, "x^2 psi": 0.5}


@dataclass
class Snapshots:
    grid: Grid
    t: np.ndarray
    psi: np.ndarray  # (T, N)


def simulate(
    noise: float = 0.0, seed: int = 0, frames: int = 400, dt_frame: float = 0.02
) -> Snapshots:
    """psi(x, t) of a displaced, moving packet in the harmonic well, with optional noise.

    Noise is complex Gaussian with standard deviation `noise` times max|psi|,
    added independently to every sample: the model of a measured field.
    """
    grid = Grid(-12.0, 12.0, 256)
    V = harmonic(grid.x)
    psi0 = gaussian_packet(grid, x0=2.0, k0=1.0, sigma=0.8)
    sub = 10
    ev = split_operator(grid, V, psi0, dt_frame / sub, frames * sub, every=sub)
    psi = ev.psi
    if noise > 0:
        rng = np.random.default_rng(seed)
        scale = noise * np.abs(psi).max() / np.sqrt(2)
        psi = psi + scale * (
            rng.standard_normal(psi.shape) + 1j * rng.standard_normal(psi.shape)
        )
    return Snapshots(grid, ev.t, psi)


def _denoise_space(grid: Grid, psi: np.ndarray) -> tuple[np.ndarray, float]:
    """Spectral low-pass at the wavenumber where signal power meets the noise floor.

    The cut is read off the data's own mean power spectrum: the floor is the
    median power in the top quarter of wavenumbers, and the cut is the
    largest |k| whose power is still ten times above it. No knowledge of the
    true state is used.
    """
    F = np.fft.fft(psi, axis=1)
    power = (np.abs(F) ** 2).mean(axis=0)
    ak = np.abs(grid.k)
    floor = np.median(power[ak > 0.75 * ak.max()])
    above = ak[power > 10 * floor]
    k_cut = float(above.max()) if above.size else float(ak.max())
    F[:, ak > k_cut] = 0
    return F, k_cut


def library(snap: Snapshots, smooth: bool = True):
    """The time derivative and the candidate terms, flattened over (t, x).

    Spatial derivatives are spectral; the time derivative is a Savitzky-Golay
    derivative along t. Points where |psi| is below 5% of its maximum are
    excluded, because there the terms are mostly noise.
    """
    g = snap.grid
    if smooth:
        F, k_cut = _denoise_space(g, snap.psi)
    else:
        F, k_cut = np.fft.fft(snap.psi, axis=1), float(np.abs(g.k).max())
    ik = 1j * g.k
    psi = np.fft.ifft(F, axis=1)
    d = [np.fft.ifft(F * ik**n, axis=1) for n in range(1, 5)]
    dt = snap.t[1] - snap.t[0]
    window = 15 if smooth else 5
    psi_t = savgol_filter(
        psi.real, window, 4, deriv=1, delta=dt, axis=0
    ) + 1j * savgol_filter(psi.imag, window, 4, deriv=1, delta=dt, axis=0)
    x = g.x[None, :]
    cols = [
        psi,
        x * psi,
        x**2 * psi,
        x**3 * psi,
        x**4 * psi,
        *d,
        np.abs(psi) ** 2 * psi,
    ]
    edge = window // 2
    inner = slice(edge, -edge)
    mask = np.abs(psi[inner]) > 0.05 * np.abs(psi).max()
    Theta = np.stack([c[inner][mask] for c in cols], axis=1)
    target = 1j * psi_t[inner][mask]
    return target, Theta, k_cut


def stlsq(
    target: np.ndarray, Theta: np.ndarray, threshold: float = 0.05, iters: int = 10
) -> np.ndarray:
    """Sequentially thresholded least squares on column-normalised terms.

    Columns are scaled to unit norm before thresholding so that a small
    coefficient on a large term is not confused with a small contribution.
    """
    scale = np.linalg.norm(Theta, axis=0)
    A = Theta / scale
    xi = np.linalg.lstsq(A, target, rcond=None)[0]
    for _ in range(iters):
        contribution = np.abs(xi) / np.abs(xi).max()
        keep = contribution >= threshold
        xi = np.zeros_like(xi)
        xi[keep] = np.linalg.lstsq(A[:, keep], target, rcond=None)[0]
    return xi / scale


def discover(noise: float = 0.0, seed: int = 0, threshold: float = 0.05) -> dict:
    snap = simulate(noise=noise, seed=seed)
    target, Theta, k_cut = library(snap, smooth=noise > 0)
    xi = stlsq(target, Theta, threshold)
    resid = np.linalg.norm(target - Theta @ xi) / np.linalg.norm(target)
    found = {name: complex(c) for name, c in zip(TERMS, xi, strict=True) if c != 0}
    return {
        "noise": noise,
        "seed": seed,
        "k_cut": k_cut,
        "rel_residual": float(resid),
        "terms": found,
    }


def equation_string(terms: dict[str, complex], digits: int = 4) -> str:
    parts = []
    for name, c in terms.items():
        val = c.real if abs(c.imag) < 1e-3 * max(abs(c), 1e-12) else c
        parts.append(
            f"{val:+.{digits}f} {name}"
            if isinstance(val, float)
            else f"({val:.{digits}f}) {name}"
        )
    return "i psi_t = " + " ".join(parts)
