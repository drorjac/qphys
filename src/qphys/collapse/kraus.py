"""Hidden Quantum Markov Models: the observation IS the collapse.

A classical HMM carries a probability vector. An HQMM carries a density
matrix, and observing symbol `x` applies a Kraus operator and renormalises:

    rho -> K_x rho K_x^dag / p(x),   p(x) = tr(K_x rho K_x^dag)

with `sum_x K_x^dag K_x = I`. That constraint is the whole implementation
problem, and the QR trick below solves it for free: stack the operators into
an (n_out*d, d) complex matrix and take its QR decomposition. The orthonormal
columns satisfy the completeness relation to machine precision, so
unconstrained gradient descent stays on the Stiefel manifold without a
penalty term or a projection step.

Monras/Beige/Wiesner (2010); Srinivasan/Gordon/Boots (ICLR 2018).

Honesty rule 1: nothing here claims a series "is quantum". The only claim
under test is whether this model is more efficient than the best classical
model at matched capacity -- and on the Seattle data it is not (see the
README claims table).
"""

from __future__ import annotations

import numpy as np

from qphys.common.metrics import nll_bits


def kraus_random(d: int, n_out: int, rng: np.random.Generator) -> list:
    """`n_out` Kraus operators on a d-dimensional state, complete by QR."""
    a = rng.normal(size=(n_out * d, d)) + 1j * rng.normal(size=(n_out * d, d))
    q, _ = np.linalg.qr(a)
    return [q[x * d : (x + 1) * d, :] for x in range(n_out)]


def kraus_from_flat(theta: np.ndarray, d: int, n_out: int) -> list:
    """The same construction from a flat real vector, for optimisers.

    The QR is inside the parameterisation, so any real vector maps to a valid
    completely positive trace-preserving map and the optimiser never has to
    respect a constraint it does not know about.
    """
    t = np.asarray(theta, float).reshape(2, n_out * d, d)
    q, _ = np.linalg.qr(t[0] + 1j * t[1])
    return [q[x * d : (x + 1) * d, :] for x in range(n_out)]


def completeness_error(ks) -> float:
    """max |sum_x K_x^dag K_x - I|. The gate is 1e-12; QR gives ~2e-16."""
    d = ks[0].shape[1]
    total = sum(k.conj().T @ k for k in ks)
    return float(np.max(np.abs(total - np.eye(d))))


def rho_initial(d: int) -> np.ndarray:
    """The maximally mixed state: no knowledge before the first symbol."""
    return np.eye(d, dtype=complex) / d


def step_probabilities(ks, rho: np.ndarray) -> np.ndarray:
    return np.array([np.real(np.trace(k @ rho @ k.conj().T)) for k in ks])


def sequence_log_likelihood(ks, symbols, rho0=None) -> np.ndarray:
    """Per-symbol natural-log likelihood along the sequence.

    Returns one value per symbol rather than a sum, so a caller can score a
    test segment while conditioning the state on the training segment --
    which is what a temporal split requires (honesty rule 4).
    """
    d = ks[0].shape[1]
    rho = rho_initial(d) if rho0 is None else np.array(rho0, dtype=complex)
    out = np.empty(len(symbols))
    for i, x in enumerate(np.asarray(symbols, int)):
        k = ks[x]
        unnorm = k @ rho @ k.conj().T
        p = float(np.real(np.trace(unnorm)))
        if p <= 1e-300:
            out[i] = -np.inf
            return out
        out[i] = np.log(p)
        rho = unnorm / p
    return out


def final_state(ks, symbols, rho0=None) -> np.ndarray:
    d = ks[0].shape[1]
    rho = rho_initial(d) if rho0 is None else np.array(rho0, dtype=complex)
    for x in np.asarray(symbols, int):
        k = ks[x]
        unnorm = k @ rho @ k.conj().T
        p = float(np.real(np.trace(unnorm)))
        if p <= 1e-300:
            return rho
        rho = unnorm / p
    return rho


def nll_bits_per_symbol(ks, symbols, rho0=None) -> float:
    return nll_bits(sequence_log_likelihood(ks, symbols, rho0))


def sample(ks, n: int, rng: np.random.Generator, rho0=None) -> np.ndarray:
    d = ks[0].shape[1]
    rho = rho_initial(d) if rho0 is None else np.array(rho0, dtype=complex)
    out = np.empty(n, dtype=int)
    for i in range(n):
        p = step_probabilities(ks, rho)
        p = np.clip(p, 0.0, None)
        p /= p.sum()
        x = int(rng.choice(len(ks), p=p))
        out[i] = x
        k = ks[x]
        unnorm = k @ rho @ k.conj().T
        rho = unnorm / float(np.real(np.trace(unnorm)))
    return out
