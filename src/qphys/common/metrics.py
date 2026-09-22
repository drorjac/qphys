"""Scoring, in bits, at matched capacity.

Honesty rule 3: models are compared at equal free-parameter count, and the
count is stated. `free_params` is therefore a function with a docstring
rather than a number typed into a table.
"""

from __future__ import annotations

import numpy as np

LOG2 = np.log(2.0)


def nll_bits(log_probs) -> float:
    """Mean negative log-likelihood in bits per symbol.

    Bits rather than nats so the numbers sit beside entropies (`C_mu`, `C_q`)
    and an i.i.d. baseline of ~1 bit on a near-balanced binary series reads
    as what it is.
    """
    lp = np.asarray(log_probs, float)
    if not np.all(np.isfinite(lp)):
        return float("inf")
    return float(-np.mean(lp) / LOG2)


def h2(p: float) -> float:
    """Binary Shannon entropy in bits, with the 0 log 0 = 0 convention."""
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def shannon_bits(probs) -> float:
    p = np.asarray(probs, float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def von_neumann_bits(rho) -> float:
    """Entropy of a density matrix, in bits: the quantum memory cost."""
    w = np.linalg.eigvalsh(np.asarray(rho))
    w = np.real(w)
    w = w[w > 1e-15]
    return float(-np.sum(w * np.log2(w)))


def free_params(model: str, **kw) -> int:
    """Free real parameters, counted explicitly so rule 3 can be checked.

    `hqmm`: the Kraus stack is an (n_out*d, d) complex matrix with
    orthonormal columns, i.e. a point on the complex Stiefel manifold
    V_d(C^(n_out*d)), whose real dimension is 2*n_out*d^2 - d^2. From that
    subtract one phase per Kraus operator (K_x -> e^(i phi) K_x leaves the
    probabilities unchanged) and the d^2 - 1 dimensions of the conjugation
    gauge (rho -> U rho U^dag applied to every operator at once).

    `markov`: 2^order transition rows, one free probability each.
    `hmm`: k^2 - k transition + k - 1 initial + k emission, for binary output.
    """
    if model == "hqmm":
        d, n_out = int(kw["d"]), int(kw["n_out"])
        return 2 * n_out * d**2 - d**2 - n_out - (d**2 - 1)
    if model == "markov":
        return int(2 ** int(kw["order"]))
    if model == "hmm":
        k = int(kw["k"])
        return (k * k - k) + (k - 1) + k
    if model == "iid":
        return 1
    raise ValueError(f"unknown model {model!r}")
