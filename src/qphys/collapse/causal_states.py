"""Statistical complexity, classical and quantum.

`C_mu` is the Shannon entropy of the minimal classical predictive states of
an epsilon-machine. `C_q` is the von Neumann entropy of the quantum causal
states, which may be NON-ORTHOGONAL and therefore compress below the
classical bound:

    C_q <= C_mu, with equality only in the trivial cases.

Gu, Wiesner, Rieper, Vedral, Nat. Commun. 3:762 (2012).

The state-merging step is the part that breaks silently. Two histories
belong to the same causal state when they predict the SAME future
distribution; at p = 0.5 the perturbed coin's two states become predictively
identical and merge, so `C_mu` drops discontinuously to 0 rather than
staying at 1. An implementation that returns 1 there has no merging.

This entire module is a statement about an idealised construction from a
known or estimated transition matrix. It says nothing about whether a fitted
model predicts well -- see the README claims table, which keeps the
representation result and the prediction result strictly apart.
"""

from __future__ import annotations

import numpy as np

from qphys.common.metrics import shannon_bits, von_neumann_bits

MERGE_TOL = 1e-9


def merge_states(conditionals: np.ndarray, tol: float = MERGE_TOL):
    """Group rows that predict the same future. Returns (labels, unique rows).

    This is the epsilon-machine's defining equivalence relation, and the
    reason `C_mu` is discontinuous at p = 0.5.
    """
    rows = np.atleast_2d(np.asarray(conditionals, float))
    labels = np.full(len(rows), -1, dtype=int)
    uniq: list = []
    for i, r in enumerate(rows):
        for j, u in enumerate(uniq):
            if np.max(np.abs(r - u)) <= tol:
                labels[i] = j
                break
        else:
            labels[i] = len(uniq)
            uniq.append(r)
    return labels, np.array(uniq)


def stationary(transition: np.ndarray) -> np.ndarray:
    """The stationary distribution, as the left eigenvector at eigenvalue 1."""
    p = np.asarray(transition, float)
    w, v = np.linalg.eig(p.T)
    i = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, i])
    pi = np.abs(pi)
    return pi / pi.sum()


def c_mu(transition: np.ndarray, tol: float = MERGE_TOL) -> float:
    """Classical statistical complexity of a first-order chain, in bits.

    Merges predictively identical states BEFORE taking the entropy. Without
    that step this returns 1 bit for a fair coin, which is wrong: a fair coin
    has no memory at all.
    """
    p = np.asarray(transition, float)
    labels, uniq = merge_states(p, tol)
    if len(uniq) <= 1:
        return 0.0
    pi = stationary(p)
    merged = np.zeros(len(uniq))
    for i, lab in enumerate(labels):
        merged[lab] += pi[i]
    return shannon_bits(merged)


def quantum_causal_states(transition: np.ndarray) -> np.ndarray:
    """|s_i> = sum_j sqrt(P(j|i)) |j>, one pure state per causal state.

    For a unifilar chain whose output names the next state, the overlap
    <s_i|s_j> = sum_k sqrt(P(k|i) P(k|j)) is nonzero whenever two rows share
    support -- and that non-orthogonality is exactly the compression.
    """
    p = np.asarray(transition, float)
    return np.sqrt(np.clip(p, 0.0, None))


def rho_causal(transition: np.ndarray, tol: float = MERGE_TOL) -> np.ndarray:
    p = np.asarray(transition, float)
    labels, uniq = merge_states(p, tol)
    if len(uniq) <= 1:
        return np.ones((1, 1))
    pi = stationary(p)
    weights = np.zeros(len(uniq))
    for i, lab in enumerate(labels):
        weights[lab] += pi[i]
    kets = quantum_causal_states(uniq)
    return sum(w * np.outer(k, k) for w, k in zip(weights, kets))


def c_q(transition: np.ndarray, tol: float = MERGE_TOL) -> float:
    """Quantum statistical complexity, in bits. Never exceeds `c_mu`."""
    return von_neumann_bits(rho_causal(transition, tol))


# --- the perturbed coin, where both quantities are closed form -------------


def perturbed_coin_transition(p: float) -> np.ndarray:
    """Flip with probability `p`; the output is the current state."""
    return np.array([[1.0 - p, p], [p, 1.0 - p]])


def perturbed_coin_overlap(p: float) -> float:
    return float(2.0 * np.sqrt(p * (1.0 - p)))


def perturbed_coin_c_q(p: float) -> float:
    """H2((1 + 2 sqrt(p(1-p))) / 2), the closed form."""
    from qphys.common.metrics import h2

    return h2((1.0 + perturbed_coin_overlap(p)) / 2.0)


def perturbed_coin_eigenvalues(p: float) -> np.ndarray:
    c = perturbed_coin_overlap(p)
    return np.array([(1.0 - c) / 2.0, (1.0 + c) / 2.0])
