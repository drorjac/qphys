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
# Significance of the two-proportion test used when counts are supplied.
# 2.5 sigma: loose enough that sampling noise does not split a real state,
# tight enough that a genuinely different prediction survives.
MERGE_Z = 2.5


def merge_states(conditionals: np.ndarray, tol: float = MERGE_TOL, counts=None):
    """Group rows that predict the same future. Returns (labels, unique rows).

    This is the epsilon-machine's defining equivalence relation, and the
    reason `C_mu` is discontinuous at p = 0.5.

    **With `counts`, equality is a statistical test rather than float
    equality.** Two conditional distributions estimated from finite samples
    are never exactly equal, so an exact-equality rule merges nothing and
    `C_mu` climbs toward `order` bits for ANY process -- on Mersenne-Twister
    bits, which have no structure whatsoever, it returned 1.0, 2.0, 3.0, 4.0,
    4.7 and 5.5 bits at orders one to six. That is the state proliferation
    this estimate is famous for, and an exact-equality merge guarantees it.

    Passing the counts behind each row merges when the rows agree to within
    the sampling error of a two-proportion comparison,
    `z * sqrt(p(1-p) (1/n1 + 1/n2))`, which is the test CSSR makes.
    """
    rows = np.atleast_2d(np.asarray(conditionals, float))
    n = None if counts is None else np.asarray(counts, float)
    labels = np.full(len(rows), -1, dtype=int)
    uniq: list = []
    uniq_n: list = []
    for i, r in enumerate(rows):
        for j, u in enumerate(uniq):
            if n is None:
                same = np.max(np.abs(r - u)) <= tol
            else:
                pooled = (r * n[i] + u * uniq_n[j]) / (n[i] + uniq_n[j])
                se = np.sqrt(
                    np.clip(pooled * (1.0 - pooled), 0.0, None)
                    * (1.0 / n[i] + 1.0 / uniq_n[j])
                )
                same = bool(np.all(np.abs(r - u) <= MERGE_Z * se + tol))
            if same:
                labels[i] = j
                if n is not None:
                    w = uniq_n[j] / (uniq_n[j] + n[i])
                    uniq[j] = w * u + (1.0 - w) * r
                    uniq_n[j] = uniq_n[j] + n[i]
                break
        else:
            labels[i] = len(uniq)
            uniq.append(r)
            if n is not None:
                uniq_n.append(n[i])
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
    return sum(w * np.outer(k, k) for w, k in zip(weights, kets, strict=True))


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


# --- history length, and the failure mode it exposes -----------------------


def conditional_table(symbols, order: int, n_symbols: int = 2, min_count: int = 1):
    """P(next | history of `order` symbols), plus each history's weight.

    Histories seen fewer than `min_count` times are dropped rather than
    trusted: with 1461 points and order 8 there are 256 histories, most of
    them seen once or twice, and a distribution estimated from one sample is
    not a distribution.
    """
    s = np.asarray(symbols, int)
    n = len(s) - order
    if n <= 0:
        return np.zeros((0, n_symbols)), np.zeros(0), np.zeros(0)
    idx = np.zeros(n, dtype=int)
    for j in range(order):
        idx = idx * n_symbols + s[j : j + n]
    counts = np.zeros((n_symbols**order, n_symbols))
    np.add.at(counts, (idx, s[order:]), 1.0)
    totals = counts.sum(axis=1)
    keep = totals >= min_count
    counts, totals = counts[keep], totals[keep]
    if not len(totals):
        return np.zeros((0, n_symbols)), np.zeros(0), np.zeros(0)
    return counts / totals[:, None], totals / totals.sum(), totals


def complexity_at_order(
    symbols,
    order: int,
    n_symbols: int = 2,
    min_count: int = 1,
    statistical: bool = True,
):
    """`C_mu` and `C_q` estimated from histories of a given length.

    `statistical=False` restores the float-equality merge, which is kept so
    the failure mode can be DRAWN rather than described: on structureless
    bits it makes `C_mu` climb toward `order` bits.
    """
    cond, weight, counts = conditional_table(symbols, order, n_symbols, min_count)
    if not len(weight):
        return {"order": order, "n_states": 0, "C_mu": np.nan, "C_q": np.nan}
    labels, uniq = merge_states(cond, counts=counts if statistical else None)
    merged = np.zeros(len(uniq))
    for i, lab in enumerate(labels):
        merged[lab] += weight[i]
    kets = quantum_causal_states(uniq)
    rho = sum(w * np.outer(k, k) for w, k in zip(merged, kets, strict=True))
    return {
        "order": order,
        "n_states": len(uniq),
        "C_mu": shannon_bits(merged),
        "C_q": von_neumann_bits(rho),
    }


def complexity_vs_history(
    symbols, orders=(1, 2, 3, 4, 5, 6), min_count: int = 5, statistical: bool = True
):
    """The diagnostic sweep. **Both curves must plateau.**

    If `C_mu` keeps climbing with history length, the estimate is measuring
    finite-sample state proliferation rather than the process: at order `k`
    there are `2^k` possible histories, and once most of them are seen a
    handful of times each, no two of their conditional distributions are
    exactly equal, nothing merges, and the entropy of the resulting state
    distribution rises toward `k` bits whatever the process is.

    This is the most common way a statistical-complexity estimate goes wrong,
    so the sweep is part of the library rather than something to remember to
    do.
    """
    import pandas as pd

    return pd.DataFrame(
        [
            complexity_at_order(
                symbols, o, min_count=min_count, statistical=statistical
            )
            for o in orders
        ]
    )
