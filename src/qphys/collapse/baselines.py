"""The classical models the quantum one has to beat, at matched capacity.

Honesty rule 3 makes this module load-bearing: a comparison against an
under-tuned baseline is not evidence. The HMM in particular collapses to a
degenerate single-emission solution from a bad initialisation and returns
exactly the i.i.d. likelihood, which would flatter the HQMM. So it gets
multiple random restarts with best-training-likelihood selection, and the
restart count is a parameter of the experiment rather than a default hidden
in a library call.
"""

from __future__ import annotations

import warnings

import numpy as np

from qphys.common.metrics import free_params, nll_bits

LAPLACE = 1e-9  # keeps a zero count from producing an infinite test NLL


def fit_iid(train, test) -> dict:
    p = float(np.mean(train))
    lp = np.where(np.asarray(test, int) == 1, np.log(p), np.log1p(-p))
    return {"model": "iid", "params": free_params("iid"), "nll": nll_bits(lp)}


def fit_markov(train, test, order: int = 1, n_symbols: int = 2) -> dict:
    """Maximum-likelihood order-k chain with Laplace smoothing.

    The test sequence is scored with its own history, including the last
    `order` symbols of TRAIN as the initial context -- otherwise the first
    test symbols are scored by a prior rather than by the model.
    """
    tr = np.asarray(train, int)
    te = np.asarray(test, int)
    k = n_symbols**order
    counts = np.full((k, n_symbols), LAPLACE)

    def ctx(seq, i):
        idx = 0
        for j in range(order):
            idx = idx * n_symbols + seq[i - order + j]
        return idx

    for i in range(order, len(tr)):
        counts[ctx(tr, i), tr[i]] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)

    joined = np.concatenate([tr[-order:], te])
    lp = [
        np.log(probs[ctx(joined, i), joined[i]])
        for i in range(order, len(joined))
    ]
    return {
        "model": f"markov-{order}",
        "params": free_params("markov", order=order),
        "nll": nll_bits(lp),
    }


def fit_hmm(train, test, k: int = 2, n_restarts: int = 20, seed: int = 0) -> dict:
    """Categorical HMM with random restarts, selected on TRAINING likelihood.

    Without the restarts, k = 2 and k = 3 both converge to a degenerate
    single-emission solution and return exactly the i.i.d. NLL. That is not
    the HMM's best fit, and reporting it would make the quantum comparison
    unfair in the HQMM's favour.
    """
    from hmmlearn import hmm

    tr = np.asarray(train, int).reshape(-1, 1)
    te = np.asarray(test, int).reshape(-1, 1)
    best, best_ll = None, -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(n_restarts):
            m = hmm.CategoricalHMM(
                n_components=k,
                n_iter=200,
                tol=1e-6,
                random_state=seed + r,
                init_params="ste",
            )
            try:
                m.fit(tr)
                ll = m.score(tr)
            except (ValueError, RuntimeError):
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
    if best is None:
        return {"model": f"hmm-{k}", "params": free_params("hmm", k=k), "nll": np.inf}

    # score the test segment conditioned on the training segment, so the
    # latent state is not reset to the prior at the split
    joined = np.vstack([tr, te])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll_joined = best.score(joined)
        ll_train = best.score(tr)
    nll = -(ll_joined - ll_train) / len(te) / np.log(2.0)
    return {
        "model": f"hmm-{k}",
        "params": free_params("hmm", k=k),
        "nll": float(nll),
        "restarts": n_restarts,
        "train_ll": float(best_ll),
    }


def fit_hqmm(
    train, test, d: int = 2, n_out: int = 2, n_restarts: int = 12, seed: int = 0
) -> dict:
    """Fit the Kraus stack by unconstrained minimisation of the training NLL.

    Unconstrained is the point: the QR inside `kraus_from_flat` means every
    real vector maps to a valid CPTP map, so there is no constraint for the
    optimiser to violate and no penalty term to tune.

    Scored on test with the state carried over from training, matching the
    HMM treatment above.
    """
    from scipy.optimize import minimize

    from qphys.collapse import kraus

    tr = np.asarray(train, int)
    te = np.asarray(test, int)
    size = 2 * n_out * d * d

    def objective(theta):
        ks = kraus.kraus_from_flat(theta, d, n_out)
        lp = kraus.sequence_log_likelihood(ks, tr)
        v = -np.mean(lp)
        return 1e6 if not np.isfinite(v) else v

    best, best_val = None, np.inf
    for r in range(n_restarts):
        rng = np.random.default_rng(seed + r)
        res = minimize(
            objective,
            rng.normal(size=size),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if res.fun < best_val:
            best, best_val = res.x, res.fun

    ks = kraus.kraus_from_flat(best, d, n_out)
    rho_after_train = kraus.final_state(ks, tr)
    nll = kraus.nll_bits_per_symbol(ks, te, rho_after_train)
    return {
        "model": f"hqmm-d{d}",
        "params": free_params("hqmm", d=d, n_out=n_out),
        "nll": float(nll),
        "restarts": n_restarts,
        "train_nll": float(best_val / np.log(2.0)),
    }
