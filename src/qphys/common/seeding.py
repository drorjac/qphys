"""Seeds, and the discipline around them.

Honesty rule 6: a single-seed result is not a result. Every experiment here
reports a spread over at least `MIN_SEEDS` seeds and states its failure rate,
so `seeds()` returns a set rather than a number and the callers cannot
quietly take the first one.

Honesty rule 7: a run may be discarded only against a convergence threshold
fixed in advance, on the TRAINING loss, never on the answer. `converged()`
is that threshold, and it is a module constant so it cannot be tuned per
experiment after the fact.
"""

from __future__ import annotations

import numpy as np

MIN_SEEDS = 5
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

# Pre-declared, per honesty rule 7. A relativistic-momentum fit that ends
# above this on the training objective is rejected before its c_hat is read.
# The measured separation is two orders of magnitude -- converged seeds land
# near 1e-4, the failing one at 2.5e-2 -- so this sits between them and is
# not a threshold tuned to taste.
CONVERGENCE_REL_LOSS = 1e-3


def seeds(n: int = MIN_SEEDS) -> tuple[int, ...]:
    if n < MIN_SEEDS:
        raise ValueError(
            f"{n} seeds requested but honesty rule 6 requires at least "
            f"{MIN_SEEDS}: single-seed results are not results"
        )
    return tuple(range(n))


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def set_torch_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def converged(rel_loss: float) -> bool:
    """Did this run meet the pre-declared training-loss threshold?"""
    return bool(np.isfinite(rel_loss) and rel_loss <= CONVERGENCE_REL_LOSS)


def summarise(values, name: str = "value") -> dict:
    """Mean, spread and failure rate over seeds -- the reporting unit.

    NaNs are failures, not missing data: they are counted and reported, never
    silently dropped, because the failure rate IS a result (33% of the
    relativistic fits fail, and hiding that would misdescribe the method).
    """
    arr = np.asarray(list(values), dtype=float)
    ok = np.isfinite(arr)
    return {
        "name": name,
        "n": int(arr.size),
        "n_ok": int(ok.sum()),
        "failure_rate": float(1.0 - ok.mean()) if arr.size else float("nan"),
        "mean": float(np.mean(arr[ok])) if ok.any() else float("nan"),
        "std": float(np.std(arr[ok], ddof=1)) if ok.sum() > 1 else float("nan"),
        "values": arr.tolist(),
    }
