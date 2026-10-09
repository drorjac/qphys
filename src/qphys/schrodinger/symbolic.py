"""Closed forms found by symbolic regression from numerical solutions.

Three questions, each answered by the solver and then handed to a symbolic
regressor that is not told the answer:

    box_law        E(n, L) for the infinite well. Expected: pi^2 n^2 / (2 L^2).
    quartic_law    E_n for V = x^4. WKB predicts C (n + 1/2)^(4/3), exact
                   only as n grows; the regressor sees exact numerics.
    potential      V(x) recovered from the ground-state DENSITY alone, via
                   V - E0 = (sqrt rho)'' / (2 sqrt rho). Expected for the
                   double well: x^4/4 - x^2 + const.

The regressor is PySR (genetic programming over expression trees, Julia
backend) when it is installed. Without it each task falls back to the
regression that needs only numpy: a log-linear fit for the power laws and a
sparse polynomial fit for the potential. These are restricted to a family
chosen in advance, so their results are reported under a different backend
name and are not the same claim.
"""

from __future__ import annotations

import tempfile

import numpy as np

from qphys.schrodinger import grid as G
from qphys.schrodinger import stationary as S
from qphys.schrodinger.discover import stlsq


def has_pysr() -> bool:
    try:
        import pysr  # noqa: F401
    except ImportError:
        return False
    return True


# --- Data -------------------------------------------------------------------


def box_data(n_levels: int = 6, lengths=(1.0, 1.5, 2.0, 3.0, 4.0), points: int = 400):
    rows = []
    for L in lengths:
        g = G.box_grid(L, points)
        E, _ = S.eigenstates(g, np.zeros(points), n_levels)
        rows += [(n + 1, L, e) for n, e in enumerate(E)]
    a = np.array(rows)
    return a[:, :2], a[:, 2]


def quartic_data(n_levels: int = 20):
    g = G.Grid(-12.0, 12.0, 3000)
    E, _ = S.eigenstates(g, G.quartic(g.x), n_levels)
    return np.arange(n_levels, dtype=float)[:, None], E


def potential_data(a: float = 1.0, b: float = 0.25, x_range: float = 2.5):
    """(x, V_rec) from the double-well ground-state density only.

    rho is the only input. Its square root is the ground state (nodeless and
    real), and the time-independent equation rearranged gives V up to the
    unknown constant E0. Only the region where rho is not tiny is kept,
    because there the ratio is division by noise.
    """
    g = G.Grid(-8.0, 8.0, 1600)
    _, states = S.eigenstates(g, G.double_well(g.x, a, b), 1)
    rho = states[0] ** 2
    amp = np.sqrt(rho)
    lap = np.gradient(np.gradient(amp, g.dx), g.dx)
    V_rec = lap / (2 * amp)
    keep = np.abs(g.x) <= x_range
    return g.x[keep][:, None], V_rec[keep]


# --- Regressors ---------------------------------------------------------------


def _pysr(X, y, names, seed=0, niterations=60, extra=None):
    from pysr import PySRRegressor

    kw = {
        "niterations": niterations,
        "binary_operators": ["+", "-", "*", "/", "^"],
        "unary_operators": [],
        "maxsize": 16,
        "populations": 15,
        "constraints": {"^": (-1, 3)},
        "model_selection": "best",
        "deterministic": True,
        "parallelism": "serial",
        "random_state": seed,
        "progress": False,
        "verbosity": 0,
        "temp_equation_file": True,
        "output_directory": tempfile.mkdtemp(prefix="qphys-pysr-"),
    }
    kw.update(extra or {})
    model = PySRRegressor(**kw)
    model.fit(X, y, variable_names=names)
    pred = model.predict(X)
    return str(model.sympy()), float(np.sqrt(np.mean((pred - y) ** 2)))


def _round(expr: str, digits: int = 4) -> str:
    import sympy

    e = sympy.sympify(expr)
    e = e.xreplace(
        {f: sympy.Float(round(float(f), digits)) for f in e.atoms(sympy.Float)}
    )
    return str(sympy.simplify(e)) if e.free_symbols else str(e)


def box_law(backend: str = "auto", seed: int = 0) -> dict:
    X, y = box_data()
    true = "pi^2 n^2 / (2 L^2) = 4.9348 n^2 / L^2"
    if _use(backend) == "pysr":
        expr, rmse = _pysr(X, y, ["n", "L"], seed)
        return {
            "task": "box_law",
            "backend": "pysr",
            "expression": _round(expr),
            "truth": true,
            "rmse": rmse,
        }
    A = np.column_stack([np.log(X), np.ones(len(y))])
    p, q, c = np.linalg.lstsq(A, np.log(y), rcond=None)[0]
    pred = np.exp(c) * X[:, 0] ** p * X[:, 1] ** q
    expr = f"{np.exp(c):.4f} * n^{p:.4f} * L^{q:.4f}"
    return {
        "task": "box_law",
        "backend": "loglinear",
        "expression": expr,
        "truth": true,
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
    }


def _asymptote(f, n=1e6) -> tuple[float, float]:
    """(p, C) such that f(n) ~ C n^p as n grows, read off numerically."""
    p = float(np.log(f(2 * n) / f(n)) / np.log(2))
    return p, float(f(n) / n**p)


def quartic_law(backend: str = "auto", seed: int = 0) -> dict:
    """E_n for V = x^4, fitted on n = 2..39, where WKB is within 0.6%.

    The WKB form is only asymptotic, so a regressor may well return a
    different expression that fits as well. What is compared is the
    expression's large-n behaviour C n^p against WKB's 1.3765 n^(4/3).
    """
    X, y = quartic_data(40)
    m = X[:, 0] >= 2
    X, y = X[m], y[m]
    C = float(S.quartic_wkb_energy(0) / 0.5 ** (4 / 3))
    true = f"WKB: {C:.4f} (n + 1/2)^(4/3)"
    if _use(backend) == "pysr":
        import sympy

        expr, rmse = _pysr(X, y, ["n"], seed, niterations=200)
        f = sympy.lambdify(sympy.Symbol("n"), sympy.sympify(expr), "numpy")
        backend_name, expression = "pysr", _round(expr)
    else:
        A = np.column_stack([np.log(X[:, 0] + 0.5), np.ones(len(y))])
        p, c = np.linalg.lstsq(A, np.log(y), rcond=None)[0]

        def f(n):
            return np.exp(c) * (n + 0.5) ** p

        rmse = float(np.sqrt(np.mean((f(X[:, 0]) - y) ** 2)))
        backend_name, expression = "loglinear", f"{np.exp(c):.4f} * (n + 0.5)^{p:.4f}"
    p_inf, C_inf = _asymptote(f)
    return {
        "task": "quartic_law",
        "backend": backend_name,
        "expression": expression,
        "truth": true,
        "rmse": rmse,
        "asymptotic_exponent": p_inf,
        "asymptotic_prefactor": C_inf,
    }


def potential(backend: str = "auto", seed: int = 0) -> dict:
    X, y = potential_data()
    true = "0.25 x^4 - x^2 + const"
    if _use(backend) == "pysr":
        expr, rmse = _pysr(X, y, ["x"], seed)
        return {
            "task": "potential",
            "backend": "pysr",
            "expression": _round(expr),
            "truth": true,
            "rmse": rmse,
        }
    x = X[:, 0]
    Theta = np.column_stack([x**k for k in range(7)])
    xi = stlsq(y.astype(complex), Theta.astype(complex), threshold=0.02).real
    terms = [f"{c:+.4f} x^{k}" for k, c in enumerate(xi) if c != 0]
    pred = Theta @ xi
    return {
        "task": "potential",
        "backend": "sparse-poly",
        "expression": " ".join(terms),
        "truth": true,
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
    }


def _use(backend: str) -> str:
    if backend == "auto":
        return "pysr" if has_pysr() else "fallback"
    return backend


TASKS = (box_law, quartic_law, potential)
