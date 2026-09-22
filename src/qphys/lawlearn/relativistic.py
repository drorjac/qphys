"""The relativistic particle, symbolically -- and the 1/8 vs 3/8 test.

Natural units m = c = 1 unless stated. The Lagrangian of a free relativistic
particle in a potential is

    L = -m c^2 sqrt(1 - v^2/c^2) - V(x)

and the whole content of special relativity, for this project, is a statement
about the canonical momentum:

    p = dL/dv = gamma m v          Newton says p = m v

## The test this module exists for

Expanding in v, the Lagrangian and the kinetic energy share their v^2 term
and differ at v^4:

    L = -m c^2 + (1/2) m v^2 + (1/8) m v^4/c^2 + ...
    T = m c^2 (gamma - 1) = (1/2) m v^2 + (3/8) m v^4/c^2 + ...

So a network that recovers 3/8 has learned the kinetic ENERGY, not the
Lagrangian, and its Euler-Lagrange residual has a sign or factor error. The
v^2 term cannot tell the two apart, which is exactly why this check is worth
having: the leading order agrees and only the correction disagrees.
"""

from __future__ import annotations

import sympy as sp

v, c, m, x = sp.symbols("v c m x", real=True, positive=True)


def lagrangian_series(order: int = 6):
    """Series of L = -m c^2 sqrt(1 - v^2/c^2) in v, about v = 0."""
    lag = -m * c**2 * sp.sqrt(1 - v**2 / c**2)
    return sp.series(lag, v, 0, order + 1).removeO().expand()


def kinetic_series(order: int = 6):
    """Series of T = m c^2 (gamma - 1) in v, about v = 0."""
    kin = m * c**2 * (1 / sp.sqrt(1 - v**2 / c**2) - 1)
    return sp.series(kin, v, 0, order + 1).removeO().expand()


def quartic_coefficients() -> tuple:
    """(L's v^4 coefficient, T's v^4 coefficient) = (1/8, 3/8), in m/c^2."""
    lag = sp.simplify(lagrangian_series(4).coeff(v, 4) * c**2 / m)
    kin = sp.simplify(kinetic_series(4).coeff(v, 4) * c**2 / m)
    return sp.nsimplify(lag), sp.nsimplify(kin)


def momentum_symbolic():
    """dL/dv, simplified. Must equal gamma m v."""
    lag = -m * c**2 * sp.sqrt(1 - v**2 / c**2)
    return sp.simplify(sp.diff(lag, v))


def legendre_energy():
    """E = p v - L, simplified. Must equal gamma m c^2."""
    lag = -m * c**2 * sp.sqrt(1 - v**2 / c**2)
    p = sp.diff(lag, v)
    return sp.simplify(p * v - lag)


def gamma_expr():
    return 1 / sp.sqrt(1 - v**2 / c**2)


# --- the numerical side, used by the learned-momentum readout --------------


def c_hat_from_momentum_coefficients(b1: float, b3: float) -> float:
    """c = sqrt(b1 / (2 b3)) from p(v) ~= b1 v + b3 v^3.

    With p = gamma m v = m v + (m/(2 c^2)) v^3 + ..., we have b1 = m and
    b3 = m / (2 c^2), so b1/(2 b3) = c^2. Both the overall scale of L and the
    sign gauge cancel in that ratio, which is why `c_hat` is the quantity to
    report and `m` is not.

    Returns nan when the cubic term has the wrong sign or vanishes -- that is
    the Newtonian control's expected answer, not an error.
    """
    import numpy as np

    if b1 == 0.0 or b3 == 0.0:
        return float("nan")
    if b1 < 0.0:  # sign gauge: (p, V) -> (-p, -V) leaves the dynamics alone
        b1, b3 = -b1, -b3
    ratio = b1 / (2.0 * b3)
    if ratio <= 0.0:
        return float("nan")
    return float(np.sqrt(ratio))
