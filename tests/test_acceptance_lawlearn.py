"""Acceptance tests 12, 13, 16-19 from the build prompt, Project 2.

The orbit numbers are targets known in advance, which is what makes them an
acceptance test. They are NOT measured data -- see the docstring of
`qphys.lawlearn.orbit` and the README.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from qphys.common.units import GM_SUN_AU, MERCURY_PERIOD_YR
from qphys.lawlearn import orbit as O
from qphys.lawlearn import relativistic as R


@pytest.fixture(scope="module")
def trajectory():
    return O.integrate_orbit()


# --- 12: the test that catches learning the energy instead of the law ----
def test_lagrangian_and_kinetic_energy_differ_at_fourth_order():
    """1/8 versus 3/8. The v^2 terms agree, so only the quartic can tell the
    Lagrangian from the kinetic energy -- recovering 3/8 means the
    Euler-Lagrange residual has a sign or factor error."""
    lag_quartic, kin_quartic = R.quartic_coefficients()
    assert lag_quartic == sp.Rational(1, 8)
    assert kin_quartic == sp.Rational(3, 8)
    assert lag_quartic != kin_quartic


def test_second_order_terms_agree():
    """Both give (1/2) m v^2, which is exactly why the quartic is needed."""
    assert sp.simplify(
        R.lagrangian_series(4).coeff(R.v, 2) - R.kinetic_series(4).coeff(R.v, 2)
    ) == 0


def test_sixth_order_coefficients():
    """1/16 and 5/16, the next term of the same comparison."""
    lag = sp.nsimplify(sp.simplify(R.lagrangian_series(6).coeff(R.v, 6) * R.c**4 / R.m))
    kin = sp.nsimplify(sp.simplify(R.kinetic_series(6).coeff(R.v, 6) * R.c**4 / R.m))
    assert lag == sp.Rational(1, 16)
    assert kin == sp.Rational(5, 16)


# --- 13: Legendre consistency -------------------------------------------
def test_canonical_momentum_is_gamma_m_v():
    """The entire content of special relativity, for this project."""
    assert sp.simplify(R.momentum_symbolic() - R.gamma_expr() * R.m * R.v) == 0


def test_legendre_transform_gives_gamma_m_c_squared():
    assert sp.simplify(R.legendre_energy() - R.gamma_expr() * R.m * R.c**2) == 0


# --- 14: the gauge the readout has to survive ---------------------------
def test_c_hat_is_invariant_under_the_sign_gauge():
    """(p, V) -> (-p, -V) leaves the acceleration unchanged, so a run that
    returns b1 < 0 is not wrong -- but c_hat only survives if the readout
    flips both. A run really did come back with b1 = -0.436."""
    assert R.c_hat_from_momentum_coefficients(1.0, 0.5) == pytest.approx(1.0)
    assert R.c_hat_from_momentum_coefficients(-1.0, -0.5) == pytest.approx(1.0)


def test_c_hat_is_invariant_under_the_scale_gauge():
    """L is fixed only up to overall scale, so only scale-invariant
    quantities may be reported. c_hat is one; m is not."""
    for lam in (0.1, 2.0, 37.0):
        assert R.c_hat_from_momentum_coefficients(lam * 1.0, lam * 0.5) == (
            pytest.approx(1.0)
        )


# --- 15: the Newtonian control ------------------------------------------
def test_newtonian_control_returns_nan():
    """With no relativistic correction the cubic coefficient is ~0 or of the
    wrong sign, and c_hat must be nan rather than a large number that could
    be mistaken for a measurement."""
    assert np.isnan(R.c_hat_from_momentum_coefficients(1.0, -1e-9))
    assert np.isnan(R.c_hat_from_momentum_coefficients(1.0, 0.0))
    assert np.isnan(R.c_hat_from_momentum_coefficients(1.0, -0.5))


# --- 16, 17, 18, 19: the orbit ------------------------------------------
def test_kepler_third_law(trajectory):
    period = O.kepler_period()
    assert period == pytest.approx(MERCURY_PERIOD_YR, rel=1e-5)


def test_force_law_exponent_is_minus_two(trajectory):
    assert O.force_law_exponent(*trajectory) == pytest.approx(-2.0, abs=1e-4)


def test_recovered_gm(trajectory):
    assert O.recover_gm(*trajectory) == pytest.approx(GM_SUN_AU, rel=1e-6)


def test_conservation(trajectory):
    """Relative peak-to-peak drift over six orbits at rtol 1e-12."""
    c = O.conservation(*trajectory)
    assert c["L_drift"] < 1e-8, c
    assert c["E_drift"] < 1e-8, c


def test_mercury_precession_analytic():
    assert O.precession_analytic() == pytest.approx(42.98, abs=0.01)


@pytest.mark.slow
def test_mercury_precession_measured_by_root_finding():
    """Measured from the integrated orbit, not from the formula. The
    perihelion is located by brentq on du/dphi: the shift is ~5e-7 rad per
    orbit and fitting a parabola to a sampled grid is good only to ~1e-6,
    i.e. larger than the effect being measured."""
    assert O.precession_measured() == pytest.approx(42.98, abs=0.02)


def test_the_two_precession_routes_agree():
    assert O.precession_measured() == pytest.approx(O.precession_analytic(), rel=1e-3)


# --- the conceptual guard ------------------------------------------------
def test_mercury_is_far_too_slow_for_the_momentum_route():
    """1.97e-4 c, against the ~0.4c the quartic needs to be identifiable.

    This is the test that stops someone pointing the c_hat machinery at real
    orbital data and believing the answer. GR enters here through the
    SECULAR precession, not the instantaneous momentum.
    """
    v_over_c = O.mercury_speed_over_c()
    assert v_over_c == pytest.approx(1.97e-4, rel=0.02)
    assert v_over_c < 0.4, "the identifiability floor from the section 10 sweep"
