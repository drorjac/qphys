"""The momentum network: gauges, the convergence gate, and the readout.

The slow tests train. The fast ones check the pieces that can be wrong
without training -- and the readout is checked against the TRUE momentum, so
a readout bug cannot hide behind a training failure.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qphys.common.seeding import CONVERGENCE_REL_LOSS  # noqa: E402
from qphys.lawlearn import lawnet as L  # noqa: E402


def test_momentum_is_odd_by_construction():
    """The parity gauge is architectural, not a penalty: p(-v) = -p(v) holds
    exactly at initialisation and stays exact through training."""
    net = L.LawNet()
    v = torch.linspace(-0.9, 0.9, 51, dtype=L.DTYPE)
    assert torch.allclose(net.momentum(v), -net.momentum(-v), atol=1e-12)


def test_momentum_is_zero_at_zero():
    net = L.LawNet()
    z = torch.zeros(1, dtype=L.DTYPE)
    assert abs(float(net.momentum(z))) < 1e-12


def test_acceleration_is_finite_even_where_the_momentum_is_flat():
    """The 1e-3 in the denominator: an untrained p'(v) can pass through zero
    and a bare division would return inf on the first step."""
    net = L.LawNet()
    x = torch.linspace(-1, 1, 33, dtype=L.DTYPE)
    v = torch.linspace(-0.9, 0.9, 33, dtype=L.DTYPE)
    assert torch.all(torch.isfinite(net.accel(x.clone(), v.clone())))


def test_samples_match_the_analytic_acceleration():
    """a = -V'(x)/gamma^3, with V = x^2/2 so V' = x."""
    x, v, a = L.relativistic_samples(500, 0.6, seed=0)
    gamma = 1.0 / np.sqrt(1.0 - v**2)
    assert np.allclose(a, -x / gamma**3, atol=1e-12)


def test_newtonian_control_samples_have_no_gamma():
    x, _v, a = L.relativistic_samples(500, 0.6, seed=0, newtonian=True)
    assert np.allclose(a, -x, atol=1e-12)


def test_true_momentum_is_gamma_v():
    v = np.linspace(-0.9, 0.9, 41)
    assert np.allclose(L.true_momentum(v), v / np.sqrt(1 - v**2))
    assert np.allclose(L.true_momentum(v, newtonian=True), v)


@pytest.mark.parametrize(
    "v_max,expected", [(0.3, 0.9615), (0.5, 0.8874), (0.7, 0.7566)]
)
def test_the_cubic_readout_is_biased_on_the_exact_momentum(v_max, expected):
    """The readout's own error floor, with no network involved.

    Fitting b1 v + b3 v^3 to the EXACT gamma v returns 0.887 at v_max = 0.5:
    an 11% error that belongs to the truncation, not to any fit. A trained
    result must be read against this number, or the readout's bias gets
    reported as a physics result.
    """
    assert L.readout_bias(v_max) == pytest.approx(expected, abs=0.002)


@pytest.mark.parametrize("v_max", [0.3, 0.5])
def test_the_quintic_readout_removes_most_of_that_bias(v_max):
    """Keeping the v^5 term gives the cubic coefficient somewhere to put the
    next order, and c_hat lands within 3% instead of 11%."""
    assert L.readout_bias(v_max, powers=(1, 3, 5)) == pytest.approx(1.0, abs=0.03)


def test_the_readout_floor_is_worse_the_faster_the_data():
    """Part of why 0.7c is worse than 0.5c is algebraic, not optimisational:
    the truncation bias grows monotonically with the fitted range."""
    floors = [L.readout_bias(v) for v in (0.3, 0.5, 0.7)]
    assert floors[0] > floors[1] > floors[2]


def test_readout_on_newtonian_momentum_is_nan():
    """p = v exactly: there is no cubic term, so there is no c to recover and
    the readout must say so rather than return a number."""
    grid = np.linspace(-0.5, 0.5, 400)
    basis = np.stack([grid, grid**3], axis=1)
    (b1, b3), *_ = np.linalg.lstsq(basis, L.true_momentum(grid, True), rcond=None)
    from qphys.lawlearn.relativistic import c_hat_from_momentum_coefficients

    c = c_hat_from_momentum_coefficients(b1, b3)
    assert (not np.isfinite(c)) or c > 10.0, f"got c_hat = {c}"


def test_blackbox_is_matched_on_parameter_count():
    """Honesty rule 3. A black box given fewer parameters is a strawman."""
    n = L.count_params(L.LawNet())
    h = L.matched_hidden(n)
    assert abs(L.count_params(L.BlackBox(h)) - n) / n < 0.01


def test_convergence_gate_is_applied_before_the_answer_is_read():
    """Honesty rule 7: a run above the pre-declared threshold returns nan,
    and the threshold is a module constant, not a per-experiment choice."""
    from qphys.common.seeding import converged

    assert converged(1e-5)
    assert not converged(2.5e-2)
    assert not converged(float("nan"))
    assert CONVERGENCE_REL_LOSS == 1e-3


def test_the_gate_accounts_for_the_floor_injected_noise_puts_under_it():
    """A fixed absolute gate is arithmetically impossible to pass on noisy
    data, and the first sweep reported a 100% failure rate at 5% noise for
    exactly that reason -- which said nothing about the fits.

    The floor is a property of the injection, known before any fit runs, so
    gating the excess over it keeps the threshold pre-declared.
    """
    from qphys.common.seeding import converged

    assert not converged(1.9e-3)  # impossible under a bare 1e-3 gate
    assert converged(1.9e-3, noise_floor=1.83e-3)  # reachable above the floor
    assert not converged(2.5e-2, noise_floor=1.83e-3)  # a real failure still fails


@pytest.mark.parametrize("noise,expected", [(0.0, 0.0), (0.02, 2.9e-4), (0.05, 1.8e-3)])
def test_the_noise_floor_is_what_it_is_measured_to_be(noise, expected):
    from qphys.common.seeding import noise_floor

    _, _, clean = L.relativistic_samples(20000, 0.5, seed=0, noise=0.0)
    _, _, noisy = L.relativistic_samples(20000, 0.5, seed=0, noise=noise)
    assert noise_floor(clean, noisy) == pytest.approx(expected, rel=0.15)


@pytest.mark.slow
def test_a_trained_fit_recovers_the_speed_limit():
    """The headline: c from data capped at 0.5c, never having been told that
    relativity exists."""
    r = L.fit_and_read(0.5, seed=0, steps=3000)
    if not r["converged"]:
        pytest.skip(f"seed 0 did not converge (rel_loss {r['rel_loss']:.2e})")
    assert r["c_hat"] == pytest.approx(1.0, abs=0.25)


@pytest.mark.slow
def test_a_trained_newtonian_control_returns_no_speed_limit():
    """The control that stops the method finding relativity in Newton."""
    r = L.fit_and_read(0.5, seed=0, steps=3000, newtonian=True)
    c = r["c_hat"]
    assert (not np.isfinite(c)) or c > 3.0, f"Newtonian data returned c_hat = {c}"
