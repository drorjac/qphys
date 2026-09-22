"""Acceptance tests 1-11 from the build prompt, Project 1.

Every number asserted here was measured before the spec was written, and is
re-measured on this machine rather than recalled. Honesty rule 5: a number in
a report comes from a cell that ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from qphys.collapse import causal_states as cs
from qphys.collapse import data as D
from qphys.collapse import kraus
from qphys.collapse import leggett_garg as lg
from qphys.collapse import processes as pr
from qphys.common.metrics import free_params


# --- 1, 2: the Kraus construction ----------------------------------------
@pytest.mark.parametrize("d,n_out", [(2, 2), (3, 2), (4, 3), (2, 4)])
def test_kraus_completeness(d, n_out, rng):
    """sum_x K_x^dag K_x = I, to machine precision, for free via QR."""
    ks = kraus.kraus_random(d, n_out, rng)
    assert kraus.completeness_error(ks) < 1e-12


def test_kraus_from_flat_is_also_complete(rng):
    """Any real vector maps to a valid CPTP map, so an optimiser cannot
    leave the manifold even in principle."""
    d, n_out = 3, 2
    theta = rng.normal(size=2 * n_out * d * d)
    assert kraus.completeness_error(kraus.kraus_from_flat(theta, d, n_out)) < 1e-12


def test_step_probabilities_sum_to_one(rng):
    ks = kraus.kraus_random(3, 2, rng)
    p = kraus.step_probabilities(ks, kraus.rho_initial(3))
    assert abs(p.sum() - 1.0) < 1e-12
    assert np.all(p >= -1e-15)


def test_probabilities_stay_normalised_along_a_sequence(rng):
    """Not just at the start: renormalisation after each collapse must keep
    the state a state."""
    ks = kraus.kraus_random(3, 2, rng)
    symbols = kraus.sample(ks, 200, rng)
    rho = kraus.rho_initial(3)
    for x in symbols:
        p = kraus.step_probabilities(ks, rho)
        assert abs(p.sum() - 1.0) < 1e-10
        assert abs(np.trace(rho).real - 1.0) < 1e-10
        rho = kraus.final_state(ks, [x], rho)


# --- 3: the likelihood ordering gate -------------------------------------
@pytest.mark.slow
def test_true_model_beats_iid_and_a_wrong_model(rng):
    """true < iid < random. If this fails the likelihood is wrong."""
    d, n_out, n = 3, 2, 20_000
    true = kraus.kraus_random(d, n_out, np.random.default_rng(1))
    symbols = kraus.sample(true, n, np.random.default_rng(2))
    wrong = kraus.kraus_random(d, n_out, np.random.default_rng(99))

    nll_true = kraus.nll_bits_per_symbol(true, symbols)
    q = float(np.mean(symbols))
    nll_iid = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    nll_wrong = kraus.nll_bits_per_symbol(wrong, symbols)

    assert nll_true < nll_iid, f"true {nll_true:.4f} !< iid {nll_iid:.4f}"
    assert nll_true < nll_wrong, f"true {nll_true:.4f} !< random {nll_wrong:.4f}"


# --- 4, 5, 6: the perturbed coin -----------------------------------------
PERTURBED_COIN_TABLE = [
    # p,    C_mu, overlap, eigenvalues,      C_q
    (0.05, 1.0, 0.4359, (0.2821, 0.7179), 0.8582),
    (0.10, 1.0, 0.6000, (0.2000, 0.8000), 0.7219),
    (0.20, 1.0, 0.8000, (0.1000, 0.9000), 0.4690),
    (0.30, 1.0, 0.9165, (0.0417, 0.9583), 0.2502),
    (0.40, 1.0, 0.9798, (0.0101, 0.9899), 0.0815),
    (0.50, 0.0, 1.0000, (0.0000, 1.0000), 0.0000),
]


@pytest.mark.parametrize("p,c_mu,overlap,eigs,c_q", PERTURBED_COIN_TABLE)
def test_perturbed_coin_row(p, c_mu, overlap, eigs, c_q):
    t = cs.perturbed_coin_transition(p)
    assert cs.c_mu(t) == pytest.approx(c_mu, abs=1e-4)
    assert cs.perturbed_coin_overlap(p) == pytest.approx(overlap, abs=1e-4)
    assert cs.perturbed_coin_eigenvalues(p) == pytest.approx(eigs, abs=1e-4)
    assert cs.perturbed_coin_c_q(p) == pytest.approx(c_q, abs=1e-4)


def test_state_merging_at_p_half():
    """THE regression test. At p = 0.5 the output is i.i.d., the two causal
    states become predictively identical and MERGE, so C_mu drops
    discontinuously to 0 -- not to 1. C_mu = 1 here means no merging."""
    assert cs.c_mu(cs.perturbed_coin_transition(0.5)) == pytest.approx(0.0, abs=1e-9)
    assert cs.c_mu(cs.perturbed_coin_transition(0.49)) == pytest.approx(1.0, abs=1e-9)


def test_closed_form_matches_the_general_construction():
    """The perturbed-coin formula and the general density-matrix route must
    agree, or one of them is wrong."""
    for p in (0.05, 0.1, 0.2, 0.3, 0.4):
        t = cs.perturbed_coin_transition(p)
        assert cs.c_q(t) == pytest.approx(cs.perturbed_coin_c_q(p), abs=1e-9)


@pytest.mark.parametrize("p", [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
def test_c_q_never_exceeds_c_mu(p):
    """The theorem: quantum causal states are never more expensive."""
    t = cs.perturbed_coin_transition(p)
    assert cs.c_q(t) <= cs.c_mu(t) + 1e-12


def test_c_q_below_c_mu_on_a_random_chain(rng):
    for _ in range(20):
        row = rng.dirichlet([1.0, 1.0], size=2)
        assert cs.c_q(row) <= cs.c_mu(row) + 1e-12


# --- 7, 8, 9, 10: Leggett-Garg as a guardrail ----------------------------
def test_classical_bound_is_exactly_one():
    assert lg.classical_bound_exhaustive() == pytest.approx(1.0, abs=1e-15)


def test_quantum_maximum_is_three_halves_at_sixty_degrees():
    value, angle = lg.quantum_maximum()
    assert value == pytest.approx(1.5, abs=1e-6)
    assert angle == pytest.approx(np.pi / 3.0, abs=1e-4)


@pytest.mark.parametrize("lag", [1, 2, 3, 5, 8])
def test_aligned_k3_never_exceeds_one(lag, rng):
    """Across five process types. K3 <= 1 is an algebraic identity for an
    aligned single-window estimator, so exceeding it is a windowing bug,
    never a discovery."""
    series = {
        "iid": pr.iid_bits(20_000, rng),
        "period2": pr.periodic(20_000, 2),
        "period3": pr.periodic(20_000, 3),
        "ar2": pr.ar2_sign(20_000, rng),
        "walk": pr.random_walk_sign(20_000, rng),
    }
    for name, bits in series.items():
        k3 = lg.k3_aligned(pr.to_pm1(bits), lag)
        assert k3 <= 1.0 + 1e-9, f"{name} at lag {lag} returned K3 = {k3}"


@pytest.mark.parametrize("lag", [0.1, 0.3, 0.7, 1.0, 1.5])
def test_van_vleck_saturates_the_bound_exactly(lag):
    """A classical Gaussian oscillator, sign-dichotomised, sits exactly ON
    the macrorealist bound at every lag. The classical process imitates the
    quantum shape maximally, which is why K3 proves nothing here."""
    assert lg.k3_van_vleck(1.0, lag) == pytest.approx(1.0, abs=1e-12)


# --- capacity accounting (honesty rule 3) --------------------------------
@pytest.mark.parametrize("d,n_out,expected", [(2, 2, 7), (3, 2, 17)])
def test_hqmm_free_parameter_count(d, n_out, expected):
    """Stiefel real dimension, minus per-operator phases, minus the
    conjugation gauge. Comparisons are at matched capacity, so this count is
    part of the result rather than a footnote."""
    assert free_params("hqmm", d=d, n_out=n_out) == expected


def test_classical_free_parameter_counts():
    assert free_params("iid") == 1
    assert free_params("markov", order=1) == 2
    assert free_params("markov", order=2) == 4
    assert free_params("markov", order=3) == 8
    assert free_params("hmm", k=2) == 5
    assert free_params("hmm", k=3) == 11
    assert free_params("hmm", k=4) == 19


# --- state proliferation: the most common way this estimate goes wrong ----
def test_a_structureless_process_has_no_statistical_complexity():
    """THE negative control. Mersenne-Twister bits carry no structure, so
    C_mu must stay near zero at every history length.

    With float-equality merging it did not: nothing merged, and C_mu climbed
    1.0, 2.0, 3.0, 4.0, 4.7, 5.5 bits at orders one to six -- pure
    finite-sample state proliferation, reported as complexity.
    """
    mt = np.random.RandomState(0).randint(0, 2, 1461)
    df = cs.complexity_vs_history(mt, orders=(1, 2, 3, 4))
    assert df.C_mu.max() < 0.6, (
        f"structureless bits returned up to {df.C_mu.max():.2f} bits of "
        "classical complexity; the state merge is not merging"
    )
    assert df.C_q.max() < 0.1


def test_complexity_plateaus_with_history_length_on_real_data():
    """Both curves must flatten. A C_mu that keeps climbing with order is
    measuring the number of histories, not the process."""
    series = D.seattle_wet_dry()
    df = cs.complexity_vs_history(series.values, orders=(2, 3, 4))
    assert df.C_mu.std() < 0.1, f"C_mu has not plateaued: {df.C_mu.tolist()}"
    assert df.C_q.std() < 0.05, f"C_q has not plateaued: {df.C_q.tolist()}"


def test_the_quantum_saving_survives_at_the_plateau():
    """The 72% saving is quoted from the order-1 chain. At the plateau, where
    the estimate has stopped moving, it is larger still."""
    series = D.seattle_wet_dry()
    df = cs.complexity_vs_history(series.values, orders=(2, 3, 4))
    saving = 1.0 - df.C_q.mean() / df.C_mu.mean()
    assert saving > 0.7, f"saving at the plateau is only {saving:.1%}"


def test_merging_is_a_statistical_test_when_counts_are_given():
    """Two rows that differ by less than their sampling error are the same
    state; two that differ by more are not."""
    close = np.array([[0.50, 0.50], [0.52, 0.48]])
    _, uniq = cs.merge_states(close, counts=np.array([100.0, 100.0]))
    assert len(uniq) == 1, "rows within sampling error should merge"
    _, uniq = cs.merge_states(close, counts=np.array([100000.0, 100000.0]))
    assert len(uniq) == 2, "with enough counts the same gap is significant"
