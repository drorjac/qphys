# Hypotheses — the claims this project tests

Every experiment in the repository exists to test one of the claims below.
Each carries what would **refute** it and the test that re-checks it on every
run, and its status is updated whichever way the evidence goes. Numbers live
in [`FINDINGS.md`](FINDINGS.md) and in [`results/`](../results); this page
carries none, so it cannot drift.

| # | Claim | Status |
|---|---|---|
| C1 | A quantum model **predicts** a real series better than the best classical model at matched capacity | **not shown — it loses** |
| C2 | Quantum causal states **represent** a process with less memory than classical ones | **holds** — a theorem, checked on real data |
| C3 | A Leggett–Garg violation in a recorded series would be evidence of anything | **false, provably** |
| C4 | The speed limit is recoverable from slow motion alone | **holds** in simulation, with a stated failure rate |
| C4b | A physics-structured network extrapolates better than a black box | **refuted** — only the extracted formula does |
| C5 | That route works on real orbital data | **false** — Mercury is far too slow |

C1 and C2 are about different objects and must never be conflated: a large
memory saving and a worse test NLL are both true at once.

---

## C1 · Prediction at matched capacity

**Test.** Seattle wet/dry days, one temporal split declared up front; i.i.d.,
Markov-1…4, HMM-2…4 and HQMM-d2/d3, each stochastic row over at least five
seeds, compared at their free-parameter count.

**Refuted if** some HQMM row, averaged over its seeds, beats every classical
model with no more free parameters than it has — on a fresh fit, not only in
the committed table.

**Guarded by** `tests/test_acceptance_seattle.py`
(`test_reported_table_still_shows_the_hqmm_losing`,
`test_hqmm_loses_on_a_fresh_fit_too`, `test_the_best_model_overall_is_classical`).

## C2 · Representation with less memory

**Test.** `C_q ≤ C_μ` for the estimated causal states, on closed-form
processes and on Seattle at the plateau of history length.

**Refuted if** `C_q` exceeded `C_μ` on any chain (it cannot, if the
construction is right), or if the saving on Seattle vanished under a change
of symbolisation or history length.

**Guarded by** `tests/test_acceptance_collapse.py`
(`test_c_q_never_exceeds_c_mu`, `test_the_quantum_saving_survives_at_the_plateau`,
`test_a_structureless_process_has_no_statistical_complexity`) and
`tests/test_acceptance_seattle.py`
(`test_memory_saving_survives_a_change_of_symbolisation`).

## C3 · Leggett–Garg on a recorded series

**Test.** For a passively recorded series with an aligned estimator, `K3 ≤ 1`
is an algebraic identity; a classical Gaussian process saturates it.

**Refuted if** an aligned estimator on any classical recorded series returned
`K3 > 1`. That would be a windowing bug, not physics, which is why `K3`
ships as a unit test on the estimator.

**Guarded by** `test_aligned_k3_never_exceeds_one`,
`test_van_vleck_saturates_the_bound_exactly`.

## C4 · The speed limit from slow motion

**Test.** Train the momentum network on `(x, v, a)` from relativistic
motion below `v_max`, read `c` out of the cubic term, over at least five
seeds with the convergence gate applied before the answer is read.

**Refuted if** the recovered `ĉ` fell outside what a perfect momentum
function gives under the same readout (the readout floor), or if the
Newtonian control returned a finite speed limit.

**Guarded by** `tests/test_lawnet.py`
(`test_a_trained_fit_recovers_the_speed_limit`,
`test_a_trained_newtonian_control_returns_no_speed_limit`,
`test_convergence_gate_is_applied_before_the_answer_is_read`).

## C4b · What extrapolates

**Test.** Train inside `|v| < 0.5c`, score in `[0.7c, 0.95c]`: the
structured network, a black box matched on parameter count, and the formula
extracted from the structured network.

**Refuted if** the structured network extrapolated better than the
parameter-matched black box. It does not; the formula is what extrapolates.

**Guarded by** `test_only_the_formula_extrapolates`,
`test_blackbox_is_matched_on_parameter_count`.

## C5 · Real orbital data

**Test.** Mercury's perihelion speed against the speeds at which C4 works.

**Refuted if** the cubic term were readable at Mercury's speed. It is not.

**Guarded by** `tests/test_acceptance_lawlearn.py`
(`test_mercury_is_far_too_slow_for_the_momentum_route`).
