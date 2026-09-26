# Decisions

Research decisions, what was weighed, and why. The claims they serve are in
[`HYPOTHESES.md`](HYPOTHESES.md). Open items are at the bottom.

## Decided

| date | decision | alternatives weighed | reason |
|---|---|---|---|
| 2026-09 | **Compare at matched free-parameter count, not state count.** | Compare an HQMM of dimension d with an HMM of d states. | A d-dimensional HQMM has more free parameters than a d-state HMM (7 against 5 at d = 2, 17 against 11 at d = 3), so matching states would hand the quantum side extra capacity. CONTRIBUTING rule 3. |
| 2026-09 | **Merge causal states by a two-proportion test at 2.5σ, as CSSR does.** | Float equality of conditional distributions. | Float equality merges nothing, so `C_μ` counted histories; on structureless bits it reported 5.5 bits. The negative control exposed it. |
| 2026-09 | **Report every stochastic row over ≥ 5 seeds, with its failure rate.** | Single fits. | One HQMM fit appeared to overturn the headline; over five seeds it did not. |
| 2026-09 | **Gate convergence on the excess of training loss over the noise floor.** | A fixed loss threshold. | Injected noise puts a floor under the achievable loss; a fixed threshold discarded every noisy run by arithmetic. |
| 2026-09-26 | **Keep `qphys` and `physprior` as separate sibling projects.** | Merge `lawlearn` into `physprior`. | Both touch Mercury, but they ask different questions: physprior measures what a known law's prior is worth; qphys asks whether quantum formalism is an efficient modelling language and whether a law can be read out of a network. Their rules and protocols differ, and a merge would blur both. |
| 2026-09-26 | **Name the generated-table directory `results/`, and require code behind every file in it.** | Keep `reports/`. | The same layout as physprior. Three tables had been committed with no code that wrote them; `tests/test_cli.py` now fails on that. |

## Open

| item | what it blocks |
|---|---|
| Whether to add notebooks (one overview per sub-project) or leave the docs pages as the narrative. | Nothing; the docs pages carry the figures. |
| Public release and a Zenodo DOI. | Citation by DOI; decided together with physprior. |
