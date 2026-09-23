# Changelog

All notable changes to this project are documented here. Dates are the date
the work landed; every measured number cited is regenerated into `reports/`
by the experiment that produced it.

## [Unreleased]

### Added
- Seven generated figures (`make figures`) and a README built around them.
- Palette validator (`make palette`): the chart colours pass the lightness
  band, chroma floor, colour-vision-deficiency separation, normal-vision and
  contrast gates at the strictest adjacency setting.
- CI on Python 3.11 and 3.12, running the suite **offline** so a fresh clone
  behind a firewall is exercised on every push.
- `QPHYS_OFFLINE=1` to force the fallback loaders even where a network exists.
- MIT licence, contributing rules, pre-commit hooks, Makefile.

### Fixed
- **State merging compared conditional distributions for float equality**, so
  nothing ever merged and `C_mu` counted histories rather than structure — on
  structureless bits it reported up to 5.5 bits. Now a two-proportion test at
  2.5σ, as CSSR does. Seattle's complexity plateaus at 1.44 bits and the
  quantum saving at the plateau is 78.5%.
- **The convergence gate was unreachable on noisy data.** Injected noise puts
  a floor under the achievable training loss (2.9e-04, 1.8e-03, 7.3e-03 at
  2%, 5%, 10%), so a fixed 1e-03 threshold discarded every run at 5% and 10%
  by arithmetic rather than by performance. The gate now measures the excess
  over that floor.
- **The Seattle table reported single fits.** Every stochastic row is now a
  mean over five seeds with its standard deviation. One HQMM fit had scored
  0.8464 and appeared to overturn the headline; the same configuration
  averages 0.8785 ± 0.0267.
- Tests that assert properties of the real series now skip loudly offline
  instead of failing against the synthetic fallback.

## [0.1.0] - 2026-09-23

### Added
- `collapse`: Kraus/HQMM likelihood and sampling, classical and quantum
  causal states, Leggett–Garg guardrail, synthetic processes and controls,
  classical baselines with restart selection, the Seattle loader.
- `lawlearn`: symbolic relativistic checks, the Mercury orbit acceptance
  suite, and the momentum network with all three gauges fixed.
- The acceptance suite: every previously verified number re-measured here
  rather than recalled.
