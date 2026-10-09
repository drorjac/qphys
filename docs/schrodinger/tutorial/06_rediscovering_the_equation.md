# 6 · Rediscovering the equation

[Previous](05_tunneling.md) · [Topic page](../README.md)

Parts 1 to 5 went from the equation to its predictions. This part goes the
other way. A program is given numerical data and no formula, and is asked
for the law. Two methods are used. Sparse regression chooses terms from a
fixed list of candidates. Symbolic regression searches over all expressions
built from a few operators.

## A · The time-dependent equation from psi(x, t)

### Setup

`discover.simulate` evolves a displaced, moving Gaussian in the harmonic
well and records 400 snapshots of psi on 256 points. Optional complex
Gaussian noise of standard deviation `noise * max|psi|` is added to every
sample. That is all the regression sees.

The question is posed as

$$ i\,\partial_t\psi = \sum_j \xi_j\,\theta_j[\psi] , $$

with ten candidate terms theta_j: psi, x psi, x^2 psi, x^3 psi, x^4 psi,
psi_x, psi_xx, psi_xxx, psi_xxxx and |psi|^2 psi. The cubic term would make
the equation non-linear (it appears in the Gross-Pitaevskii equation for
condensates); first-order and third-order derivatives would break the
symmetry between left and right. The truth uses two of the ten.

### Method

1. **Derivatives.** Spatial derivatives are taken spectrally. With noise,
   white noise fills all wavenumbers equally while the signal occupies the
   low ones, so the mean power spectrum is cut where it falls to ten times
   its high-k floor. The cut is read off the data. Time derivatives use a
   Savitzky-Golay filter along t.
2. **Mask.** Points where |psi| is under 5% of its maximum are dropped;
   there every term is mostly noise.
3. **Sparse regression.** Least squares on column-normalised terms, then
   repeatedly drop every term whose normalised coefficient is under 5% of
   the largest and refit (STLSQ, the method of SINDy and PDE-FIND).

### Result

![recovered coefficients against noise](../../../figures/13_discovery.png)

| noise | terms kept | psi_xx | x^2 psi | spurious |
|---|---|---|---|---|
| 0 | 2 | -0.5000 | 0.5000 | none |
| 0.5% | 2 | -0.4993 | 0.4994 | none |
| 1% | 2 | -0.4992 | 0.4993 | none |
| 2% | 2 | -0.4988 | 0.4988 | none |
| 5% | 3 | -0.4898 | 0.5354 | x^4 psi: -0.0046 |

Means over five noise seeds, from `results/schrodinger_discovery.csv`. The
seed-to-seed spread is at most 1e-4 up to 2% and 2e-3 at 5%.

Up to 2% noise the regression returns exactly the Schrodinger equation, with
both coefficients within 0.25% of the truth. The psi_xx coefficient is
-hbar^2/2m in physical units, so this is a measurement of hbar/m from the
motion of a wave.

At 5% it fails in an informative way. The extra term is x^4 psi, a
correction to the shape of the potential, and it comes with a larger x^2
coefficient that partly compensates. The packet only explores |x| below
about 3.5, and over that range x^2 and a combination of x^2 and x^4 are
hard to tell apart once the derivatives are noisy. The regression found a
potential that fits the region the data covers. Data that covers more of
the well, or less noise, separates them.

## B · Closed forms with PySR

[PySR](https://github.com/MilesCranmer/PySR) evolves a population of
expression trees built from `+ - * / ^` and constants, and returns the best
trade-off between accuracy and size. It is not given a family of forms.
Each task below runs with the same settings (`symbolic._pysr`), seed 0, and
`deterministic=True`, so the result is reproducible.

Install it with `pip install 'qphys[sr]'`. The first import downloads Julia.
Without PySR, `symbolic.py` falls back to regressions chosen in advance: a
log-linear fit for the power laws and a sparse polynomial fit for the
potential. Those rows carry a different `backend` name in
`results/schrodinger_symbolic.csv`, because they answer a narrower
question.

### B1 · The infinite well

Input: 30 numerical eigenvalues, E for n = 1..6 and L in {1, 1.5, 2, 3, 4}.

PySR: `4.9348*n**2/L**2`. The constant is pi^2 / 2 = 4.9348. The program
found the n^2 and the 1/L^2 from numbers alone.

### B2 · The quartic oscillator

Input: E_n for V = x^4, n = 2..39, where WKB is accurate to 0.6% or better.

PySR: `n**0.3321*(1.38266334*n + 0.919038)`.

This is not the WKB form 1.3765 (n + 1/2)^(4/3), and a physicist would not
write it. Its behaviour at large n is 1.3827 n^1.3321, computed from the
expression (`symbolic._asymptote`). The exponent is 0.1% from 4/3 and the
prefactor 0.5% from WKB. So the regressor found the scaling law. It
expressed the low-n corrections in its own way, because with 38 points
many expressions fit equally well.

A first attempt with n = 0..19 and a shorter search returned
`0.9137*n + 0.8453*n**1.4137 + 0.6094`. That also fits, and its large-n
exponent is 1.41, which is wrong. Including n = 0 and 1, where WKB fails,
pulled the fit toward a different shape. Restricting the data to the regime
where the law holds, and searching longer, gave the right exponent. The
input data and its range are part of the result.

The log-linear fallback, which assumes the form c (n + 1/2)^p in advance,
gives 1.3814 (n + 1/2)^1.3322.

### B3 · The potential from the density

Input: only the ground-state density rho(x) of the double well
V = x^4/4 - x^2, on |x| <= 2.5.

For a real, nodeless ground state, psi = sqrt(rho), and the stationary
equation rearranges to

$$ V(x) - E_0 = \frac{(\sqrt\rho)''}{2\sqrt\rho} . $$

So the density determines the potential up to the constant E_0. This is
the one-dimensional case of the Hohenberg-Kohn statement that underlies
density functional theory.

PySR on (x, (sqrt rho)'' / (2 sqrt rho)): `x**2*(0.2499*x**2 - 0.9998) + 0.2994`.

The constant is -E_0: the numerical ground-state energy is -0.2995. The
regressor recovered the potential and the ground-state energy together from
a single density profile.

## What this does and does not show

It shows that the equation, the coefficient hbar^2/2m, a spectrum's scaling
law and a potential can be recovered from data generated by the
Schrodinger equation, and how much noise each step tolerates. It does not
show that a machine would have discovered quantum mechanics. Part A's
candidate list already contained the answer, and B ran on clean numerical
solutions. Choosing the variables (psi rather than |psi|^2, i psi_t rather
than psi_t) was the conceptual step, and that was done by hand.

```python
from qphys.schrodinger import discover as D, symbolic as Y

r = D.discover(noise=0.01, seed=0)
D.equation_string(r["terms"])      # i psi_t = +0.4993 x^2 psi -0.4992 psi_xx

Y.potential(backend="pysr")        # needs qphys[sr]
```
