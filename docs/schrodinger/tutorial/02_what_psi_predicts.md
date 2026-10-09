# 2 · What psi predicts

[Previous](01_from_waves_to_the_equation.md) · [Topic page](../README.md) · Next: [3 · Stationary states](03_stationary_states.md)

The Schrodinger equation is deterministic. Given psi now, psi later is fixed.
But psi is not the particle's position, and a single measurement of position
is not predictable. This part says exactly what is predicted.

## The Born rule

If a position detector covers the whole line, the probability that it fires
in [x, x + dx] is

$$ P(x)\,dx = |\psi(x, t)|^2\,dx . $$

This needs the integral of |psi|^2 to be 1, which is the normalisation
condition. Part 1 showed that the equation conserves that integral, so a
wavefunction normalised once stays normalised.

Averages over many identically prepared runs follow:

$$ \langle x\rangle = \int x\,|\psi|^2\,dx, \qquad
   \langle p\rangle = \int \psi^*\,(-i\,\partial_x)\,\psi\,dx . $$

The momentum average is easier in Fourier space, where |psi(k)|^2 is the
momentum distribution. `evolve.expect_x` and `evolve.expect_p` compute both.

## Uncertainty

For a Gaussian packet of width sigma in position, the momentum distribution
is a Gaussian of width 1/(2 sigma). The product is 1/2, the minimum allowed
by Heisenberg's relation. `evolve.gaussian_packet(grid, x0, k0, sigma)`
builds exactly this state. A narrower packet in x is wider in p, and the
wider momentum spread makes it spread faster in time.

## The test: draw detections and compare

`born.sample_positions` simulates n independent runs, each ending in one
detection drawn from |psi|^2. The state is two packets moving toward each
other at the moment they overlap. Their sum has fringes of spacing pi / k0
that neither packet has alone. If the sampler got |psi|^2 slightly wrong, the
fringes would wash out.

![histograms converging to |psi|^2](../../../figures/11_born_rule.png)

At 300 detections the histogram shows two humps and no fringes. At 100,000
the fringes are resolved. The total-variation distance between the histogram
and |psi|^2 (same 80 bins) is 0.204 at n = 100 and 0.0081 at n = 100,000,
mean over five seeds. A fit gives n^-0.485. Independent samples give
n^-1/2, and the small shortfall at large n comes from the finite number of
bins in the comparison.

This is what "the Schrodinger equation predicts the particle's location"
means operationally: it predicts the distribution of outcomes, and that
prediction is tested by histograms of repeated runs, never by a single one.

## After a detection

A real detector has finite resolution. If it fires at x_d with resolution
s, the state just afterwards is psi multiplied by a window of width s around
x_d and renormalised (`born.collapse`). That is a new, narrow packet. It has
a wide momentum distribution, so it spreads quickly again under the
Schrodinger equation. The measurement does not follow from the equation; it
is a second rule applied on top of it. The `collapse` sub-project of this
repository models that second rule with Kraus operators.

```python
from qphys.schrodinger import born
from qphys.schrodinger.experiments import interference_state

g, psi = interference_state()
x = born.sample_positions(g, psi, 10_000, seed=0)
born.total_variation(g, psi, x)   # about 0.02
```
