# 5 · Tunnelling

[Previous](04_time_evolution.md) · [Topic page](../README.md) · Next: [6 · Rediscovering the equation](06_rediscovering_the_equation.md)

## The plane-wave calculation

A barrier of height V0 occupies |x| < a/2. A wave of energy E comes from the
left. In each of the three regions the solution is a sum of exponentials:
oscillating, exp(+-ikx) with k = sqrt(2E), outside; and growing or decaying,
exp(+-kappa x) with kappa = sqrt(2 (V0 - E)), inside when E < V0. Requiring
psi and psi' to be continuous at both edges gives four linear equations,
and the transmitted fraction is

$$ T(E) = \Big[1 + \frac{V_0^2\,\sinh^2(\kappa a)}{4E\,(V_0 - E)}\Big]^{-1}, \qquad E < V_0 . $$

For a thick barrier T is close to 16 E (V0 - E) / V0^2 times
exp(-2 kappa a): exponentially small, never zero. Above the barrier sinh
becomes sin, and T reaches exactly 1 at resonances, where the barrier holds
a whole number of half wavelengths. Classically the answer is a step: T = 0
below V0 and T = 1 above. `tunneling.transmission_plane_wave` implements the
formula.

## A packet has no single energy

A plane wave is not normalisable and is not a particle. A real particle is a
packet with a spread of momenta, and each momentum component is transmitted
with its own T. The prediction for a packet is the average

$$ T_\text{packet} = \int |\tilde\psi_0(k)|^2\,T(k^2/2)\,dk \Big/ \int |\tilde\psi_0(k)|^2\,dk \quad (k > 0) . $$

`tunneling.tunnel(k0)` sends a Gaussian packet (width 4, starting at
x = -30) at a barrier with V0 = 1 and a = 1. It evolves until the reflected
and transmitted parts have separated, then integrates |psi|^2 to the right
of the barrier.

![a packet splitting at the barrier](../../../figures/09_tunneling_spacetime.png)

![animated](../../../figures/10_tunneling.gif)

## The result, and the correction it needed

![transmission against energy](../../../figures/12_tunneling.png)

Over 21 mean energies from 0.18 to 3.38, the simulated transmission agrees
with the packet average to at most 8.0e-4. The single-energy formula is off
by up to 4.0e-3: the momentum spread of the packet is a real, measurable
effect, and comparing a simulation with the wrong prediction would have
hidden it.

The first run did not agree. Every simulated T was 0.2% to 3.0% below the
prediction, and the gap did not shrink with smaller dt or a wider packet.
The cause was the barrier itself. Sampled pointwise on a grid with
dx = 0.078, |x| < 0.5 contains 13 grid points, which is a barrier of width
1.016, not 1. Transmission falls as exp(-2 kappa a), and at E = 0.32 a
1.6% wider barrier transmits about 3% less. `grid.barrier` now gives each cell
the fraction of the barrier it actually overlaps, and the gap fell from
8.9e-3 to 3.5e-4 at that energy.

The general lesson: a sharp feature on a grid has an effective size that
differs from the nominal one by up to a grid cell, and any quantity that
depends exponentially on that size will show it.

## Measuring a tunnelling probability

In a real experiment T is measured exactly as in part 2: send many particles
one at a time and count how many arrive on the far side. The fraction
converges to T with a standard error sqrt(T (1 - T) / n). Scanning
tunnelling microscopes and alpha decay both use this exponential
sensitivity to barrier width.

```python
from qphys.schrodinger.tunneling import tunnel

r = tunnel(k0=1.2)            # E0 = 0.72, below the barrier
r["T_sim"], r["T_packet"]     # about 0.543 and 0.543
```
