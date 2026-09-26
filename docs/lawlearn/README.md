# lawlearn — learning the law

Hand a network only `(x, v, a)`. Never mention relativity, `γ`, or a speed
limit. Then read the speed of light out of what it learned.

Code: [`src/qphys/lawlearn`](../../src/qphys/lawlearn) ·
Tables: [`results/`](../../results)

---

## Learn the momentum, not the Lagrangian

![the learned momentum against Newton and against truth](../../figures/06_momentum.png)

The textbook LNN parameterises `L(x, v)` and inverts the Euler–Lagrange
equation. **That was tried and it failed**: relative loss stalled at 0.19 and
`ĉ` came back at 0.70. The cause is gauge freedom — `L` is fixed only up to
scale, an additive constant and any total time derivative — and under the
scale gauge the optimiser drifts to tiny `λ` where the Hessian regulariser
dominates and gradients die.

The fix is to parameterise the **canonical momentum** `p(v) = ∂L/∂v` and the
potential `V(x)` as two small networks. Euler–Lagrange then collapses to

```
a = -V'(x) / p'(v)
```

First derivatives only, no Hessian to invert: relative loss `1.1e-05` against
`1.5e-02`. It is also the physically correct object — **special relativity is
precisely a statement about `p(v)`**. Newton says `p = mv`; relativity says
`p = γmv`. That is the entire content.

All three gauges are fixed explicitly: **parity** architecturally by
antisymmetrising the momentum, **sign** at readout by flipping `(p, V)`
together (a run really did return `b1 = −0.34`), and **scale** by reporting
only scale-invariant quantities — `ĉ` is one, `m` is not.

## What actually limits the answer

![recovered c against the readout's own error floor](../../figures/05_readout_floor.png)

The recovered `ĉ` tracks, to within **1.4% at every speed**, what a *perfect*
momentum function would give under the same cubic readout. So the deficit
belongs to the readout's truncation, not to the fit — the network learns
`p(v)` well everywhere from 0.2c to 0.7c.

Three consequences, each overturning a natural reading of the raw `ĉ` column:

- **The quartic is identifiable below 0.4c.** At 0.2c the converged seeds
  give 0.979, within 0.4% of the best that readout can do.
- **0.5c is not a sweet spot.** Its bias (−11%) is worse than 0.3c's (−4%).
- **0.7c is worse than 0.5c for an algebraic reason**, not only a
  conditioning one: the truncation bias grows monotonically with range.

Keeping the `v⁵` term trades bias for variance, and pays only where there is
bias to remove — at 0.5c it moves `ĉ` from 0.8813 ± 0.0022 to
**0.9870 ± 0.0184**; at 0.3c it is marginally worse.

**Noise barely touches any of this.** From clean data to 10% noise `ĉ` moves
0.6%, while the failure rate stays flat at ~40%.

## Only the formula extrapolates

![in-range and out-of-range error for each arm](../../figures/07_extrapolation.png)

Trained inside `|v| < 0.5c`, scored in `[0.7c, 0.95c]`, with the black box
matched on parameter count (8737 against 8705):

- **In-range the black box wins.** The physics prior costs a little
  in-distribution, which is what a prior should cost.
- **Out of range only the formula generalises**, and it is an order of
  magnitude more stable across seeds.
- **The physics-structured network extrapolates worse than the plain black
  box.** Structure does not make a network extrapolate; it makes a formula
  extractable, and the formula extrapolates.
- **A truncated series is not a law.** The cubic readout carries the same
  information as the γ-form and is the worst arm out there.

## The orbit, and the trap it documents

`orbit.py` is the acceptance suite: Kepler's third law to 0.0002%, a force-law
exponent of **−2.0000**, `GM` exact, and Mercury's perihelion precession at
**42.98 arcsec/century** measured two independent ways — the closed form, and
root-finding `du/dφ` on the integrated orbit.

It is **not measured data**. It is a high-precision integration from real
published constants, which is exactly what makes it a good acceptance test.

Mercury's perihelion speed is 59 km/s = **1.97e-4 c**, three orders below the
identifiability floor, so the `ĉ` route **cannot** work on orbital data.
General relativity enters there through the *secular precession* — a
cumulative effect over 415 orbits per century — not the instantaneous
momentum. Confusing the two would produce a confident wrong answer.
