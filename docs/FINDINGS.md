# Findings, in detail

The README carries the figures and the claims table. This file carries the numbers behind them, and the corrections each one needed before it
meant anything. Every table here is regenerated into `results/` by the
experiment that produced it; none is typed by hand.

---

### Claim 2 needed a fix before its number meant anything

The 72% saving is quoted from the estimated order-1 chain. Whether it
describes the *process* depends on `C_mu` having stopped moving with history
length, and the first implementation here failed that badly.

With float-equality merging, nothing ever merges: two conditional
distributions estimated from finite counts are never exactly equal. On
**Mersenne-Twister bits**, which carry no structure at all, `C_mu` climbed
1.0, 2.0, 3.0, 4.0, 4.7 and 5.5 bits at orders one to six -- it was measuring
the number of histories, not the process. Seattle showed the same ramp, so
its order-1 number could not be trusted either.

Merging now uses a two-proportion test at 2.5 sigma, as CSSR does. The
control collapses to ~0 bits at every order, and the real data plateaus:

| order | states | `C_mu` | `C_q` |
|---|---|---|---|
| 1 | 2 | 0.984 | 0.276 |
| 2 | 3 | 1.444 | 0.310 |
| 3 | 3 | 1.444 | 0.310 |
| 4 | 3 | 1.444 | 0.309 |

So the process carries about **1.44 bits** of classical predictive state, not
the 0.98 that order 1 alone suggests, and the saving at the plateau is
**78.5%** rather than 71.9%. Both numbers are defensible about different
objects -- 71.9% is a property of the first-order *model*, 78.5% of the
process as far as this data can see it -- and the README quotes both rather
than picking the flattering one.

### Claims 1 and 2 must never be conflated

| Claim | What it is about | Status |
|---|---|---|
| Representation (2) | an idealised construction from a *known* transition matrix | holds |
| Prediction (1) | *estimation* from finite data | fails |

Keeping these apart is the single most important piece of intellectual
hygiene in this project. A 72% memory saving on the estimated chain and a
worse test NLL are both true at once, and they say different things.

### What is never claimed

**No time series here "is quantum."** The only claim under test is whether a
model built from the quantum formalism is more efficient or more accurate
than the best classical model at matched capacity. That is a modeling claim,
not a physics claim (honesty rule 1).

### Claim 4: what limits `c_hat` is the readout, not the fit

Five seeds per row, converged runs only, with the failure rate beside the
mean because a third of these fits do not converge:

| v_max | noise | converged | c_hat | sd | readout floor | gap to floor |
|---|---|---|---|---|---|---|
| 0.2 | 0 | 3/5 | 0.9792 | 0.0092 | 0.9831 | 0.4% |
| 0.3 | 0 | 3/5 | 0.9485 | 0.0024 | 0.9615 | 1.4% |
| 0.5 | 0 | 3/5 | 0.8813 | 0.0022 | 0.8874 | 0.7% |
| 0.5 | 2% | 3/5 | 0.8802 | 0.0030 | 0.8874 | 0.8% |
| 0.5 | 5% | 3/5 | 0.8786 | 0.0043 | 0.8874 | 1.0% |
| 0.5 | 10% | 3/5 | 0.8764 | 0.0064 | 0.8874 | 1.2% |
| 0.7 | 0 | 3/5 | 0.7526 | 0.0007 | 0.7566 | 0.5% |

**Noise barely touches it.** From clean data to 10% noise `c_hat` moves by
0.6%, from 0.8813 to 0.8764, while its spread grows from 0.0022 to 0.0064 and
the failure rate does not move at all. The method is limited by the readout
and by whether a seed converges, not by measurement noise.

The **readout floor** is `c_hat` recovered from the EXACT `gamma v` by the
same cubic fit -- the error a perfect momentum function would still incur.
The recovered value tracks it to within 1.4% everywhere, so **effectively all
of the deviation from `c = 1` belongs to the readout, not to the network**.
The network learns `p(v)` well at every speed tested.

Three consequences, each of which contradicts a plausible reading of the raw
`c_hat` column:

- **The quartic IS identifiable below 0.4c.** At 0.2c the converged seeds give
  0.979, within 0.4% of the best that readout can do.
- **0.5c is not a sweet spot.** Its cubic bias (-11%) is worse than 0.3c's
  (-4%). A sweep that reported ~1.00 there was averaging scatter across the
  true value, not measuring it.
- **0.7c is worse than 0.5c for an algebraic reason**, not only a
  conditioning one: the truncation bias grows monotonically with the fitted
  range, from -2% at 0.2c to -24% at 0.7c.

Keeping the `v^5` term trades that bias for variance, and the trade is only
worth it where there is bias to remove:

| v_max | cubic | quintic | better |
|---|---|---|---|
| 0.3 | 0.9485 ± 0.0024 (5.2% off) | 0.9407 ± 0.0065 (5.9% off) | cubic |
| 0.5 | 0.8813 ± 0.0022 (11.9% off) | **0.9870 ± 0.0184 (1.3% off)** | quintic |

So the best configuration measured here is **`v_max = 0.5` read with a
quintic**, at `c_hat = 0.9870 ± 0.0184`. The cubic readout is the default
because it is lower variance and it is the method specified; the quintic is
one keyword away.

### Claim 4b: only the formula extrapolates

Section 14, trained inside `|v| < 0.5c` and scored in `[0.7c, 0.95c]`, over
the converged seeds of a 5-seed run, with the black box matched on parameter
count (8737 against 8705, a 0.4% difference):

| model | free params | in-range | out-of-range |
|---|---|---|---|
| BlackBox | 8737 | **0.00304 ± 0.0008** | 0.743 ± 0.075 |
| LawNet (neural) | 8705 | 0.00476 ± 0.0018 | 1.162 ± 0.080 |
| symbolic, gamma-form | 2 | 0.0427 ± 0.0007 | **0.559 ± 0.007** |
| symbolic, cubic truncation | 2 | – | 1.219 ± 0.017 |

- **In-range the black box wins.** The physics prior costs a little
  in-distribution, which is what it should cost.
- **Out of range only the formula generalises**, and it is also an order of
  magnitude more stable across seeds (0.007 against 0.075).
- **The physics-structured network extrapolates worse than the plain black
  box.** Structure does not make a network extrapolate; it makes a formula
  extractable, and the formula extrapolates. The pipeline is *network
  interpolates, symbolic fit, extrapolate the formula* -- never the network
  alone.
- **A truncated series is not a law.** The cubic readout, carrying the same
  information as the gamma-form, is the worst arm out of range.

### The convergence gate had to be fixed before the noisy rows meant anything

The first sweep reported a **100% failure rate** at 5% and 10% noise. That was
the gate, not the fits: injected noise puts a floor under the achievable
training loss -- 2.9e-04, 1.8e-03 and 7.3e-03 at 2%, 5% and 10% -- so a fixed
1e-03 threshold is reachable at 2% and *arithmetically impossible* above it.
The gate now measures the excess over that floor, which keeps honesty rule 7
intact because the floor is fixed by the injection's design, not read off the
answers.

