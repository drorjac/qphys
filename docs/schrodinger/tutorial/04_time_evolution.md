# 4 · Time evolution

[Previous](03_stationary_states.md) · [Topic page](../README.md) · Next: [5 · Tunnelling](05_tunneling.md)

## The formal solution

Because H does not depend on time, the solution is

$$ \psi(t) = e^{-iHt}\,\psi(0) . $$

H is Hermitian, so exp(-iHt) is unitary and preserves the norm. A numerical
method that is not exactly unitary leaks or creates probability. Explicit
Euler, the first method one might try, multiplies the norm by
sqrt(1 + dt^2 E^2) every step and blows up.

## Split-operator

The kinetic part of H is diagonal in momentum space, the potential part in
position space. Neither alone is hard to exponentiate. The Strang splitting

$$ e^{-iH\,dt} = e^{-iV\,dt/2}\;e^{-iT\,dt}\;e^{-iV\,dt/2} + O(dt^3) $$

is applied with an FFT between the factors:

```
psi <- exp(-i V dt/2) * psi
psi <- IFFT( exp(-i k^2 dt/2) * FFT(psi) )
psi <- exp(-i V dt/2) * psi
```

Every factor multiplies each component by a pure phase, so each step is
exactly unitary, whatever dt is. The O(dt^3) error per step comes from V and
T not commuting (the Baker-Campbell-Hausdorff commutator terms). Over a
fixed time it accumulates to O(dt^2).

`evolve.split_operator(grid, V, psi0, dt, steps)` implements this.

## Measuring the integrator

A Gaussian packet of width 1 is displaced to x = 3 in the harmonic well and
evolved for one classical period, 2 pi. The reference is the eigenstate
expansion of part 3, built from the spectral Hamiltonian that the
integrator actually uses, so that only the time-stepping error is measured.

| dt | norm drift | energy drift | state error | Ehrenfest residual |
|---|---|---|---|---|
| 0.20 | 4.0e-15 | 9.5e-3 | 0.070 | 1.0e-2 |
| 0.10 | 1.0e-14 | 2.4e-3 | 0.017 | 2.5e-3 |
| 0.05 | 1.5e-14 | 5.9e-4 | 4.4e-3 | 6.3e-4 |
| 0.02 | 4.2e-14 | 9.5e-5 | 7.0e-4 | 1.0e-4 |
| 0.01 | 1.1e-13 | 2.4e-5 | 1.75e-4 | 2.5e-5 |

From `results/schrodinger_conservation.csv`. The norm is conserved to
rounding error at every step size; its slow growth with the number of steps
is accumulated floating-point error. The state error falls by 4.0 for each
halving of dt, the dt^2 of Strang splitting. The first version of this check
used eigenstates of the finite-difference Hamiltonian as the reference; its
error stopped falling at 4.3e-4, which was the difference between the two
discretisations of the kinetic term, not a property of the integrator.

## Back to Newton: Ehrenfest's theorem

Differentiating the expectation values and using the equation gives

$$ \frac{d\langle x\rangle}{dt} = \langle p\rangle, \qquad
   \frac{d\langle p\rangle}{dt} = -\Big\langle \frac{dV}{dx}\Big\rangle . $$

The mean obeys Newton's law with the force averaged over |psi|^2. For the
harmonic well the force is linear, its average is the force at the mean,
and the centre of the packet follows the classical orbit exactly: after one
period <x> is back at 2.99998 for dt = 0.01. For a non-linear force the mean
force differs from the force at the mean, and quantum and classical motion
separate. The last column of the table is the residual of the momentum
equation, relative to the size of the force. It falls as dt^2.

## Spreading

A free Gaussian of initial width sigma widens as

$$ \sigma(t) = \sigma\sqrt{1 + \big(t / 2\sigma^2\big)^2} . $$

The narrower the packet, the wider its momentum distribution and the faster
it spreads. This is the dispersion relation of part 1 at work. In a harmonic
well the spreading is undone every half period and the packet breathes
instead.

```python
from qphys.schrodinger import evolve as Ev, grid as G

g = G.Grid(-12, 12, 512)
V = G.harmonic(g.x)
psi0 = Ev.gaussian_packet(g, x0=3.0, k0=0.0, sigma=1.0)
run = Ev.split_operator(g, V, psi0, dt=0.01, steps=628)
Ev.expect_x(g, run.psi[-1]), Ev.norm(g, run.psi[-1])
```
