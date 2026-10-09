# 1 · From waves to the equation

[Back to the topic page](../README.md) · Next: [2 · What psi predicts](02_what_psi_predicts.md)

The Schrodinger equation is not derived from anything more basic. It is
constructed so that two experimental facts hold, and then tested. This part
follows the construction.

## Two facts from experiment

Light of frequency omega comes in quanta of energy (Planck, Einstein):

$$ E = \hbar\,\omega . $$

Electrons diffract like waves, with a wavelength set by their momentum
(de Broglie, confirmed by Davisson and Germer):

$$ p = \hbar\,k, \qquad k = 2\pi/\lambda . $$

## A free particle as a wave

A wave with one wavenumber and one frequency is

$$ \psi(x, t) = e^{i(kx - \omega t)} . $$

Taking derivatives pulls the numbers down:

$$ -i\hbar\,\partial_x \psi = \hbar k\,\psi = p\,\psi, \qquad
   i\hbar\,\partial_t \psi = \hbar\omega\,\psi = E\,\psi . $$

So on this wave, momentum acts as the operator `p = -i hbar d/dx` and energy
as `E = i hbar d/dt`. A free non-relativistic particle has E = p^2 / 2m.
Asking the wave to satisfy that relation as an operator identity gives

$$ i\hbar\,\partial_t \psi = -\frac{\hbar^2}{2m}\,\partial_x^2 \psi . $$

That is the free Schrodinger equation. Its dispersion relation is
omega = hbar k^2 / 2m: different wavelengths travel at different speeds, so
a localised packet spreads. Part 4 measures that spreading.

## Adding a force

A particle in a potential V(x) has E = p^2/2m + V(x). The same substitution
gives

$$ i\hbar\,\partial_t \psi = \Big[-\frac{\hbar^2}{2m}\,\partial_x^2 + V(x)\Big]\psi \equiv \hat H \psi . $$

This step is a guess. Nothing forces V to enter as a multiplication. It is
justified by its consequences: it reproduces the hydrogen spectrum, and in
part 4 it reproduces Newton's law for the mean position (Ehrenfest).

From here on hbar = m = 1. In those units the coefficient of psi_xx is
-1/2, and part 6 measures it from data.

## Why the equation is complex

A real wave equation, such as the one for a string, is second order in
time. You must give both psi and its time derivative at t = 0. The
Schrodinger equation is first order in time, so psi(0) alone fixes the
future. A real first-order equation of the form psi_t = c psi_xx is the heat
equation: it smooths and decays, and the total of psi is not conserved.
The factor i turns decay into rotation. Each Fourier mode evolves as
exp(-i k^2 t / 2), a phase, so nothing decays.

What is conserved is the integral of |psi|^2. Multiplying the equation by
psi* and subtracting the complex conjugate gives a continuity equation

$$ \partial_t |\psi|^2 + \partial_x j = 0, \qquad
   j = \operatorname{Im}\!\big(\psi^*\,\partial_x\psi\big) . $$

Probability density |psi|^2 flows with current j and is neither created nor
destroyed. This is why |psi|^2 can be a probability (part 2). It holds
because H is Hermitian, and part 4 checks that the integrator keeps this
property exactly.

## Linearity

If psi_1 and psi_2 are solutions, so is any a psi_1 + b psi_2. This single
property produces interference (part 2), stationary-state expansions
(part 3), and the double-well splitting (part 3). It is also what part 6
exploits: the right-hand side is a linear combination of terms, so finding
the equation is a linear regression problem.

## In code

```python
from qphys.schrodinger.grid import Grid, hamiltonian, harmonic

g = Grid(-10, 10, 1000)          # x_j = -10 + j dx
H = hamiltonian(g, harmonic(g.x))  # sparse, real, symmetric
```

On a grid, psi is a vector, d2/dx2 is a banded matrix, V is a diagonal
matrix, and the equation becomes i dpsi/dt = H psi, a linear ODE.
