"""Learning the law by learning the canonical momentum.

## The design decision this module exists to encode

The textbook Lagrangian Neural Network parameterises `L_theta(x, v)` and
inverts the Euler-Lagrange equation,

    a = (d2L/dv2)^-1 [ dL/dx - (d2L/dx dv) v ]

**That was tried and it failed**: relative loss stalled at 0.19, the
recovered quadratic coefficient came back as 0.0053 against a true 0.5, and
`c_hat` at 0.70. The cause is gauge freedom -- `L` is fixed only up to an
overall scale, an additive constant, and any total time derivative dF(x)/dt.
Under the scale gauge the optimiser drifts toward tiny `lambda`, where the
Hessian regulariser dominates and the gradients die.

The fix, for separable systems, is to parameterise the **canonical momentum**
`p_theta(v) = dL/dv` and the potential `V_phi(x)` as two small networks.
Euler-Lagrange then collapses to `dp/dt = -V'(x)`, so

    a = -V'(x) / p'(v)

First derivatives only, no Hessian to invert. Measured: relative loss 1.1e-5
against 1.5e-2, and `c_hat` from 0.70 to 0.999.

This is also the physically correct object to learn. **Special relativity is
precisely a statement about p(v)**: Newton says `p = m v`, relativity says
`p = gamma m v`. That is the entire content, and reading `c` out of the
quartic term of slow motion is reading it off the one function that carries
it.

## Gauges, all three fixed explicitly

  sign     (p, V) -> (-p, -V) leaves `a` unchanged. Enforced at readout by
           flipping both when b1 < 0; a run really did return b1 = -0.436,
           and `c_hat` survives only because of that flip.
  scale    L is defined up to overall scale, so only scale-invariant
           quantities may be reported. `c_hat` is one; `m` is not.
  parity   p(-v) = -p(v), enforced architecturally by antisymmetrising the
           network. Free accuracy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from qphys.common.seeding import CONVERGENCE_REL_LOSS, converged
from qphys.lawlearn.relativistic import c_hat_from_momentum_coefficients

DTYPE = torch.float64


def _mlp(hidden: int, bias_out: bool) -> nn.Module:
    """tanh, never ReLU: a ReLU network has zero second derivative almost
    everywhere, which is fatal when the loss differentiates the output."""
    return nn.Sequential(
        nn.Linear(1, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1, bias=bias_out),
    ).to(DTYPE)


class LawNet(nn.Module):
    """p_theta(v) and V_phi(x), with parity on the momentum built in."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.p_net = _mlp(hidden, bias_out=False)
        self.v_net = _mlp(hidden, bias_out=True)

    def momentum(self, v: torch.Tensor) -> torch.Tensor:
        u = v.unsqueeze(-1)
        return (self.p_net(u) - self.p_net(-u)).squeeze(-1) / 2.0

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_net(x.unsqueeze(-1)).squeeze(-1)

    def accel(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """a = -V'(x) / p'(v). The 1e-3 keeps the denominator away from zero
        without biasing it, since p'(v) >= 1 for any physical momentum."""
        x = x.requires_grad_(True)
        v = v.requires_grad_(True)
        dp = torch.autograd.grad(self.momentum(v).sum(), v, create_graph=True)[0]
        dv = torch.autograd.grad(self.potential(x).sum(), x, create_graph=True)[0]
        return -dv / (dp.abs() + 1e-3)


# --- the system being learned ---------------------------------------------


def relativistic_samples(
    n: int,
    v_max: float,
    seed: int,
    noise: float = 0.0,
    k: float = 1.0,
    x_max: float = 1.0,
    newtonian: bool = False,
):
    """(x, v, a) for a particle in V(x) = k x^2 / 2, natural units m = c = 1.

    a = -V'(x) / gamma^3 relativistically, a = -V'(x) in the Newtonian
    control. The network is handed these three columns and nothing else --
    no gamma, no mention of a speed limit, no functional form.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-x_max, x_max, n)
    v = rng.uniform(-v_max, v_max, n)
    gamma = 1.0 if newtonian else 1.0 / np.sqrt(1.0 - v**2)
    a = -(k * x) / gamma**3
    if noise > 0.0:
        a = a + rng.normal(0.0, noise * np.mean(np.abs(a)), n)
    return x, v, a


def true_momentum(v, newtonian: bool = False):
    v = np.asarray(v, float)
    return v if newtonian else v / np.sqrt(1.0 - v**2)


# --- training --------------------------------------------------------------


def train(
    x,
    v,
    a,
    seed: int = 0,
    hidden: int = 64,
    steps: int = 5000,
    lr: float = 2e-3,
    batch: int = 1024,
) -> dict:
    """Fit p(v) and V(x) to (x, v, a). Returns the model and its final loss.

    The loss is normalised by mean(a^2) so it is scale-free and the
    pre-declared convergence threshold means the same thing at every v_max.
    """
    torch.manual_seed(seed)
    xt = torch.tensor(np.asarray(x, float), dtype=DTYPE)
    vt = torch.tensor(np.asarray(v, float), dtype=DTYPE)
    at = torch.tensor(np.asarray(a, float), dtype=DTYPE)
    scale = float(torch.mean(at**2))

    net = LawNet(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = len(at)
    gen = torch.Generator().manual_seed(seed)

    loss_value = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen)
        opt.zero_grad()
        pred = net.accel(xt[idx].clone(), vt[idx].clone())
        loss = torch.mean((pred - at[idx]) ** 2) / scale
        loss.backward()
        opt.step()
        sched.step()
        loss_value = float(loss.detach())

    return {"net": net, "rel_loss": loss_value, "seed": seed}


# The readout truncates a series, and the truncation itself is biased. Fitting
# b1 v + b3 v^3 to the EXACT gamma v over |v| < 0.5 returns c_hat = 0.887, an
# 11% error with no network involved at all, because the v^5 term of gamma v
# has nowhere to go but into b3. Measured on the exact momentum:
#
#     v_max    cubic    +v^5     +v^7
#     0.3     0.9615   1.0021   0.9999
#     0.5     0.8874   1.0222   0.9972
#     0.7     0.7566   1.1703   0.9605
#
# So the cubic readout is LOW VARIANCE AND BIASED, and the quintic is nearly
# unbiased but leans harder on the learned momentum being right far from the
# origin. DEFAULT_POWERS keeps the cubic, which is the method the build
# prompt specifies and the one its table reports; `powers=(1, 3, 5)` is the
# debiased variant, and the sweep reports both.
DEFAULT_POWERS = (1, 3)


def read_momentum_coefficients(
    net: LawNet, v_max: float, n: int = 400, powers=DEFAULT_POWERS
) -> tuple:
    """Fit p_hat(v) ~= sum_k b_k v^k on the learned momentum.

    Read from the NETWORK rather than from the data, so it is the learned law
    being interrogated, not the samples it was trained on.
    """
    grid = np.linspace(-v_max, v_max, n)
    with torch.no_grad():
        p = net.momentum(torch.tensor(grid, dtype=DTYPE)).numpy()
    basis = np.stack([grid**k for k in powers], axis=1)
    coeffs, *_ = np.linalg.lstsq(basis, p, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def readout_bias(v_max: float, powers=DEFAULT_POWERS, n: int = 4000) -> float:
    """c_hat recovered from the EXACT momentum: the readout's own error floor.

    Any trained result should be read against this. A fit that returns 0.87
    at v_max = 0.5 under the cubic readout has not missed by 13% -- it has
    landed within 2% of what a perfect momentum function would give.
    """
    grid = np.linspace(-v_max, v_max, n)
    p = true_momentum(grid)
    basis = np.stack([grid**k for k in powers], axis=1)
    coeffs, *_ = np.linalg.lstsq(basis, p, rcond=None)
    return c_hat_from_momentum_coefficients(float(coeffs[0]), float(coeffs[1]))


def recover_c(net: LawNet, v_max: float, powers=DEFAULT_POWERS) -> float:
    b1, b3 = read_momentum_coefficients(net, v_max, powers=powers)
    return c_hat_from_momentum_coefficients(b1, b3)


def fit_and_read(
    v_max: float,
    seed: int,
    noise: float = 0.0,
    n: int = 4000,
    steps: int = 5000,
    newtonian: bool = False,
) -> dict:
    """One seed, end to end, with the convergence gate applied BEFORE the
    answer is read (honesty rule 7).

    A run above the pre-declared threshold returns c_hat = nan and is counted
    as a failure. The failure is detectable without looking at the answer,
    which is what makes the discard legitimate rather than cherry-picking:
    at v_max = 0.5c the failing seed finishes two orders of magnitude worse
    on training loss than the converged ones.
    """
    x, v, a = relativistic_samples(n, v_max, seed, noise, newtonian=newtonian)
    fit = train(x, v, a, seed=seed, steps=steps)
    ok = converged(fit["rel_loss"])
    b1, b3 = read_momentum_coefficients(fit["net"], v_max)
    return {
        "v_max": v_max,
        "noise": noise,
        "seed": seed,
        "rel_loss": fit["rel_loss"],
        "converged": ok,
        "b1": b1,
        "b3": b3,
        "c_hat": recover_c(fit["net"], v_max) if ok else float("nan"),
        # the same fit read with the debiased quintic, and the readout's own
        # error floor, so a result can be separated from its readout
        "c_hat_quintic": (
            recover_c(fit["net"], v_max, powers=(1, 3, 5)) if ok else float("nan")
        ),
        "readout_floor": readout_bias(v_max),
        "readout_floor_quintic": readout_bias(v_max, powers=(1, 3, 5)),
        "threshold": CONVERGENCE_REL_LOSS,
        "net": fit["net"],
    }


# --- section 14: the metric that actually matters --------------------------


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class BlackBox(nn.Module):
    """a = f(x, v), no physics. The thing the structured model must beat.

    Its width is chosen so its parameter count matches `LawNet`'s, because a
    black box given fewer parameters is not a baseline, it is a strawman.
    """

    def __init__(self, hidden: int = 91):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        ).to(DTYPE)

    def forward(self, x, v):
        return self.net(torch.stack([x, v], dim=-1)).squeeze(-1)


def matched_hidden(target_params: int) -> int:
    """Smallest hidden width whose BlackBox is closest to `target_params`."""
    best, best_gap = 1, np.inf
    for h in range(2, 400):
        gap = abs(count_params(BlackBox(h)) - target_params)
        if gap < best_gap:
            best, best_gap = h, gap
    return best


def train_blackbox(x, v, a, seed=0, hidden=91, steps=5000, lr=2e-3, batch=1024):
    torch.manual_seed(seed)
    xt = torch.tensor(np.asarray(x, float), dtype=DTYPE)
    vt = torch.tensor(np.asarray(v, float), dtype=DTYPE)
    at = torch.tensor(np.asarray(a, float), dtype=DTYPE)
    scale = float(torch.mean(at**2))
    net = BlackBox(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(seed)
    n = len(at)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen)
        opt.zero_grad()
        loss = torch.mean((net(xt[idx], vt[idx]) - at[idx]) ** 2) / scale
        loss.backward()
        opt.step()
        sched.step()
    return net, float(loss.detach())


def potential_gradient_coefficient(net: LawNet, x_max: float = 1.0) -> float:
    """Fit V'(x) ~= k x on the learned potential, for the symbolic readout."""
    grid = torch.linspace(-x_max, x_max, 200, dtype=DTYPE).requires_grad_(True)
    dv = torch.autograd.grad(net.potential(grid).sum(), grid)[0].detach().numpy()
    g = grid.detach().numpy()
    return float(np.sum(g * dv) / np.sum(g * g))


def symbolic_accel(net: LawNet, v_max: float, x, v, relativistic: bool = True):
    """Extrapolate the FORMULA, not the network.

    Networks do not extrapolate; a fitted formula does. Two readouts:

    `relativistic=True`  reconstruct p(v) = b1 v / sqrt(1 - v^2 / c_hat^2),
                         which is what a physicist writes down once c_hat is
                         in hand. This is the pipeline that generalises.
    `relativistic=False` keep the cubic truncation p = b1 v + b3 v^3. It is
                         the same information, and it fails outside the
                         fitted range -- a truncated series is not a law.
    """
    b1, b3 = read_momentum_coefficients(net, v_max)
    if b1 < 0:
        b1, b3 = -b1, -b3
    k = abs(potential_gradient_coefficient(net))
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    if relativistic:
        c = c_hat_from_momentum_coefficients(b1, b3)
        if not np.isfinite(c):
            return np.full_like(v, np.nan)
        s = 1.0 - (v / c) ** 2
        s = np.where(s <= 1e-12, np.nan, s)
        dp = b1 / s**1.5  # d/dv of gamma-form momentum
    else:
        dp = b1 + 3.0 * b3 * v**2
    return -(k * x) / dp


def _nrmse(truth, pred) -> float:
    truth = np.asarray(truth, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(pred)
    if not ok.any():
        return float("nan")
    return float(
        np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2)) / np.std(truth[ok])
    )


def extrapolation_benchmark(
    seed: int = 0, v_train: float = 0.5, v_out=(0.7, 0.95), n: int = 4000, steps=4000
) -> dict:
    """Train inside |v| < v_train, score inside and in [0.7, 0.95] c.

    Reports the neural momentum model, both symbolic readouts, and a
    parameter-matched black box, so the claim "the network interpolates and
    the formula extrapolates" is measured rather than asserted.
    """
    x, v, a = relativistic_samples(n, v_train, seed)
    fit = train(x, v, a, seed=seed, steps=steps)
    net = fit["net"]
    bb, _ = train_blackbox(x, v, a, seed=seed, hidden=matched_hidden(count_params(net)))

    rng = np.random.default_rng(seed + 500)
    xo = rng.uniform(-1.0, 1.0, n)
    sign = rng.choice([-1.0, 1.0], n)
    vo = sign * rng.uniform(*v_out, n)
    ao = -xo / (1.0 / np.sqrt(1.0 - vo**2)) ** 3

    def neural(xx, vv):
        return (
            net.accel(
                torch.tensor(xx, dtype=DTYPE).clone(),
                torch.tensor(vv, dtype=DTYPE).clone(),
            )
            .detach()
            .numpy()
        )

    def black(xx, vv):
        with torch.no_grad():
            return bb(
                torch.tensor(xx, dtype=DTYPE), torch.tensor(vv, dtype=DTYPE)
            ).numpy()

    return {
        "seed": seed,
        "rel_loss": fit["rel_loss"],
        "converged": converged(fit["rel_loss"]),
        "lawnet_params": count_params(net),
        "blackbox_params": count_params(bb),
        "in_lawnet": _nrmse(a, neural(x, v)),
        "in_blackbox": _nrmse(a, black(x, v)),
        "in_symbolic": _nrmse(a, symbolic_accel(net, v_train, x, v)),
        "out_lawnet": _nrmse(ao, neural(xo, vo)),
        "out_blackbox": _nrmse(ao, black(xo, vo)),
        "out_symbolic_relativistic": _nrmse(ao, symbolic_accel(net, v_train, xo, vo)),
        "out_symbolic_cubic": _nrmse(
            ao, symbolic_accel(net, v_train, xo, vo, relativistic=False)
        ),
        "c_hat": recover_c(net, v_train),
    }
