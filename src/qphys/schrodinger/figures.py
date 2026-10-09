"""Figures for docs/schrodinger, from results/ and from short simulations.

python -m qphys.schrodinger.figures      # all of them, into figures/
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qphys.common.plotting import CATEGORICAL, INK, label_at, note, use_style
from qphys.config import FIGURES_DIR, RESULTS_DIR
from qphys.schrodinger import born as B
from qphys.schrodinger import evolve as Ev
from qphys.schrodinger import grid as G
from qphys.schrodinger import stationary as S
from qphys.schrodinger import tunneling as T

NUMERIC, PREDICTION, SPURIOUS, OTHER = CATEGORICAL


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(FIGURES_DIR.parent)}")
    return path


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def fig_eigenstates():
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    cases = [
        (
            "Harmonic well: equally spaced levels",
            G.Grid(-6, 6, 600),
            G.harmonic,
            (-5, 5),
            5,
        ),
        (
            "Double well (depth 4): levels come in tunnel-split pairs",
            G.Grid(-7, 7, 700),
            lambda x: G.double_well(x, a=2.0),
            (-4.2, 4.2),
            6,
        ),
    ]
    for ax, (title, g, Vf, xlim, k) in zip(axes, cases, strict=True):
        V = Vf(g.x)
        E, states = S.eigenstates(g, V, k)
        ax.plot(g.x, V, color=INK["secondary"], lw=1.4)
        gaps = np.diff(E)
        scale = 0.42 * np.median(gaps) / np.abs(states).max()
        for n in range(k):
            ax.axhline(E[n], color=INK["grid"], lw=0.8, zorder=0)
            ax.fill_between(
                g.x,
                E[n],
                E[n] + scale * states[n] ** 2 * 2.2,
                color=NUMERIC,
                alpha=0.18,
                lw=0,
            )
            ax.plot(g.x, E[n] + scale * states[n], color=NUMERIC, lw=1.4)
            if n == 0 or E[n] - E[n - 1] > 0.3 * np.median(gaps):
                label_at(
                    ax,
                    xlim[0],
                    E[n],
                    f"E{n} = {E[n]:.3f}",
                    INK["secondary"],
                    dx=2,
                    dy=7,
                    size=7.5,
                )
            else:
                label_at(
                    ax,
                    xlim[0],
                    E[n - 1],
                    f"E{n} - E{n - 1} = {E[n] - E[n - 1]:.1e}",
                    INK["secondary"],
                    dx=2,
                    dy=-9,
                    size=7.5,
                )
        ax.set_xlim(*xlim)
        ax.set_ylim(V.min() - 0.3 if V.min() < 0 else -0.2, E[-1] + 1.0)
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("energy")
    note(
        axes[0],
        "Line: phi_n(x), drawn at height E_n. Shade: |phi_n|^2, where the particle is found.",
    )
    return _save(fig, "08_eigenstates.png")


def _tunnel_run(k0=1.2, frames=240):
    g = G.Grid(-80.0, 80.0, 2048)
    V = G.barrier(g.x, 1.0, 1.0)
    psi0 = Ev.gaussian_packet(g, -25.0, k0, 3.0)
    t_end = 50.0 / k0
    steps = 2400
    return (
        g,
        V,
        Ev.split_operator(g, V, psi0, t_end / steps, steps, every=steps // frames),
    )


def fig_spacetime():
    use_style()
    g, _, ev = _tunnel_run()
    rho = np.abs(ev.psi) ** 2
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    keep = (g.x > -45) & (g.x < 45)
    ax.imshow(
        rho[:, keep].T,
        origin="lower",
        aspect="auto",
        cmap="Blues",
        extent=[ev.t[0], ev.t[-1], g.x[keep][0], g.x[keep][-1]],
        vmax=np.percentile(rho, 99.5),
    )
    ax.axhspan(-0.5, 0.5, color=PREDICTION, alpha=0.5, lw=0)
    ax.grid(False)
    ax.set_xlabel("time")
    ax.set_ylabel("x")
    T_end = (rho[-1, g.x > 0.5]).sum() * g.dx
    ax.set_title(
        f"A packet with E = 0.72 V0 splits at the barrier; {T_end:.0%} tunnels through"
    )
    label_at(ax, ev.t[-1], 0, "barrier", PREDICTION, dx=-48, dy=8)
    note(
        ax,
        "|psi(x, t)|^2. Classically every particle with E < V0 is reflected.",
        y=-0.16,
    )
    return _save(fig, "09_tunneling_spacetime.png")


def gif_tunneling():
    use_style()
    from matplotlib.animation import FuncAnimation, PillowWriter

    g, V, ev = _tunnel_run(frames=120)
    keep = (g.x > -45) & (g.x < 45)
    x = g.x[keep]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ymax = 1.15 * (np.abs(ev.psi[0]) ** 2).max()
    ax.fill_between(x, 0, V[keep] * ymax * 0.9, color=PREDICTION, alpha=0.35, lw=0)
    (re,) = ax.plot(x, np.zeros_like(x), color=INK["muted"], lw=0.8)
    (dens,) = ax.plot(x, np.zeros_like(x), color=NUMERIC, lw=1.8)
    ax.set_ylim(-ymax * 0.6, ymax)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("x")
    ax.set_yticks([])
    title = ax.set_title("")

    def draw(i):
        psi = ev.psi[i][keep]
        dens.set_ydata(np.abs(psi) ** 2)
        re.set_ydata(0.5 * np.sqrt(ymax) * psi.real)
        T_now = (np.abs(ev.psi[i][g.x > 0.5]) ** 2).sum() * g.dx
        title.set_text(
            f"|psi|^2 (blue), Re psi (grey).  t = {ev.t[i]:5.1f}   P(x > barrier) = {T_now:.2f}"
        )
        return dens, re, title

    anim = FuncAnimation(fig, draw, frames=len(ev.t), blit=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "10_tunneling.gif"
    anim.save(path, writer=PillowWriter(fps=24), dpi=90)
    plt.close(fig)
    print(f"wrote {path.relative_to(FIGURES_DIR.parent)}")
    return path


def fig_born():
    use_style()
    from qphys.schrodinger.experiments import interference_state

    g, psi = interference_state()
    rho = np.abs(psi) ** 2
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1, 0.9]}
    )
    for ax, n in zip(axes[:2], (300, 100_000), strict=True):
        x = B.sample_positions(g, psi, n, seed=0)
        ax.hist(
            x,
            bins=np.linspace(-6, 6, 121),
            density=True,
            color=NUMERIC,
            alpha=0.55,
            lw=0,
        )
        ax.plot(g.x, rho, color=INK["primary"], lw=1.2)
        ax.set_xlim(-6, 6)
        ax.set_title(f"{n:,} detections")
        ax.set_xlabel("x")
    axes[0].set_ylabel("density")
    label_at(axes[1], 1.2, rho.max() * 0.98, "|psi|^2", INK["primary"])

    df = _csv("schrodinger_born.csv")
    m = df.groupby("n_samples").tv_distance.agg(["mean", "std"])
    ax = axes[2]
    ax.errorbar(m.index, m["mean"], m["std"], fmt="o", color=NUMERIC, capsize=2)
    ref = m["mean"].iloc[0] * np.sqrt(m.index[0] / m.index.to_numpy())
    ax.plot(m.index, ref, color=INK["muted"], lw=1, ls="--")
    label_at(ax, m.index[2], ref[2], "n^(-1/2)", INK["muted"], dy=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("detections n")
    ax.set_title(f"Histogram -> |psi|^2 as n^{df.fitted_exponent.iloc[0]:.2f}")
    ax.set_ylabel("total-variation distance")
    note(
        axes[0],
        "Two packets overlapping. Each run gives one point; the fringes appear only in aggregate.",
    )
    return _save(fig, "11_born_rule.png")


def fig_tunneling():
    use_style()
    df = _csv("schrodinger_tunneling.csv")
    V0, a = df.V0.iloc[0], df.a.iloc[0]
    E = np.linspace(0.02, 3.6, 600)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(E / V0, T.transmission_plane_wave(E, V0, a), color=INK["muted"], lw=1.2)
    ax.plot(df.E0 / V0, df.T_packet, color=PREDICTION, lw=1.8)
    ax.plot(df.E0 / V0, df.T_sim, "o", color=NUMERIC, ms=5)
    ax.step(
        [0, 1, 1, 3.6], [0, 0, 1, 1], where="post", color=INK["secondary"], lw=1, ls=":"
    )
    ax.axvline(1, color=INK["grid"], lw=1)
    label_at(
        ax,
        2.9,
        float(T.transmission_plane_wave(2.9 * V0, V0, a)[0]),
        "plane wave",
        INK["muted"],
        dy=-10,
    )
    label_at(
        ax,
        df.E0.iloc[3],
        df.T_packet.iloc[3],
        "packet prediction",
        PREDICTION,
        dx=10,
        dy=-6,
    )
    label_at(ax, df.E0.iloc[6], df.T_sim.iloc[6], "simulation", NUMERIC, dx=-62, dy=8)
    label_at(ax, 0.05, 0.03, "classical", INK["secondary"])
    ax.set_xlabel("mean energy / barrier height")
    ax.set_ylabel("transmission probability")
    worst = df.abs_err_packet.max()
    ax.set_title(
        f"Simulated transmission matches the packet-averaged formula to {worst:.0e}"
    )
    note(
        ax,
        "Barrier V0 = 1, width 1. The plane-wave line is the single-energy idealisation.",
        y=-0.16,
    )
    return _save(fig, "12_tunneling.png")


def fig_discovery():
    use_style()
    df = _csv("schrodinger_discovery.csv")
    terms = [c[3:-1] for c in df if c.startswith("re[")]
    g = df.groupby("noise")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    noise = np.array(sorted(df.noise.unique()))
    true = {"psi_xx": -0.5, "x^2 psi": 0.5}
    for term in terms:
        col = f"re[{term}]"
        mean, std = g[col].mean().to_numpy(), g[col].std().fillna(0).to_numpy()
        if not np.any(mean):
            continue
        color = (
            NUMERIC if term == "psi_xx" else (PREDICTION if term in true else SPURIOUS)
        )
        err = mean - true.get(term, 0.0)
        ax.errorbar(100 * noise, err, std, fmt="o-", color=color, capsize=2, lw=1.4)
        label_at(
            ax,
            100 * noise[-1],
            err[-1],
            term if term in true else f"{term} (spurious)",
            color,
        )
    ax.axhline(0, color=INK["muted"], lw=1, ls="--")
    ax.set_xlabel("noise on psi (% of max |psi|)")
    ax.set_ylabel("recovered - true coefficient")
    ok = df.groupby("noise").exact_support.mean()
    clean = ok[ok == 1].index.max()
    ax.set_title(
        f"Of {len(terms)} candidate terms, exactly the right two survive up to {100 * clean:.0f}% noise"
    )
    note(
        ax,
        "Target i psi_t. Truth: -1/2 psi_xx + 1/2 x^2 psi. Every term not drawn was set to zero.",
        y=-0.16,
    )
    return _save(fig, "13_discovery.png")


def fig_numerics():
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    conv = _csv("schrodinger_convergence.csv")
    for order, color in ((2, PREDICTION), (4, NUMERIC)):
        sub = conv[conv.order == order]
        ax.loglog(sub.dx, sub.max_rel_err, "o-", color=color)
        label_at(
            ax,
            sub.dx.iloc[0],
            sub.max_rel_err.iloc[0],
            f"{order + 1 if order == 2 else 5}-point: dx^{sub.fitted_order.iloc[0]:.2f}",
            color,
            dx=-10,
            dy=-12,
            ha="right",
        )
    ax.set_xlabel("grid spacing dx")
    ax.set_ylabel("max relative error, E1..E4")
    ax.set_title("Infinite well: errors fall at the stencil's order")

    ax = axes[1]
    spec = _csv("schrodinger_spectrum.csv")
    q = spec[spec.potential == "quartic"]
    ax.semilogy(q.n, q.rel_err.abs(), "o-", color=NUMERIC)
    ax.set_xlabel("level n")
    ax.set_ylabel("|E_WKB / E_exact - 1|")
    ax.set_title(
        f"Quartic well: WKB is {q.rel_err.abs().iloc[0]:.0%} off at n = 0, {q.rel_err.abs().iloc[-1]:.2%} at n = {q.n.iloc[-1]}"
    )
    note(
        axes[1],
        "WKB is a semiclassical approximation; it improves as the wavelength shrinks.",
        y=-0.2,
    )
    return _save(fig, "14_numerics.png")


ALL = (
    fig_eigenstates,
    fig_spacetime,
    gif_tunneling,
    fig_born,
    fig_tunneling,
    fig_discovery,
    fig_numerics,
)


def build(only: str | None = None) -> None:
    for fn in ALL:
        if only and only not in fn.__name__:
            continue
        fn()


if __name__ == "__main__":
    import sys

    build(sys.argv[1] if len(sys.argv) > 1 else None)
