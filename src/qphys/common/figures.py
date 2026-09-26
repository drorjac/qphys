"""Every figure in the repo, generated from `results/` and from closed forms.

    python -m qphys.common.figures          # all of them, into figures/

Each figure answers one question and says its answer in the title, because a
chart whose point has to be explained in prose is not carrying its weight.
Forms are chosen by the data's job -- magnitude, identity, polarity -- and
the palette is validated by `qphys.common.palette`, never eyeballed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qphys.common.plotting import INK, ROLE, label_at, note, use_style
from qphys.config import FIGURES_DIR, RESULTS_DIR


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(FIGURES_DIR.parent)}")
    return path


# --- Project 1 ------------------------------------------------------------


def fig_compression_gap() -> Path:
    """Why quantum causal states can be cheaper: they may be non-orthogonal.

    Form: two lines over a continuous parameter, with the gap between them
    shaded -- the gap IS the result, so it gets the ink.
    """
    from qphys.collapse import causal_states as cs

    use_style()
    p = np.linspace(0.001, 0.5, 400)
    c_mu = np.array([cs.c_mu(cs.perturbed_coin_transition(x)) for x in p])
    c_q = np.array([cs.perturbed_coin_c_q(x) for x in p])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.fill_between(p, c_q, c_mu, color=ROLE["quantum"], alpha=0.10, linewidth=0)
    ax.plot(p, c_mu, color=ROLE["classical"], label="classical  $C_\\mu$")
    ax.plot(p, c_q, color=ROLE["quantum"], label="quantum  $C_q$")

    ax.annotate(
        "the saving",
        xy=(0.36, 0.5 * (c_mu[288] + c_q[288])),
        color=INK["secondary"],
        fontsize=8.5,
        ha="center",
    )
    ax.plot([0.5], [0.0], "o", color=ROLE["classical"], zorder=5)
    ax.annotate(
        "at $p=0.5$ the states merge\nand $C_\\mu$ drops to 0",
        xy=(0.5, 0.0),
        xytext=(-10, 34),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=INK["secondary"],
        arrowprops={"arrowstyle": "-", "color": INK["muted"], "linewidth": 0.8},
    )
    ax.set_xlabel("flip probability $p$")
    ax.set_ylabel("predictive state cost (bits)")
    ax.set_title("Non-orthogonal states cost less to remember")
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="lower left")
    note(ax, "Perturbed coin, closed form. $C_q \\leq C_\\mu$ always (Gu et al. 2012).")
    return _save(fig, "01_compression_gap.png")


def fig_state_proliferation() -> Path:
    """The most common way this estimate goes wrong, drawn.

    Form: small multiples. The two panels are two METHODS, so they are
    panels rather than series; within each, the two series are real data and
    a structureless control.
    """
    from qphys.collapse import causal_states as cs
    from qphys.collapse import data as D

    use_style()
    orders = (1, 2, 3, 4, 5, 6)
    real = D.seattle_wet_dry().values
    control = np.random.RandomState(0).randint(0, 2, len(real))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, statistical, title in zip(
        axes,
        (False, True),
        ("Exact-equality merge — broken", "Two-proportion merge — correct"),
        strict=True,
    ):
        for series, role, name in (
            (real, "classical", "Seattle wet/dry"),
            (control, "control", "random bits (no structure)"),
        ):
            df = cs.complexity_vs_history(series, orders, statistical=statistical)
            ax.plot(df.order, df.C_mu, "-o", color=ROLE[role], label=name)
        ax.plot(orders, orders, ls=(0, (3, 3)), lw=1.2, color=INK["muted"], zorder=1)
        ax.set_title(title)
        ax.set_xlabel("history length (symbols)")
    axes[0].annotate(
        "the dashed line is one bit per symbol:\ncounting histories, not structure",
        xy=(0.03, 0.03),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK["muted"],
    )
    axes[0].set_ylabel("$C_\\mu$ (bits)")
    axes[0].legend(loc="upper left")
    note(
        axes[0],
        "Structureless bits must give $C_\\mu = 0$ at every length. On the left "
        "they give up to 5.5 bits.",
    )
    return _save(fig, "02_state_proliferation.png")


def fig_capacity() -> Path:
    """The headline negative result: the HQMM does not earn its complexity.

    Form: a dot plot against free parameters, because the claim is about
    capacity and the x axis has to be capacity. Error bars are the seed
    spread, which is the whole reason the single-fit version misled.

    The y axis is clipped to where the models actually differ. The i.i.d.
    baseline sits far above at 0.984 and would squash everything else into a
    fifth of the panel; it is annotated rather than dropped, because a
    reader is entitled to know what "no model at all" scores.
    """
    import pandas as pd

    use_style()
    path = RESULTS_DIR / "seattle_table.csv"
    if not path.exists():
        raise FileNotFoundError("run qphys.collapse.experiments.seattle_comparison")
    df = pd.read_csv(path)
    df["family"] = np.where(df.model.str.startswith("hqmm"), "quantum", "classical")
    shown = df[df.model != "iid"]
    iid = df[df.model == "iid"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    best = df.loc[df.nll.idxmin()]
    ax.axhline(best.nll, color=INK["muted"], lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.annotate(
        f"best of all: {best.model}, {best.params} parameters",
        xy=(0.015, best.nll),
        xycoords=("axes fraction", "data"),
        xytext=(0, -13),
        textcoords="offset points",
        ha="left",
        fontsize=8.5,
        color=INK["muted"],
    )
    for family, role in (("classical", "classical"), ("quantum", "quantum")):
        sub = shown[shown.family == family]
        ax.errorbar(
            sub.params,
            sub.nll,
            yerr=sub.nll_std,
            fmt="o",
            color=ROLE[role],
            ecolor=ROLE[role],
            elinewidth=1.4,
            capsize=3,
            markeredgecolor="#fcfcfb",
            markeredgewidth=1.2,
            markersize=8,
            label=f"{family} models",
            zorder=3,
        )
    for _, r in shown.iterrows():
        if r.model in ("hmm-2", "hqmm-d2", "hqmm-d3"):
            dy = 13 if r.model == "hmm-2" else -14 if r.model == "hqmm-d2" else 0
            ha = "center" if dy else "left"
            label_at(
                ax,
                r.params,
                r.nll,
                r.model,
                ROLE["quantum" if r.family == "quantum" else "classical"],
                dx=0 if dy else 9,
                dy=dy,
                ha=ha,
            )
    ax.annotate(
        f"i.i.d. baseline, no model: {iid.nll:.3f}  \u2191",
        xy=(0.01, 0.985),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK["muted"],
        va="top",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.841, 0.895)
    # Parameter counts are small integers; scientific tick labels on a log
    # axis make the reader decode "2 x 10^0" to get back to "2".
    ax.set_xticks([2, 4, 8, 16], ["2", "4", "8", "16"])
    ax.minorticks_off()
    ax.set_xlabel("free parameters (log scale)")
    ax.set_ylabel("test NLL (bits/symbol) \u2014 lower is better")
    ax.set_title("At matched capacity, the quantum model loses")
    ax.legend(loc="upper right")
    note(
        ax,
        "Seattle wet/dry, temporal 70/30 split. Bars are the spread over 5 "
        "seeds \u2014 the single-fit version of this chart had a different winner.",
    )
    return _save(fig, "03_capacity.png")


def fig_leggett_garg() -> Path:
    """Why a Leggett-Garg violation in a recorded series proves nothing.

    Form: one line (the quantum curve) against a bound, with measured
    processes as marks on it. The bound is the point, so it is drawn as a
    rule rather than a series.
    """
    from qphys.collapse import leggett_garg as lg
    from qphys.collapse import processes as pr

    use_style()
    wt = np.linspace(0, np.pi, 600)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axhspan(1.0, 1.62, color=ROLE["quantum"], alpha=0.07, linewidth=0)
    ax.plot(wt, lg.k3_quantum(wt), color=ROLE["quantum"], label="ideal quantum $K_3$")
    ax.axhline(1.0, color=INK["secondary"], lw=1.4)
    ax.annotate(
        "macrorealist bound $K_3 \\leq 1$",
        xy=(2.55, 1.0),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=INK["secondary"],
    )
    peak, angle = lg.quantum_maximum()
    ax.plot([angle], [peak], "o", color=ROLE["quantum"], zorder=5)
    label_at(ax, angle, peak, f"  {peak:.2f} at 60°", ROLE["quantum"])

    rng = np.random.default_rng(0)
    series = {
        "i.i.d. bits": pr.iid_bits(20000, rng),
        "period-2": pr.periodic(20000, 2),
        "AR(2) sign": pr.ar2_sign(20000, rng),
        "random walk": pr.random_walk_sign(20000, rng),
    }
    xs = np.linspace(0.35, 2.75, len(series))
    for x, (name, bits) in zip(xs, series.items(), strict=True):
        best = max(lg.k3_aligned(pr.to_pm1(bits), lag) for lag in (1, 2, 3, 5))
        ax.plot([x], [best], "D", color=ROLE["classical"], markersize=7, zorder=4)
        ax.annotate(
            name,
            xy=(x, best),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK["secondary"],
        )
    ax.plot(
        [], [], "D", color=ROLE["classical"], label="recorded series (max over lags)"
    )
    ax.set_xlabel("measurement angle $\\omega t$ (rad)")
    ax.set_ylabel("$K_3$")
    ax.set_ylim(-0.1, 1.62)
    ax.set_title("A recorded series can never violate the bound")
    ax.legend(loc="lower left")
    note(
        ax,
        "For an aligned single-window estimator $K_3 \\leq 1$ is an algebraic "
        "identity, so exceeding it is a windowing bug, not a discovery.",
    )
    return _save(fig, "04_leggett_garg.png")


# --- Project 2 ------------------------------------------------------------


def fig_readout_floor() -> Path:
    """What limits the recovered speed of light: the readout, not the fit.

    Form: measured points against the floor they are being read against,
    with the true value as a rule. The floor is a reference, not a competing
    entity, so it is muted ink rather than a categorical hue.
    """
    import pandas as pd

    from qphys.lawlearn.lawnet import readout_bias

    use_style()
    path = RESULTS_DIR / "c_hat_sweep.csv"
    if not path.exists():
        raise FileNotFoundError("run qphys.lawlearn.experiments.c_hat_sweep")
    d = pd.read_csv(path)
    d = d[(d.noise == 0) & d.converged]
    g = d.groupby("v_max").c_hat.agg(["mean", "std"]).reset_index()
    grid = np.linspace(0.15, 0.75, 120)

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ax.axhline(1.0, color=INK["secondary"], lw=1.4)
    ax.annotate(
        "true $c = 1$",
        xy=(0.16, 1.0),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=8.5,
        color=INK["secondary"],
    )
    ax.plot(
        grid,
        [readout_bias(v) for v in grid],
        color=INK["muted"],
        ls=(0, (4, 3)),
        lw=1.6,
        label="cubic readout floor (exact momentum)",
    )
    ax.plot(
        grid,
        [readout_bias(v, powers=(1, 3, 5)) for v in grid],
        color=ROLE["symbolic"],
        ls=(0, (1, 2)),
        lw=1.6,
        label="quintic readout floor",
    )
    ax.errorbar(
        g.v_max,
        g["mean"],
        yerr=g["std"],
        fmt="o",
        color=ROLE["structured"],
        markersize=8,
        capsize=3,
        elinewidth=1.4,
        markeredgecolor="#fcfcfb",
        markeredgewidth=1.2,
        label="recovered $\\hat{c}$ (5 seeds)",
        zorder=4,
    )
    ax.set_xlabel("fastest speed in the training data, $v_{max}/c$")
    ax.set_ylabel("recovered $\\hat{c}$")
    ax.set_title("The recovered speed limit tracks the readout, not the fit")
    ax.set_ylim(0.6, 1.25)
    ax.legend(loc="lower left")
    note(
        ax,
        "Every point sits within 1.4% of the floor a PERFECT momentum "
        "function would give, so the deficit is the truncation, not the network.",
    )
    return _save(fig, "05_readout_floor.png")


def fig_momentum() -> Path:
    """What the network actually learned, against Newton and against truth.

    Form: three curves on one axis, the two laws plus what was recovered.
    """
    import pandas as pd

    use_style()
    path = RESULTS_DIR / "c_hat_sweep.csv"
    d = pd.read_csv(path)
    row = d[(d.v_max == 0.5) & (d.noise == 0) & d.converged].iloc[0]
    b1, b3 = float(row.b1), float(row.b3)
    if b1 < 0:
        b1, b3 = -b1, -b3

    v = np.linspace(0, 0.95, 300)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axvspan(0, 0.5, color=ROLE["structured"], alpha=0.06, linewidth=0)
    ax.annotate(
        "trained here",
        xy=(0.25, 3.0),
        ha="center",
        fontsize=8.5,
        color=INK["secondary"],
    )
    ax.plot(
        v,
        v / np.sqrt(1 - v**2),
        color=INK["secondary"],
        lw=2.4,
        label="truth  $p = \\gamma v$",
    )
    ax.plot(v, v, color=ROLE["control"], ls=(0, (4, 3)), label="Newton  $p = v$")
    ax.plot(
        v, b1 * v + b3 * v**3, color=ROLE["structured"], label="learned, cubic readout"
    )
    ax.set_xlabel("$v/c$")
    ax.set_ylabel("canonical momentum $p$")
    ax.set_ylim(0, 3.3)
    ax.set_title("Relativity is a statement about $p(v)$, and it is learnable")
    ax.legend(loc="upper left")
    note(
        ax,
        "The network is shown only $(x, v, a)$ -- never $\\gamma$, never a "
        "speed limit. The cubic term of slow motion carries $c$.",
    )
    return _save(fig, "06_momentum.png")


def fig_extrapolation() -> Path:
    """Only the formula extrapolates.

    Form: a dumbbell, not bars. The measure spans three decades so the axis
    has to be logarithmic, and a bar encodes length from a zero baseline that
    a log axis does not have. Two dots joined by a line put the reader on the
    quantity that matters anyway -- how far each arm moves when it leaves the
    data it was trained on.
    """
    import pandas as pd

    use_style()
    path = RESULTS_DIR / "extrapolation.csv"
    if not path.exists():
        raise FileNotFoundError("run qphys.lawlearn.lawnet.extrapolation_benchmark")
    d = pd.read_csv(path)
    d = d[d.converged]
    arms = [
        ("symbolic readout", "in_symbolic", "out_symbolic_relativistic", "symbolic"),
        ("physics net", "in_lawnet", "out_lawnet", "structured"),
        ("black box", "in_blackbox", "out_blackbox", "blackbox"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for i, (_name, cin, cout, role) in enumerate(arms):
        a, b = d[cin].mean(), d[cout].mean()
        ax.plot(
            [a, b],
            [i, i],
            color=ROLE[role],
            lw=2.4,
            alpha=0.45,
            zorder=2,
            solid_capstyle="round",
        )
        ax.plot(
            [a],
            [i],
            "o",
            color=ROLE[role],
            markersize=7,
            zorder=3,
            markerfacecolor="#fcfcfb",
            markeredgewidth=2.2,
            markeredgecolor=ROLE[role],
        )
        ax.plot(
            [b],
            [i],
            "o",
            color=ROLE[role],
            markersize=9,
            zorder=3,
            markeredgecolor="#fcfcfb",
            markeredgewidth=1.6,
        )
        label_at(ax, b, i, f"  {b:.2f}", ROLE[role])
        ax.annotate(
            f"x{b / a:.0f}",
            xy=(np.sqrt(a * b), i),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK["muted"],
        )

    ax.set_yticks(range(len(arms)), [a[0] for a in arms])
    ax.set_xscale("log")
    ax.set_xlim(1.4e-3, 4.0)
    ax.set_ylim(-0.6, len(arms) - 0.25)
    ax.set_xlabel("nRMSE (log scale) — lower is better")
    ax.set_title("Networks interpolate; only the extracted formula extrapolates")
    ax.grid(axis="y", visible=False)
    ax.plot(
        [],
        [],
        "o",
        markerfacecolor="#fcfcfb",
        markeredgecolor=INK["muted"],
        markeredgewidth=2.2,
        markersize=7,
        linestyle="none",
        label="inside $|v| < 0.5c$  (trained)",
    )
    ax.plot(
        [],
        [],
        "o",
        color=INK["muted"],
        markersize=9,
        linestyle="none",
        label="outside, $|v| \\in [0.7c, 0.95c]$",
    )
    ax.legend(loc="lower left", ncols=2)
    note(
        ax,
        "At matched capacity: 8737 parameters against 8705. The "
        "physics-structured network extrapolates WORSE than the plain black box.",
        y=-0.26,
    )
    return _save(fig, "07_extrapolation.png")


ALL = (
    fig_compression_gap,
    fig_state_proliferation,
    fig_capacity,
    fig_leggett_garg,
    fig_readout_floor,
    fig_momentum,
    fig_extrapolation,
)


def build(only: str | None = None) -> None:
    for fn in ALL:
        if only and only not in fn.__name__:
            continue
        fn()


if __name__ == "__main__":
    import sys

    build(sys.argv[1] if len(sys.argv) > 1 else None)
