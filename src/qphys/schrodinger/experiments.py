"""Every measured number in docs/schrodinger, as functions that write results/.

    spectrum       numerical eigenvalues against closed forms
    convergence    eigenvalue error against grid spacing, both stencils
    conservation   norm, energy and accuracy of the integrator against dt
    born           histogram of detections against |psi|^2, over sample size
    tunneling      packet transmission against the plane-wave formula
    discovery      the equation recovered from psi(x, t), over noise and seeds
    symbolic       closed forms from symbolic regression

Each one runs in seconds to a few minutes, except `symbolic` with PySR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qphys.common.seeding import seeds
from qphys.config import RESULTS_DIR
from qphys.schrodinger import born as B
from qphys.schrodinger import discover as D
from qphys.schrodinger import evolve as Ev
from qphys.schrodinger import grid as G
from qphys.schrodinger import stationary as S
from qphys.schrodinger import symbolic as Y
from qphys.schrodinger import tunneling as T


def _write(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = Path(RESULTS_DIR) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    return df


def spectrum(k: int = 8) -> pd.DataFrame:
    rows = []
    g = G.Grid(-10.0, 10.0, 1000)
    E, _ = S.eigenstates(g, G.harmonic(g.x), k)
    ref = S.harmonic_energy(np.arange(k))
    rows += [
        ("harmonic", n, e, r, "exact")
        for n, (e, r) in enumerate(zip(E, ref, strict=True))
    ]

    gb = G.box_grid(2.0, 400)
    E, _ = S.eigenstates(gb, np.zeros(gb.n), k)
    ref = S.box_energy(np.arange(1, k + 1), 2.0)
    rows += [
        ("box", n + 1, e, r, "exact")
        for n, (e, r) in enumerate(zip(E, ref, strict=True))
    ]

    gq = G.Grid(-12.0, 12.0, 3000)
    E, _ = S.eigenstates(gq, G.quartic(gq.x), k)
    ref = S.quartic_wkb_energy(np.arange(k))
    rows += [
        ("quartic", n, e, r, "wkb") for n, (e, r) in enumerate(zip(E, ref, strict=True))
    ]

    gd = G.Grid(-8.0, 8.0, 1600)
    E, _ = S.eigenstates(gd, G.double_well(gd.x), k)
    rows += [("double_well", n, e, np.nan, "none") for n, e in enumerate(E)]

    df = pd.DataFrame(
        rows, columns=["potential", "n", "E_numeric", "E_reference", "reference"]
    )
    df["rel_err"] = (df.E_numeric - df.E_reference) / df.E_reference
    return _write(df, "schrodinger_spectrum.csv")


def convergence() -> pd.DataFrame:
    rows = []
    ref = S.box_energy(np.arange(1, 5), 2.0)
    for order in (2, 4):
        for n in (50, 100, 200, 400, 800):
            g = G.box_grid(2.0, n)
            E, _ = S.eigenstates(g, np.zeros(n), 4, order=order)
            rows.append(
                {
                    "order": order,
                    "points": n,
                    "dx": g.dx,
                    "max_rel_err": float(np.abs(E / ref - 1).max()),
                }
            )
    df = pd.DataFrame(rows)
    for order, sub in df.groupby("order"):
        slope = np.polyfit(np.log(sub.dx), np.log(sub.max_rel_err), 1)[0]
        df.loc[sub.index, "fitted_order"] = slope
        print(f"stencil {order}: error ~ dx^{slope:.2f}")
    return _write(df, "schrodinger_convergence.csv")


def conservation(t_end: float = 2 * np.pi) -> pd.DataFrame:
    """A displaced packet in the harmonic well for one classical period."""
    g = G.Grid(-12.0, 12.0, 512)
    V = G.harmonic(g.x)
    psi0 = Ev.gaussian_packet(g, x0=3.0, k0=0.0, sigma=1.0)
    E, vecs = np.linalg.eigh(G.spectral_hamiltonian(g, V))
    states = (vecs / np.sqrt(g.dx)).T
    rows = []
    for dt in (0.2, 0.1, 0.05, 0.02, 0.01):
        steps = round(t_end / dt)
        ev = Ev.split_operator(g, V, psi0, dt, steps)
        exact = Ev.eigen_expansion(g, E, states, psi0, ev.t)
        err = np.sqrt((np.abs(ev.psi - exact) ** 2).sum(axis=1) * g.dx).max()
        norms = np.array([Ev.norm(g, f) for f in ev.psi])
        energies = np.array([Ev.energy(g, V, f) for f in ev.psi])
        rows.append(
            {
                "dt": dt,
                "steps": steps,
                "max_norm_drift": float(np.abs(norms - 1).max()),
                "max_energy_drift": float(
                    np.abs(energies - energies[0]).max() / energies[0]
                ),
                "max_state_error": float(err),
                "ehrenfest_residual": Ev.ehrenfest_residual(ev),
                "x_after_period": Ev.expect_x(g, ev.psi[-1]),
            }
        )
    return _write(pd.DataFrame(rows), "schrodinger_conservation.csv")


def interference_state():
    """Two packets moving toward each other, at the moment they overlap.

    The density shows fringes with spacing pi / k0 that neither packet has
    alone. It is the state used to test the Born rule, because a sampler that
    got |psi|^2 slightly wrong would wash the fringes out.
    """
    g = G.Grid(-30.0, 30.0, 2048)
    k0 = 3.0
    psi0 = Ev.gaussian_packet(g, -8.0, k0, 1.5) + Ev.gaussian_packet(g, 8.0, -k0, 1.5)
    psi0 = g.normalise(psi0)
    t = 8.0 / k0
    ev = Ev.split_operator(g, np.zeros(g.n), psi0, t / 400, 400, every=400)
    return g, ev.psi[-1]


def born(sizes=(100, 300, 1000, 3000, 10_000, 30_000, 100_000)) -> pd.DataFrame:
    g, psi = interference_state()
    rows = []
    for n in sizes:
        for s in seeds():
            x = B.sample_positions(g, psi, n, seed=s)
            rows.append(
                {"n_samples": n, "seed": s, "tv_distance": B.total_variation(g, psi, x)}
            )
    df = pd.DataFrame(rows)
    means = df.groupby("n_samples").tv_distance.mean()
    slope = np.polyfit(np.log(means.index), np.log(means.values), 1)[0]
    df["fitted_exponent"] = slope
    print(f"TV distance ~ n^{slope:.3f}")
    return _write(df, "schrodinger_born.csv")


def tunneling(k0s=None) -> pd.DataFrame:
    k0s = np.round(np.linspace(0.6, 2.6, 21), 3) if k0s is None else k0s
    df = pd.DataFrame([T.tunnel(float(k)) for k in k0s])
    df["abs_err_packet"] = (df.T_sim - df.T_packet).abs()
    df["abs_err_plane_wave"] = (df.T_sim - df.T_plane_wave).abs()
    return _write(df, "schrodinger_tunneling.csv")


def discovery(noises=(0.0, 0.005, 0.01, 0.02, 0.05)) -> pd.DataFrame:
    rows = []
    for noise in noises:
        for s in seeds() if noise > 0 else (0,):
            r = D.discover(noise=noise, seed=s)
            row = {
                "noise": noise,
                "seed": s,
                "k_cut": r["k_cut"],
                "rel_residual": r["rel_residual"],
                "n_terms": len(r["terms"]),
                "exact_support": set(r["terms"]) == set(D.TRUTH),
            }
            for name in D.TERMS:
                c = r["terms"].get(name, 0j)
                row[f"re[{name}]"] = c.real
                row[f"im[{name}]"] = c.imag
            rows.append(row)
    return _write(pd.DataFrame(rows), "schrodinger_discovery.csv")


def symbolic(backends=("fallback", "pysr")) -> pd.DataFrame:
    rows = []
    for backend in backends:
        if backend == "pysr" and not Y.has_pysr():
            print("PySR is not installed; skipping the pysr rows")
            continue
        for task in Y.TASKS:
            r = task(backend=backend)
            print(r)
            rows.append(r)
    return _write(pd.DataFrame(rows), "schrodinger_symbolic.csv")


ALL = (spectrum, convergence, conservation, born, tunneling, discovery, symbolic)


def run_all() -> None:
    for fn in ALL:
        fn()
