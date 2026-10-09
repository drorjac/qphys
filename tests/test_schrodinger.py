"""The schrodinger sub-project: each check the docs cite, as a test."""

from __future__ import annotations

import numpy as np
import pytest

from qphys.schrodinger import born as B
from qphys.schrodinger import discover as D
from qphys.schrodinger import evolve as Ev
from qphys.schrodinger import grid as G
from qphys.schrodinger import stationary as S
from qphys.schrodinger import symbolic as Y
from qphys.schrodinger import tunneling as T


def test_harmonic_levels_are_n_plus_half():
    g = G.Grid(-10, 10, 1000)
    E, phi = S.eigenstates(g, G.harmonic(g.x), 6)
    assert np.allclose(E, S.harmonic_energy(np.arange(6)), rtol=1e-6)
    assert np.allclose(phi[2], S.harmonic_state(g.x, 2), atol=1e-5)


@pytest.mark.parametrize(("order", "expected"), [(2, 2.0), (4, 4.0)])
def test_box_converges_at_the_stencil_order(order, expected):
    ref = S.box_energy(np.arange(1, 4), 2.0)
    errs, dxs = [], []
    for n in (100, 200, 400):
        g = G.box_grid(2.0, n)
        E, _ = S.eigenstates(g, np.zeros(n), 3, order=order)
        errs.append(np.abs(E / ref - 1).max())
        dxs.append(g.dx)
    slope = np.polyfit(np.log(dxs), np.log(errs), 1)[0]
    assert slope == pytest.approx(expected, abs=0.05)


def test_split_operator_is_unitary_and_second_order():
    g = G.Grid(-12, 12, 256)
    V = G.harmonic(g.x)
    psi0 = Ev.gaussian_packet(g, 3.0, 0.0, 1.0)
    E, vecs = np.linalg.eigh(G.spectral_hamiltonian(g, V))
    states = (vecs / np.sqrt(g.dx)).T
    errs = []
    for dt in (0.1, 0.05):
        ev = Ev.split_operator(
            g, V, psi0, dt, round(np.pi / dt), every=round(np.pi / dt)
        )
        assert abs(Ev.norm(g, ev.psi[-1]) - 1) < 1e-12
        exact = Ev.eigen_expansion(g, E, states, psi0, ev.t[-1:])[0]
        errs.append(np.sqrt((np.abs(ev.psi[-1] - exact) ** 2).sum() * g.dx))
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.1)


def test_gaussian_packet_sits_on_the_heisenberg_bound():
    g = G.Grid(-30, 30, 2048)
    psi = Ev.gaussian_packet(g, 1.0, 2.0, 1.5)
    assert Ev.expect_x(g, psi) == pytest.approx(1.0, abs=1e-8)
    assert Ev.expect_p(g, psi) == pytest.approx(2.0, abs=1e-8)
    assert Ev.spread_x(g, psi) * Ev.spread_p(g, psi) == pytest.approx(0.5, rel=1e-6)


def test_born_sampling_converges_to_psi_squared():
    g = G.Grid(-10, 10, 1024)
    psi = Ev.gaussian_packet(g, -2, 0, 0.7) + Ev.gaussian_packet(g, 2, 0, 0.7)
    psi = g.normalise(psi)
    tv = [
        B.total_variation(g, psi, B.sample_positions(g, psi, n, seed=0))
        for n in (1000, 100_000)
    ]
    assert tv[1] < tv[0] / 5


def test_plane_wave_transmission_is_continuous_at_the_barrier_top():
    below, at, above = T.transmission_plane_wave([1 - 1e-6, 1.0, 1 + 1e-6], 1.0, 1.0)
    assert below == pytest.approx(at, rel=1e-4)
    assert above == pytest.approx(at, rel=1e-4)


def test_barrier_is_cell_averaged_to_its_nominal_width():
    g = G.Grid(-160, 160, 4096)
    V = G.barrier(g.x, 1.0, 1.0)
    assert V.sum() * g.dx == pytest.approx(1.0, rel=1e-12)


@pytest.mark.slow
def test_tunneling_matches_the_packet_average():
    r = T.tunnel(1.0)
    assert abs(r["T_sim"] - r["T_packet"]) < 1e-3


def test_discovery_recovers_the_equation_exactly_without_noise():
    r = D.discover(noise=0.0)
    assert set(r["terms"]) == set(D.TRUTH)
    for name, value in D.TRUTH.items():
        assert r["terms"][name] == pytest.approx(value, abs=1e-3)


def test_discovery_survives_one_percent_noise():
    r = D.discover(noise=0.01, seed=1)
    assert set(r["terms"]) == set(D.TRUTH)
    assert r["terms"]["psi_xx"].real == pytest.approx(-0.5, abs=5e-3)


def test_density_determines_the_potential():
    r = Y.potential(backend="fallback")
    assert r["expression"] == "+0.2994 x^0 -0.9998 x^2 +0.2499 x^4"


def test_box_law_power_fit():
    r = Y.box_law(backend="fallback")
    assert r["expression"] == "4.9348 * n^2.0000 * L^-2.0000"
