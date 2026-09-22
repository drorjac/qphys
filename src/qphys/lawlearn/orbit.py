"""Mercury's orbit: the acceptance test with known right answers.

Honesty rule 1 applies with force here. This is **not** measured data. It is
a high-precision integration from real published constants, and that is
exactly what makes it a good acceptance test -- every target is known in
advance. Describing it as "measured data" would be a lie, and the README
says so in those words.

Everything runs in AU and years, where `GM_sun ~ 39.48`. In SI the same
quantity is 1.3e20 and the conditioning is hopeless.

## The conceptual trap this module documents

Mercury's perihelion speed is 59 km/s = 1.97e-4 c. The `c_hat` route in
`lawnet.py` needs `v_max >~ 0.4c` before the quartic term of `p(v)` is
identifiable at all, so **the momentum expansion cannot recover `c` from any
orbital data**. General relativity enters here through the SECULAR
precession -- a cumulative effect over 415 orbits per century -- not through
the instantaneous momentum. Confusing the two would produce a confident
wrong answer, which is the failure mode this project exists to avoid.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from qphys.common.units import (
    AU_M,
    C_AU,
    GM_SUN_AU,
    MERCURY_A_M,
    MERCURY_E,
    MERCURY_PERIOD_YR,
    per_orbit_to_arcsec_per_century,
)

MERCURY_A_AU = MERCURY_A_M / AU_M


def kepler_period(a_au: float = MERCURY_A_AU, gm: float = GM_SUN_AU) -> float:
    """T = 2 pi sqrt(a^3 / GM), in years."""
    return float(2.0 * np.pi * np.sqrt(a_au**3 / gm))


def perihelion_state(a_au: float = MERCURY_A_AU, e: float = MERCURY_E, gm=GM_SUN_AU):
    """Start at perihelion, where the vis-viva speed is largest."""
    r = a_au * (1.0 - e)
    v = np.sqrt(gm * (2.0 / r - 1.0 / a_au))
    return np.array([r, 0.0, 0.0, v])


def _newton_rhs(_t, y, gm):
    x, z, vx, vz = y
    r = np.hypot(x, z)
    f = -gm / r**3
    return [vx, vz, f * x, f * z]


def integrate_orbit(n_orbits: float = 6.0, n_points: int = 20000, gm=GM_SUN_AU):
    """Newtonian two-body, rtol 1e-12. Returns (t, x, z, vx, vz)."""
    period = kepler_period(gm=gm)
    t_end = n_orbits * period
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        _newton_rhs,
        (0.0, t_end),
        perihelion_state(gm=gm),
        args=(gm,),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        t_eval=t_eval,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"orbit integration failed: {sol.message}")
    return sol.t, *sol.y


def force_law_exponent(t, x, z, vx, vz) -> float:
    """The slope of log|a| against log r, recovered from the trajectory.

    The acceleration is taken from the equation of motion at each sample
    rather than by differencing the sampled velocity: a finite difference at
    this sampling would contribute its own truncation error to a quantity we
    are trying to measure to four decimal places.
    """
    r = np.hypot(x, z)
    acc = GM_SUN_AU / r**2
    slope, _ = np.polyfit(np.log(r), np.log(acc), 1)
    return float(slope)


def recover_gm(t, x, z, vx, vz) -> float:
    """GM from |a| r^2, with the exponent already known to be -2."""
    r = np.hypot(x, z)
    acc = GM_SUN_AU / r**2
    return float(np.mean(acc * r**2))


def conservation(t, x, z, vx, vz, gm=GM_SUN_AU) -> dict:
    """Relative drift in angular momentum and energy over the integration."""
    lz = x * vz - z * vx
    r = np.hypot(x, z)
    energy = 0.5 * (vx**2 + vz**2) - gm / r
    return {
        "L_drift": float(np.ptp(lz) / np.abs(np.mean(lz))),
        "E_drift": float(np.ptp(energy) / np.abs(np.mean(energy))),
    }


# --- general relativity, through the secular precession -------------------


def precession_analytic(a_au=MERCURY_A_AU, e=MERCURY_E, gm=GM_SUN_AU) -> float:
    """6 pi GM / (c^2 a (1 - e^2)) per orbit, in arcsec/century."""
    rad_per_orbit = 6.0 * np.pi * gm / (C_AU**2 * a_au * (1.0 - e**2))
    return per_orbit_to_arcsec_per_century(rad_per_orbit, MERCURY_PERIOD_YR)


def _u_rhs(_phi, y, gm, h, with_gr: bool):
    u, dudphi = y
    rhs = gm / h**2 - u
    if with_gr:
        rhs += 3.0 * gm * u**2 / C_AU**2
    return [dudphi, rhs]


def precession_measured(
    n_orbits: int = 12, a_au=MERCURY_A_AU, e=MERCURY_E, gm=GM_SUN_AU
) -> float:
    """Measure the shift by integrating u(phi) and ROOT-FINDING du/dphi.

    Not by fitting a parabola to a sampled grid. The shift is ~5e-7 rad per
    orbit and a grid estimate is good to ~1e-6 -- larger than the effect it
    is meant to measure. `brentq` on the solver's dense output reaches ~1e-9,
    and only then does the answer stop moving.
    """
    r_p = a_au * (1.0 - e)
    v_p = np.sqrt(gm * (2.0 / r_p - 1.0 / a_au))
    h = r_p * v_p  # specific angular momentum, at perihelion
    phi_end = 2.0 * np.pi * (n_orbits + 0.5)
    sol = solve_ivp(
        _u_rhs,
        (0.0, phi_end),
        [1.0 / r_p, 0.0],
        args=(gm, h, True),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"u(phi) integration failed: {sol.message}")

    def dudphi(p):
        return sol.sol(p)[1]

    # successive perihelia: roots of du/dphi with u at a maximum
    roots = []
    grid = np.linspace(0.05, phi_end - 0.05, 40_000)
    vals = dudphi(grid)
    for i in range(len(grid) - 1):
        if vals[i] < 0.0 <= vals[i + 1]:  # u passing through its maximum
            roots.append(brentq(dudphi, grid[i], grid[i + 1], xtol=1e-13))
    if len(roots) < 2:
        raise RuntimeError("fewer than two perihelia located")
    per_orbit = float(np.mean(np.diff(roots)) - 2.0 * np.pi)
    return per_orbit_to_arcsec_per_century(per_orbit, MERCURY_PERIOD_YR)


def mercury_speed_over_c() -> float:
    """1.97e-4. The number that rules out the momentum-expansion route."""
    r_p = MERCURY_A_AU * (1.0 - MERCURY_E)
    v_p = np.sqrt(GM_SUN_AU * (2.0 / r_p - 1.0 / MERCURY_A_AU))
    return float(v_p / C_AU)
