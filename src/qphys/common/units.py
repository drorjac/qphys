"""Natural units inside, physical units only at the reporting boundary.

`c = m = hbar = 1` internally. This is not cosmetic: in SI, `c^2 ~ 9e16`, and
a loss surface carrying that factor is unusable -- gradients in the quartic
term of `p(v)` are 17 orders of magnitude from those in the linear term.

Orbital work uses AU and years, where `GM_sun ~ 39.48` rather than `1.3e20`,
for the same reason.
"""

from __future__ import annotations

import numpy as np

# --- exact or published constants, each with its source --------------------
C_SI = 299_792_458.0  # m/s, exact by definition (SI, 2019 redefinition)
GM_SUN_SI = 1.32712440018e20  # m^3/s^2, IAU 2009/2015 nominal
AU_M = 1.495978707e11  # m, IAU 2012 exact
YEAR_S = 365.25 * 86400.0  # Julian year, s
ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0

# Mercury, published elements (IAU / JPL)
MERCURY_A_M = 5.7909050e10  # m
MERCURY_E = 0.205630
MERCURY_PERIOD_YR = 0.2408467

# GM_sun in AU^3 / yr^2 -- the working value for the orbit track.
GM_SUN_AU = GM_SUN_SI * YEAR_S**2 / AU_M**3
C_AU = C_SI * YEAR_S / AU_M  # speed of light in AU/yr


def m_to_au(x):
    return np.asarray(x, float) / AU_M


def au_to_m(x):
    return np.asarray(x, float) * AU_M


def rad_to_arcsec(x):
    return np.asarray(x, float) * ARCSEC_PER_RAD


def per_orbit_to_arcsec_per_century(rad_per_orbit: float, period_yr: float) -> float:
    """A secular rate: the per-orbit shift times the orbits in a century."""
    return rad_to_arcsec(rad_per_orbit) * (100.0 / period_yr)
