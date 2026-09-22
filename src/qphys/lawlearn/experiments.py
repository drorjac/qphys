"""The section 10 sweeps, as functions rather than a loose script.

Logic lives in `src/`; a notebook or a shell one-liner calls these. Rows are
written as they finish, so a partial run is still a usable result -- these
take hours and a crash at row six should not cost rows one to five.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qphys.common.seeding import seeds, summarise
from qphys.lawlearn import lawnet as L

REPORTS = Path(__file__).resolve().parents[3] / "reports"

# The identifiability sweep: how slow can the data be before the quartic term
# of p(v) stops being readable, and how much noise can it carry.
IDENTIFIABILITY_ROWS = [
    (0.2, 0.0),
    (0.3, 0.0),
    (0.5, 0.0),
    (0.7, 0.0),
    (0.5, 0.02),
    (0.5, 0.05),
    (0.5, 0.10),
]


def c_hat_sweep(rows=None, n_seeds: int = 5, steps: int = 4000, out=None):
    """Recover c at each (v_max, noise), over at least five seeds.

    Every row reports its FAILURE RATE beside its mean, because a third of
    these fits do not converge and a table that hid that would misdescribe
    the method (honesty rule 6).
    """
    rows = rows or IDENTIFIABILITY_ROWS
    out = Path(out) if out else REPORTS / "c_hat_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    collected: list = []
    for v_max, noise in rows:
        for seed in seeds(n_seeds):
            r = L.fit_and_read(v_max, seed=seed, noise=noise, steps=steps)
            r.pop("net", None)
            collected.append(r)
            pd.DataFrame(collected).to_csv(out, index=False)
            print(
                f"v_max={v_max} noise={noise} seed={seed} "
                f"excess={r['rel_loss_excess']:.2e} conv={r['converged']} "
                f"c_hat={r['c_hat']:.4f}",
                flush=True,
            )
        sub = [
            r["c_hat"] for r in collected if r["v_max"] == v_max and r["noise"] == noise
        ]
        s = summarise(sub, f"v_max={v_max} noise={noise}")
        print(
            f"  -> mean={s['mean']:.4f} std={s['std']:.4f} "
            f"failure_rate={s['failure_rate']:.0%}",
            flush=True,
        )
    return pd.DataFrame(collected)


def readout_comparison(v_maxes=(0.3, 0.5), n_seeds: int = 5, steps: int = 4000):
    """Cubic against quintic readout, on the SAME trained fits.

    Reading the same network two ways is the only way to separate the
    readout's contribution from the network's.
    """
    rows = []
    for v_max in v_maxes:
        for seed in seeds(n_seeds):
            r = L.fit_and_read(v_max, seed=seed, steps=steps)
            r.pop("net", None)
            rows.append(r)
    df = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / "readout_comparison.csv", index=False)
    return df


def summarise_sweep(path=None) -> pd.DataFrame:
    """The sweep table as it belongs in a report, against the readout floor."""
    path = Path(path) if path else REPORTS / "c_hat_sweep.csv"
    d = pd.read_csv(path)
    out = []
    for (v_max, noise), grp in d.groupby(["v_max", "noise"]):
        ok = grp[grp.converged]
        out.append(
            {
                "v_max": v_max,
                "noise": noise,
                "seeds": len(grp),
                "converged": len(ok),
                "failure_rate": 1.0 - len(ok) / len(grp),
                "c_hat": ok.c_hat.mean() if len(ok) else float("nan"),
                "c_hat_sd": ok.c_hat.std() if len(ok) > 1 else float("nan"),
                "readout_floor": L.readout_bias(v_max),
            }
        )
    df = pd.DataFrame(out)
    df["gap_to_floor"] = (df.c_hat / df.readout_floor - 1.0).abs()
    return df
