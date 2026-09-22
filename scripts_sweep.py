"""The section 10 identifiability sweep, at >= 5 seeds (honesty rule 6).

Rows are appended as they finish, so a partial run is still a usable result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from qphys.common.seeding import seeds, summarise  # noqa: E402
from qphys.lawlearn import lawnet as L  # noqa: E402

OUT = Path(__file__).parent / "reports" / "c_hat_sweep.csv"
ROWS = [
    (0.2, 0.0), (0.3, 0.0), (0.5, 0.0), (0.7, 0.0),
    (0.5, 0.02), (0.5, 0.05), (0.5, 0.10),
]

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for v_max, noise in ROWS:
        for seed in seeds(5):
            r = L.fit_and_read(v_max, seed=seed, noise=noise, steps=4000)
            r.pop("net")
            out.append(r)
            pd.DataFrame(out).to_csv(OUT, index=False)
            print(
                f"v_max={v_max} noise={noise} seed={seed} "
                f"rel_loss={r['rel_loss']:.2e} conv={r['converged']} "
                f"c_hat={r['c_hat']:.4f}",
                flush=True,
            )
        sub = [r["c_hat"] for r in out if r["v_max"] == v_max and r["noise"] == noise]
        s = summarise(sub, f"v_max={v_max} noise={noise}")
        print(
            f"  -> mean={s['mean']:.4f} std={s['std']:.4f} "
            f"failure_rate={s['failure_rate']:.0%}",
            flush=True,
        )

if __name__ == "__main__":
    main()
