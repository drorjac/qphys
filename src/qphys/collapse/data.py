"""Real series, cached once, with an offline fallback for every loader.

The environment this project was first verified in could reach only PyPI and
raw.githubusercontent.com -- SILSO, USGS, Stooq, ANU QRNG and JPL Horizons
all failed. So **the full pipeline must run with no network**: each loader
downloads once into `data/raw/`, records the byte count and SHA-256, and
falls back to a bundled or generated series when the network is absent.

A fallback is never silently substituted for real data: `Series.is_real` says
which one you got, and the reports print it.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from qphys.config import RAW_DIR, offline

RAW = RAW_DIR
TIMEOUT = 20

SOURCES = {
    "seattle-weather": (
        "https://raw.githubusercontent.com/vega/vega-datasets/main/data/"
        "seattle-weather.csv"
    ),
    "monthly-sunspots": (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "monthly-sunspots.csv"
    ),
    "daily-min-temperatures": (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "daily-min-temperatures.csv"
    ),
}


@dataclass
class Series:
    """A loaded series and the provenance that makes it citable."""

    name: str
    values: np.ndarray
    is_real: bool
    provenance: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.values)


def _cache_path(name: str) -> Path:
    return RAW / f"{name}.csv"


def _meta_path(name: str) -> Path:
    return RAW / f"{name}.meta.json"


def fetch(name: str, force: bool = False) -> Path | None:
    """Download once into data/raw/, recording bytes and SHA-256.

    Returns None when the network is unavailable, so callers fall back rather
    than raising: the pipeline must complete offline.
    """
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; known: {sorted(SOURCES)}")
    path = _cache_path(name)
    if path.exists() and not force:
        return path
    # QPHYS_OFFLINE=1 forces the fallback path even where a network exists,
    # so CI exercises the same code an adopter behind a firewall will hit.
    if offline():
        return None
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(SOURCES[name], timeout=TIMEOUT) as r:
            if r.status != 200:
                return None
            blob = r.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    path.write_bytes(blob)
    _meta_path(name).write_text(
        json.dumps(
            {
                "url": SOURCES[name],
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            },
            indent=2,
        )
    )
    return path


def provenance(name: str) -> dict:
    p = _meta_path(name)
    return json.loads(p.read_text()) if p.exists() else {}


def _synthetic_wet_dry(n: int = 1461, seed: int = 0) -> np.ndarray:
    """The offline stand-in: a two-state chain with Seattle-like persistence.

    Deliberately NOT a copy of the real series. It exists so the pipeline
    runs end to end without a network, and every report that uses it is
    marked `is_real=False`.
    """
    rng = np.random.default_rng(seed)
    p = np.array([[0.7563, 0.2437], [0.3274, 0.6726]])
    out = np.empty(n, dtype=int)
    s = 0
    for i in range(n):
        out[i] = s
        s = int(rng.random() > p[s, 0])
    return out


def seattle_wet_dry() -> Series:
    """Seattle daily weather 2012-2015, symbolised wet/dry at precipitation > 0.

    1461 daily records. Chosen because it is a real precipitation sequence --
    the same wet/dry structure that matters for rainfall sensing -- and small
    enough to iterate on.
    """
    path = fetch("seattle-weather")
    if path is None:
        return Series(
            name="seattle-weather",
            values=_synthetic_wet_dry(),
            is_real=False,
            provenance={"note": "network unavailable; synthetic fallback"},
        )
    df = pd.read_csv(path)
    bits = (df["precipitation"].to_numpy(float) > 0.0).astype(int)
    return Series(
        name="seattle-weather",
        values=bits,
        is_real=True,
        provenance=provenance("seattle-weather") | {"n": len(bits)},
    )


def load_csv_series(name: str, column: str) -> Series:
    path = fetch(name)
    if path is None:
        return Series(name, _synthetic_wet_dry(), False, {"note": "offline fallback"})
    df = pd.read_csv(path)
    return Series(name, df[column].to_numpy(float), True, provenance(name))


def temporal_split(values, train_frac: float = 0.7):
    """A temporal split, declared before the first fit (honesty rule 4).

    Not a random split: the whole question is prediction forward in time, and
    a shuffled split would leak the future into the training set.
    """
    a = np.asarray(values)
    cut = int(train_frac * len(a))  # truncate, so 1461 -> 1022 / 439
    return a[:cut], a[cut:]


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def read_bytes_as_frame(blob: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(blob))
