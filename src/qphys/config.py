"""Where things are read from and written to.

Every path is resolved here, once, and each can be overridden by an
environment variable, so the package behaves the same from a checkout, from
an installed wheel, or in CI with a scratch directory:

    QPHYS_DATA_DIR      downloads and processed data   (default <root>/data)
    QPHYS_RESULTS_DIR   generated tables               (default <root>/results)
    QPHYS_FIGURES_DIR   generated figures              (default <root>/figures)
    QPHYS_OFFLINE       "1" forces the fallback loaders even with a network

`<root>` is the repository root when running from a checkout, and the current
working directory otherwise; an installed copy must not write inside
site-packages. The variables are read at import time.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    # <root>/src/qphys/config.py -> <root>
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


def _from_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


ROOT = _repo_root()
DATA_DIR = _from_env("QPHYS_DATA_DIR", ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = _from_env("QPHYS_RESULTS_DIR", ROOT / "results")
FIGURES_DIR = _from_env("QPHYS_FIGURES_DIR", ROOT / "figures")


def offline() -> bool:
    """Read on every call, so a test can set it after import."""
    return os.environ.get("QPHYS_OFFLINE") == "1"
