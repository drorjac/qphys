"""A single plot style, so every figure in the reports matches."""

from __future__ import annotations

PALETTE = {
    "classical": "#2a78d6",
    "quantum": "#eb6834",
    "baseline": "#6b6a67",
    "truth": "#1baf7a",
    "warn": "#4a3aa7",
}


def use_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
