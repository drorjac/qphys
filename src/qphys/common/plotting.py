"""One chart style, applied everywhere, with a palette that was validated.

The colour choice is computable, so it is computed rather than eyeballed:
`python -m qphys.common.palette "#2a78d6,#eb6834,#1baf7a,#4a3aa7" light all`
runs the lightness-band, chroma-floor, colour-vision-deficiency separation,
normal-vision and contrast gates. The four categorical hues below pass all of
them at the strictest adjacency setting, with one contrast RELIEF on the
green -- which is why every chart in this repo carries a legend AND direct
labels, so identity is never colour alone.

Rules this module enforces, from the data-visualisation method:

  * categorical hues are assigned in FIXED ORDER and never cycled; a fifth
    series folds into a facet rather than inventing a hue;
  * colour follows the entity, not its rank, so a chart that drops a series
    does not repaint the survivors;
  * text wears ink tokens, never the series colour;
  * grid and axes are recessive; marks are thin;
  * a reference or ground-truth line is NOT a categorical series -- it is
    muted ink, because it is not competing with the data.
"""

from __future__ import annotations

# Categorical identity, in fixed assignment order.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")

# Stable role -> hue map, so the same entity is the same colour in every
# figure in the repo and across both projects.
ROLE = {
    "classical": CATEGORICAL[0],
    "quantum": CATEGORICAL[1],
    "structured": CATEGORICAL[0],
    "blackbox": CATEGORICAL[1],
    "symbolic": CATEGORICAL[2],
    "control": CATEGORICAL[3],
}

SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
INK = {
    "primary": "#1a1a19",
    "secondary": "#52514e",
    "muted": "#8a8985",
    "grid": "#e3e2de",
}


def use_style() -> None:
    """Recessive furniture, thin marks, no chartjunk."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": SURFACE["light"],
            "axes.facecolor": SURFACE["light"],
            "savefig.facecolor": SURFACE["light"],
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK["secondary"],
            "axes.edgecolor": INK["grid"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK["secondary"],
            "text.color": INK["primary"],
        }
    )


def label_at(ax, x, y, text, color, dx=6, dy=0, va="center", ha="left", size=8.5):
    """A direct label beside the mark it names.

    The label is in the series colour only where it IS the identity cue; a
    coloured mark plus ink text is preferred, so this is used sparingly and
    never on every point.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        va=va,
        ha=ha,
        fontsize=size,
    )


def note(ax, text, y=-0.22):
    """One line of secondary ink under a panel: what the reader should take."""
    ax.annotate(
        text,
        xy=(0, y),
        xycoords="axes fraction",
        color=INK["secondary"],
        fontsize=8.5,
        va="top",
    )


def ring(artists, width=2.0):
    """A 2px surface ring on overlapping marks, so they stay separable."""
    import matplotlib.patheffects as pe

    for a in artists if isinstance(artists, list | tuple) else [artists]:
        a.set_path_effects(
            [pe.withStroke(linewidth=width + 2, foreground=SURFACE["light"])]
        )
