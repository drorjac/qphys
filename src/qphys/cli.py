"""Command-line interface.

    qphys run collapse     the Seattle comparison: table, per-seed rows, memory
    qphys run lawlearn     the c_hat sweep, its noise rows, the readout
                           comparison, and the extrapolation benchmark
    qphys run all          both (hours)
    qphys figures [NAME]   every figure, or one, from results/
    qphys info             resolved paths, and whether the loaders are offline

Every `run` target writes to results/ and nothing else. The tables there are
what the README and docs cite; this is how they are regenerated.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence

from qphys import __version__, config


def _collapse() -> None:
    from qphys.collapse import experiments as E

    E.seattle_comparison()
    E.seattle_memory()


def _lawlearn() -> None:
    from qphys.lawlearn import experiments as E

    E.c_hat_sweep()
    E.noise_rows()
    E.readout_comparison()
    E.extrapolation()


TARGETS: dict[str, Callable[[], None]] = {
    "collapse": _collapse,
    "lawlearn": _lawlearn,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qphys",
        description="Quantum formalism as a modeling language: "
        "collapse models for time series, and learning the law from trajectories.",
    )
    parser.add_argument("--version", action="version", version=f"qphys {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="re-run the measured results into results/")
    run.add_argument("target", choices=[*TARGETS, "all"])

    fig = sub.add_parser("figures", help="regenerate figures from results/")
    fig.add_argument("name", nargs="?", help="one figure, e.g. capacity")

    sub.add_parser("info", help="resolved paths and switches")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        names = list(TARGETS) if args.target == "all" else [args.target]
        for name in names:
            t0 = time.perf_counter()
            TARGETS[name]()
            print(f"{name}: done in {time.perf_counter() - t0:.0f} s")
        return 0

    if args.command == "figures":
        from qphys.common.figures import build

        build(args.name)
        return 0

    if args.command == "info":
        print(f"qphys {__version__}")
        for label, path in [
            ("root", config.ROOT),
            ("data", config.DATA_DIR),
            ("results", config.RESULTS_DIR),
            ("figures", config.FIGURES_DIR),
        ]:
            print(f"  {label:<8} {path}")
        print(f"  offline  {config.offline()}")
        return 0

    return 1  # unreachable: argparse requires a command


if __name__ == "__main__":
    sys.exit(main())
