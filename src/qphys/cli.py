"""Command-line interface.

    qphys run collapse     the Seattle comparison: table, per-seed rows, memory
    qphys run lawlearn     the c_hat sweep, its noise rows, the readout
                           comparison, and the extrapolation benchmark
    qphys run all          both (hours)
    qphys figures [NAME]   every figure, or one, from results/
    qphys verify CANDIDATE compare a regenerated results directory with results/
    qphys info             resolved paths, and whether the loaders are offline

Every `run` target writes to results/ and nothing else. The tables there are
what the README and docs cite; this is how they are regenerated.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

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


# The packages whose version can move a number in results/.
PACKAGES = ("numpy", "scipy", "pandas", "torch", "hmmlearn", "scikit-learn")


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def environment() -> dict:
    """What produced a run: code, interpreter, libraries, and real or fallback data."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "qphys": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in PACKAGES},
        "offline": config.offline(),
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _record_environment() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "environment.json"
    path.write_text(json.dumps(environment(), indent=2, sort_keys=True) + "\n")


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

    ver = sub.add_parser("verify", help="compare regenerated results with results/")
    ver.add_argument("candidate", help="the regenerated results directory")
    ver.add_argument("--reference", default=None, help="default: results/")

    sub.add_parser("info", help="resolved paths and switches")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        names = list(TARGETS) if args.target == "all" else [args.target]
        _record_environment()
        for name in names:
            t0 = time.perf_counter()
            TARGETS[name]()
            print(f"{name}: done in {time.perf_counter() - t0:.0f} s")
        return 0

    if args.command == "figures":
        from qphys.common.figures import build

        build(args.name)
        return 0

    if args.command == "verify":
        from qphys import verify

        reference = Path(args.reference) if args.reference else config.RESULTS_DIR
        candidate = Path(args.candidate)
        for d in (reference, candidate):
            if not d.is_dir():
                print(f"not a directory: {d}", file=sys.stderr)
                return 2
        results = verify.compare_dirs(reference, candidate)
        for r in results:
            if r.status != "identical":
                print(f"  {r.status:16s} {r.path}  {r.detail}".rstrip())
        print(verify.summary(results))
        return 1 if any(r.failed for r in results) else 0

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
