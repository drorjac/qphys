"""The command line, and the rule that every table has code behind it.

A table in `results/` that no function writes cannot be regenerated, and a
number that cannot be regenerated is not a result. Three tables were once
committed that way; this file stops it happening again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qphys import __version__, cli, config

SRC = Path(cli.__file__).resolve().parent


def test_version_is_the_installed_one():
    assert __version__ != "0+unknown"


def test_info_prints_the_resolved_paths(capsys):
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert str(config.RESULTS_DIR) in out


def test_run_rejects_an_unknown_target():
    with pytest.raises(SystemExit):
        cli.main(["run", "nonsense"])


@pytest.mark.parametrize(
    "table",
    sorted(p.name for p in config.RESULTS_DIR.glob("*") if p.is_file()),
)
def test_every_result_is_written_by_code(table):
    code = "\n".join(p.read_text() for p in SRC.rglob("*.py"))
    assert f'"{table}"' in code, (
        f"results/{table} is committed but no module under src/qphys writes it"
    )
