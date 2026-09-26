"""The command line, and the rule that every table has code behind it.

A table in `results/` that no function writes cannot be regenerated. Three
tables were once committed that way; `test_every_result_is_written_by_code`
prevents it.
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


# --- verify: the reproduction contract ---------------------------------------


def test_verify_accepts_a_copy_and_catches_a_moved_number(tmp_path, capsys):
    import shutil

    import pandas as pd

    from qphys import verify

    ref, cand = tmp_path / "ref", tmp_path / "cand"
    ref.mkdir()
    pd.DataFrame({"seed": [0, 1], "c_hat": [0.8813, 0.8802]}).to_csv(
        ref / "sweep.csv", index=False
    )
    shutil.copytree(ref, cand)
    (cand / "environment.json").write_text("{}")  # expected to differ; ignored
    assert cli.main(["verify", str(cand), "--reference", str(ref)]) == 0

    pd.DataFrame({"seed": [0, 1], "c_hat": [0.8813, 0.8802 * (1 + 1e-6)]}).to_csv(
        cand / "sweep.csv", index=False
    )
    results = verify.compare_dirs(ref, cand)
    assert [r.status for r in results] == ["differs"]
    assert cli.main(["verify", str(cand), "--reference", str(ref)]) == 1
    assert "c_hat[1]" in capsys.readouterr().out


def test_the_environment_record_is_serialisable():
    import json

    env = cli.environment()
    assert env["packages"]["numpy"]
    json.dumps(env)
