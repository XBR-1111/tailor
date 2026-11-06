"""Smoke tests: package imports and CLI contract hold before any real feature exists."""

from pathlib import Path

from typer.testing import CliRunner

import tailor
from tailor.cli import app

runner = CliRunner()


def _pyproject_version() -> str:
    import tomllib
    from pathlib import Path as _P

    data = tomllib.loads((_P(__file__).parent.parent / "pyproject.toml").read_text())
    return str(data["project"]["version"])


def test_version_exports() -> None:
    """__version__ used to be a stale hardcoded string (0.5.0 through
    v0.9.0 releases). It must equal pyproject.toml, the release truth."""
    assert tailor.__version__ == _pyproject_version()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert tailor.__version__ in result.stdout


def test_cli_check_exit_code_is_error_while_unimplemented() -> None:
    # exit 2 keeps scripts and CI from mistaking a stub for a passing check
    result = runner.invoke(app, ["check", "/nonexistent"])
    assert result.exit_code == 2


def test_version_command() -> None:
    from typer.testing import CliRunner

    from tailor import __version__
    from tailor.cli import app

    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_provenance_empty_ledger(tmp_path) -> None:
    from typer.testing import CliRunner

    from tailor.cli import app
    from test_expectations import make_fixture

    ds = make_fixture(tmp_path)
    result = CliRunner().invoke(app, ["provenance", str(ds)])
    assert result.exit_code == 0
    assert "has not been filtered" in result.stdout


def test_fail_on_skip_exit_code(tmp_path: Path) -> None:
    """--fail-on-skip turns a SKIP-laden green run into exit 3: a CI gate
    over a meta-only checkout is not a full pass."""
    from typer.testing import CliRunner

    from tailor.cli import app
    from test_expectations import make_fixture

    ds = make_fixture(tmp_path)
    runner = CliRunner()
    # clean dataset still SKIPs deep checks (video_content default-off) —
    # enough SKIPs that --fail-on-skip must flag the incomplete verdict
    r = runner.invoke(app, ["check", str(ds), "--fail-on-skip"])
    assert r.exit_code == 3, (r.exit_code, r.output)
    r2 = runner.invoke(app, ["check", str(ds)])
    assert r2.exit_code == 0  # without the flag a no-FAIL run stays green


def test_json_stdout_stays_pure_with_markdown(tmp_path: Path) -> None:
    """--json --markdown together: stdout must parse with plain json.load.
    The 'markdown report:' line used to follow the JSON on stdout, and every
    downstream json.load / jq in CI choked on it ("Extra data")."""
    import json as _json

    from typer.testing import CliRunner

    from tailor.cli import app
    from test_expectations import make_fixture

    ds = make_fixture(tmp_path)
    runner = CliRunner()  # click>=8.2: stdout/stderr already separated
    r = runner.invoke(app, ["check", str(ds), "--json", "--markdown", str(tmp_path / "r.md")])
    assert r.exit_code == 0, (r.exit_code, r.output)
    data = _json.loads(r.stdout)  # plain loads, no raw_decode crutch
    assert data["verdict"] in ("accepted", "rejected")
    assert (tmp_path / "r.md").is_file()
