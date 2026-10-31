"""Smoke tests: package imports and CLI contract hold before any real feature exists."""

from typer.testing import CliRunner

import tailor
from tailor.cli import app

runner = CliRunner()


def test_version_exports() -> None:
    assert tailor.__version__ == "0.3.0"


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert tailor.__version__ in result.stdout


def test_cli_check_exit_code_is_error_while_unimplemented() -> None:
    # exit 2 keeps scripts and CI from mistaking a stub for a passing check
    result = runner.invoke(app, ["check", "/nonexistent"])
    assert result.exit_code == 2
