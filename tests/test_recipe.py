import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tailor.cli import app
from tailor.filtering import DropDecision
from tailor.recipe import load_recipe
from test_expectations import MAKE_FIXTURE, make_fixture


def _recipe(root: Path, dataset: str, **settings: Any) -> Path:
    path = root / "recipe.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "dataset": dataset, "output": "out", **settings}))
    return path


def test_recipe_preview_uses_relative_paths_and_inline_checks(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, "nan", fmt="v3.0")
    path = _recipe(tmp_path, ds.name, checks={"nan_inf": {"enabled": True}})
    cli = CliRunner().invoke(app, ["run", str(path), "--json"])
    assert cli.exit_code == 0, cli.output
    result = json.loads(cli.stdout)
    assert result["status"] == "planned"
    assert result["plan"]["drop"] == [{"episode_index": 1, "reasons": ["nan_inf (FAIL)"]}]
    assert result["output_checks"] is None
    assert not (tmp_path / "out").exists()


def test_recipe_applies_shared_plan_and_keeps_auditable_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    ds = make_fixture(tmp_path, "nan", fmt="v3.0")
    clean = tmp_path / "clean"
    subprocess.run(
        [sys.executable, str(MAKE_FIXTURE), str(clean), "--format", "v3.0", "--episodes", "2"],
        check=True,
        capture_output=True,
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text("checks:\n  fps: {allowed: [10]}\n")
    path = _recipe(tmp_path, ds.name, profile=profile.name)

    def rewrite(source: Path, output: Path, decision: DropDecision) -> None:
        assert source == ds and set(decision.drop) == {1}
        assert output.parent.name.startswith(".tailor-stage-")
        shutil.copytree(clean, output)
        print("writer progress")

    monkeypatch.setattr("tailor.recipe.rewrite_filtered_dataset", rewrite)
    cli = CliRunner().invoke(app, ["run", str(path), "--apply", "--json"])
    assert cli.exit_code == 0, cli.output
    result = json.loads(cli.stdout)
    assert result["status"] == "accepted" and result["output_checks"]
    ledger = json.loads((tmp_path / "out/meta/tailor_ledger.jsonl").read_text())
    assert ledger["dropped"] == {"1": ["nan_inf (FAIL)"]}
    assert ledger["recipe"] == result["provenance"]
    assert ledger["recipe"]["recipe_sha256"] == load_recipe(path).digest
    assert "writer progress" in cli.stderr


def test_failed_recheck_never_publishes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    ds = make_fixture(tmp_path, "nan", fmt="v3.0")
    path = _recipe(tmp_path, ds.name)
    monkeypatch.setattr(
        "tailor.recipe.rewrite_filtered_dataset",
        lambda source, output, decision: shutil.copytree(source, output),
    )
    cli = CliRunner().invoke(app, ["run", str(path), "--apply", "--json"])
    assert cli.exit_code == 1, cli.output
    assert json.loads(cli.stdout)["status"] == "rejected"
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".tailor-stage-*"))


def test_recipe_copies_an_accepted_dataset_when_plan_is_empty(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, fmt="v3.0")
    path = _recipe(tmp_path, ds.name)
    cli = CliRunner().invoke(app, ["run", str(path), "--apply", "--json"])
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.stdout)["plan"]["drop"] == []
    output = tmp_path / "out"
    assert (output / "meta/info.json").read_bytes() == (ds / "meta/info.json").read_bytes()
    assert CliRunner().invoke(app, ["run", str(path), "--apply"]).exit_code == 2


@pytest.mark.parametrize(
    "settings",
    [
        {"version": 2},
        {"filter": {"min_length": "100"}},
        {"filter": {"drop_episodes": [True]}},
        {"filter": {"from_check": False, "drop_warn": True}},
        {"profile": "name.yaml", "checks": {}},
    ],
)
def test_bad_recipe_exits_before_accessing_input(tmp_path: Path, settings: dict) -> None:
    path = _recipe(tmp_path, "missing", **settings)
    cli = CliRunner().invoke(app, ["run", str(path), "--apply", "--json"])
    assert cli.exit_code == 2 and not cli.stdout, cli.output
    assert "recipe error" in cli.stderr
    assert not (tmp_path / "out").exists()


def test_remote_recipe_rejects_a_movable_ref(tmp_path: Path) -> None:
    path = _recipe(tmp_path, "lerobot/example@main")
    cli = CliRunner().invoke(app, ["run", str(path)])
    assert cli.exit_code == 2 and "commit SHA" in cli.stderr
