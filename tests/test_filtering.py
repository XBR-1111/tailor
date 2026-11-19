"""Filter-rule tests: pure-function drop-set logic, no lerobot needed."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from tailor.cli import app
from tailor.expectations import check_duplicate_episodes, check_timestamps, run_all
from tailor.filtering import collect_drops, parse_episode_list, render_decision
from tailor.loader import load_snapshot
from tailor.model import DatasetSnapshot
from tailor.profile import load_profile
from test_expectations import make_fixture


def test_filter_preview_reports_exact_episode_plan_without_writing(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, "nan", fmt="v3.0")
    out = tmp_path / "[review] filtered output"
    result = CliRunner().invoke(
        app, ["filter", str(ds), "--from-check", "--dry-run", "--json", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["schema_version"] == 1
    assert plan["output"] == str(out)
    assert plan["keep"] == [0, 2]
    assert plan["drop"] == [{"episode_index": 1, "reasons": ["nan_inf (FAIL)"]}]
    assert plan["error"] is None
    assert not out.exists()


def test_filter_preview_reports_global_failures_in_json(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, "frame-mismatch", fmt="v3.0")
    result = CliRunner().invoke(app, ["filter", str(ds), "--from-check", "--dry-run", "--json"])
    assert result.exit_code == 1, result.output
    plan = json.loads(result.stdout)
    assert plan["drop"] == []
    assert any(item["name"] == "total_frames" for item in plan["unassigned_failures"])
    assert plan["error"]


def test_filter_preview_rejects_an_empty_output_and_preserves_the_plan(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, fmt="v3.0")
    result = CliRunner().invoke(
        app, ["filter", str(ds), "--min-length", "100", "--dry-run", "--json"]
    )
    assert result.exit_code == 1, result.output
    plan = json.loads(result.stdout)
    assert plan["keep"] == [] and len(plan["drop"]) == 3
    assert plan["error"] == "filter would remove every episode"
    assert not Path(plan["output"]).exists()


def test_parse_episode_list() -> None:
    assert parse_episode_list("3,17,23") == {3, 17, 23}
    assert parse_episode_list(" 5 ") == {5}


def test_parse_rejects_junk() -> None:
    with pytest.raises(ValueError, match="invalid episode"):
        parse_episode_list("1,two,3")


def test_min_length_rule(tmp_path: Path) -> None:
    snap = load_snapshot(make_fixture(tmp_path, "short-episode"))
    d = collect_drops(snap, min_length=10)
    assert set(d.drop) == {2}  # short-episode corrupts ep 2 to length 1
    assert "min-length" in d.drop[2][0]


def test_explicit_rule(tmp_path: Path) -> None:
    snap = load_snapshot(make_fixture(tmp_path))
    d = collect_drops(snap, explicit={0, 2})
    assert set(d.drop) == {0, 2}
    assert d.keep == {1}


def test_deduke_keeps_first(tmp_path: Path) -> None:
    snap = load_snapshot(make_fixture(tmp_path, "duplicate-episode"))
    d = collect_drops(snap, dedupe=True)
    assert set(d.drop) == {2}  # ep 2 copies ep 0; first occurrence survives
    assert any("duplicate of episode 0" in r for r in d.drop[2])


def test_from_check_uses_fail_level_only(tmp_path: Path) -> None:
    # frame-mismatch FAILs but its details carry no episode list -> no drops;
    # dropped-frames FAILs with episode details -> ep 0 dropped
    snap = load_snapshot(make_fixture(tmp_path, "dropped-frames"))
    results = run_all(snap)
    d = collect_drops(snap, from_check=results)
    assert set(d.drop) == {0}
    assert any("frame_continuity" in r for r in d.drop[0])


def test_from_check_drop_warn(tmp_path: Path) -> None:
    snap = load_snapshot(make_fixture(tmp_path, "duplicate-episode"))
    results = run_all(snap)
    # duplicate_episodes is WARN: not dropped by default, dropped with --drop-warn
    assert collect_drops(snap, from_check=results).n_drop == 0
    assert collect_drops(snap, from_check=results, drop_warn=True).n_drop == 1


def test_from_check_drops_episode_with_nonfinite_features(tmp_path: Path) -> None:
    snap = load_snapshot(make_fixture(tmp_path, "nan", fmt="v3.0"))
    decision = collect_drops(snap, from_check=run_all(snap))
    assert decision.drop == {1: ["nan_inf (FAIL)"]}
    assert decision.keep == {0, 2}


def test_from_check_returns_failure_if_findings_cannot_be_filtered(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, "frame-mismatch", fmt="v3.0")
    result = CliRunner().invoke(app, ["filter", str(ds), "--from-check"])
    assert result.exit_code == 1, result.output
    assert "REJECTED" in result.stdout


@pytest.mark.parametrize(
    ("name", "rule", "length", "examples", "limit"),
    [
        ("episode_lengths", {"min": 100, "severity": "fail"}, 50, "short", 20),
        ("episode_lengths", {}, 0, "bad", 20),
        ("training_readiness", {"chunk_size": 100}, 50, "too_short", 10),
    ],
)
def test_yaml_filters_all_episodes_identified_by_length_rules(
    tmp_path: Path, name: str, rule: dict, length: int, examples: str, limit: int
) -> None:
    episodes = [{"episode_index": i, "length": length} for i in range(25)]
    episodes.append({"episode_index": 25, "length": 150})
    snap = DatasetSnapshot(root=tmp_path, episodes=episodes, metadata_only=True)
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({"checks": {name: rule}}))

    result = next(r for r in run_all(snap, load_profile(profile)) if r.name == name)
    decision = collect_drops(snap, from_check=[result])

    assert result.details[examples] == list(range(limit))
    assert result.details["episodes"] == list(range(25))
    assert set(decision.drop) == set(range(25))
    assert decision.keep == {25}


def test_from_check_uses_complete_duplicate_group(tmp_path: Path) -> None:
    frames = pd.DataFrame({"action": [[0.0], [1.0]]})
    snap = DatasetSnapshot(
        root=tmp_path,
        episodes=[{"episode_index": i} for i in range(53)],
        frames=dict.fromkeys(range(53), frames),
    )
    decision = collect_drops(snap, from_check=[check_duplicate_episodes(snap)], drop_warn=True)
    assert set(decision.drop) == set(range(1, 53))
    assert decision.keep == {0}


def test_duplicate_rules_keep_a_copy_after_other_removals(tmp_path: Path) -> None:
    snap = load_snapshot(make_fixture(tmp_path, "duplicate-episode"))
    decision = collect_drops(snap, explicit={0}, dedupe=True)
    assert set(decision.drop) == {0} and decision.keep == {1, 2}
    decision = collect_drops(snap, explicit={0}, from_check=run_all(snap), drop_warn=True)
    assert set(decision.drop) == {0} and decision.keep == {1, 2}


def test_from_check_uses_complete_timestamp_warnings(tmp_path: Path) -> None:
    irregular = pd.DataFrame({"timestamp": [0.0, 0.2]})
    regular = pd.DataFrame({"timestamp": [0.0, 0.1]})
    snap = DatasetSnapshot(
        root=tmp_path,
        info={"fps": 10},
        episodes=[{"episode_index": i} for i in range(13)],
        frames={**dict.fromkeys(range(12), irregular), 12: regular},
    )
    decision = collect_drops(snap, from_check=[check_timestamps(snap)], drop_warn=True)
    assert set(decision.drop) == set(range(12))
    assert decision.keep == {12}


def test_filter_refuses_to_write_when_rules_drop_every_episode(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, fmt="v3.0")
    profile = tmp_path / "profile.yaml"
    profile.write_text("checks:\n  episode_lengths: {min: 21, severity: fail}\n")
    dest = tmp_path / "filtered"

    result = CliRunner().invoke(
        app, ["filter", str(ds), "--from-check", "-p", str(profile), "-o", str(dest)]
    )

    assert result.exit_code == 1, result.output
    assert "would remove every episode" in result.stderr
    assert not dest.exists()


def test_filter_rejects_an_unknown_explicit_id_before_rewriting(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, fmt="v3.0")
    dest = tmp_path / "filtered"

    result = CliRunner().invoke(app, ["filter", str(ds), "--drop-episodes", "999", "-o", str(dest)])

    assert result.exit_code == 2, result.output
    assert "unknown episode indices: 999" in result.stderr
    assert not dest.exists()


def test_render_decision_lists_reasons(tmp_path: Path) -> None:
    snap = load_snapshot(make_fixture(tmp_path))
    d = collect_drops(snap, explicit={1})
    text = render_decision(d, 3)
    assert "drop ep 1" in text and "explicit" in text


def test_parse_episode_list_rejects_negative() -> None:
    with pytest.raises(ValueError):
        parse_episode_list("3,-7,12")
