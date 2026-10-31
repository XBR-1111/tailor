"""Filter-rule tests: pure-function drop-set logic, no lerobot needed."""

from pathlib import Path

from tailor.expectations import run_all
from tailor.filtering import collect_drops, parse_episode_list, render_decision
from tailor.loader import load_snapshot
from test_expectations import make_fixture


def test_parse_episode_list() -> None:
    assert parse_episode_list("3,17,23") == {3, 17, 23}
    assert parse_episode_list(" 5 ") == {5}


def test_parse_rejects_junk() -> None:
    import pytest

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


def test_render_decision_lists_reasons(tmp_path: Path) -> None:
    snap = load_snapshot(make_fixture(tmp_path))
    d = collect_drops(snap, explicit={1})
    text = render_decision(d, 3)
    assert "drop ep 1" in text and "explicit" in text


def test_parse_episode_list_rejects_negative() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_episode_list("3,-7,12")
