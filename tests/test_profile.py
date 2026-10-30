"""Acceptance-profile tests: defaults unchanged, overrides bite, typos fail fast."""

from pathlib import Path

import pytest

from tailor.expectations import run_all
from tailor.loader import load_snapshot
from tailor.model import Status
from tailor.profile import ProfileError, default_profile, load_profile
from test_expectations import make_fixture


def _profile_file(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "req.yaml"
    p.write_text(content)
    return p


def test_default_profile_matches_builtin_behavior(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path)
    snap = load_snapshot(ds)
    without = {r.name: r.status for r in run_all(snap)}
    with_default = {r.name: r.status for r in run_all(snap, default_profile())}
    assert without == with_default  # defaults untouched by the profile machinery


def test_episode_min_threshold_overrides(tmp_path: Path) -> None:
    # fixture episodes are 20 frames; requiring 30 must flag all of them
    prof = _profile_file(tmp_path, "checks:\n  episode_lengths:\n    min: 30\n")
    snap = load_snapshot(make_fixture(tmp_path / "a"))
    results = {r.name: r for r in run_all(snap, load_profile(prof))}
    assert results["episode_lengths"].status == Status.WARN
    assert "<30 frames" in results["episode_lengths"].message


def test_severity_remap_warn_to_fail(tmp_path: Path) -> None:
    # short-episode normally WARNs; a strict profile makes it a FAIL (exit 1)
    prof = _profile_file(tmp_path, "checks:\n  episode_lengths:\n    min: 30\n    severity: fail\n")
    ds = make_fixture(tmp_path / "b")
    snap = load_snapshot(ds)
    results = {r.name: r for r in run_all(snap, load_profile(prof))}
    assert results["episode_lengths"].status == Status.FAIL


def test_fps_allowed_set(tmp_path: Path) -> None:
    # fixture runs at 10 fps; allowing only [30, 50] must fail it
    prof = _profile_file(tmp_path, "checks:\n  fps:\n    allowed: [30, 50]\n")
    snap = load_snapshot(make_fixture(tmp_path / "c"))
    results = {r.name: r for r in run_all(snap, load_profile(prof))}
    assert results["fps"].status == Status.FAIL


def test_codebase_version_allowlist(tmp_path: Path) -> None:
    prof = _profile_file(tmp_path, "checks:\n  codebase_version:\n    allowed: [v2.1]\n")
    snap = load_snapshot(make_fixture(tmp_path / "d", fmt="v3.0"))
    results = {r.name: r for r in run_all(snap, load_profile(prof))}
    assert results["codebase_version"].status == Status.FAIL


def test_unknown_check_name_rejected(tmp_path: Path) -> None:
    prof = _profile_file(tmp_path, "checks:\n  teleportation:\n    min: 1\n")
    with pytest.raises(ProfileError, match="unknown check 'teleportation'"):
        load_profile(prof)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    prof = _profile_file(tmp_path, "checks:\n  fps:\n    max_speed: 88\n")
    with pytest.raises(ProfileError, match="no configurable key 'max_speed'"):
        load_profile(prof)


def test_invalid_severity_rejected(tmp_path: Path) -> None:
    prof = _profile_file(tmp_path, "checks:\n  fps:\n    severity: MAYBE\n")
    with pytest.raises(ProfileError, match="severity"):
        load_profile(prof)


def test_default_profile_shape() -> None:
    prof = default_profile()
    assert prof["timestamps"]["spacing_tolerance"] == 0.1
    assert prof["action_quality"]["clip_run"] == 5


def test_frozen_threshold_override(tmp_path: Path) -> None:
    # fixture injects 11 consecutive identical actions (run of 11 diffs);
    # raising the bar to 15 must silence the finding
    ds = make_fixture(tmp_path / "x", corrupt="frozen-action")
    snap = load_snapshot(ds)
    strict = {"action_quality": {"frozen_frames": 15}}
    results = {r.name: r for r in run_all(snap, strict)}
    assert results["action_quality"].status == Status.PASS
    default_results = {r.name: r for r in run_all(snap)}
    assert default_results["action_quality"].status == Status.WARN
