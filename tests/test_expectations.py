"""Acceptance-suite tests: a clean synthetic fixture passes, each corruption is caught."""

import subprocess
import sys
from pathlib import Path

import pytest

from tailor.expectations import run_all
from tailor.loader import load_snapshot
from tailor.model import Status

REPO = Path(__file__).resolve().parent.parent
MAKE_FIXTURE = REPO / "scripts" / "make_fixture.py"


def make_fixture(tmp_path: Path, corrupt: str | None = None, fmt: str = "v2.1") -> Path:
    out = tmp_path / f"ds_{fmt}_{corrupt or 'clean'}"
    subprocess.run(
        [sys.executable, str(MAKE_FIXTURE), str(out), "--format", fmt]
        + (["--corrupt", corrupt] if corrupt else []),
        check=True,
        capture_output=True,
    )
    return out


def statuses(dataset: Path) -> dict[str, Status]:
    snap = load_snapshot(dataset)
    return {r.name: r.status for r in run_all(snap)}


def test_clean_fixture_is_accepted(tmp_path: Path) -> None:
    st = statuses(make_fixture(tmp_path))
    bad = {
        n: s
        for n, s in st.items()
        if s != Status.PASS and not (n == "training_readiness" and s == Status.SKIP)
    }
    assert not bad, bad
    snap = load_snapshot(make_fixture(tmp_path))
    results = run_all(snap)
    assert len(results) == 22


def test_nan_is_caught(tmp_path: Path) -> None:
    assert statuses(make_fixture(tmp_path, "nan"))["nan_inf"] == Status.FAIL


def test_missing_file_is_caught(tmp_path: Path) -> None:
    st = statuses(make_fixture(tmp_path, "missing-file"))
    assert st["episode_files"] == Status.FAIL


def test_frame_mismatch_is_caught(tmp_path: Path) -> None:
    assert statuses(make_fixture(tmp_path, "frame-mismatch"))["total_frames"] == Status.FAIL


def test_bad_version_is_caught(tmp_path: Path) -> None:
    assert statuses(make_fixture(tmp_path, "bad-version"))["codebase_version"] == Status.FAIL


def test_short_episode_warns(tmp_path: Path) -> None:
    assert statuses(make_fixture(tmp_path, "short-episode"))["episode_lengths"] == Status.WARN


def test_dup_timestamp_is_caught(tmp_path: Path) -> None:
    assert statuses(make_fixture(tmp_path, "dup-timestamp"))["timestamps"] == Status.FAIL


def test_unreadable_dataset(tmp_path: Path) -> None:
    snap = load_snapshot(tmp_path)  # empty dir
    assert not snap.readable


def test_cli_check_exit_codes(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tailor.cli import app

    runner = CliRunner()
    clean = make_fixture(tmp_path)
    assert runner.invoke(app, ["check", str(clean)]).exit_code == 0
    bad = make_fixture(tmp_path, "nan")
    assert runner.invoke(app, ["check", str(bad)]).exit_code == 1
    assert runner.invoke(app, ["check", str(tmp_path / "nope")]).exit_code == 2
    json_out = runner.invoke(app, ["check", str(bad), "--json"])
    assert json_out.exit_code == 1
    assert '"verdict": "rejected"' in json_out.stdout


@pytest.mark.parametrize("fmt", ["v2.1", "v3.0"])
def test_clean_fixture_is_accepted_both_formats(tmp_path: Path, fmt: str) -> None:
    st = statuses(make_fixture(tmp_path, fmt=fmt))
    # training_readiness SKIPs by design until chunk_size is configured
    bad = {
        n: s
        for n, s in st.items()
        if s != Status.PASS and not (n == "training_readiness" and s == Status.SKIP)
    }
    assert not bad, bad


@pytest.mark.parametrize(
    "corrupt,expect_name",
    [
        ("nan", "nan_inf"),
        ("missing-file", "episode_files"),
        ("frame-mismatch", "total_frames"),
        ("bad-version", "codebase_version"),
        ("short-episode", "episode_lengths"),
        ("dup-timestamp", "timestamps"),
        ("frozen-action", "action_quality"),
        ("stats-mismatch", "stats_recompute"),
        ("missing-video", "video_files"),
        ("video-mismatch", "video_streams"),
        ("zero-variance", "zero_variance"),
        ("dropped-frames", "frame_continuity"),
        ("duplicate-episode", "duplicate_episodes"),
        ("portability-junk", "portability"),
    ],
)
@pytest.mark.parametrize("fmt", ["v2.1", "v3.0"])
def test_every_corruption_has_a_catcher(
    tmp_path: Path, corrupt: str, expect_name: str, fmt: str
) -> None:
    st = statuses(make_fixture(tmp_path, corrupt, fmt=fmt))
    assert st[expect_name] in (Status.FAIL, Status.WARN)


def test_schema_drop_is_caught_v2(tmp_path: Path) -> None:
    # schema corruption is v2-only in the fixture (v3 episodes share one shard)
    st = statuses(make_fixture(tmp_path, "schema-drop", fmt="v2.1"))
    assert st["schema_consistency"] == Status.FAIL
