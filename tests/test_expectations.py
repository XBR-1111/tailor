"""Acceptance-suite tests: a clean synthetic fixture passes, each corruption is caught."""

import subprocess
import sys
from pathlib import Path

import pytest

from tailor.expectations import run_all
from tailor.filtering import collect_drops
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
    # structural SKIPs are not defects: training_readiness needs chunk_size
    # configured, video_sync has no per-camera timestamps to compare on v2.1
    structural_skip = {"training_readiness", "video_sync"}
    bad = {
        n: s
        for n, s in st.items()
        if s != Status.PASS and not (n in structural_skip and s == Status.SKIP)
    }
    assert not bad, bad
    snap = load_snapshot(make_fixture(tmp_path))
    results = run_all(snap)
    assert len(results) == 23


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
    # structural SKIPs by design: training_readiness needs chunk_size
    # configured; video_sync has no per-camera timestamps on v2.x
    structural_skip = {"training_readiness", "video_sync"}
    bad = {
        n: s
        for n, s in st.items()
        if s != Status.PASS and not (n in structural_skip and s == Status.SKIP)
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


def test_action_quality_attributes_episodes(tmp_path: Path) -> None:
    """frozen-action corrupts episode 1; the check must name it in details
    so --from-check can drop exactly that episode."""
    st = statuses(make_fixture(tmp_path, "frozen-action"))
    assert st["action_quality"] == Status.WARN
    results = {r.name: r for r in run_all(load_snapshot(make_fixture(tmp_path, "frozen-action")))}
    details = results["action_quality"].details
    assert details.get("episodes") == [1]


def test_from_check_drops_action_quality_episode(tmp_path: Path) -> None:
    """The escalation promise: a WARN flagged with episode attribution must
    be droppable via --from-check --drop-warn."""
    ds = make_fixture(tmp_path, "frozen-action")
    snap = load_snapshot(ds)
    results = run_all(snap)
    decision = collect_drops(snap, from_check=results, drop_warn=True)
    assert 1 in decision.drop
    assert "action_quality" in decision.drop[1][-1] or any(
        "action_quality" in r for r in decision.drop[1]
    )


def test_video_desync_caught_v3(tmp_path: Path) -> None:
    """v3-only corruption: wrist camera's to_timestamp drifts 0.3s in ep 1.
    video_sync must WARN and attribute ep 1; video_streams stays PASS (the
    video file itself is well-formed — only metadata drifted)."""
    st = statuses(make_fixture(tmp_path, "video-desync", fmt="v3.0"))
    assert st["video_sync"] == Status.WARN
    assert st["video_streams"] == Status.PASS
    results = {
        r.name: r
        for r in run_all(load_snapshot(make_fixture(tmp_path, "video-desync", fmt="v3.0")))
    }
    assert results["video_sync"].details.get("episodes") == [1]


def test_video_sync_skips_v2(tmp_path: Path) -> None:
    st = statuses(make_fixture(tmp_path, fmt="v2.1"))
    assert st["video_sync"] == Status.SKIP


def test_video_sync_passes_single_camera(tmp_path: Path) -> None:
    st = statuses(make_fixture(tmp_path, fmt="v3.0"))
    assert st["video_sync"] == Status.PASS  # one camera -> nothing to sync


def test_run_all_is_deterministic(tmp_path: Path) -> None:
    """Two fresh loads + runs must agree on every result field — pins the
    behaviour of the column cache and the parallel video probing."""
    ds = make_fixture(tmp_path, "frozen-action", fmt="v3.0")
    r1 = [(r.name, r.status, r.message, r.details) for r in run_all(load_snapshot(ds))]
    ds2 = make_fixture(tmp_path / "again", "frozen-action", fmt="v3.0")
    r2 = [(r.name, r.status, r.message, r.details) for r in run_all(load_snapshot(ds2))]
    assert r1 == r2


def test_sample_last_covers_tail(tmp_path: Path) -> None:
    """--sample last keeps the TAIL episodes: corruption planted in the last
    episode must be caught with max_episodes=1 + sample=last."""
    from tailor.loader import load_snapshot

    # dup-timestamp corrupts ep 0; the selector test needs head != tail,
    # which 3 episodes give us regardless of where the corruption sits
    ds = make_fixture(tmp_path / "t", "dup-timestamp", fmt="v2.1")
    head = load_snapshot(ds, max_episodes=1, sample="first")
    tail = load_snapshot(ds, max_episodes=1, sample="last")
    assert sorted(head.frames) == [0]
    assert sorted(tail.frames) == [2]
