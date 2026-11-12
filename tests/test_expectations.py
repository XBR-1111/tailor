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


def make_fixture(
    tmp_path: Path, corrupt: str | None = None, fmt: str = "v2.1", frames: int | None = None
) -> Path:
    out = tmp_path / f"ds_{fmt}_{corrupt or 'clean'}_f{frames or 20}"
    subprocess.run(
        [sys.executable, str(MAKE_FIXTURE), str(out), "--format", fmt]
        + (["--corrupt", corrupt] if corrupt else [])
        + (["--frames", str(frames)] if frames else []),
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
    structural_skip = {"training_readiness", "video_sync", "video_content"}
    bad = {
        n: s
        for n, s in st.items()
        if s != Status.PASS and not (n in structural_skip and s == Status.SKIP)
    }
    assert not bad, bad
    snap = load_snapshot(make_fixture(tmp_path))
    results = run_all(snap)
    assert len(results) == 26  # one more check per release; keep in sync with EXPECTATIONS


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
    structural_skip = {"training_readiness", "video_sync", "video_content"}
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
        ("video-fps", "video_streams"),
        ("video-blackout", "video_black"),
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


def test_rotation_degrees_reads_displaymatrix_and_legacy_tag() -> None:
    """90° displaymatrix, identity matrix and legacy 'rotate' tag."""
    from types import SimpleNamespace

    from tailor.expectations import _rotation_degrees

    cw90 = SimpleNamespace(side_data={"displaymatrix": [0, 1, 0, -1, 0, 0, 0, 0, 1]}, metadata={})
    assert _rotation_degrees(cw90) == 90
    upright = SimpleNamespace(side_data={"displaymatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]}, metadata={})
    assert _rotation_degrees(upright) is None
    legacy = SimpleNamespace(side_data={}, metadata={"rotate": "180"})
    assert _rotation_degrees(legacy) == 180


def test_interlaced_flag_detection() -> None:
    from types import SimpleNamespace

    from tailor.expectations import _interlaced

    def stream(field_order: object) -> SimpleNamespace:
        return SimpleNamespace(codec_context=SimpleNamespace(field_order=field_order))

    assert _interlaced(stream("TT")) is True
    assert _interlaced(stream("PROGRESSIVE")) is False
    assert _interlaced(stream(2)) is True  # FFmpeg enum: 2 = TT
    assert _interlaced(stream(1)) is False  # 1 = progressive
    assert _interlaced(stream(None)) is None


def test_duration_gap_ratio_math() -> None:
    from tailor.expectations import _duration_gap_ratio

    # 100 frames at 10 fps = 10s of content; header claims 12s → 1/6 short
    assert _duration_gap_ratio(100, 10.0, 12.0) == pytest.approx(1 / 6)
    assert _duration_gap_ratio(100, 10.0, 10.05) == pytest.approx(0.05 / 10.05)
    assert _duration_gap_ratio(100, None, 12.0) is None
    assert _duration_gap_ratio(0, 10.0, 12.0) is None
    assert _duration_gap_ratio(100, 10.0, None) is None


@pytest.mark.parametrize("corrupt", ["video-freeze", "video-blur"])
def test_content_defects_caught_by_deep_content_check(tmp_path: Path, corrupt: str) -> None:
    """video_content is OFF by default (full decode) and must be opted in.
    Covers repeated frames (stalled camera) and whole-segment defocus."""
    for fmt in ("v2.1", "v3.0"):
        ds = make_fixture(tmp_path / fmt, corrupt, fmt=fmt)
        snap = load_snapshot(ds)
        off = {r.name: r.status for r in run_all(snap)}
        assert off["video_content"] == Status.SKIP
        on = {r.name: r.status for r in run_all(snap, {"video_content": {"enabled": True}})}
        assert on["video_content"] == Status.FAIL, f"{corrupt} {fmt}"


def test_video_content_clean_passes_with_profile(tmp_path: Path) -> None:
    """No false positives: a clean fixture passes the deep scan when enabled."""
    ds = make_fixture(tmp_path, fmt="v2.1")
    snap = load_snapshot(ds)
    on = {r.name: r.status for r in run_all(snap, {"video_content": {"enabled": True}})}
    assert on["video_content"] == Status.PASS


def test_mid_black_requires_lit_lead_in() -> None:
    """A mid-file black span is a fault only when the scene was still lit just
    before it: a wrist camera dimming into occlusion (real ep 247: mean 15-28
    a second before black) must not be flagged, a lit→black cut must be."""
    import numpy as np

    from tailor.expectations import _content_problems

    n = 60
    t = np.arange(n) / 10.0  # 6 s at 10 fps
    mean = np.full(n, 100.0)
    mean[30:40] = 3.0  # black span 3.0-3.9 s, lead-in lit at 100
    dark = (mean < 24).astype(float)
    base = {
        "t": t.tolist(),
        "mean": mean.tolist(),
        "std": (np.where(dark > 0, 1.0, 30.0)).tolist(),
        "dark": dark.tolist(),
        "mad": [0.05] * n,
        "ident": [False] * n,
        "exact": [False] * n,
        "lap_t": t[::5].tolist(),
        "lap": [1000.0] * len(t[::5]),
        "lap99": [200.0] * len(t[::5]),
    }
    lit_cut = _content_problems(dict(base), {"black_onset": 40.0})
    assert any(p.startswith("black") for p in lit_cut), lit_cut

    dim = dict(base)
    mean2 = np.array(base["mean"])
    mean2[20:30] = 20.0  # one second of dimming before the black span
    dim["mean"] = mean2.tolist()
    occluded = _content_problems(dim, {"black_onset": 40.0})
    assert not any(p.startswith("black") for p in occluded), occluded


def test_orphan_shards_fail_both_pointer_checks(tmp_path: Path) -> None:
    """Unreferenced data/video shards (the unitreeh1 aftermath: real frames
    sitting in a file no episode points at) must fail episode_files and
    video_files even though every declared file exists with enough frames."""
    from tailor.expectations import check_episode_files, check_video_files

    ds = make_fixture(tmp_path / "v3", "orphan-shard", fmt="v3.0")
    snap = load_snapshot(ds)
    ef = check_episode_files(snap)
    vf = check_video_files(snap)
    assert ef.status == Status.FAIL, ef.message
    assert "not referenced" in ef.message
    assert vf.status == Status.FAIL, vf.message
    assert any("orphan" in str(p) for p in vf.details.get("problems", []))


def test_clean_v3_has_no_orphans_or_overlaps(tmp_path: Path) -> None:
    """The clean v3 fixture must stay green under the new pointer checks."""
    from tailor.expectations import check_episode_files, check_video_files

    ds = make_fixture(tmp_path / "v3", fmt="v3.0")
    snap = load_snapshot(ds)
    assert check_episode_files(snap).status == Status.PASS
    assert check_video_files(snap).status == Status.PASS


def test_overlapping_video_spans_fail(tmp_path: Path) -> None:
    """Two episodes claiming the same seconds of one shard is never legal."""
    import pandas as pd

    from tailor.expectations import check_video_files

    ds = make_fixture(tmp_path / "v3", fmt="v3.0")
    eps_path = ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(eps_path)
    fcol = "videos/observation.images.top/from_timestamp"
    tcol = "videos/observation.images.top/to_timestamp"
    # episode 1 starts inside episode 0's span on the same shard
    df.loc[1, fcol] = 1.0
    df.loc[1, tcol] = float(df.loc[0, tcol]) + 1.0
    df.to_parquet(eps_path)
    snap = load_snapshot(ds)
    r = check_video_files(snap)
    assert r.status == Status.FAIL, r.message
    assert any("overlaps" in str(p) for p in r.details.get("problems", []))


def test_nan_inf_covers_extra_numeric_columns(tmp_path: Path) -> None:
    """A NaN in any numeric observation.*/action column (extra scalar
    sensors, joint positions) is a failure, not just state/action."""
    import numpy as np
    import pandas as pd

    from tailor.expectations import check_nan_inf

    snap = load_snapshot(make_fixture(tmp_path))
    idx = min(snap.frames)
    df = snap.frames[idx].copy()
    df["observation.elbow_pos"] = np.linspace(0.0, 1.0, len(df))
    df["observation.language"] = pd.Series(["grab the cup"] * len(df), dtype=object)
    df.loc[df.index[3], "observation.elbow_pos"] = np.nan
    snap.frames[idx] = df
    r = check_nan_inf(snap)
    assert r.status == Status.FAIL, r.message
    assert any("elbow_pos" in str(b) for b in r.details.get("bad", []))


def test_frozen_position_classes_and_ignore(tmp_path: Path) -> None:
    """Prefix/suffix stalls (collection-side holds) get position tags; a
    profile's frozen_ignore keeps them out of the verdict while mid stalls
    stay failing. Real case: DexJoCo keeps ~2 s after success on every
    episode — 134 edge stalls flooded the report and REJECTED the dataset."""
    from tailor.expectations import check_video_content

    # default: tail stall fails, tagged suffix
    ds = make_fixture(tmp_path / "v2s", "video-freeze-suffix", frames=40)
    snap = load_snapshot(ds)
    r = check_video_content(snap, {"enabled": True, "frozen_edge_s": 1.0})
    assert r.status == Status.FAIL, r.message
    assert any("frozen[suffix]" in str(p) for p in r.details.get("problems", []))
    assert r.details["frozen_by_position"]["suffix"] >= 1

    # profile ignores suffix: verdict clean, count stays visible
    r2 = check_video_content(
        snap, {"enabled": True, "frozen_edge_s": 1.0, "frozen_ignore": ["suffix"]}
    )
    assert r2.status == Status.PASS, r2.message
    assert r2.details["frozen_by_position"]["suffix"] >= 1
    assert r2.details["frozen_ignored"] >= 1

    # mid stall (existing corruption) stays failing under the same profile
    ds2 = make_fixture(tmp_path / "v2m", "video-freeze", frames=40)
    snap2 = load_snapshot(ds2)
    r3 = check_video_content(
        snap2, {"enabled": True, "frozen_edge_s": 1.0, "frozen_ignore": ["suffix"]}
    )
    assert r3.status == Status.FAIL, r3.message
    assert any("frozen[mid]" in str(p) for p in r3.details.get("problems", []))

    # clean stays clean with the counter at zero (message unchanged)
    ds3 = make_fixture(tmp_path / "v2c")
    snap3 = load_snapshot(ds3)
    r4 = check_video_content(snap3, {"enabled": True})
    assert r4.status == Status.PASS, r4.message
    assert r4.details["frozen_by_position"] == {"prefix": 0, "suffix": 0, "mid": 0}


def test_reordered_rows_classified_not_holes(tmp_path: Path) -> None:
    """Issue #11: a complete permutation must report `reordering`, never `holes`."""
    from tailor.expectations import check_frame_continuity

    for fmt in ("v2.1", "v3.0"):
        snap = load_snapshot(make_fixture(tmp_path / fmt, "reordered-rows", fmt))
        r = check_frame_continuity(snap)
        assert r.status == Status.FAIL, (fmt, r.message)
        assert "reordering" in r.message, (fmt, r.message)
        assert "holes" not in r.message, (fmt, r.message)
        assert "length_mismatch" not in r.message, (fmt, r.message)


def test_reordered_rows_index_column_stays_consistent(tmp_path: Path) -> None:
    """Whole-row shuffle keeps index == offset + frame_index per row (order-free)."""
    from tailor.expectations import check_index_column

    for fmt in ("v2.1", "v3.0"):
        snap = load_snapshot(make_fixture(tmp_path / fmt, "reordered-rows", fmt))
        r = check_index_column(snap)
        assert r.status == Status.PASS, (fmt, r.message)


def test_index_mislabel_caught_and_old_checks_blind(tmp_path: Path) -> None:
    """Issue #11 blind spot: wrong index column passes frame_continuity/timestamps,
    only the new index_column check catches it."""
    from tailor.expectations import check_frame_continuity, check_index_column, check_timestamps

    for fmt in ("v2.1", "v3.0"):
        snap = load_snapshot(make_fixture(tmp_path / fmt, "index-mislabel", fmt))
        r = check_index_column(snap)
        assert r.status == Status.FAIL, (fmt, r.message)
        assert "1" in r.message and "mislabeled" in r.message, (fmt, r.message)
        # the pre-existing suite stays green on this corruption
        assert check_frame_continuity(snap).status == Status.PASS, fmt
        assert check_timestamps(snap).status == Status.PASS, fmt


def test_clean_fixture_index_column_passes(tmp_path: Path) -> None:
    from tailor.expectations import check_index_column

    for fmt in ("v2.1", "v3.0"):
        snap = load_snapshot(make_fixture(tmp_path / fmt, None, fmt))
        r = check_index_column(snap)
        assert r.status == Status.PASS, (fmt, r.message)


def test_stats_drift_severity_tracks_coverage(tmp_path: Path) -> None:
    """Issue #11: sampled-run drift is WARN; full-run drift keeps FAIL teeth."""
    ds = make_fixture(tmp_path / "s", "stats-mismatch", "v2.1")
    snap = load_snapshot(ds, max_episodes=1)
    sampled = next(r for r in run_all(snap) if r.name == "stats_recompute")
    assert sampled.status == Status.WARN, sampled.message
    assert "coverage" in sampled.message, sampled.message

    full = next(r for r in run_all(load_snapshot(ds)) if r.name == "stats_recompute")
    assert full.status == Status.FAIL, full.message


def test_multi_task_v3_loads_and_checks_run(tmp_path: Path) -> None:
    """Regression (issue #12 candidate): episodes whose v3 `tasks` cell is a
    multi-element ndarray used to crash the loader's truthiness test, silently
    turning the whole dataset into "unreadable". Legal shape — must load."""
    ds = make_fixture(tmp_path / "mt", "multi-task", "v3.0")
    snap = load_snapshot(ds)
    assert snap.readable, "multi-task episode must not make the dataset unreadable"
    assert snap.episodes is not None and len(snap.episodes) == 3
    st = {r.name: r.status for r in run_all(snap)}
    assert "frame_continuity" in st  # the suite ran end-to-end
