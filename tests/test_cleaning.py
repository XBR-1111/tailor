"""Clean-stage tests: stats recompute + timestamp rebuild fix what checks flag."""

import json
from pathlib import Path

from tailor.cleaning import rebuild_timestamps, recompute_stats
from tailor.loader import load_snapshot
from tailor.model import Status
from test_expectations import make_fixture, statuses


def _status_of(ds: Path, name: str) -> Status:
    return statuses(ds)[name]


def test_stats_mismatch_fixed_by_recompute(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, "stats-mismatch")
    assert _status_of(ds, "stats_recompute") == Status.FAIL
    snap = load_snapshot(ds)
    fresh = recompute_stats(snap)
    (ds / "meta" / "stats.json").write_text(json.dumps(fresh, indent=2))
    assert _status_of(ds, "stats_recompute") == Status.PASS


def test_dup_timestamps_fixed_by_rebuild(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, "dup-timestamp")
    assert _status_of(ds, "timestamps") in (Status.FAIL, Status.WARN)
    snap = load_snapshot(ds)
    fps = float(snap.info["fps"])
    files, rows = rebuild_timestamps(snap, fps)
    assert rows > 0
    assert _status_of(ds, "timestamps") == Status.PASS


def test_recompute_preserves_video_stats(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path)
    snap = load_snapshot(ds)
    fresh = recompute_stats(snap)
    # fixture stats.json declares no video entries, so none are carried over
    assert "observation.images.top" not in fresh
    assert fresh["observation.state"]["count"] == [60, 60, 60, 60, 60, 60]


def test_rebuild_is_idempotent(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path, "dup-timestamp")
    snap = load_snapshot(ds)
    fps = float(snap.info["fps"])
    rebuild_timestamps(snap, fps)
    files2, rows2 = rebuild_timestamps(load_snapshot(ds), fps)
    assert files2 == [] and rows2 == 0


def test_clean_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tailor.cli import app

    ds = make_fixture(tmp_path, "stats-mismatch")
    before = (ds / "meta" / "stats.json").read_text()
    runner = CliRunner()
    result = runner.invoke(app, ["clean", str(ds)])
    assert result.exit_code == 0
    assert "dry-run" in result.stdout
    assert (ds / "meta" / "stats.json").read_text() == before
    assert not (ds.parent / f"{ds.name}_pre_clean").exists()


def test_clean_cli_apply_fixes_and_rechecks(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tailor.cli import app

    ds = make_fixture(tmp_path, "stats-mismatch")
    runner = CliRunner()
    result = runner.invoke(app, ["clean", str(ds), "--apply"])
    assert result.exit_code == 0
    # backup exists outside the dataset root
    backup = ds.parent / f"{ds.name}_pre_clean" / "meta" / "stats.json"
    assert backup.is_file()
    # and the dataset now passes its own stats check
    assert _status_of(ds, "stats_recompute") == Status.PASS
