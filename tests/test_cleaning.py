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
    _files, rows = rebuild_timestamps(snap, fps)
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


def _dir_digest(root: Path) -> dict[str, bytes]:
    import hashlib

    out = {}
    for f in sorted(root.rglob("*.parquet")):
        out[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).digest()
    return out


def test_dry_run_leaves_parquet_untouched(tmp_path: Path) -> None:
    """The regression test for the dry-run-that-wrote bug: dup-timestamp is
    the corruption whose clean path rewrites parquet, so a dry-run over it
    must leave every parquet byte-identical."""
    from typer.testing import CliRunner

    from tailor.cli import app

    ds = make_fixture(tmp_path, "dup-timestamp")
    before = _dir_digest(ds / "data")
    assert before, "fixture has parquet files"
    runner = CliRunner()
    result = runner.invoke(app, ["clean", str(ds)])
    assert result.exit_code == 0
    assert "dry-run" in result.stdout
    assert _dir_digest(ds / "data") == before


def test_apply_backs_up_originals_byte_equal(tmp_path: Path) -> None:
    """The regression test for the backup-after-write bug: <dataset>_pre_clean
    must contain the PRE-clean bytes, not the already-rewritten ones."""
    import hashlib

    from typer.testing import CliRunner

    from tailor.cli import app

    ds = make_fixture(tmp_path, "dup-timestamp")
    originals = {
        str(f.relative_to(ds)): hashlib.sha256(f.read_bytes()).digest()
        for f in sorted((ds / "data").rglob("*.parquet"))
    }
    runner = CliRunner()
    result = runner.invoke(app, ["clean", str(ds), "--apply"])
    assert result.exit_code == 0
    backup_dir = ds.parent / f"{ds.name}_pre_clean"
    backed_up = {
        str(f.relative_to(backup_dir)): hashlib.sha256(f.read_bytes()).digest()
        for f in sorted(backup_dir.rglob("*.parquet"))
    }
    assert backed_up, "timestamp repairs must be backed up"
    assert backed_up == {k: v for k, v in originals.items() if k in backed_up}


def test_running_stats_matches_numpy(tmp_path: Path) -> None:
    """Streaming (Chan merge) must match numpy's full-array stats exactly
    on sane data — this is the contract both stats paths rely on."""
    import numpy as np

    from tailor.stats import RunningStats

    rng = np.random.default_rng(0)
    full = rng.standard_normal((1500, 7)) * 3 + 5
    running = RunningStats()
    for start in range(0, 1500, 137):  # uneven chunks on purpose
        running.update(full[start : start + 137])
    assert np.allclose(running.mean, full.mean(axis=0), rtol=1e-12)
    assert np.allclose(running.std, full.std(axis=0), rtol=1e-10)
    assert np.allclose(running.min, full.min(axis=0))
    assert np.allclose(running.max, full.max(axis=0))
    assert running.count.tolist() == [1500] * 7


def test_apply_creates_missing_stats_json(tmp_path: Path) -> None:
    """A dataset without meta/stats.json: --apply must CREATE it (backing up
    nothing, since there is no original) instead of crashing on copy2."""

    from typer.testing import CliRunner

    from tailor.cli import app

    ds = make_fixture(tmp_path / "ds")
    (ds / "meta" / "stats.json").unlink()
    runner = CliRunner()
    result = runner.invoke(app, ["clean", str(ds), "--apply"])
    assert result.exit_code == 0, result.output
    assert (ds / "meta" / "stats.json").is_file()
    # re-check passes with the fresh stats
    assert _status_of(ds, "stats_recompute") == Status.PASS


def test_dryrun_reports_missing_stats_honestly(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tailor.cli import app

    ds = make_fixture(tmp_path / "ds")
    (ds / "meta" / "stats.json").unlink()
    result = CliRunner().invoke(app, ["clean", str(ds)])
    assert result.exit_code == 0
    assert "missing" in result.stdout
    assert "--apply would create it" in result.stdout
