"""Clean-stage tests: stats recompute + timestamp rebuild fix what checks flag."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tailor.cleaning import apply_clean, rebuild_timestamps, recompute_stats
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


@pytest.mark.parametrize("fmt", ["v2.1", "v3.0"])
def test_recompute_preserves_stats_and_refreshes_quantiles(tmp_path: Path, fmt: str) -> None:
    ds = make_fixture(tmp_path, fmt=fmt)
    path = ds / "meta" / "stats.json"
    previous = json.loads(path.read_text())
    previous["observation.images.top"] = {"count": [30], "mean": [[[0.5]], [[0.4]], [[0.3]]]}
    previous["next.done"] = {"count": [60], "mean": [0.05]}
    levels = {"q01": 0.01, "q50": 0.50, "q99": 0.99}
    previous["action"].update({name: [-10.0] * 6 for name in levels})
    path.write_text(json.dumps(previous))

    fresh = recompute_stats(load_snapshot(ds))
    assert fresh["observation.images.top"] == previous["observation.images.top"]
    assert fresh["next.done"] == previous["next.done"]
    assert fresh["action"]["count"] == [60]
    assert fresh["observation.state"]["count"] == [60]
    action = np.concatenate(
        [np.stack(pd.read_parquet(f)["action"]) for f in ds.glob("data/*/*.parquet")]
    )
    for name, level in levels.items():
        assert fresh["action"][name] == pytest.approx(np.quantile(action, level, axis=0), abs=0.001)


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


def test_clean_recomputes_stats_from_repaired_timestamp_values(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tailor.cli import app

    ds = make_fixture(tmp_path, "dup-timestamp", fmt="v3.0")
    path = next(ds.glob("data/*/*.parquet"))
    original = pq.read_table(path)
    index = original.schema.get_field_index("timestamp")
    # Replicate LeRobot's float32 timestamp schema and inject a null timestamp.
    timestamps = original["timestamp"].to_pylist()
    timestamps[25] = None
    field = original.schema.field(index).with_type(pa.float32())
    modified = original.set_column(index, field, pa.array(timestamps, type=pa.float32()))
    pq.write_table(modified, path)
    stats_path = ds / "meta" / "stats.json"
    previous = json.loads(stats_path.read_text())
    previous["timestamp"] = {"q99": [0.0]}
    stats_path.write_text(json.dumps(previous))

    result = CliRunner().invoke(app, ["clean", str(ds), "--apply"])
    assert result.exit_code == 0, result.output
    rebuilt = pq.read_table(path)
    assert rebuilt.schema.field(index) == field
    assert rebuilt.schema.metadata == modified.schema.metadata
    rebuilt_times = rebuilt["timestamp"].to_numpy().astype(float)
    assert rebuilt_times == pytest.approx(rebuilt["frame_index"].to_numpy() / 10)
    stats = json.loads(stats_path.read_text())["timestamp"]
    assert stats["count"] == [len(rebuilt_times)]
    assert stats["mean"] == [pytest.approx(np.mean(rebuilt_times))]
    assert stats["q99"] == [pytest.approx(np.quantile(rebuilt_times, 0.99))]


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


def test_clean_keeps_originals_when_staging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_fixture(tmp_path)
    files = sorted(ds.glob("data/*/*.parquet"))[:2]
    for f in files:
        df = pd.read_parquet(f)
        df["timestamp"] = 0.0
        df.to_parquet(f, index=False)
    originals = {f: f.read_bytes() for f in [*files, ds / "meta" / "stats.json"]}
    write_table = pq.write_table
    writes = 0

    def fail_second_write(table: pa.Table, path: Path) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full while staging")
        write_table(table, path)

    monkeypatch.setattr(pq, "write_table", fail_second_write)
    with pytest.raises(OSError, match="disk full"):
        apply_clean(load_snapshot(ds), stats=True, timestamps=True)
    assert writes == 2
    assert {f: f.read_bytes() for f in originals} == originals


def test_running_stats_matches_numpy(tmp_path: Path) -> None:
    """Streaming (Chan merge) must match numpy's full-array stats exactly
    on sane data — this is the contract both stats paths rely on."""
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
