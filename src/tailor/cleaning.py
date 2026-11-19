"""Deterministic repairs (the clean stage): stats recompute + timestamp rebuild.

Everything is metadata-level: parquet timestamps and meta/stats.json. No
lerobot dependency — the CLI runs a dry-run by default and only writes with
--apply, backing up every replaced file outside the dataset root so
portability checks stay clean.
"""

import json
import mmap
import re
import shutil
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from typing import BinaryIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tailor.model import DatasetSnapshot
from tailor.stats import RunningStats

_NUMERIC_COLS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "index",
    "episode_index",
    "task_index",
)
_QUANTILE_KEY = re.compile(r"q(?:[0-9]{2}|100)")


@dataclass
class CleanReport:
    stats_rewritten: bool = False
    timestamp_files: list[Path] = field(default_factory=list)
    timestamp_rows: int = 0
    backup_dir: Path | None = None


def _stats_path(snap: DatasetSnapshot) -> Path:
    return snap.root / "meta" / "stats.json"


def _data_files(snap: DatasetSnapshot) -> list[Path]:
    """All parquet shard/episode files in data/."""
    data = snap.root / "data"
    return sorted(data.glob("chunk-*/*.parquet")) if data.is_dir() else []


def recompute_stats(
    snap: DatasetSnapshot, *, timestamp_files: Sequence[Path] = (), fps: float = 0.0
) -> dict:
    """Recompute numerical statistics and any existing LeRobot quantiles.

    Retain statistics for features this repair does not process. When both
    repairs are requested, compute from the timestamps that will be written.
    """
    p = _stats_path(snap)
    out = json.loads(p.read_text()) if p.is_file() else {}
    if not isinstance(out, dict):
        raise ValueError(f"expected an object in {p}")
    files = _data_files(snap)
    to_repair = set(timestamp_files)

    def _columns(columns: Sequence[str]) -> Iterator[tuple[str, np.ndarray]]:
        for path in files:
            df = pd.read_parquet(path)
            if df.empty:
                continue
            for col in columns:
                if col not in df.columns:
                    continue
                if col == "timestamp" and path in to_repair:
                    arr, _ = _timestamp_values(df, fps)
                else:
                    arr = np.stack([np.asarray(v, dtype=float) for v in df[col]])
                if not np.isfinite(arr).all():
                    raise ValueError(f"cannot recompute {col!r}: non-finite values in {path}")
                yield col, arr.reshape(len(arr), -1)

    quantile_levels = {
        col: {k: int(k[1:]) / 100 for k in previous if _QUANTILE_KEY.fullmatch(k)}
        for col in _NUMERIC_COLS
        if isinstance(previous := out.get(col), dict)
    }
    running: dict[str, RunningStats] = {}
    widths: dict[str, int] = {}
    quantiles: dict[str, dict[str, list[float]]] = {}
    samples: dict[str, BinaryIO] = {}
    with ExitStack() as scratch:
        for col, arr in _columns(_NUMERIC_COLS):
            width = arr.shape[1]
            if widths.setdefault(col, width) != width or width == 0:
                raise ValueError(f"cannot recompute {col!r}: inconsistent feature width")
            running.setdefault(col, RunningStats()).update(arr)
            levels = quantile_levels.get(col)
            if not levels:
                continue
            if len(files) == 1:
                values = np.quantile(arr, list(levels.values()), axis=0)
                quantiles[col] = dict(zip(levels, values.tolist(), strict=True))
            else:
                if col not in samples:
                    samples[col] = scratch.enter_context(TemporaryFile(mode="w+b"))
                np.asarray(arr, dtype=np.float64).tofile(samples[col])

        for col, stats in running.items():
            count = int(stats.count[0])
            if col in samples:
                levels = quantile_levels[col]
                # Work on a private disk-backed array. NumPy partitions the
                # temporary samples in place, without copying the full matrix.
                with mmap.mmap(samples[col].fileno(), 0, access=mmap.ACCESS_WRITE) as mapped:
                    matrix: np.ndarray = np.ndarray(
                        (count, widths[col]), dtype=np.float64, buffer=mapped
                    )
                    values = np.quantile(
                        matrix, list(levels.values()), axis=0, overwrite_input=True
                    )
                    del matrix
                quantiles[col] = dict(zip(levels, values.tolist(), strict=True))
            out[col] = {
                "min": stats.min.tolist(),
                "max": stats.max.tolist(),
                "mean": stats.mean.tolist(),
                "std": stats.std.tolist(),
                "count": [count],
                **quantiles.get(col, {}),
            }
    return out


def _timestamp_values(df: pd.DataFrame, fps: float) -> tuple[np.ndarray, np.ndarray]:
    old = df["timestamp"].to_numpy()
    expected = df["frame_index"].to_numpy(dtype=float) / fps
    if old.dtype.kind == "f":
        expected = expected.astype(old.dtype)
    changed = np.isfinite(expected) & ~np.isclose(old, expected, rtol=0, atol=1e-6)
    return np.where(changed, expected, old), changed


def detect_timestamp_changes(snap: DatasetSnapshot, fps: float) -> tuple[list[Path], int]:
    """Which files/rows would change under a timestamp rebuild. Writes nothing.

    Dry-runs must go through this, never through the writer: the whole point
    of a dry-run is that disk state is untouched.
    """
    changed_files: list[Path] = []
    changed_rows = 0
    for f in _data_files(snap):
        df = pd.read_parquet(f)
        if "frame_index" not in df.columns or "timestamp" not in df.columns:
            continue
        _, diff = _timestamp_values(df, fps)
        if not diff.any():
            continue
        changed_files.append(f)
        changed_rows += int(diff.sum())
    return changed_files, changed_rows


@contextmanager
def _replacement_file(path: Path) -> Iterator[Path]:
    """Write and close the replacement before publishing it at the original path."""
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        replacement = Path(tmp.name)
    try:
        yield replacement
        if path.is_file():
            shutil.copymode(path, replacement)
        replacement.replace(path)
    finally:
        replacement.unlink(missing_ok=True)


def _stage_timestamps(fps: float, files: list[Path], pending: ExitStack) -> None:
    """Prepare corrected files for publication when the transaction stack closes."""
    for f in files:
        table = pq.read_table(f)
        fixed, _ = _timestamp_values(table.select(["frame_index", "timestamp"]).to_pandas(), fps)
        index = table.schema.get_field_index("timestamp")
        timestamp = table.schema.field(index)
        if not pa.types.is_floating(timestamp.type):
            timestamp = timestamp.with_type(pa.float64())
        table = table.set_column(index, timestamp, pa.array(fixed, type=timestamp.type))
        replacement = pending.enter_context(_replacement_file(f))
        pq.write_table(table, replacement)


def rebuild_timestamps(snap: DatasetSnapshot, fps: float) -> tuple[list[Path], int]:
    """Detect + write in one call. Only for callers that own the backup step.

    apply_clean does backup-then-write via detect + _stage_timestamps; this
    convenience wrapper exists for programmatic use where detection and the
    write happen in the same breath.
    """
    files, rows = detect_timestamp_changes(snap, fps)
    with ExitStack() as pending:
        _stage_timestamps(fps, files, pending)
    return files, rows


def apply_clean(snap: DatasetSnapshot, *, stats: bool, timestamps: bool) -> CleanReport:
    """Back up then rewrite. Backups land in <root>_pre_clean/ (outside the root)."""
    report = CleanReport()
    if not (stats or timestamps):
        return report
    fps = float((snap.info or {}).get("fps", 0)) if timestamps else 0.0
    files, rows = detect_timestamp_changes(snap, fps) if 0 < fps < np.inf else ([], 0)
    stats_path = _stats_path(snap)
    serialized = None
    if stats:
        fresh = recompute_stats(snap, timestamp_files=files, fps=fps)
        serialized = json.dumps(fresh, indent=2, allow_nan=False).encode()

    root = snap.root.resolve()
    backup = root.parent / f"{root.name}_pre_clean"
    backup.mkdir(parents=True, exist_ok=True)
    report.backup_dir = backup

    def _backup(p: Path) -> None:
        rel = p.relative_to(snap.root)
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            with _replacement_file(dst) as replacement:
                shutil.copy2(p, replacement)

    if serialized is not None and stats_path.is_file():
        _backup(stats_path)
    for path in files:
        _backup(path)
    # Prepare every output before replacing any originals. ExitStack publishes
    # in reverse order, so stats become visible only after all timestamps do.
    with ExitStack() as pending:
        if serialized is not None:
            replacement = pending.enter_context(_replacement_file(stats_path))
            replacement.write_bytes(serialized)
        _stage_timestamps(fps, files, pending)
    report.stats_rewritten = serialized is not None
    report.timestamp_files = files
    report.timestamp_rows = rows
    return report
