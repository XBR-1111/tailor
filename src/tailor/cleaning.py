"""Deterministic repairs (the clean stage): stats recompute + timestamp rebuild.

Everything is metadata-level: parquet timestamps and meta/stats.json. No
lerobot dependency — the CLI runs a dry-run by default and only writes with
--apply, backing up every replaced file outside the dataset root so
portability checks stay clean.
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tailor.model import DatasetSnapshot

_NUMERIC_COLS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "index",
    "episode_index",
    "task_index",
)


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


def recompute_stats(snap: DatasetSnapshot) -> dict:
    """Fresh stats over every numeric parquet column; video keys preserved
    from the previous stats.json (pixel stats need full decode — out of scope)."""
    old = {}
    p = _stats_path(snap)
    if p.is_file():
        old = json.loads(p.read_text())
    video_keys = [
        k for k, f in (snap.info or {}).get("features", {}).items() if f.get("dtype") == "video"
    ]
    out: dict = {k: old[k] for k in video_keys if k in old}
    for col in _NUMERIC_COLS:
        rows = [
            np.asarray(v, dtype=float)
            for f in _data_files(snap)
            for df in [pd.read_parquet(f)]
            if col in df.columns
            for v in df[col]
        ]
        if not rows:
            continue
        arr = np.stack(rows)
        dims = 1 if arr.ndim == 1 else arr.shape[1]
        out[col] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])] * dims,
        }
    return out


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
        new_ts = df["frame_index"].astype(float) / fps
        diff = (df["timestamp"].astype(float) - new_ts).abs() > 1e-6
        if not diff.any():
            continue
        changed_files.append(f)
        changed_rows += int(diff.sum())
    return changed_files, changed_rows


def _rewrite_timestamps(snap: DatasetSnapshot, fps: float, files: list[Path]) -> None:
    """Write frame_index/fps timestamps into the given files (already detected).

    Callers must back the files up BEFORE this — after the write the on-disk
    original is gone.
    """
    for f in files:
        df = pd.read_parquet(f)
        df["timestamp"] = df["frame_index"].astype(float) / fps
        df.to_parquet(f, index=False)
    if files:
        # loader caches v2 episode frames; the rewrite must invalidate it or
        # same-process re-checks would judge stale data
        from tailor.loader import _read_v2_episode_frames

        _read_v2_episode_frames.cache_clear()


def rebuild_timestamps(snap: DatasetSnapshot, fps: float) -> tuple[list[Path], int]:
    """Detect + write in one call. Only for callers that own the backup step.

    apply_clean does backup-then-write via detect + _rewrite_timestamps; this
    convenience wrapper exists for programmatic use where detection and the
    write happen in the same breath.
    """
    files, rows = detect_timestamp_changes(snap, fps)
    _rewrite_timestamps(snap, fps, files)
    return files, rows


def apply_clean(snap: DatasetSnapshot, *, stats: bool, timestamps: bool) -> CleanReport:
    """Back up then rewrite. Backups land in <root>_pre_clean/ (outside the root)."""
    report = CleanReport()
    if not (stats or timestamps):
        return report
    backup = snap.root.parent / f"{snap.root.name}_pre_clean"
    backup.mkdir(parents=True, exist_ok=True)
    report.backup_dir = backup

    def _backup(p: Path) -> None:
        rel = p.relative_to(snap.root)
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(p, dst)

    if stats:
        p = _stats_path(snap)
        _backup(p)
        p.write_text(json.dumps(recompute_stats(snap), indent=2))
        report.stats_rewritten = True
    if timestamps:
        fps = float((snap.info or {}).get("fps", 0))
        if fps > 0:
            # detect first, back up the ORIGINALS, then write — writing before
            # the backup would archive the already-rewritten data
            files, rows = detect_timestamp_changes(snap, fps)
            for f in files:
                _backup(f)
            _rewrite_timestamps(snap, fps, files)
            report.timestamp_files = files
            report.timestamp_rows = rows
    return report
