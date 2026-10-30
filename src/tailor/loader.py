"""Defensive reader for LeRobot v2.x dataset metadata.

Any structural problem (missing file, bad JSON) leaves the corresponding field
None so expectations report a clean FAIL instead of an exception. Parquet frames
are read lazily per episode; a broken parquet marks that episode and moves on.
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from tailor.model import DatasetSnapshot

SUPPORTED_VERSIONS = ("v2.0", "v2.1")


def load_snapshot(root: Path, max_episodes: int | None = None) -> DatasetSnapshot:
    """Read info.json / episodes.jsonl / tasks.jsonl and sample parquet frames."""
    snap = DatasetSnapshot(root=root)
    meta = root / "meta"

    info_path = meta / "info.json"
    if info_path.is_file():
        try:
            snap.info = json.loads(info_path.read_text())
        except json.JSONDecodeError:
            snap.info = None
    else:
        snap.info = None

    snap.episodes = _read_jsonl(meta / "episodes.jsonl")
    snap.tasks = _read_jsonl(meta / "tasks.jsonl")

    if snap.episodes is not None:
        episodes = snap.episodes if max_episodes is None else snap.episodes[:max_episodes]
        for ep in episodes:
            idx = ep.get("episode_index")
            df = _read_episode_frames(str(root), idx)
            if df is not None:
                snap.frames[idx] = df
    return snap


def _read_jsonl(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    rows = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except json.JSONDecodeError:
        return None
    return rows


def _episode_parquet_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


@lru_cache(maxsize=128)
def _read_episode_frames(root_str: str, episode_index: int | None) -> pd.DataFrame | None:
    """Read one episode's parquet; None means unreadable (corrupt or missing)."""
    if episode_index is None:
        return None
    path = _episode_parquet_path(Path(root_str), episode_index)
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None
