"""Defensive reader for LeRobot v2.x and v3.0 dataset metadata.

Any structural problem (missing file, bad JSON) leaves the corresponding field
None so expectations report a clean FAIL instead of an exception.

v2.x and v3.0 differ on disk but are normalized here into one shape:
episodes -> [{episode_index, length, task_index, ...}], tasks -> [{task_index}],
frames -> {episode_index: DataFrame}. Expectations never branch on version.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import cast

import pandas as pd

from tailor.model import DatasetSnapshot

SUPPORTED_VERSIONS = ("v2.0", "v2.1", "v3.0")


def load_snapshot(
    root: Path, max_episodes: int | None = None, sample: str = "first"
) -> DatasetSnapshot:
    """Read metadata and sample per-episode frames for a LeRobot dataset.

    sample picks WHICH N episodes --max-episodes keeps: "first" (default,
    cheap convenience) or "last" (tail-end episodes are a common corruption
    site — the real-dataset defect this suite found first lived in the last
    episode).
    """
    if sample not in ("first", "last"):
        raise ValueError(f"sample must be 'first' or 'last', got {sample!r}")
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

    version = (snap.info or {}).get("codebase_version")
    if version == "v3.0":
        snap.episodes = _load_v3_episodes(meta)
        snap.tasks = _load_v3_tasks(meta)
    else:
        snap.episodes = _read_jsonl(meta / "episodes.jsonl")
        snap.tasks = _read_jsonl(meta / "tasks.jsonl")

    if snap.episodes:
        if max_episodes is None:
            episodes = snap.episodes
        elif sample == "last":
            episodes = snap.episodes[-max_episodes:]
        else:
            episodes = snap.episodes[:max_episodes]
        if version == "v3.0":
            # v3 frames live in shared chunk files; group once per file.
            # v0.1 reads whole chunk files — fine at this scale; slicing via
            # dataset_from_index is the follow-up for huge datasets.
            seen = {ep.get("episode_index") for ep in episodes}
            for f in sorted((root / "data").glob("chunk-*/file-*.parquet")):
                try:
                    df = pd.read_parquet(f)
                except Exception:
                    continue
                for raw_idx, g in df.groupby("episode_index"):
                    # groupby keys are typed as a wide pandas union; by
                    # construction episode_index is an integer column
                    idx = int(cast("int", raw_idx))
                    if idx in seen and idx not in snap.frames:
                        snap.frames[idx] = g
        else:
            # v2: one parquet per episode — reading them in parallel pays off
            # from a few dozen episodes up (pyarrow releases the GIL on read)
            import os
            from concurrent.futures import ThreadPoolExecutor

            wanted = []
            for ep in episodes:
                raw = ep.get("episode_index")
                if raw is not None:
                    wanted.append(int(raw))
            workers = min(4, os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                frames = pool.map(lambda i: (i, _read_v2_episode_frames(str(root), i)), wanted)
                for idx, ep_df in frames:
                    if ep_df is not None:
                        snap.frames[idx] = ep_df
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


def _load_v3_episodes(meta: Path) -> list[dict] | None:
    files = sorted(meta.glob("episodes/chunk-*/file-*.parquet"))
    if not files:
        return None
    rows = []
    try:
        for f in files:
            for r in pd.read_parquet(f).to_dict("records"):
                tasks = r.get("tasks") or []
                # v3 stores task TEXT per episode; only integer index lists
                # yield a task_index — text lists leave None and the range
                # check degrades to the total_tasks consistency check.
                if hasattr(tasks, "tolist"):
                    tasks = tasks.tolist()
                r["task_index"] = (
                    min(tasks)
                    if isinstance(tasks, list) and tasks and all(isinstance(t, int) for t in tasks)
                    else None
                )
                rows.append(r)
    except Exception:
        return None
    rows.sort(key=lambda r: r.get("episode_index", -1))
    return rows


def _load_v3_tasks(meta: Path) -> list[dict] | None:
    path = meta / "tasks.parquet"
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path).to_dict("records")
    except Exception:
        return None


def _episode_parquet_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


@lru_cache(maxsize=128)
def _read_v2_episode_frames(root_str: str, episode_index: int | None) -> pd.DataFrame | None:
    """Read one v2 episode parquet; None means unreadable (corrupt or missing)."""
    if episode_index is None:
        return None
    path = _episode_parquet_path(Path(root_str), episode_index)
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None
