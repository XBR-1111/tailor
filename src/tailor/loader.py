"""Defensive reader for LeRobot v2.x and v3.0 dataset metadata.

Any structural problem (missing file, bad JSON) leaves the corresponding field
None so expectations report a clean FAIL instead of an exception.

v2.x and v3.0 differ on disk but are normalized here into one shape:
episodes -> [{episode_index, length, task_index, ...}], tasks -> [{task_index}],
frames -> {episode_index: DataFrame}. Expectations never branch on version.
"""

import json
import logging
from pathlib import Path
from typing import cast

import pandas as pd

from tailor.model import DatasetSnapshot

logger = logging.getLogger(__name__)

SUPPORTED_VERSIONS = ("v2.0", "v2.1", "v3.0")


def load_snapshot(
    root: Path,
    max_episodes: int | None = None,
    sample: str = "first",
    *,
    metadata_only: bool = False,
) -> DatasetSnapshot:
    """Read metadata and sample per-episode frames for a LeRobot dataset.

    sample picks WHICH N episodes --max-episodes keeps: "first" (default,
    cheap convenience) or "last" (tail-end episodes are a common corruption
    site — the real-dataset defect this suite found first lived in the last
    episode).
    """
    if sample not in ("first", "last"):
        raise ValueError(f"sample must be 'first' or 'last', got {sample!r}")
    snap = DatasetSnapshot(root=root, metadata_only=metadata_only)
    meta = root / "meta"

    info_path = meta / "info.json"
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text())
            if isinstance(info, dict):
                snap.info = info
        except (OSError, ValueError):
            pass

    version = (snap.info or {}).get("codebase_version")
    if version == "v3.0":
        snap.episodes = _load_v3_episodes(meta)
        snap.tasks = _load_v3_tasks(meta)
    else:
        snap.episodes = _read_jsonl(meta / "episodes.jsonl")
        snap.tasks = _read_jsonl(meta / "tasks.jsonl")

    if snap.episodes and not metadata_only:
        if max_episodes is None:
            episodes = snap.episodes
        elif sample == "last":
            episodes = snap.episodes[-max_episodes:]
        else:
            episodes = snap.episodes[:max_episodes]
        if version == "v3.0":
            _load_v3_frames(snap, episodes)
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
                frames = pool.map(lambda i: (i, _read_v2_episode_frames(root, i)), wanted)
                for idx, ep_df in frames:
                    if ep_df is not None:
                        snap.frames[idx] = ep_df
                    else:
                        path = _episode_parquet_path(root, idx)
                        if path.is_file():
                            snap.frame_errors[str(path.relative_to(root))] = "unreadable Parquet"
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
    for f in files:
        try:
            for r in pd.read_parquet(f).to_dict("records"):
                tasks = r.get("tasks")
                if tasks is None:
                    tasks = []
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
        except Exception as exc:
            # swallowing is deliberate (defensive loader contract: unreadable
            # metadata becomes a clean FAIL), but never silently — name the
            # file so an "unreadable" verdict can be traced to its cause
            logger.warning(
                "v3 episodes parquet unreadable, ignoring %s (%s: %s)", f, type(exc).__name__, exc
            )
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


def _read_v2_episode_frames(root: Path, episode_index: int) -> pd.DataFrame | None:
    """Read one v2 episode parquet; None means unreadable (corrupt or missing)."""
    try:
        return pd.read_parquet(_episode_parquet_path(root, episode_index))
    except Exception:
        return None


def _v3_episode_shard(root: Path, episode: dict) -> Path | None:
    chunk, file = episode.get("data/chunk_index"), episode.get("data/file_index")
    if not isinstance(chunk, int) or not isinstance(file, int):
        return None
    return root / "data" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"


def _load_v3_frames(snap: DatasetSnapshot, episodes: list[dict]) -> None:
    # LeRobot resolves each episode's declared shard. Looking for its rows in
    # other shards would hide a bad pointer from acceptance.
    wanted_by_shard: dict[Path, set[int]] = {}
    for ep in episodes:
        path = _v3_episode_shard(snap.root, ep)
        idx = ep.get("episode_index")
        if path is not None and isinstance(idx, int):
            wanted_by_shard.setdefault(path, set()).add(idx)
    for path, wanted in sorted(wanted_by_shard.items()):
        if not path.is_file():
            continue  # the file-existence check names missing shards
        relative = str(path.relative_to(snap.root))
        try:
            df = pd.read_parquet(path)
        except Exception:
            snap.frame_errors[relative] = "unreadable Parquet"
            continue
        if "episode_index" not in df.columns:
            snap.frame_errors[relative] = "episode_index column missing"
            continue
        if not pd.api.types.is_integer_dtype(df["episode_index"]):
            snap.frame_errors[relative] = "episode_index column must contain integers"
            continue
        found = set()
        for key, rows in df.groupby("episode_index"):
            if key in wanted:
                idx = int(cast("int", key))
                snap.frames[idx] = rows
                found.add(idx)
        if missing := sorted(wanted - found):
            snap.frame_errors[relative] = (
                f"missing rows for {len(missing)} declared episode(s): {missing[:10]}"
            )
