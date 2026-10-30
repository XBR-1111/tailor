"""Acceptance expectations for LeRobot v2.x datasets.

Each expectation is a pure function DatasetSnapshot -> CheckResult, registered
by name in EXPECTATIONS. Expectations degrade to SKIP when the inputs they
need were unreadable, so one broken file never hides other findings.

The v0.1 suite targets distributed datasets ("does the package match the
label"): format compliance, metadata-vs-data consistency, integrity.
"""

import math
from collections.abc import Callable

from tailor.loader import SUPPORTED_VERSIONS, _episode_parquet_path
from tailor.model import CheckResult, DatasetSnapshot, Status

Expectation = Callable[[DatasetSnapshot], CheckResult]


def _ok(name: str, message: str = "") -> CheckResult:
    return CheckResult(name, Status.PASS, message)


def _fail(name: str, message: str, **details) -> CheckResult:
    return CheckResult(name, Status.FAIL, message, details)


def _warn(name: str, message: str, **details) -> CheckResult:
    return CheckResult(name, Status.WARN, message, details)


def _skip(name: str, message: str) -> CheckResult:
    return CheckResult(name, Status.SKIP, message)


def check_info_json(snap: DatasetSnapshot) -> CheckResult:
    if snap.info is None:
        return _fail("info_json", "meta/info.json missing or not valid JSON")
    return _ok("info_json")


def check_codebase_version(snap: DatasetSnapshot) -> CheckResult:
    if snap.info is None:
        return _skip("codebase_version", "info.json unreadable")
    version = snap.info.get("codebase_version")
    if version is None:
        return _fail("codebase_version", "codebase_version field missing")
    if version not in SUPPORTED_VERSIONS:
        return _fail(
            "codebase_version",
            f"{version} not supported yet (supported: {', '.join(SUPPORTED_VERSIONS)}); "
            "v3.x is on the roadmap",
            version=version,
        )
    return _ok("codebase_version", version)


def check_fps(snap: DatasetSnapshot) -> CheckResult:
    if snap.info is None:
        return _skip("fps", "info.json unreadable")
    fps = snap.info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        return _fail("fps", f"fps must be a positive number, got {fps!r}")
    if fps < 1:
        return _warn("fps", f"unusual fps={fps} (<1)")
    return _ok("fps", str(fps))


def check_episodes_metadata(snap: DatasetSnapshot) -> CheckResult:
    if snap.episodes is None:
        return _fail("episodes_metadata", "meta/episodes.jsonl missing or not valid JSONL")
    if len(snap.episodes) == 0:
        return _fail("episodes_metadata", "episodes.jsonl is empty")
    return _ok("episodes_metadata", f"{len(snap.episodes)} episodes")


def check_episode_files_exist(snap: DatasetSnapshot) -> CheckResult:
    if snap.episodes is None:
        return _skip("episode_files", "episodes.jsonl unreadable")
    missing = [
        ep.get("episode_index")
        for ep in snap.episodes
        if not _episode_parquet_path(snap.root, ep.get("episode_index", -1)).is_file()
    ]
    if missing:
        return _fail(
            "episode_files",
            f"{len(missing)} episode parquet file(s) missing",
            missing=missing[:20],
        )
    return _ok("episode_files")


def check_total_frames(snap: DatasetSnapshot) -> CheckResult:
    if snap.info is None or snap.episodes is None:
        return _skip("total_frames", "info.json or episodes.jsonl unreadable")
    declared = snap.info.get("total_frames")
    actual = sum(ep.get("length", 0) for ep in snap.episodes)
    if not isinstance(declared, int) or declared != actual:
        return _fail(
            "total_frames",
            f"info.json declares total_frames={declared}, sum of episode lengths={actual}",
            declared=declared,
            actual=actual,
        )
    return _ok("total_frames", str(actual))


def check_episode_lengths(snap: DatasetSnapshot) -> CheckResult:
    if snap.episodes is None:
        return _skip("episode_lengths", "episodes.jsonl unreadable")
    bad = [ep.get("episode_index") for ep in snap.episodes if not ep.get("length", 0) > 0]
    if bad:
        return _fail("episode_lengths", f"{len(bad)} episode(s) with length<=0", bad=bad[:20])
    short = [ep.get("episode_index") for ep in snap.episodes if 0 < ep.get("length", 0) < 2]
    if short:
        return _warn(
            "episode_lengths",
            f"{len(short)} very short episode(s) (<2 frames) will break chunked sampling",
            short=short[:20],
        )
    return _ok("episode_lengths")


def check_task_indices(snap: DatasetSnapshot) -> CheckResult:
    if snap.episodes is None or snap.tasks is None:
        return _skip("task_indices", "episodes.jsonl or tasks.jsonl unreadable")
    total_tasks = len(snap.tasks)
    declared = snap.info.get("total_tasks") if snap.info else None
    bad = [
        ep.get("episode_index") for ep in snap.episodes if ep.get("task_index", 0) >= total_tasks
    ]
    if bad:
        return _fail(
            "task_indices",
            f"{len(bad)} episode(s) reference task_index >= {total_tasks}",
            bad=bad[:20],
        )
    if isinstance(declared, int) and declared != total_tasks:
        return _fail(
            "task_indices",
            f"info.json declares total_tasks={declared}, tasks.jsonl has {total_tasks} rows",
            declared=declared,
            actual=total_tasks,
        )
    return _ok("task_indices", f"{total_tasks} tasks")


def _feature_dim(info: dict, name: str) -> int | None:
    feat = info.get("features", {}).get(name, {})
    shape = feat.get("shape")
    if isinstance(shape, list) and shape:
        return int(shape[-1])
    return None


def check_state_action_dims(snap: DatasetSnapshot) -> CheckResult:
    if snap.info is None or not snap.frames:
        return _skip("state_action_dims", "no readable frames to compare against info.json")
    problems = []
    for idx, df in sorted(snap.frames.items()):
        for col, key in (("observation.state", "state"), ("action", "action")):
            declared = _feature_dim(snap.info, key)
            if col not in df.columns:
                problems.append((idx, col, "column missing"))
                continue
            val = df[col].iloc[0]
            dim = len(val) if hasattr(val, "__len__") else 1
            if declared is not None and dim != declared:
                problems.append((idx, col, f"dim {dim} != declared {declared}"))
    if problems:
        return _fail(
            "state_action_dims",
            f"{len(problems)} dim/column mismatch(es) vs info.json features",
            problems=problems[:10],
        )
    return _ok("state_action_dims")


def check_nan_inf(snap: DatasetSnapshot) -> CheckResult:
    if not snap.frames:
        return _skip("nan_inf", "no readable frames")
    bad = []
    for idx, df in sorted(snap.frames.items()):
        for col in ("observation.state", "action"):
            if col not in df.columns:
                continue
            series = df[col].explode()
            n_bad = int(
                series.map(lambda v: not isinstance(v, float) or not math.isfinite(v)).sum()
            )
            if n_bad:
                bad.append((idx, col, n_bad))
    if bad:
        return _fail(
            "nan_inf",
            "non-finite values found in state/action",
            bad=bad[:10],
        )
    return _ok("nan_inf")


def check_timestamps(snap: DatasetSnapshot) -> CheckResult:
    if snap.info is None or not snap.frames:
        return _skip("timestamps", "no readable frames or fps unknown")
    fps = snap.info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        return _skip("timestamps", "fps invalid, cannot check spacing")
    expected_dt = 1.0 / fps
    problems = []
    spacing_warns = []
    for idx, df in sorted(snap.frames.items()):
        if "timestamp" not in df.columns:
            problems.append((idx, "timestamp column missing"))
            continue
        ts = df["timestamp"].to_list()
        diffs = [b - a for a, b in zip(ts, ts[1:], strict=False)]
        if any(d <= 0 for d in diffs):
            problems.append((idx, "non-monotonic timestamps"))
            continue
        if any(abs(d - expected_dt) > 0.1 * expected_dt for d in diffs):
            spacing_warns.append(idx)
    if problems:
        return _fail("timestamps", "timestamp problems", problems=problems[:10])
    if spacing_warns:
        return _warn(
            "timestamps",
            f"{len(spacing_warns)} episode(s) with frame spacing deviating >10% from 1/fps",
            episodes=spacing_warns[:10],
        )
    return _ok("timestamps")


# Registry order is the report order.
EXPECTATIONS: dict[str, Expectation] = {
    f.__name__.removeprefix("check_"): f
    for f in (
        check_info_json,
        check_codebase_version,
        check_fps,
        check_episodes_metadata,
        check_episode_files_exist,
        check_total_frames,
        check_episode_lengths,
        check_task_indices,
        check_state_action_dims,
        check_nan_inf,
        check_timestamps,
    )
}


def run_all(snap: DatasetSnapshot) -> list[CheckResult]:
    return [fn(snap) for fn in EXPECTATIONS.values()]
