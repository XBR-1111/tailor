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

Expectation = Callable[[DatasetSnapshot, dict], CheckResult]


def _ok(name: str, message: str = "") -> CheckResult:
    return CheckResult(name, Status.PASS, message)


def _fail(name: str, message: str, **details) -> CheckResult:
    return CheckResult(name, Status.FAIL, message, details)


def _warn(name: str, message: str, **details) -> CheckResult:
    return CheckResult(name, Status.WARN, message, details)


def _skip(name: str, message: str) -> CheckResult:
    return CheckResult(name, Status.SKIP, message)


def check_info_json(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.info is None:
        return _fail("info_json", "meta/info.json missing or not valid JSON")
    return _ok("info_json")


def check_codebase_version(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.info is None:
        return _skip("codebase_version", "info.json unreadable")
    allowed = tuple(cfg.get("allowed", SUPPORTED_VERSIONS))
    version = snap.info.get("codebase_version")
    if version is None:
        return _fail("codebase_version", "codebase_version field missing")
    if version not in allowed:
        return _fail(
            "codebase_version",
            f"{version} not allowed by this profile (allowed: {', '.join(allowed)}); "
            "v3.x is on the roadmap",
            version=version,
        )
    return _ok("codebase_version", version)


def check_fps(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.info is None:
        return _skip("fps", "info.json unreadable")
    fps = snap.info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        return _fail("fps", f"fps must be a positive number, got {fps!r}")
    allowed = cfg.get("allowed")
    if allowed is not None and fps not in allowed:
        return _fail("fps", f"fps={fps} outside allowed set {sorted(allowed)}")
    if fps < 1:
        return _warn("fps", f"unusual fps={fps} (<1)")
    return _ok("fps", str(fps))


def check_episodes_metadata(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.episodes is None:
        return _fail("episodes_metadata", "meta/episodes.jsonl missing or not valid JSONL")
    if len(snap.episodes) == 0:
        return _fail("episodes_metadata", "episodes.jsonl is empty")
    return _ok("episodes_metadata", f"{len(snap.episodes)} episodes")


def check_episode_files_exist(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.episodes is None:
        return _skip("episode_files", "episode metadata unreadable")
    version = (snap.info or {}).get("codebase_version")
    if version == "v3.0":
        # v3 stores episodes back-to-back in chunk/file shards; episode meta
        # carries the shard pointer, so existence is checked per shard.
        missing_shards = set()
        for ep in snap.episodes:
            c, f = ep.get("data/chunk_index"), ep.get("data/file_index")
            if not isinstance(c, int) or not isinstance(f, int):
                missing_shards.add("unresolved shard pointer")
                continue
            shard = snap.root / "data" / f"chunk-{c:03d}" / f"file-{f:03d}.parquet"
            if not shard.is_file():
                missing_shards.add(str(shard.relative_to(snap.root)))
        if missing_shards:
            return _fail(
                "episode_files",
                f"{len(missing_shards)} data shard(s) missing",
                missing=sorted(missing_shards)[:20],
            )
        return _ok("episode_files")
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


def check_total_frames(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
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


def check_episode_lengths(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.episodes is None:
        return _skip("episode_lengths", "episodes.jsonl unreadable")
    bad = [ep.get("episode_index") for ep in snap.episodes if not ep.get("length", 0) > 0]
    if bad:
        return _fail("episode_lengths", f"{len(bad)} episode(s) with length<=0", bad=bad[:20])
    minimum = cfg.get("min", 2)
    short = [ep.get("episode_index") for ep in snap.episodes if 0 < ep.get("length", 0) < minimum]
    if short:
        return _warn(
            "episode_lengths",
            f"{len(short)} very short episode(s) (<{minimum} frames) will break chunked sampling",
            short=short[:20],
        )
    return _ok("episode_lengths")


def check_task_indices(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.episodes is None or snap.tasks is None:
        return _skip("task_indices", "episodes.jsonl or tasks.jsonl unreadable")
    total_tasks = len(snap.tasks)
    declared = snap.info.get("total_tasks") if snap.info else None
    bad = [
        ep.get("episode_index")
        for ep in snap.episodes
        if (ti := ep.get("task_index")) is not None and ti >= total_tasks
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


def check_state_action_dims(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
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


def check_nan_inf(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if not snap.frames:
        return _skip("nan_inf", "no readable frames")
    bad = []
    for idx, df in sorted(snap.frames.items()):
        for col in ("observation.state", "action"):
            if col not in df.columns:
                continue
            series = df[col].explode()

            # numpy scalars are not python float; coerce first so np.float32
            # values are judged by value, not by type
            def _finite(v) -> bool:
                try:
                    return math.isfinite(float(v))
                except (TypeError, ValueError):
                    return False

            n_bad = int(series.map(lambda v: not _finite(v)).sum())
            if n_bad:
                bad.append((idx, col, n_bad))
    if bad:
        return _fail(
            "nan_inf",
            "non-finite values found in state/action",
            bad=bad[:10],
        )
    return _ok("nan_inf")


def check_timestamps(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.info is None or not snap.frames:
        return _skip("timestamps", "no readable frames or fps unknown")
    fps = snap.info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        return _skip("timestamps", "fps invalid, cannot check spacing")
    expected_dt = 1.0 / fps
    tolerance = cfg.get("spacing_tolerance", 0.1)
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
        if any(abs(d - expected_dt) > tolerance * expected_dt for d in diffs):
            spacing_warns.append(idx)
    if problems:
        return _fail("timestamps", "timestamp problems", problems=problems[:10])
    if spacing_warns:
        return _warn(
            "timestamps",
            f"{len(spacing_warns)} episode(s) with frame spacing deviating"
            f" >{tolerance:.0%} from 1/fps",
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


def run_all(snap: DatasetSnapshot, profile: dict | None = None) -> list[CheckResult]:
    """Run every expectation; profile remaps severities per check."""
    profile = profile or {}
    results = []
    for name, fn in EXPECTATIONS.items():
        cfg = profile.get(name, {})
        r = fn(snap, cfg)
        target = cfg.get("severity")
        if target and r.status in (Status.WARN, Status.FAIL):
            r = CheckResult(r.name, Status(target), r.message, r.details)
        results.append(r)
    return results
