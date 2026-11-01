"""Acceptance expectations for LeRobot v2.x datasets.

Each expectation is a pure function DatasetSnapshot -> CheckResult, registered
by name in EXPECTATIONS. Expectations degrade to SKIP when the inputs they
need were unreadable, so one broken file never hides other findings.

The v0.1 suite targets distributed datasets ("does the package match the
label"): format compliance, metadata-vs-data consistency, integrity.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tailor.loader import SUPPORTED_VERSIONS, _episode_parquet_path
from tailor.model import CheckResult, DatasetSnapshot, Status

Expectation = Callable[[DatasetSnapshot, dict], CheckResult]


def _ok(name: str, message: str = "") -> CheckResult:
    return CheckResult(name, Status.PASS, message)


def _fail(name: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(name, Status.FAIL, message, details)


def _warn(name: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(name, Status.WARN, message, details)


def _skip(name: str, message: str) -> CheckResult:
    return CheckResult(name, Status.SKIP, message)


def check_info_json(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.info is None:
        return _fail("info_json", "meta/info.json missing or not valid JSON")
    return _ok("info_json")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


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
            f"{version} not allowed by this profile (allowed: {', '.join(allowed)})",
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
    if not (snap.root / "data").is_dir():
        return _skip("episode_files", "data/ not on disk (meta-only run? use --full)")
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
        return _fail(
            "episode_lengths",
            f"{len(bad)} episode(s) with length<=0",
            bad=bad[:20],
            episodes=bad[:20],
        )
    minimum = cfg.get("min", 2)
    short = [ep.get("episode_index") for ep in snap.episodes if 0 < ep.get("length", 0) < minimum]
    if short:
        return _warn(
            "episode_lengths",
            f"{len(short)} very short episode(s) (<{minimum} frames) will break chunked sampling",
            short=short[:20],
            episodes=short[:20],
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
    return _ok("task_indices", _plural(total_tasks, "task"))


def _feature_dim(info: dict, name: str) -> int | None:
    feat = info.get("features", {}).get(name, {})
    shape = feat.get("shape")
    if isinstance(shape, list) and shape:
        return int(shape[-1])
    return None


def _longest_true_run(mask: np.ndarray) -> int:
    """Longest run of consecutive True values in a 1-D boolean array."""
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    if edges.size < 2:  # all True
        return int(mask.size)
    return int((edges[1::2] - edges[0::2]).max())


def check_state_action_dims(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.info is None or not snap.frames:
        return _skip("state_action_dims", "no readable frames to compare against info.json")
    problems = []
    for idx in sorted(snap.frames):
        df = snap.frames[idx]
        if len(df) == 0:
            # a readable-but-empty episode parquet: no rows to infer dims,
            # and an empty episode is itself a defect worth naming
            problems.append((idx, "-", "empty episode file (0 rows)"))
            continue
        for col, key in (("observation.state", "state"), ("action", "action")):
            declared = _feature_dim(snap.info, key)
            if col not in df.columns:
                problems.append((idx, col, "column missing"))
                continue
            # dims from the first row only — converting the whole column
            # here would cost a full np.stack for a number we can read off
            # one element (col_array stays for the checks that need values)
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
    for idx in sorted(snap.frames):
        for col in ("observation.state", "action"):
            arr = snap.col_array(idx, col)
            if arr is None:
                continue
            n_bad = int((~np.isfinite(arr)).sum())
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
    for idx in sorted(snap.frames):
        df = snap.frames[idx]
        if "timestamp" not in df.columns:
            problems.append((idx, "timestamp column missing"))
            continue
        ts = np.asarray(df["timestamp"], dtype=float)
        if ts.size < 2:
            continue
        diffs = np.diff(ts)
        if (diffs <= 0).any():
            problems.append((idx, "non-monotonic timestamps"))
            continue
        if (np.abs(diffs - expected_dt) > tolerance * expected_dt).any():
            spacing_warns.append(idx)
    if problems:
        return _fail(
            "timestamps",
            "timestamp problems",
            problems=problems[:10],
            episodes=sorted({p[0] for p in problems if isinstance(p[0], int)}),
        )
    if spacing_warns:
        return _warn(
            "timestamps",
            f"{len(spacing_warns)} episode(s) with frame spacing deviating"
            f" >{tolerance:.0%} from 1/fps",
            episodes=spacing_warns[:10],
        )
    return _ok("timestamps")


def _declared_stats(snap: DatasetSnapshot) -> dict | None:
    """meta/stats.json if present (both v2.x and v3.0 keep it there)."""
    path = snap.root / "meta" / "stats.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def check_stats_recompute(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if not snap.frames:
        return _skip("stats_recompute", "no readable frames")
    declared = _declared_stats(snap)
    if declared is None:
        return _skip("stats_recompute", "meta/stats.json missing (informational)")
    mean_tol = cfg.get("mean_tolerance", 0.05)
    from tailor.stats import RunningStats

    drift_report = []
    for col in ("observation.state", "action"):
        stats = declared.get(col)
        if not stats or col not in next(iter(snap.frames.values())).columns:
            continue
        running = RunningStats()
        for idx in snap.frames:
            arr = snap.col_array(idx, col)
            if arr is not None:
                running.update(arr)  # O(one episode) memory, never concatenates
        for stat_name, actual in (
            ("mean", running.mean),
            ("std", running.std),
        ):
            decl = np.asarray(stats[stat_name], dtype=float)
            scale = np.maximum(np.abs(decl), 1e-9)
            drift = float((np.abs(decl - actual) / scale).max())
            if drift > mean_tol:
                drift_report.append((col, stat_name, round(drift, 4)))
    if drift_report:
        return _fail(
            "stats_recompute",
            f"declared stats drift beyond {mean_tol:.0%} of scale (recompute mismatch)",
            drift=drift_report[:10],
        )
    return _ok("stats_recompute")


def check_action_quality(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if not snap.frames:
        return _skip("action_quality", "no readable frames")
    frozen_frames = cfg.get("frozen_frames", 10)
    clip_run = cfg.get("clip_run", 5)
    jump_sigma = cfg.get("jump_sigma", 8)
    jump_ratio_max = cfg.get("jump_ratio", 0.01)

    # one read of stats.json for the whole check, not one per episode
    action_stats = (_declared_stats(snap) or {}).get("action", {})
    range_edges: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if "min" in action_stats and "max" in action_stats:
        lo = np.asarray(action_stats["min"], dtype=float)
        hi = np.asarray(action_stats["max"], dtype=float)
        range_edges = (lo, hi, np.maximum(hi - lo, 1e-9))

    frozen_eps: list[int] = []
    frozen_best: dict[int, int] = {}
    clipped_eps: list[int] = []
    jump_eps: list[int] = []
    for idx in sorted(snap.frames):
        actions = snap.col_array(idx, "action")
        if actions is None or len(actions) < 2:
            continue
        # frozen: longest run of identical consecutive action vectors
        same = (np.diff(actions, axis=0) == 0).all(axis=1)
        longest = _longest_true_run(same)
        if longest >= frozen_frames:
            frozen_eps.append(idx)
            frozen_best[idx] = longest
        # clipped: saturation shows as one dim STAYING at the declared range
        # edge, not as any single frame near it (smooth trajectories touch
        # edges naturally). Longest per-dim run within 1% of the declared
        # range (stats.json) must stay under clip_run frames. In-sample
        # min/max would flag extreme frames by construction, so no declared
        # stats means this sub-check is skipped.
        if range_edges is not None:
            lo, hi, span = range_edges
            at_edge = (actions <= lo + 0.01 * span) | (actions >= hi - 0.01 * span)
            for dim in range(at_edge.shape[1]):
                if _longest_true_run(at_edge[:, dim]) >= clip_run:
                    clipped_eps.append(idx)
                    break
        # jumps: fraction of frame-to-frame deltas beyond k sigma
        deltas = np.abs(np.diff(actions, axis=0))
        sigma = np.maximum(actions.std(axis=0), 1e-9)
        if (deltas > jump_sigma * sigma).any(axis=1).mean() > jump_ratio_max:
            jump_eps.append(idx)

    findings: list[str] = []
    if frozen_eps:
        findings.append(
            f"frozen runs in {len(frozen_eps)} episode(s) "
            f"(worst: ep{max(frozen_eps, key=lambda e: frozen_best[e])}, "
            f"{max(frozen_best.values())} identical actions)"
        )
    if clipped_eps:
        findings.append(f"range-edge pegging in {len(clipped_eps)} episode(s)")
    if jump_eps:
        findings.append(f"sudden jumps beyond {jump_sigma} sigma in {len(jump_eps)} episode(s)")
    if findings:
        episodes = sorted(set(frozen_eps) | set(clipped_eps) | set(jump_eps))
        return _warn(
            "action_quality",
            "; ".join(findings),
            findings=findings,
            # 'episodes' drives --from-check: these are the episodes whose
            # removal eliminates the finding
            episodes=episodes,
        )
    return _ok("action_quality")


def _video_keys(info: dict) -> list[str]:
    return [key for key, feat in info.get("features", {}).items() if feat.get("dtype") == "video"]


def check_video_files(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if not (snap.root / "videos").is_dir():
        return _skip("video_files", "videos/ not on disk (meta-only run? use --full)")
    if snap.info is None or snap.episodes is None:
        return _skip("video_files", "info.json or episode metadata unreadable")
    keys = _video_keys(snap.info)
    if not keys:
        return _ok("video_files", "no video streams declared")
    version = (snap.info or {}).get("codebase_version")
    missing = set()
    for ep in snap.episodes:
        for key in keys:
            if version == "v3.0":
                c = ep.get(f"videos/{key}/chunk_index")
                f = ep.get(f"videos/{key}/file_index")
                if not isinstance(c, int) or not isinstance(f, int):
                    missing.add(f"{key}: unresolved shard pointer")
                    continue
                path = snap.root / "videos" / key / f"chunk-{c:03d}" / f"file-{f:03d}.mp4"
            else:
                idx = ep.get("episode_index", -1)
                path = (
                    snap.root
                    / "videos"
                    / key
                    / f"chunk-{idx // 1000:03d}"
                    / f"episode_{idx:06d}.mp4"
                )
            if not path.is_file():
                missing.add(f"{key}: {path.relative_to(snap.root)}")
    if missing:
        return _fail(
            "video_files", f"{len(missing)} video file(s) missing", missing=sorted(missing)[:10]
        )
    return _ok("video_files", f"{len(keys)} stream(s)")


def check_video_streams(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if not (snap.root / "videos").is_dir():
        return _skip("video_streams", "videos/ not on disk (meta-only run? use --full)")
    if snap.info is None or snap.episodes is None:
        return _skip("video_streams", "info.json or episode metadata unreadable")
    keys = _video_keys(snap.info)
    if not keys:
        return _ok("video_streams", "no video streams declared")
    try:
        import av
    except ImportError:
        return _skip("video_streams", "PyAV not installed (pip install av)")
    tolerance = cfg.get("frame_tolerance", 1)
    version = (snap.info or {}).get("codebase_version")

    def _probe(path: Path) -> tuple[int, int, int]:
        # packet count == frame count only for CFR streams without B-frames;
        # B-frame encodes may need decoded-frame counting (frame_tolerance
        # absorbs the usual off-by-a-couple, not systematic differences).
        # Container nb_frames metadata is trusted when present (MP4/H.264
        # carries it): reading the header beats demuxing every packet.
        container = av.open(str(path))
        stream = container.streams.video[0]
        nb = int(getattr(stream, "frames", 0) or 0)
        frames = nb if nb > 0 else sum(1 for p in container.demux(video=0) if p.dts is not None)
        w, h = stream.codec_context.width, stream.codec_context.height
        container.close()
        return frames, w, h

    # collect probe jobs first, run them in parallel (av releases the GIL),
    # then judge — video probing dominates wall time on video-heavy datasets
    jobs: list[tuple[str, str, Path, int]] = []  # (key, label, path, expected)
    for key in keys:
        shape = snap.info["features"][key].get("shape", [])
        if version == "v3.0":
            # episodes share shards: expected frames = sum of lengths pointing
            # at the same (chunk, file)
            shards: dict[tuple, list] = {}
            for ep in snap.episodes:
                c, f = ep.get(f"videos/{key}/chunk_index"), ep.get(f"videos/{key}/file_index")
                if isinstance(c, int) and isinstance(f, int):
                    shards.setdefault((c, f), []).append(ep.get("length", 0))
            for (c, f), lengths in shards.items():
                path = snap.root / "videos" / key / f"chunk-{c:03d}" / f"file-{f:03d}.mp4"
                if not path.is_file():
                    continue  # already reported by video_files
                jobs.append((key, path.name, path, sum(lengths)))
        else:
            for ep in snap.episodes:
                idx = ep.get("episode_index", -1)
                path = (
                    snap.root
                    / "videos"
                    / key
                    / f"chunk-{idx // 1000:03d}"
                    / f"episode_{idx:06d}.mp4"
                )
                if not path.is_file():
                    continue
                jobs.append((key, f"ep {idx}", path, ep.get("length", 0)))

    problems: list[str] = []
    import os
    from concurrent.futures import ThreadPoolExecutor

    workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        probed = list(pool.map(lambda j: _safe_probe(_probe, j[2]), jobs))
    for (key, label, _path, expected), result in zip(jobs, probed, strict=True):
        shape = snap.info["features"][key].get("shape", [])
        if isinstance(result, Exception):
            problems.append(f"{key}: {label} video undecodable")
            continue
        frames, w, h = result
        if abs(frames - expected) > tolerance:
            problems.append(f"{key} {label}: {frames} frames vs {expected} episode frames")
        if len(shape) >= 2 and (h, w) != (shape[0], shape[1]):
            problems.append(f"{key} {label}: {w}x{h} vs declared {shape[1]}x{shape[0]}")
    if problems:
        return _fail("video_streams", f"{len(problems)} stream problem(s)", problems=problems[:10])
    return _ok("video_streams", f"{len(keys)} stream(s) probed")


def _safe_probe(
    probe: "Callable[[Path], tuple[int, int, int]]", path: Path
) -> tuple[int, int, int] | Exception:
    try:
        return probe(path)
    except Exception as e:  # undecodable container must fail the check, not the run
        return e


def check_zero_variance(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if not snap.frames:
        return _skip("zero_variance", "no readable frames")
    eps = cfg.get("eps", 1e-6)
    flat = []
    for idx in sorted(snap.frames):
        for col in ("observation.state", "action"):
            arr = snap.col_array(idx, col)
            if arr is None:
                continue
            dead = np.where(arr.std(axis=0) < eps)[0].tolist()
            if dead:
                for dim in dead:
                    flat.append((idx, col, dim))
    if flat:
        return _warn(
            "zero_variance",
            f"{len(flat)} dim(s) never move (std < {eps:g}): normalization would divide by zero",
            dims=flat[:10],
            episodes=sorted({t[0] for t in flat}),
        )
    return _ok("zero_variance")


def check_schema_consistency(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if len(snap.frames) < 2:
        return _skip("schema_consistency", "need >=2 readable episodes to compare")
    problems = []
    ref_idx = min(snap.frames)
    ref = snap.frames[ref_idx]
    ref_cols = list(ref.columns)
    ref_dtypes = {c: str(ref[c].dtype) for c in ref.columns}
    for idx, df in sorted(snap.frames.items()):
        if idx == ref_idx:
            continue
        if list(df.columns) != ref_cols:
            missing = [c for c in ref_cols if c not in df.columns]
            extra = [c for c in df.columns if c not in ref_cols]
            problems.append((idx, "columns", f"missing={missing[:3]} extra={extra[:3]}"))
            continue
        for col, ref_dt in ref_dtypes.items():
            if str(df[col].dtype) != ref_dt:
                problems.append((idx, col, f"{df[col].dtype} vs {ref_dt} in ep {ref_idx}"))
    if problems:
        return _fail(
            "schema_consistency",
            f"{len(problems)} schema drift(s) across episodes",
            problems=problems[:10],
        )
    return _ok("schema_consistency")


def check_frame_continuity(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if not snap.frames:
        return _skip("frame_continuity", "no readable frames")
    problems = []
    declared = {ep.get("episode_index"): ep.get("length") for ep in (snap.episodes or [])}
    for idx, df in sorted(snap.frames.items()):
        n = len(df)
        if idx in declared and declared[idx] is not None and n != declared[idx]:
            problems.append((idx, f"{n} rows vs declared length {declared[idx]}"))
            continue
        if "frame_index" in df.columns:
            fi = np.asarray(df["frame_index"])
            # holes = any position where frame_index deviates from 0..n-1
            if fi.size and (fi != np.arange(fi.size)).any():
                first_hole = int(np.flatnonzero(fi != np.arange(fi.size))[0])
                problems.append((idx, f"frame_index has holes (first at {first_hole})"))
    if problems:
        return _fail(
            "frame_continuity",
            f"{len(problems)} continuity problem(s): dropped frames or row/length mismatch",
            problems=problems[:10],
            episodes=sorted({p[0] for p in problems if isinstance(p[0], int)}),
        )
    return _ok("frame_continuity")


def check_training_readiness(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.episodes is None:
        return _skip("training_readiness", "episode metadata unreadable")
    chunk_size = cfg.get("chunk_size")
    if not chunk_size:
        return _skip(
            "training_readiness",
            "configure training_readiness.chunk_size (your policy's action horizon) to enable",
        )
    lengths = [ep.get("length", 0) for ep in snap.episodes]
    too_short = [
        ep.get("episode_index") for ep in snap.episodes if ep.get("length", 0) <= chunk_size
    ]
    min_len = min(lengths) if lengths else 0
    if too_short:
        return _fail(
            "training_readiness",
            f"{len(too_short)} episode(s) shorter than chunk_size={chunk_size}+1; "
            "ACT/diffusion sampling would crash or pad",
            too_short=too_short[:10],
            min_length=min_len,
        )
    return _ok("training_readiness", f"min episode length {min_len} > chunk_size {chunk_size}")


def check_portability(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if not snap.root.is_dir():
        return _skip("portability", "dataset root unreadable")
    problems = []
    warnings = []
    # symlinks break on transfer (absolute targets die on every other machine)
    for path in snap.root.rglob("*"):
        if path.is_symlink():
            problems.append(f"symlink: {path.relative_to(snap.root)}")
    allowed_dirs = {"meta", "data", "videos"}
    allowed_files = {
        "README.md",
        "LICENSE",
        "LICENSE.txt",
        "LICENSE.md",
        ".gitattributes",
        ".gitignore",
    }
    for entry in snap.root.iterdir():
        if entry.is_dir() and entry.name not in allowed_dirs:
            warnings.append(f"unexpected top-level dir: {entry.name}")
        elif entry.is_file() and entry.name not in allowed_files:
            warnings.append(f"unexpected top-level file: {entry.name}")
    # stray files inside data/ break nothing but pollute uploads
    for entry in (snap.root / "data").rglob("*") if (snap.root / "data").is_dir() else []:
        if entry.is_file() and entry.suffix not in (".parquet",):
            warnings.append(f"stray file in data/: {entry.relative_to(snap.root)}")
    if problems:
        return _fail(
            "portability", f"{len(problems)} portability blocker(s)", problems=problems[:10]
        )
    if warnings:
        return _warn("portability", f"{len(warnings)} unusual entr(y/ies)", warnings=warnings[:10])
    return _ok("portability")


def check_duplicate_episodes(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if len(snap.frames) < 2:
        return _skip("duplicate_episodes", "need >=2 readable episodes")

    hashes: dict[str, list[int]] = {}
    for idx in sorted(snap.frames):
        h = snap.trajectory_digest(idx)
        if h is not None:
            hashes.setdefault(h, []).append(idx)
    dupes = {h: eps for h, eps in hashes.items() if len(eps) > 1}
    if dupes:
        pairs = list(dupes.values())[:10]
        # 'episodes' drives --from-check: the later members whose removal
        # breaks the duplication (first member of each group stays)
        drop_later = sorted({ep_i for grp in dupes.values() for ep_i in grp[1:]})
        return _warn(
            "duplicate_episodes",
            f"{len(dupes)} group(s) of byte-identical state/action trajectories",
            groups=pairs,
            episodes=drop_later[:50],
        )
    return _ok("duplicate_episodes")


def check_splits(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if snap.info is None:
        return _skip("splits", "info.json unreadable")
    splits = snap.info.get("splits") or {}
    if len(splits) < 2:
        return _ok("splits", "single split, no leakage possible")
    # lerobot declares splits as "start:end" episode ranges
    ranges = {}
    for name, spec in splits.items():
        if isinstance(spec, str) and ":" in spec:
            try:
                a, b = spec.split(":", 1)
                ranges[name] = (int(a), int(b))
            except ValueError:
                return _fail("splits", f"unparseable split spec {name}={spec!r}")
    overlaps = []
    names = list(ranges)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            a1, b1 = ranges[n1]
            a2, b2 = ranges[n2]
            if max(a1, a2) < min(b1, b2):
                overlaps.append((n1, n2))
    if overlaps:
        return _fail("splits", f"episode ranges overlap: {overlaps}", overlaps=overlaps)
    return _ok("splits", f"{len(ranges)} disjoint splits")


def check_video_sync(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    """Cross-camera time alignment (v3.0 metadata level).

    v3 episodes carry per-video-key from/to timestamps; stereo rigs that
    drifted (no hardware sync) show up as differing spans across keys.
    v2.x cuts one video per episode with no per-camera timestamps in
    metadata — nothing to compare without decoding, so SKIP.
    """
    cfg = cfg or {}
    if snap.info is None or snap.episodes is None:
        return _skip("video_sync", "info.json or episode metadata unreadable")
    version = (snap.info or {}).get("codebase_version")
    if version != "v3.0":
        return _skip(
            "video_sync",
            "v2.x stores no per-camera timestamps in metadata (videos are cut per episode)",
        )
    keys = _video_keys(snap.info)
    if len(keys) < 2:
        return _ok("video_sync", "single video stream, nothing to sync")
    tolerance = cfg.get("tolerance", 0.05)
    problems: list[tuple] = []
    for ep in snap.episodes:
        idx = ep.get("episode_index")
        spans = {}
        for key in keys:
            t0 = ep.get(f"videos/{key}/from_timestamp")
            t1 = ep.get(f"videos/{key}/to_timestamp")
            if t0 is None or t1 is None:
                problems.append((idx, key, "missing camera timestamps"))
                continue
            spans[key] = (float(t0), float(t1))
        if len(spans) < 2:
            continue
        starts = [v[0] for v in spans.values()]
        ends = [v[1] for v in spans.values()]
        spread = max(max(starts) - min(starts), max(ends) - min(ends))
        if spread > tolerance:
            problems.append((idx, f"camera spans drift {spread:.3f}s > {tolerance:.3f}s"))
    if problems:
        episodes = sorted({p[0] for p in problems if isinstance(p[0], int)})
        return _warn(
            "video_sync",
            f"{len(problems)} episode(s) with cross-camera time drift > {tolerance:.3f}s",
            problems=problems[:10],
            episodes=episodes,
        )
    return _ok("video_sync", f"{len(keys)} cameras aligned within {tolerance:.3f}s")


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
        check_stats_recompute,
        check_action_quality,
        check_video_files,
        check_video_streams,
        check_video_sync,
        check_zero_variance,
        check_schema_consistency,
        check_frame_continuity,
        check_training_readiness,
        check_portability,
        check_duplicate_episodes,
        check_splits,
    )
}


def run_all(snap: DatasetSnapshot, profile: dict | None = None) -> list[CheckResult]:
    """Run every expectation; profile remaps severities per check.

    A check disabled via `enabled: false` is not executed at all — it reports
    SKIP naming the profile, and contributes no episodes to --from-check.
    """
    from tailor.profile import remap

    profile = profile or {}
    results = []
    for name, fn in EXPECTATIONS.items():
        cfg = profile.get(name, {})
        if cfg.get("enabled") is False:
            results.append(_skip(name, "disabled by profile"))
            continue
        r = fn(snap, cfg)
        results.append(CheckResult(r.name, remap(r.status, cfg), r.message, r.details))
    return results
