"""Acceptance expectations for LeRobot v2.x datasets.

Each expectation is a pure function DatasetSnapshot -> CheckResult, registered
by name in EXPECTATIONS. Expectations degrade to SKIP when the inputs they
need were unreadable, so one broken file never hides other findings.

The v0.1 suite targets distributed datasets ("does the package match the
label"): format compliance, metadata-vs-data consistency, integrity.
"""

import json
import math
import re
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from tailor.loader import SUPPORTED_VERSIONS, _episode_parquet_path
from tailor.model import CheckResult, DatasetSnapshot, Status

if TYPE_CHECKING:
    import av

Expectation = Callable[[DatasetSnapshot, dict], CheckResult]

T = TypeVar("T")

# frames, width, height, fps, duration_s, interlaced, rotation_deg
ProbeResult = tuple[int, int, int, float | None, float | None, bool | None, float | None]

# black-edge thresholds: grey level counts as dark below 24/255; a frame is
# "black" when ≥98% of its pixels are dark (compression noise keeps a few alive)
BLACK_DARK_LEVEL = 24
BLACK_DARK_FRAC = 0.98

# the deep content scan samples Laplacian sharpness every Nth frame (a full
# per-frame Laplacian triples decode cost for no extra signal)
BLUR_EVERY = 5


def _ok(name: str, message: str = "", **details: Any) -> CheckResult:
    return CheckResult(name, Status.PASS, message, details)


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


def check_episode_files(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if not (snap.root / "data").is_dir():
        return _skip("episode_files", "data/ not on disk (meta-only run? use --full)")
    if snap.episodes is None:
        return _skip("episode_files", "episode metadata unreadable")
    version = (snap.info or {}).get("codebase_version")
    if version == "v3.0":
        # v3 stores episodes back-to-back in chunk/file shards; episode meta
        # carries the shard pointer, so existence is checked per shard.
        referenced: set[tuple[int, int]] = set()
        missing_shards = set()
        for ep in snap.episodes:
            c, f = ep.get("data/chunk_index"), ep.get("data/file_index")
            if not isinstance(c, int) or not isinstance(f, int):
                missing_shards.add("unresolved shard pointer")
                continue
            referenced.add((c, f))
            shard = snap.root / "data" / f"chunk-{c:03d}" / f"file-{f:03d}.parquet"
            if not shard.is_file():
                missing_shards.add(str(shard.relative_to(snap.root)))
        if missing_shards:
            return _fail(
                "episode_files",
                f"{len(missing_shards)} data shard(s) missing",
                missing=sorted(missing_shards)[:20],
            )
        # pointer self-consistency, the other half of existence: a shard on
        # disk that no episode references is data nobody will ever read. The
        # real unitreeh1 case (README) — all cam_left pointers aimed at
        # file-000 while 500 frames sat in an unreferenced file-001 — sails
        # through existence alone when the wrongly-pointed file happens to be
        # big enough; the orphan scan catches it in one move, metadata-only.
        orphans: list[str] = []
        for shard in sorted((snap.root / "data").glob("chunk-*/file-*.parquet")):
            try:
                c = int(shard.parent.name.removeprefix("chunk-"))
                f = int(shard.stem.removeprefix("file-"))
            except ValueError:
                orphans.append(str(shard.relative_to(snap.root)) + " (unparsable name)")
                continue
            if (c, f) not in referenced:
                orphans.append(str(shard.relative_to(snap.root)))
        if orphans:
            return _fail(
                "episode_files",
                f"{len(orphans)} data shard(s) on disk not referenced by any episode",
                orphans=orphans[:20],
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
    # a True inside the padded array guarantees >= 2 rising/falling edges,
    # so the pairwise edge distances below are always well-defined
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
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


def _numeric_feature_columns(df: Any) -> list[str]:
    """observation.*/action columns with a numeric dtype — a NaN is an error
    in any of them, not just state/action (joint positions, extra scalar
    sensors). String columns (language instructions) and non-feature columns
    (frame_index, timestamp) are excluded.
    """
    out = []
    for col in df.columns:
        if not (col.startswith("observation.") or col.startswith("action")):
            continue
        dt = df[col].dtype
        if np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.integer):
            out.append(col)  # plain scalar column
        elif dt.kind == "O" and len(df) and isinstance(df[col].iloc[0], np.ndarray):
            out.append(col)  # fixed_size_list columns (state/action) arrive as object
    return out


def check_nan_inf(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    cfg = cfg or {}
    if not snap.frames:
        return _skip("nan_inf", "no readable frames")
    bad = []
    for idx in sorted(snap.frames):
        for col in _numeric_feature_columns(snap.frames[idx]):
            arr = snap.col_array(idx, col)
            if arr is None:
                continue
            n_bad = int((~np.isfinite(arr)).sum())
            if n_bad:
                bad.append((idx, col, n_bad))
    if bad:
        return _fail(
            "nan_inf",
            "non-finite values found in numeric feature columns",
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
            ("min", running.min),
            ("max", running.max),
        ):
            decl = np.asarray(stats[stat_name], dtype=float)
            scale = np.maximum(np.abs(decl), 1e-9)
            drift = float((np.abs(decl - actual) / scale).max())
            if drift > mean_tol:
                drift_report.append((col, stat_name, round(drift, 4)))
    if drift_report:
        # A sampled run (e.g. --max-episodes 20 of 400k) recomputes stats over
        # a subset; drift against full-dataset stats.json is then expected and
        # unattributable. Severity tracks coverage instead of a blanket WARN,
        # which would defang full-run checks. Unknown total keeps FAIL teeth.
        total = (snap.info or {}).get("total_frames") or 0
        sampled = sum(len(df) for df in snap.frames.values())
        coverage = sampled / total if total else 1.0
        details = {"drift": drift_report[:10]}
        if coverage < cfg.get("coverage_threshold", 0.95):
            return _warn(
                "stats_recompute",
                f"declared stats drift beyond {mean_tol:.0%} of scale, but only "
                f"{sampled}/{total} frames sampled ({coverage:.0%}) — insufficient "
                f"coverage to attribute; rerun without --max-episodes",
                **details,
            )
        return _fail(
            "stats_recompute",
            f"declared stats drift beyond {mean_tol:.0%} of scale (recompute mismatch)",
            **details,
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


def _stream_fps(stream: "av.VideoStream") -> float | None:
    """Container-declared fps, preferring average_rate over codec framerate."""
    rate = getattr(stream, "average_rate", None) or stream.codec_context.framerate
    if rate is None or rate <= 0:
        return None
    return float(rate)


def _stream_duration_s(
    stream: "av.VideoStream", container: "av.container.InputContainer"
) -> float | None:
    """Duration in seconds from the stream header, else the container header."""
    if stream.duration is not None and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration) / 1_000_000.0
    return None


def _rotation_degrees(stream: "av.VideoStream") -> float | None:
    """Display rotation from side data / legacy metadata; None when upright.

    MP4 writers store a displaymatrix; older files used a plain 'rotate'
    tag. Both are honored — players apply them, most training loaders
    don't, so a rotated file silently trains on sideways frames.
    """
    side = getattr(stream, "side_data", None) or {}
    matrix = side.get("displaymatrix")
    if matrix is not None:
        identity = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        if tuple(matrix[:9]) == identity:
            return None
        return round(-math.degrees(math.atan2(matrix[3], matrix[0]))) % 360
    legacy = (getattr(stream, "metadata", None) or {}).get("rotate")
    if legacy is not None:
        try:
            deg = float(legacy) % 360
        except ValueError:
            return None
        return deg or None
    return None


def _interlaced(stream: "av.VideoStream") -> bool | None:
    """True when the stream is flagged interlaced; None when undeterminable."""
    fo = getattr(getattr(stream, "codec_context", None), "field_order", None)
    if fo is None:
        return None
    text = str(fo).upper()
    if text.isdigit():
        return int(text) not in (0, 1)  # FFmpeg: 0 unknown, 1 progressive
    return text not in ("NONE", "UNKNOWN", "PROGRESSIVE", "UNSPECIFIED")


def _duration_gap_ratio(frames: int, fps: float | None, duration_s: float | None) -> float | None:
    """|decoded-length - header-duration| / header-duration, None when unknown.

    frames/fps estimates what the file should decode to; a header claiming
    much more means the recording was cut off before it finalised.
    """
    if not frames or fps is None or fps <= 0 or duration_s is None or duration_s <= 0:
        return None
    return abs(frames / fps - duration_s) / duration_s


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
                # v2.x official template: videos/chunk-XXX/{video_key}/episode_XXXXXX.mp4
                # (the chunk wraps the key — verified against lerobot/pusht@v2.1)
                idx = ep.get("episode_index", -1)
                path = (
                    snap.root
                    / "videos"
                    / f"chunk-{idx // 1000:03d}"
                    / key
                    / f"episode_{idx:06d}.mp4"
                )
            if not path.is_file():
                missing.add(f"{key}: {path.relative_to(snap.root)}")
    if missing:
        return _fail(
            "video_files", f"{len(missing)} video file(s) missing", missing=sorted(missing)[:10]
        )
    if version == "v3.0":
        # the other half of pointer self-consistency, video side: orphan
        # shards (on disk, never referenced) and timestamp overlap on the
        # same shard. Overlap means one video span claims to belong to two
        # episodes — never legal. A gap inside a shard is only a warning:
        # deletions legally leave unreferenced spans inside a shared file.
        referenced: dict[str, set[tuple[int, int]]] = {key: set() for key in keys}
        spans: dict[tuple[str, int, int], list[tuple[float, float]]] = {}
        for ep in snap.episodes:
            for key in keys:
                c = ep.get(f"videos/{key}/chunk_index")
                f = ep.get(f"videos/{key}/file_index")
                if not isinstance(c, int) or not isinstance(f, int):
                    continue
                referenced[key].add((c, f))
                a = ep.get(f"videos/{key}/from_timestamp")
                b = ep.get(f"videos/{key}/to_timestamp")
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    spans.setdefault((key, c, f), []).append((float(a), float(b)))
        problems: list[str] = []
        for key in keys:
            vdir = snap.root / "videos" / key
            if not vdir.is_dir():
                continue
            for shard in sorted(vdir.glob("chunk-*/file-*.mp4")):
                try:
                    c = int(shard.parent.name.removeprefix("chunk-"))
                    f = int(shard.stem.removeprefix("file-"))
                except ValueError:
                    problems.append(f"{key}: {shard.relative_to(snap.root)} (unparsable name)")
                    continue
                if (c, f) not in referenced[key]:
                    problems.append(f"{key}: orphan shard {shard.relative_to(snap.root)}")
        overlaps: list[str] = []
        gaps: list[str] = []
        for (key, c, f), iv in spans.items():
            iv.sort()
            for (a1, b1), (a2, b2) in pairwise(iv):
                if a2 < b1 - 1e-6:
                    overlaps.append(
                        f"{key} chunk-{c:03d}/file-{f:03d}: "
                        f"[{a1:.1f},{b1:.1f}) overlaps [{a2:.1f},{b2:.1f})"
                    )
                elif a2 > b1 + 1e-6:
                    gaps.append(
                        f"{key} chunk-{c:03d}/file-{f:03d}: gap {a2 - b1:.1f}s at {b1:.1f}s"
                    )
        if problems or overlaps:
            return _fail(
                "video_files",
                f"{len(problems) + len(overlaps)} video shard pointer problem(s)",
                problems=(problems + overlaps)[:20],
                gaps=gaps[:20],
            )
        if gaps:
            return _warn(
                "video_files",
                f"{len(gaps)} unreferenced span(s) inside shared shards (deleted episodes?)",
                gaps=gaps[:20],
            )
    return _ok("video_files", f"{len(keys)} stream(s)")


def _video_probe_jobs(snap: DatasetSnapshot, keys: list[str]) -> list[tuple[str, str, Path, int]]:
    """(key, label, path, expected_frames) for every declared video file on disk.

    Shared by the stream probe and the head/tail decode: v3 episodes share
    shards, so expected frames are summed per shard; v2 cuts one file per
    episode. Missing files are skipped — video_files already reports them.
    """
    version = (snap.info or {}).get("codebase_version")
    jobs: list[tuple[str, str, Path, int]] = []
    if snap.episodes is None:
        return jobs
    for key in keys:
        if version == "v3.0":
            shards: dict[tuple, list] = {}
            for ep in snap.episodes:
                c, f = ep.get(f"videos/{key}/chunk_index"), ep.get(f"videos/{key}/file_index")
                if isinstance(c, int) and isinstance(f, int):
                    shards.setdefault((c, f), []).append(ep.get("length", 0))
            for (c, f), lengths in shards.items():
                path = snap.root / "videos" / key / f"chunk-{c:03d}" / f"file-{f:03d}.mp4"
                if not path.is_file():
                    continue
                jobs.append((key, path.name, path, sum(lengths)))
        else:
            for ep in snap.episodes:
                idx = ep.get("episode_index", -1)
                path = (
                    snap.root
                    / "videos"
                    / f"chunk-{idx // 1000:03d}"
                    / key
                    / f"episode_{idx:06d}.mp4"
                )
                if not path.is_file():
                    continue
                jobs.append((key, f"ep {idx}", path, ep.get("length", 0)))
    return jobs


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

    def _probe(
        path: Path,
    ) -> tuple[int, int, int, float | None, float | None, bool | None, float | None]:
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
        fps = _stream_fps(stream)
        duration = _stream_duration_s(stream, container)
        interlaced = _interlaced(stream)
        rotation = _rotation_degrees(stream)
        container.close()
        return frames, w, h, fps, duration, interlaced, rotation

    # collect probe jobs first, run them in parallel (av releases the GIL),
    # then judge — video probing dominates wall time on video-heavy datasets
    jobs = _video_probe_jobs(snap, keys)

    problems: list[str] = []
    import os
    from concurrent.futures import ThreadPoolExecutor

    workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        probed = list(pool.map(lambda j: _safe_probe(_probe, j[2]), jobs))
    fps_rel_tol = cfg.get("fps_rel_tol", 0.05)
    duration_tol = cfg.get("duration_tol", 0.02)
    for (key, label, _path, expected), result in zip(jobs, probed, strict=True):
        shape = snap.info["features"][key].get("shape", [])
        if isinstance(result, Exception):
            problems.append(f"{key}: {label} video undecodable")
            continue
        frames, w, h, fps, duration, interlaced, rotation = result
        if abs(frames - expected) > tolerance:
            problems.append(f"{key} {label}: {frames} frames vs {expected} episode frames")
        if len(shape) >= 2 and (h, w) != (shape[0], shape[1]):
            problems.append(f"{key} {label}: {w}x{h} vs declared {shape[1]}x{shape[0]}")
        # declared fps (info.json features) vs container fps: a silent unit
        # or profile mismatch breaks timestamp reconstruction downstream
        declared_fps = snap.info["features"][key].get("fps")
        if fps and declared_fps and abs(fps - declared_fps) > max(fps_rel_tol * declared_fps, 0.5):
            problems.append(
                f"{key} {label}: container {fps:g} fps vs declared {declared_fps:g} fps"
            )
        # frames/fps vs header duration: a header claiming far more than the
        # file decodes to means the recording was cut off before finalising
        gap = _duration_gap_ratio(frames, fps, duration)
        if gap is not None and gap > duration_tol:
            problems.append(
                f"{key} {label}: decodes to ~{frames / (fps or 1):.1f}s but header "
                f"says {duration:.1f}s ({gap:.0%} truncated)"
            )
        if rotation:
            problems.append(
                f"{key} {label}: rotation {rotation:g}° metadata (players rotate, loaders don't)"
            )
        if interlaced:
            problems.append(f"{key} {label}: interlaced stream (loaders assume progressive)")
    if problems:
        return _fail("video_streams", f"{len(problems)} stream problem(s)", problems=problems[:10])
    return _ok("video_streams", f"{len(keys)} stream(s) probed")


def _safe_probe(probe: "Callable[[Path], T]", path: Path) -> "T | Exception":
    try:
        return probe(path)
    except Exception as e:  # undecodable container must fail the check, not the run
        return e


def _decode_edge_frames(
    path: Path, head: int, tail: int, fps_hint: float | None
) -> tuple[list[bool], list[bool]]:
    """Per-frame 'is black' flags for the first/last frames of one file.

    Cheap content probe: only the head and tail are decoded (the classic
    'camera never started' / 'recording cut off' failures live at the
    edges), so cost is O(head+tail), not O(file).
    """
    import av

    def _black_flags(container: "av.container.InputContainer", n: int) -> list[bool]:
        flags: list[bool] = []
        for frame in container.decode(container.streams.video[0]):
            gray = frame.reformat(format="gray").to_ndarray()
            dark = float((gray < BLACK_DARK_LEVEL).mean())
            flags.append(dark >= BLACK_DARK_FRAC)
            if len(flags) >= n:
                break
        return flags

    container = av.open(str(path))
    try:
        head_flags = _black_flags(container, head)
        tail_flags: list[bool] = []
        stream = container.streams.video[0]
        duration = _stream_duration_s(stream, container)
        fps = fps_hint or _stream_fps(stream) or 30.0
        if duration and tail:
            # seek into the file by container timestamp (µs), land before the
            # target so the tail window is fully covered after keyframe realign
            target = max(0.0, duration - (tail / fps) - 1.0)
            container.seek(int(target * 1_000_000))
            tail_flags = _black_flags(container, tail * 3)[-tail:]
        return head_flags, tail_flags
    finally:
        container.close()


def check_video_black(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    """Black openings/closings: lens cap left on, camera that never started.

    Decodes only the first/last few frames of each video file — the edge
    failures that frame counts and metadata cannot see. A mid-file black
    span is out of scope (needs a full decode); this check is the cheap
    80% of that value.
    """
    cfg = cfg or {}
    if not (snap.root / "videos").is_dir():
        return _skip("video_black", "videos/ not on disk (meta-only run? use --full)")
    if snap.info is None or snap.episodes is None:
        return _skip("video_black", "info.json or episode metadata unreadable")
    keys = _video_keys(snap.info)
    if not keys:
        return _ok("video_black", "no video streams declared")
    try:
        import av  # noqa: F401 — capability probe, same gate as video_streams
    except ImportError:
        return _skip("video_black", "PyAV not installed (pip install av)")

    head = int(cfg.get("head_frames", 24))
    tail = int(cfg.get("tail_frames", 24))
    dark_share = float(cfg.get("dark_share", 0.9))
    jobs = _video_probe_jobs(snap, keys)
    problems: list[str] = []
    for key, label, path, _expected in jobs:
        result = _safe_probe(lambda p: _decode_edge_frames(p, head, tail, None), path)
        if isinstance(result, Exception):
            problems.append(f"{key}: {label} undecodable")
            continue
        head_flags, tail_flags = result
        if head_flags and (sum(head_flags) / len(head_flags)) >= dark_share:
            problems.append(
                f"{key} {label}: black opening ({sum(head_flags)}/{len(head_flags)} frames)"
            )
        if tail_flags and (sum(tail_flags) / len(tail_flags)) >= dark_share:
            problems.append(
                f"{key} {label}: black closing ({sum(tail_flags)}/{len(tail_flags)} frames)"
            )
    if problems:
        return _fail(
            "video_black", f"{len(problems)} black-edge problem(s)", problems=problems[:10]
        )
    return _ok("video_black", f"{len(keys)} stream(s), edges not black")


def _decode_content_measures(path: Path, gray_width: int = 64) -> dict | None:
    """One full decode: per-frame brightness/dark share/MAD/repeat flags,
    Laplacian variance every BLUR_EVERY frames, timestamps.

    Mirrors rdq's decode_pass, minus optical flow / phash (camera-swap and
    duplicate curation are out of scope for an acceptance gate).
    """
    import av

    out: dict = {
        "t": [],
        "mean": [],
        "std": [],
        "dark": [],
        "mad": [],
        "ident": [],
        "exact": [],
        "lap_t": [],
        "lap": [],
        "lap99": [],
    }
    try:
        container = av.open(str(path))
    except Exception:
        return None
    stream = container.streams.video[0]
    tb = float(stream.time_base) if stream.time_base else None
    prev16: np.ndarray | None = None
    prev_id: np.ndarray | None = None
    t0 = None
    for i, fr in enumerate(container.decode(stream)):
        t = (
            fr.pts * tb
            if (fr.pts is not None and tb)
            else (out["t"][-1] + 1 / 30 if out["t"] else 0.0)
        )
        t0 = t if t0 is None else t0
        t -= t0
        aw = max(16, min(gray_width, fr.width - fr.width % 2))
        h2 = max(2, int(round(fr.height * aw / fr.width / 2) * 2))
        g = fr.reformat(width=aw, height=h2, format="gray").to_ndarray()
        g16 = g.astype(np.int16)
        out["t"].append(t)
        out["mean"].append(float(g.mean()))
        out["std"].append(float(g.std()))
        out["dark"].append(float((g < BLACK_DARK_LEVEL).mean()))
        if prev16 is not None:
            out["mad"].append(float(np.abs(g16 - prev16).mean() / 255))
        else:
            out["mad"].append(float("nan"))
        if prev_id is not None and prev_id.shape == g.shape:
            dif = np.abs(g16 - prev_id)
            out["ident"].append(bool(np.count_nonzero(dif > 2) <= 0.0005 * g.size))
            out["exact"].append(bool(not dif.any()))
        else:
            out["ident"].append(False)
            out["exact"].append(False)
        if i % BLUR_EVERY == 0:
            f = g.astype(np.float32)
            lap = (
                np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f
            )
            core = lap[1:-1, 1:-1]
            out["lap_t"].append(t)
            out["lap"].append(float(core.var()))
            # p99 of |Laplacian|: an in-focus frame keeps sharp edges even when
            # the scene is mostly flat (close-up against a plain surface), so
            # p99 stays ≥~75; a defocused lens has NO edge anywhere and p99
            # collapses to the sensor-noise floor (~10 on our 64-px grayscale)
            out["lap99"].append(float(np.percentile(np.abs(core), 99)))
        prev16, prev_id = g16, g
    container.close()
    return out


def _true_runs(t: "np.ndarray", flags: "np.ndarray", min_len: float) -> list[tuple[float, float]]:
    """[(start, end)] for consecutive-True stretches lasting ≥ min_len seconds."""
    spans: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if t[i - 1] - t[start] >= min_len:
                spans.append((float(t[start]), float(t[i])))
            start = None
    if start is not None and t[len(flags) - 1] - t[start] >= min_len:
        spans.append((float(t[start]), float(t[len(flags) - 1])))
    return spans


def _content_problems(measures: dict, cfg: dict) -> list[str]:
    """Content-level problems in one decoded segment (seconds + reason)."""
    t = np.asarray(measures["t"], dtype=float)
    if len(t) < 2:
        return []
    problems: list[str] = []
    dt = float(np.median(np.diff(t)))
    dur = float(t[-1]) + dt
    dark = np.asarray(measures["dark"], dtype=float) >= 0.98
    std = np.asarray(measures["std"], dtype=float)
    mean = np.asarray(measures["mean"], dtype=float)
    ident = np.asarray(measures["ident"], dtype=bool)
    exact = np.asarray(measures["exact"], dtype=bool)
    # frozen: repeated frames, and a stalled camera's encoder emits mostly
    # bit-exact repeats — a quarter exact separates stall from a still scene.
    # Position carries the semantics: prefix/suffix stalls are usually
    # collection-side (post-reset hold, post-success tail; a real DexJoCo
    # case keeps ~2 s after success on every episode) and nearly lossless to
    # trim, a mid-episode stall is a content fault or a stopped robot.
    edge = float(cfg.get("frozen_edge_s", 2.0))
    for a, b in _true_runs(t, ident, float(cfg.get("frozen_min_s", 1.0))):
        m = (t >= a) & (t < b)
        if m.any() and float(exact[m].mean()) >= 0.25:
            d_start, d_end = a, dur - b
            pos = (
                "prefix"
                if d_start <= edge and d_start <= d_end
                else ("suffix" if d_end <= edge else "mid")
            )
            problems.append(f"frozen[{pos}] {b - a:.1f}s from {a:.1f}s (repeated frames)")
    # black spans away from the file edges (edges are video_black's job).
    # A real recorder failure cuts from a normally lit scene to black; a wrist
    # camera dimming into black has already gone dark for a second before
    # (an object moving right up against the lens). Only report when the
    # scene was still lit just before the black onset (real ep 247 case:
    # mean 15-28 for a second before going dark — occlusion, not a fault).
    onset = float(cfg.get("black_onset", 40.0))
    for a, b in _true_runs(t, dark, 0.5):
        if not (a > 0.5 and b < dur - 0.5):
            continue
        pre = mean[(t >= a - 0.5) & (t < a)]
        if len(pre) and float(np.median(pre)) < onset:
            continue
        problems.append(f"black {b - a:.1f}s from {a:.1f}s (mid-file)")
    # flat frame while lit: lens covered / featureless scene
    lit = mean[~dark]
    norm = float(np.median(lit)) if len(lit) else 0.0
    cover = (std < 8.0) & (mean < 0.7 * norm) & ~dark if norm else np.zeros(len(t), bool)
    for a, b in _true_runs(t, cover, float(cfg.get("flat_min_s", 1.0))):
        problems.append(f"covered {b - a:.1f}s from {a:.1f}s (flat and dark: lens blocked)")
    over = (mean > 245.0) & (std < 10.0)
    for a, b in _true_runs(t, over, 1.0):
        problems.append(f"overexposed {b - a:.1f}s from {a:.1f}s (washed out)")
    # blur: p99 of |Laplacian| collapses only when no edge is sharp anywhere.
    # Variance cannot separate defocus from a legitimately flat close-up (a
    # wrist camera against a plain surface sits at 5-15, the same range as
    # sensor noise), but those frames still carry sharp object contours —
    # measured p99 was 76-128 for normal cameras on a real 1.4 GB dataset vs
    # 8-10 for a defocused lens. The 40 floor sits 4-8x from both sides.
    # Judged over the WHOLE segment: a working camera sees something sharp at
    # some point (reaching in, turning); only a broken/dirty lens is flat for
    # the entire recording. Per-span detection drowned in false positives on
    # a real wrist camera (33 hits per 100 episodes, all legit close-ups).
    lap99 = np.asarray(measures.get("lap99") or [], dtype=float)
    floor = float(cfg.get("blur_p99", 40.0))
    cover_frac = float(cfg.get("blur_cover", 0.95))
    if len(lap99):
        soft_frac = float((lap99 < floor).mean())
        if soft_frac >= cover_frac:
            problems.append(f"no sharp edges in {soft_frac:.0%} of frames (defocus / dirty lens)")
    return problems


def _video_segments(
    snap: DatasetSnapshot, keys: list[str], ep_indices: set[int] | None = None
) -> list[tuple[str, str, Path, float | None, float | None]]:
    """(key, label, path, start_s, end_s) — v2 one file per episode, v3
    episodes cut from shared shards by their metadata timestamps.

    ep_indices limits scanning to the sampled episodes (--max-episodes):
    decode-heavy work must follow the sample, unlike metadata checks which
    read the full episode list for free.
    """
    version = (snap.info or {}).get("codebase_version")
    out: list[tuple[str, str, Path, float | None, float | None]] = []
    if snap.episodes is None:
        return out
    for key in keys:
        if version == "v3.0":
            for ep in snap.episodes:
                if ep_indices is not None and ep.get("episode_index") not in ep_indices:
                    continue
                c = ep.get(f"videos/{key}/chunk_index")
                f = ep.get(f"videos/{key}/file_index")
                t0, t1 = (
                    ep.get(f"videos/{key}/from_timestamp"),
                    ep.get(f"videos/{key}/to_timestamp"),
                )
                if not all(isinstance(v, (int, float)) for v in (c, f, t0, t1)):
                    continue
                ci, fi = cast(int, c), cast(int, f)
                path = snap.root / "videos" / key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
                if path.is_file():
                    out.append(
                        (
                            key,
                            f"{path.name} ep {ep.get('episode_index')}",
                            path,
                            cast(float, t0),
                            cast(float, t1),
                        )
                    )
        else:
            for ep in snap.episodes:
                if ep_indices is not None and ep.get("episode_index") not in ep_indices:
                    continue
                idx = ep.get("episode_index", -1)
                path = (
                    snap.root
                    / "videos"
                    / f"chunk-{idx // 1000:03d}"
                    / key
                    / f"episode_{idx:06d}.mp4"
                )
                if path.is_file():
                    out.append((key, f"ep {idx}", path, None, None))
    return out


def check_video_content(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    """Full-decode content scan: frozen frames, mid-file black, covered
    lens, over-exposure, persistent blur. OFF by default — a profile must
    opt in (any config entry for this check enables it), because a full
    decode is minutes per GB, not the seconds a gate can afford.
    """
    if not cfg:
        return _skip("video_content", "full-decode check; enable via a profile entry")
    if not (snap.root / "videos").is_dir():
        return _skip("video_content", "videos/ not on disk")
    if snap.info is None:
        return _skip("video_content", "info.json unreadable")
    keys = _video_keys(snap.info)
    if not keys:
        return _ok("video_content", "no video streams declared")
    try:
        import av  # noqa: F401 — same capability gate as the other video checks
    except ImportError:
        return _skip("video_content", "PyAV not installed (pip install av)")

    # decode-heavy work follows the --max-episodes sample; the absolute
    # blur floor makes per-camera baselines unnecessary (see _content_problems)
    ep_indices = set(snap.frames) if snap.frames else None
    segs = _video_segments(snap, keys, ep_indices)
    decoded: dict[Path, dict | None] = {}
    for _key, _label, path, _a, _b in segs:
        if path not in decoded:
            decoded[path] = _decode_content_measures(path)
    problems: list[str] = []
    for key, label, path, a, b in segs:
        m = decoded.get(path)
        if m is None:
            problems.append(f"{key}: {label} undecodable")
            continue
        if a is not None and b is not None:
            tt = np.asarray(m["t"], dtype=float)
            mask = (tt >= a - 1e-6) & (tt < b - 1e-6)
            if not mask.any():
                continue
            lt = np.asarray(m["lap_t"], dtype=float)
            lmask = (lt >= a - 1e-6) & (lt < b - 1e-6)
            seg = {"t": (tt[mask] - a).tolist(), "lap_t": (lt[lmask] - a).tolist()}
            seg["lap"] = np.asarray(m["lap"])[lmask].tolist()
            # per-frame series slice on the frame mask; lap is sampled every
            # BLUR_EVERY frames and gets its own mask above
            seg["lap99"] = np.asarray(m["lap99"])[lmask].tolist()
            for k2 in ("mean", "std", "dark", "mad", "ident", "exact"):
                seg[k2] = np.asarray(m[k2])[mask].tolist()
        else:
            seg = m
        for p in _content_problems(seg, cfg):
            problems.append(f"{key} {label}: {p}")
    # position-aware frozen policy: prefix/suffix stalls are collection-side
    # redundancy (post-reset hold, post-success tail), not content faults —
    # profiles can list them in frozen_ignore to keep them visible in the
    # count but out of the verdict. Counted per position either way so a
    # flood of edge stalls cannot hide the mid-episode ones in the report.
    ignore = {str(x) for x in cfg.get("frozen_ignore", [])}
    frozen_counts = {"prefix": 0, "suffix": 0, "mid": 0}
    kept: list[str] = []
    ignored = 0
    for line in problems:
        fm = re.match(r"frozen\[(prefix|suffix|mid)\]", line.split(": ", 1)[-1])
        if fm:
            frozen_counts[fm.group(1)] += 1
            if fm.group(1) in ignore:
                ignored += 1
                continue
        kept.append(line)
    details: dict[str, Any] = {"frozen_by_position": frozen_counts}
    if ignored:
        details["frozen_ignored"] = ignored
    if kept:
        return _fail(
            "video_content", f"{len(kept)} content problem(s)", problems=kept[:10], **details
        )
    if frozen_counts != {"prefix": 0, "suffix": 0, "mid": 0}:
        return _ok(
            "video_content",
            f"{len(segs)} segment(s) scanned; {ignored} edge stall(s) ignored per profile",
            **details,
        )
    return _ok(
        "video_content",
        f"{len(segs)} segment(s) scanned, content clean",
        frozen_by_position=frozen_counts,
    )


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
    # three defect classes with different repairs; order = report order
    by_class: dict[str, list[tuple[int, str]]] = {
        "length_mismatch": [],  # rows vs declared episode length
        "reordering": [],  # complete but stored out of order (a permutation)
        "holes": [],  # frame_index values genuinely missing
    }
    declared = {ep.get("episode_index"): ep.get("length") for ep in (snap.episodes or [])}
    for idx, df in sorted(snap.frames.items()):
        n = len(df)
        if idx in declared and declared[idx] is not None and n != declared[idx]:
            by_class["length_mismatch"].append(
                (idx, f"{n} rows vs declared length {declared[idx]}")
            )
            continue
        if "frame_index" in df.columns:
            fi = np.asarray(df["frame_index"])
            if fi.size and (fi != np.arange(fi.size)).any():
                if np.array_equal(np.sort(fi), np.arange(fi.size)):
                    # complete but reordered: FAIL, not WARN — official readers
                    # use row position as frame identity, so shuffled storage
                    # corrupts consumers even though no value is missing
                    by_class["reordering"].append(
                        (
                            idx,
                            "rows stored out of order (complete, sortable by frame_index); "
                            "consumers reading by row position get shuffled frames",
                        )
                    )
                else:
                    first_hole = int(np.flatnonzero(fi != np.arange(fi.size))[0])
                    by_class["holes"].append(
                        (idx, f"frame_index has holes (first at {first_hole})")
                    )
    problems = by_class["length_mismatch"] + by_class["reordering"] + by_class["holes"]
    if problems:
        parts = [f"{len(v)} {k}" for k, v in by_class.items() if v]
        return _fail(
            "frame_continuity",
            f"continuity problems: {', '.join(parts)}",
            problems=problems[:10],
            episodes=sorted({p[0] for p in problems if isinstance(p[0], int)}),
        )
    return _ok("frame_continuity")


def check_index_column(snap: DatasetSnapshot, cfg: dict | None = None) -> CheckResult:
    """Global row identity: index == episode_offset + frame_index, per row.

    The official writer stamps `index` as a global chronological counter while
    consumers address frames by row position, so an `index` column that
    disagrees with episode_offset + frame_index (order-independent, compared
    per row) means a rewrite/concatenation accident — every consumer trusting
    `index` as time order is silently wrong. Offset accumulates *declared*
    lengths, so the verdict holds under --max-episodes sampling.
    """
    cfg = cfg or {}
    if not snap.frames:
        return _skip("index_column", "no readable frames")
    if not snap.episodes:
        return _skip("index_column", "episode metadata unreadable")
    has_cols = all(
        "index" in df.columns and "frame_index" in df.columns for df in snap.frames.values()
    )
    if not has_cols:
        return _skip("index_column", "index/frame_index column absent from sampled frames")

    def ep_len(ep: dict) -> int | None:
        if ep.get("length") is not None:
            return int(ep["length"])
        raw_idx = ep.get("episode_index")
        if not isinstance(raw_idx, int):
            return None
        loaded = snap.frames.get(raw_idx)
        return len(loaded) if loaded is not None else None

    problems = []
    checked = 0
    offset = 0
    for ep in sorted(snap.episodes, key=lambda e: e.get("episode_index", -1)):
        raw_idx = ep.get("episode_index")
        if not isinstance(raw_idx, int):
            break
        idx = raw_idx
        ln = ep_len(ep)
        if ln is None:
            # offset chain broken: later episodes have no trustworthy anchor
            break
        df = snap.frames.get(idx)
        if df is not None and "index" in df.columns and "frame_index" in df.columns and len(df):
            expected = offset + np.asarray(df["frame_index"])
            actual = np.asarray(df["index"])
            checked += len(df)
            bad = int((actual != expected).sum())
            if bad:
                first = int(np.flatnonzero(actual != expected)[0])
                problems.append(
                    (
                        idx,
                        f"{bad}/{len(df)} rows ({bad / len(df):.1%}) index != "
                        f"episode_offset + frame_index (first at row {first})",
                    )
                )
            # v3.0 only: rows must also sit inside the declared global window
            f_from, f_to = ep.get("dataset_from_index"), ep.get("dataset_to_index")
            if f_from is not None and f_to is not None and (
                expected.min() < f_from or expected.max() >= f_to
            ):
                    problems.append(
                        (
                            idx,
                            f"expected range [{int(expected.min())},{int(expected.max())}] outside "
                            f"declared [{f_from},{f_to})",
                        )
                    )
        offset += ln
    if problems:
        return _fail(
            "index_column",
            f"{len(problems)} episode(s) with mislabeled index column",
            problems=problems[:10],
            episodes=sorted({p[0] for p in problems if isinstance(p[0], int)}),
        )
    if checked == 0:
        return _skip("index_column", "no anchored episodes to check")
    return _ok("index_column", f"{checked} sampled rows match episode_offset + frame_index")


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
    # .cache is huggingface_hub's local-dir bookkeeping (snapshot_download
    # side effect), not dataset pollution — every HF-downloaded dataset
    # would otherwise WARN on it
    allowed_dirs = {"meta", "data", "videos", ".cache"}
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
        check_episode_files,
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
        check_video_black,
        check_video_content,
        check_video_sync,
        check_zero_variance,
        check_schema_consistency,
        check_frame_continuity,
        check_index_column,
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
