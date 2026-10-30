"""Generate tiny synthetic LeRobot datasets (v2.1 or v3.0) for tests and dev.

Fixtures are deterministic (seeded), run in milliseconds and fit in git.
Corruptions are injected with --corrupt to build negative test cases: each
corruption must be caught by at least one expectation, on both formats.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 10
N_EPISODES = 3
EP_LENGTH = 20
STATE_DIM = 6
ACTION_DIM = 6

CORRUPTIONS = (
    "nan",  # NaN in observation.state of episode 1
    "missing-file",  # delete episode 1's parquet (v2) / the data shard (v3)
    "frame-mismatch",  # info total_frames off by 7
    "bad-version",  # codebase_version set to v9.9
    "short-episode",  # episode 2 declared length set to 1
    "dup-timestamp",  # episode 0 timestamps all zero (non-monotonic)
    "frozen-action",  # episode 1: 12 consecutive identical actions
    "stats-mismatch",  # stats.json action mean shifted by 0.5
    "missing-video",  # delete one video file (v2: ep1 / v3: the only shard)
    "video-mismatch",  # video shard short by 5 frames vs episode lengths
)


def _trajectory(rng: np.random.Generator, ep: int, corrupt: str | None) -> pd.DataFrame:
    length = EP_LENGTH
    t = np.arange(length) / FPS
    # plausible joint trajectories: sine per joint + small noise
    state = np.stack(
        [
            np.sin(2 * np.pi * (t * 0.2 + j * 0.1)) + 0.01 * rng.standard_normal(length)
            for j in range(STATE_DIM)
        ],
        axis=1,
    )
    action = state + 0.005 * rng.standard_normal((length, ACTION_DIM))
    if corrupt == "nan" and ep == 1:
        state[5, 2] = np.nan
    if corrupt == "frozen-action" and ep == 1:
        action[5:17] = action[5]
    if corrupt == "dup-timestamp" and ep == 0:
        t = np.zeros(length)
    return pd.DataFrame(
        {
            "timestamp": t,
            "episode_index": ep,
            "frame_index": np.arange(length),
            "task_index": 0,
            "observation.state": list(state),
            "action": list(action),
        }
    )


def make_v2(out: Path, corrupt: str | None) -> Path:
    rng = np.random.default_rng(42)
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for ep in range(N_EPISODES):
        df = _trajectory(rng, ep, corrupt)
        df["index"] = np.arange(total_frames, total_frames + EP_LENGTH)
        df.to_parquet(out / "data" / "chunk-000" / f"episode_{ep:06d}.parquet", index=False)
        total_frames += EP_LENGTH

    episodes = [
        {"episode_index": i, "tasks": ["pick up the cube"], "length": EP_LENGTH}
        for i in range(N_EPISODES)
    ]
    if corrupt == "short-episode":
        episodes[2]["length"] = 1

    _write_info(meta, "v2.1", corrupt, total_frames)
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes) + "\n")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up the cube"}) + "\n"
    )

    if corrupt == "missing-file":
        (out / "data" / "chunk-000" / "episode_000001.parquet").unlink()
    _write_videos(out, "v2.1", corrupt)
    _write_videos(out, "v3.0", corrupt)
    _write_stats(out, corrupt)
    return out


def make_v3(out: Path, corrupt: str | None) -> Path:
    rng = np.random.default_rng(42)
    meta = out / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    frames = []
    episodes_meta = []
    total_frames = 0
    for ep in range(N_EPISODES):
        df = _trajectory(rng, ep, corrupt)
        df["index"] = np.arange(total_frames, total_frames + EP_LENGTH)
        df["next.done"] = False
        frames.append(df)
        # short-episode corrupts only the declared length, never the rows —
        # the mismatch stays inside metadata as in real broken datasets
        length = 1 if (corrupt == "short-episode" and ep == 2) else EP_LENGTH
        t0, t1 = total_frames / FPS, (total_frames + EP_LENGTH) / FPS
        episodes_meta.append(
            {
                "episode_index": ep,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + EP_LENGTH,
                f"videos/{VIDEO_KEY}/chunk_index": 0,
                f"videos/{VIDEO_KEY}/file_index": 0,
                f"videos/{VIDEO_KEY}/from_timestamp": t0,
                f"videos/{VIDEO_KEY}/to_timestamp": t1,
                "tasks": ["pick up the cube"],
                "length": length,
            }
        )
        total_frames += EP_LENGTH

    pd.concat(frames, ignore_index=True).to_parquet(
        out / "data" / "chunk-000" / "file-000.parquet", index=False
    )
    pd.DataFrame(episodes_meta).to_parquet(
        out / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
    )
    pd.DataFrame({"task_index": [0]}).to_parquet(meta / "tasks.parquet", index=False)

    _write_info(meta, "v3.0", corrupt, total_frames)

    if corrupt == "missing-file":
        (out / "data" / "chunk-000" / "file-000.parquet").unlink()
    _write_videos(out, "v3.0", corrupt)
    _write_videos(out, "v3.0", corrupt)
    _write_stats(out, corrupt)
    return out


VIDEO_KEY = "observation.images.top"
VIDEO_SIZE = 64


def _write_videos(out: Path, fmt: str, corrupt: str | None) -> None:
    # zero frames are fine: existence, frame count and resolution are what
    # the checks verify; nobody watches fixture videos for content
    import av

    vdir = out / "videos" / VIDEO_KEY / "chunk-000"
    vdir.mkdir(parents=True, exist_ok=True)

    def encode(path: Path, nframes: int) -> None:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = stream.height = VIDEO_SIZE
        stream.pix_fmt = "yuv420p"
        blank = av.VideoFrame.from_ndarray(
            np.zeros((VIDEO_SIZE, VIDEO_SIZE, 3), dtype=np.uint8), format="rgb24"
        )
        for _ in range(nframes):
            for packet in stream.encode(blank):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)
        container.close()

    if fmt == "v2.1":
        for ep in range(N_EPISODES):
            short = 5 if (corrupt == "video-mismatch" and ep == 0) else 0
            encode(vdir / f"episode_{ep:06d}.mp4", EP_LENGTH - short)
        if corrupt == "missing-video":
            (vdir / "episode_000001.mp4").unlink()
    else:
        # v3: all episodes share one shard, cut by from/to timestamps
        short = 5 if corrupt == "video-mismatch" else 0
        encode(vdir / "file-000.mp4", N_EPISODES * EP_LENGTH - short)
        if corrupt == "missing-video":
            (vdir / "file-000.mp4").unlink()


def _write_stats(out: Path, corrupt: str | None) -> None:
    # stats computed from the actual generated frames; corrupt shifts the
    # declared action mean so the recompute check has something to catch
    import numpy as np

    files = sorted(out.glob("data/chunk-*/episode_*.parquet")) or sorted(
        out.glob("data/chunk-*/file-*.parquet")
    )
    # aggregate every file per column: per-file stats would drift from what
    # the checker recomputes over all frames together
    per_col: dict[str, list] = {}
    for f in files:
        df = pd.read_parquet(f)
        for col in ("observation.state", "action"):
            per_col.setdefault(col, []).extend(np.asarray(v, dtype=float) for v in df[col])
    stats = {}
    for col, rows in per_col.items():
        arr = np.stack(rows)
        stats[col] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        }
    if corrupt == "stats-mismatch":
        stats["action"]["mean"] = [m + 0.5 for m in stats["action"]["mean"]]
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))


def _write_info(meta: Path, fmt: str, corrupt: str | None, total_frames: int) -> None:
    version = "v9.9" if corrupt == "bad-version" else fmt
    declared_frames = total_frames + (7 if corrupt == "frame-mismatch" else 0)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": version,
                "robot_type": "fixture-arm",
                "fps": FPS,
                "total_episodes": N_EPISODES,
                "total_frames": declared_frames,
                "total_tasks": 1,
                "chunks_size": 1000,
                "splits": {"train": f"0:{N_EPISODES}"},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": (
                    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
                ),
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
                    "action": {"dtype": "float32", "shape": [ACTION_DIM]},
                    VIDEO_KEY: {"dtype": "video", "shape": [VIDEO_SIZE, VIDEO_SIZE, 3]},
                },
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="output dataset directory")
    parser.add_argument("--format", choices=("v2.1", "v3.0"), default="v2.1")
    parser.add_argument("--corrupt", choices=CORRUPTIONS, default=None)
    args = parser.parse_args()
    if args.out.exists():
        shutil.rmtree(args.out)
    maker = make_v2 if args.format == "v2.1" else make_v3
    maker(args.out, args.corrupt)
    tag = f" (format={args.format}" + (f", corrupt={args.corrupt})" if args.corrupt else ")")
    print(f"fixture written: {args.out}{tag}")


if __name__ == "__main__":
    main()
