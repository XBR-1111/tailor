"""Generate a tiny synthetic LeRobot v2.1 dataset for tests and local dev.

The fixture is deterministic (seeded), lives in milliseconds of runtime and
fits in git. Corruptions are injected with --corrupt to build negative test
cases: each corruption must be caught by at least one expectation.
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
    "missing-file",  # delete episode 1's parquet
    "frame-mismatch",  # info total_frames off by 7
    "bad-version",  # codebase_version set to v9.9
    "short-episode",  # episode 2 length set to 1
    "dup-timestamp",  # episode 0 timestamps all zero (non-monotonic)
)


def make_dataset(out: Path, corrupt: str | None = None) -> Path:
    rng = np.random.default_rng(42)
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "meta"
    (meta).mkdir(exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for ep in range(N_EPISODES):
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
        if corrupt == "dup-timestamp" and ep == 0:
            t = np.zeros(length)
        df = pd.DataFrame(
            {
                "timestamp": t,
                "episode_index": ep,
                "frame_index": np.arange(length),
                "index": np.arange(total_frames, total_frames + length),
                "task_index": 0,
                "observation.state": list(state),
                "action": list(action),
            }
        )
        df.to_parquet(out / "data" / "chunk-000" / f"episode_{ep:06d}.parquet", index=False)
        total_frames += length

    episodes = [
        {"episode_index": i, "tasks": ["pick up the cube"], "length": EP_LENGTH}
        for i in range(N_EPISODES)
    ]
    if corrupt == "short-episode":
        episodes[2]["length"] = 1

    version = "v2.1"
    if corrupt == "bad-version":
        version = "v9.9"
    declared_frames = total_frames
    if corrupt == "frame-mismatch":
        declared_frames += 7

    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": version,
                "robot_type": "fixture-arm",
                "fps": FPS,
                "total_episodes": N_EPISODES,
                "total_frames": declared_frames,
                "total_tasks": 1,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
                    "action": {"dtype": "float32", "shape": [ACTION_DIM]},
                },
            },
            indent=2,
        )
    )
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes) + "\n")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up the cube"}) + "\n"
    )

    if corrupt == "missing-file":
        (out / "data" / "chunk-000" / "episode_000001.parquet").unlink()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="output dataset directory")
    parser.add_argument("--corrupt", choices=CORRUPTIONS, default=None)
    args = parser.parse_args()
    if args.out.exists():
        shutil.rmtree(args.out)
    make_dataset(args.out, args.corrupt)
    tag = f" (corrupt={args.corrupt})" if args.corrupt else ""
    print(f"fixture written: {args.out}{tag}")


if __name__ == "__main__":
    main()
