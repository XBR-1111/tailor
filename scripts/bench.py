"""Benchmark the acceptance suite on a synthetic dataset of chosen scale.

Usage:
    PYTHONPATH=src:scripts python scripts/bench.py --episodes 300 --frames 500

Times load_snapshot and run_all separately (cold caches each round) so the
loader and the checks can be tracked independently. Fixtures persist under
/tmp between runs (delete to regenerate). Deterministic; wall-clock.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    import make_fixture

    from tailor.expectations import run_all
    from tailor.loader import _read_v2_episode_frames, load_snapshot

    make_fixture.set_scale(args.episodes, args.frames)

    for fmt in ("v2.1", "v3.0"):
        root = Path(f"/tmp/tailor-bench-{fmt}")
        if not (root / "meta" / "info.json").is_file():
            if root.exists():
                shutil.rmtree(root)
            if fmt == "v2.1":
                make_fixture.make_v2(root, None)
            else:
                make_fixture.make_v3(root, None)

        load_times, check_times = [], []
        for _ in range(args.repeat):
            _read_v2_episode_frames.cache_clear()  # cold loader cache each round
            t0 = time.perf_counter()
            snap = load_snapshot(root)
            t1 = time.perf_counter()
            results = run_all(snap)
            t2 = time.perf_counter()
            assert results, "no results"
            load_times.append(t1 - t0)
            check_times.append(t2 - t1)
        n_frames = args.episodes * args.frames
        print(
            f"{fmt}: {args.episodes} eps x {args.frames} frames ({n_frames} total) | "
            f"load {min(load_times):.3f}s | checks {min(check_times):.3f}s | "
            f"total {min(load_times) + min(check_times):.3f}s (best of {args.repeat})"
        )


if __name__ == "__main__":
    main()
