"""Core data structures shared by loader, expectations and report."""

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    """Outcome of one expectation against one dataset snapshot."""

    name: str
    status: Status
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSnapshot:
    """Everything an expectation is allowed to look at.

    Loaded lazily and defensively: a dataset that cannot even be read yields
    None fields, and expectations degrade to FAIL instead of crashing.

    Column access goes through col_array()/trajectory_digest() — both cache
    their conversions so the frame-walking checks convert each episode column
    to a numpy array exactly once per run, no matter how many checks read it.
    """

    root: Any  # pathlib.Path
    info: dict | None = None
    episodes: list[dict] | None = None
    tasks: list[dict] | None = None
    # per-episode sampled parquet frames: {episode_index: pandas.DataFrame}
    frames: dict[int, Any] = field(default_factory=dict)
    _col_cache: dict[tuple[int, str], np.ndarray | None] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def readable(self) -> bool:
        return self.info is not None and self.episodes is not None

    def col_array(self, idx: int, col: str) -> np.ndarray | None:
        """(n_frames, dim) float64 array of one episode column, cached.

        None when the episode is unreadable, empty, or lacks the column —
        callers treat that as 'cannot judge', matching the defensive style.
        """
        key = (idx, col)
        if key not in self._col_cache:
            df = self.frames.get(idx)
            if df is None or col not in df.columns or len(df) == 0:
                self._col_cache[key] = None
            else:
                self._col_cache[key] = np.stack([np.asarray(v, dtype=float) for v in df[col]])
        return self._col_cache[key]

    def trajectory_digest(self, idx: int) -> str | None:
        """Byte-level digest of one episode's state+action trajectory.

        Shared by the duplicate_episodes check (Accept) and --dedupe (Cut) so
        the two can never disagree about what counts as a duplicate. float64
        coercion first: identical trajectories stored as float32/float64 must
        hash equal, and NaN bytes hash equal to NaN bytes (which is the
        desired identity here).
        """
        h = hashlib.sha256()
        parts = 0
        for col in ("observation.state", "action"):
            arr = self.col_array(idx, col)
            if arr is not None:
                h.update(np.ascontiguousarray(arr).tobytes())
                parts += 1
        return h.hexdigest() if parts else None


def summarize(results: list[CheckResult]) -> tuple[int, int, int]:
    """(failures, warns, passes) — drives both the report footer and the exit code."""
    fails = sum(r.status == Status.FAIL for r in results)
    warns = sum(r.status == Status.WARN for r in results)
    passes = sum(r.status == Status.PASS for r in results)
    return fails, warns, passes
