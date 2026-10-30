"""Core data structures shared by loader, expectations and report."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
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
    """

    root: Any  # pathlib.Path
    info: dict | None = None
    episodes: list[dict] | None = None
    tasks: list[dict] | None = None
    # per-episode sampled parquet frames: {episode_index: pandas.DataFrame}
    frames: dict[int, Any] = field(default_factory=dict)

    @property
    def readable(self) -> bool:
        return self.info is not None and self.episodes is not None


def summarize(results: list[CheckResult]) -> tuple[int, int, int]:
    """(failures, warns, passes) — drives both the report footer and the exit code."""
    fails = sum(r.status == Status.FAIL for r in results)
    warns = sum(r.status == Status.WARN for r in results)
    passes = sum(r.status == Status.PASS for r in results)
    return fails, warns, passes
