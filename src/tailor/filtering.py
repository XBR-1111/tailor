"""Episode-level filtering rules (the Cut stage).

Pure functions only: rules + a snapshot (and optionally check results) -> a
drop set with per-episode reasons. The actual dataset rewrite lives in the CLI
layer behind the [lerobot] extra so the acceptance core stays dependency-free.
"""

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from tailor.model import CheckResult, DatasetSnapshot, Status

if TYPE_CHECKING:
    import pandas as pd


def trajectory_digest(df: "pd.DataFrame") -> str | None:
    """Byte-level digest of one episode's state+action trajectory.

    Shared by the duplicate_episodes check (Accept) and --dedupe (Cut) so the
    two can never disagree about what counts as a duplicate. float64 coercion
    first: identical trajectories stored as float32/float64 must hash equal,
    and NaN bytes hash equal to NaN bytes (which is the desired identity here).
    """
    cols = [c for c in ("observation.state", "action") if c in df.columns]
    if not cols:
        return None
    blob = b"".join(
        np.ascontiguousarray(np.asarray(v, dtype=np.float64)).tobytes() for c in cols for v in df[c]
    )
    return hashlib.sha256(blob).hexdigest()


@dataclass
class DropDecision:
    keep: set[int] = field(default_factory=set)
    drop: dict[int, list[str]] = field(default_factory=dict)

    @property
    def n_drop(self) -> int:
        return len(self.drop)


def parse_episode_list(spec: str) -> set[int]:
    """'3,17,23' -> {3,17,23}; raises ValueError on junk (negatives included:
    episode indices start at 0, a negative is always a typo)."""
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"invalid episode index {part!r}")
        out.add(int(part))
    return out


def collect_drops(
    snap: DatasetSnapshot,
    *,
    explicit: set[int] | None = None,
    min_length: int | None = None,
    dedupe: bool = False,
    from_check: list[CheckResult] | None = None,
    drop_warn: bool = False,
) -> DropDecision:
    """Combine rules into one drop set; later rules never rescue earlier drops."""
    decision = DropDecision()
    episodes = snap.episodes or []
    decision.keep = {ep["episode_index"] for ep in episodes if "episode_index" in ep}

    def _drop(idx: int, reason: str) -> None:
        decision.drop.setdefault(idx, []).append(reason)

    if explicit:
        for idx in sorted(explicit):
            _drop(idx, "explicit --drop-episodes")
    if min_length is not None:
        for ep in episodes:
            if ep.get("length", 0) < min_length:
                _drop(
                    ep["episode_index"],
                    f"length {ep.get('length', 0)} < min-length {min_length}",
                )
    if dedupe:
        # duplicate episodes are detected by value: drop every later member of
        # each identical group, keep the first
        seen: dict[str, int] = {}
        for idx, df in sorted(snap.frames.items()):
            h = trajectory_digest(df)
            if h is None:
                continue
            if h in seen:
                _drop(idx, f"duplicate of episode {seen[h]}")
            else:
                seen[h] = idx
    if from_check is not None:
        trigger = {Status.FAIL} | ({Status.WARN} if drop_warn else set())
        for r in from_check:
            if r.status not in trigger:
                continue
            eps = r.details.get("episodes") or []
            for idx in eps:
                _drop(idx, f"{r.name} ({r.status.value})")
    decision.keep -= set(decision.drop)
    return decision


def render_decision(decision: DropDecision, total: int) -> str:
    lines = [f"episodes: {total} total -> keep {len(decision.keep)}, drop {decision.n_drop}"]
    for idx in sorted(decision.drop):
        reasons = "; ".join(decision.drop[idx])
        lines.append(f"  drop ep {idx}: {reasons}")
    return "\n".join(lines)
