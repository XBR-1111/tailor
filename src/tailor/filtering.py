"""Episode-level filtering rules (the Cut stage).

Pure functions only: rules + a snapshot (and optionally check results) -> a
drop set with per-episode reasons. The actual dataset rewrite lives in the CLI
layer behind the [lerobot] extra so the acceptance core stays dependency-free.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tailor.model import CheckResult, DatasetSnapshot, Status


@dataclass
class DropDecision:
    keep: set[int] = field(default_factory=set)
    drop: dict[int, list[str]] = field(default_factory=dict)

    @property
    def n_drop(self) -> int:
        return len(self.drop)


@dataclass
class FilterPlan:
    decision: DropDecision
    findings: list[CheckResult] | None = None

    @property
    def blockers(self) -> list[CheckResult]:
        return [
            result
            for result in self.findings or []
            if result.status == Status.FAIL and not result.details.get("episodes")
        ]

    @property
    def error(self) -> str | None:
        if self.decision.n_drop and not self.decision.keep:
            return "filter would remove every episode"
        if not self.decision.n_drop and any(r.status == Status.FAIL for r in self.findings or []):
            return "source rejected and no episodes selected for removal"
        return None

    def as_dict(self, source: Path, output: Path) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": str(source),
            "output": str(output),
            "keep": sorted(self.decision.keep),
            "drop": [
                {"episode_index": index, "reasons": self.decision.drop[index]}
                for index in sorted(self.decision.drop)
            ],
            "unassigned_failures": [asdict(result) for result in self.blockers],
            "error": self.error,
        }


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
        unknown = explicit - decision.keep
        if unknown:
            raise ValueError(f"unknown episode indices: {', '.join(map(str, sorted(unknown)))}")
        for idx in sorted(explicit):
            _drop(idx, "explicit --drop-episodes")
    if min_length is not None:
        for ep in episodes:
            if ep.get("length", 0) < min_length:
                _drop(
                    ep["episode_index"],
                    f"length {ep.get('length', 0)} < min-length {min_length}",
                )
    duplicate_status = None
    if from_check is not None:
        trigger = {Status.FAIL} | ({Status.WARN} if drop_warn else set())
        for r in from_check:
            if r.status not in trigger:
                continue
            if r.name == "duplicate_episodes" and r.details.get("episodes"):
                duplicate_status = r.status
                continue
            for idx in r.details.get("episodes") or []:
                if type(idx) is int and idx in decision.keep:
                    _drop(idx, f"{r.name} ({r.status.value})")
    if dedupe or duplicate_status is not None:
        # Keep the first eligible member after the other rules. Keeping the
        # original group's first member could remove every valid copy.
        seen: dict[str, int] = {}
        for idx in sorted((decision.keep & snap.frames.keys()) - decision.drop.keys()):
            digest = snap.trajectory_digest(idx)
            if digest is None:
                continue
            if digest not in seen:
                seen[digest] = idx
                continue
            if dedupe:
                _drop(idx, f"duplicate of episode {seen[digest]}")
            if duplicate_status is not None:
                _drop(idx, f"duplicate_episodes ({duplicate_status.value})")
    decision.keep -= set(decision.drop)
    return decision


def render_decision(decision: DropDecision, total: int) -> str:
    lines = [f"episodes: {total} total -> keep {len(decision.keep)}, drop {decision.n_drop}"]
    for idx in sorted(decision.drop):
        reasons = "; ".join(decision.drop[idx])
        lines.append(f"  drop ep {idx}: {reasons}")
    return "\n".join(lines)


def rewrite_filtered_dataset(source: Path, output: Path, decision: DropDecision) -> None:
    """Use the optional LeRobot writer to apply a previously validated plan."""
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(source.name, root=str(source))
    delete_episodes(dataset, sorted(decision.drop), output_dir=str(output))
