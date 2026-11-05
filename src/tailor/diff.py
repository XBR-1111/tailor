"""Version-to-version dataset diff: what changed between two deliveries.

Compares meta-level artifacts only (info.json, episodes metadata, stats.json,
video shard pointers and file sizes) — no parquet parsing, no decoding — so a
diff stays inside the same seconds-scale budget as check.

Two deliveries of "the same" dataset differ in ways that matter downstream:
an episode silently dropped, a feature renamed, a video shard re-encoded,
normalization statistics drifting. The exit code answers "did anything
change": 0 = identical, 1 = differences found.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tailor.loader import load_snapshot

# stats drift beyond this relative change is reported (mean/std are the
# numbers a training config inherits; silent drift breaks comparability)
STATS_DRIFT_TOL = 0.10
# video file size change beyond this share suggests a re-encode even when
# pointers are unchanged (containers can be rewritten in place)
VIDEO_SIZE_TOL = 0.05


@dataclass
class DiffReport:
    """Everything that differs between two dataset versions."""

    old_root: str
    new_root: str
    structural: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    episodes_added: list[int] = field(default_factory=list)
    episodes_removed: list[int] = field(default_factory=list)
    length_changed: list[str] = field(default_factory=list)
    video_repointed: list[str] = field(default_factory=list)
    video_resized: list[str] = field(default_factory=list)
    stats_changed: list[str] = field(default_factory=list)

    def has_differences(self) -> bool:
        return bool(
            self.structural
            or self.features
            or self.episodes_added
            or self.episodes_removed
            or self.length_changed
            or self.video_repointed
            or self.video_resized
            or self.stats_changed
        )


def _summarize(root: Path) -> dict:
    """Meta-level facts of one version, without touching data files."""
    snap = load_snapshot(root)
    info = snap.info or {}
    return {
        "snap": snap,
        "info": info,
        "episodes": {ep.get("episode_index", -1): ep for ep in (snap.episodes or [])},
        "root": root,
    }


def _diff_structural(old: dict, new: dict, out: DiffReport) -> None:
    oi, ni = old["info"], new["info"]
    for key in ("codebase_version", "fps", "total_episodes", "total_frames", "total_tasks"):
        if oi.get(key) != ni.get(key):
            out.structural.append(f"{key}: {oi.get(key)!r} → {ni.get(key)!r}")
    if (oi.get("splits") or {}) != (ni.get("splits") or {}):
        out.structural.append(f"splits: {oi.get('splits')!r} → {ni.get('splits')!r}")


def _diff_features(old: dict, new: dict, out: DiffReport) -> None:
    of = old["info"].get("features") or {}
    nf = new["info"].get("features") or {}
    for key in sorted(set(of) | set(nf)):
        if key not in nf:
            out.features.append(f"{key}: removed")
        elif key not in of:
            out.features.append(f"{key}: added")
        else:
            for attr in ("dtype", "shape", "fps"):
                if of[key].get(attr) != nf[key].get(attr):
                    out.features.append(
                        f"{key}.{attr}: {of[key].get(attr)!r} → {nf[key].get(attr)!r}"
                    )


def _diff_episodes(old: dict, new: dict, out: DiffReport) -> None:
    oe, ne = old["episodes"], new["episodes"]
    out.episodes_added = sorted(set(ne) - set(oe))
    out.episodes_removed = sorted(set(oe) - set(ne))
    for idx in sorted(set(oe) & set(ne)):
        ol, nl = oe[idx].get("length"), ne[idx].get("length")
        if ol != nl:
            out.length_changed.append(f"ep {idx}: length {ol} → {nl}")
        for key in set(oe[idx]) & set(ne[idx]):
            if not key.startswith("videos/"):
                continue
            if oe[idx][key] != ne[idx][key]:
                out.video_repointed.append(f"ep {idx} {key}: {oe[idx][key]} → {ne[idx][key]}")


def _video_files(snap: Any) -> dict[tuple[str, str], Path]:
    """Every video file on disk, keyed by (video_key, relative path).

    Layout differs by version: v3 puts the key first (videos/{key}/chunk-…),
    v2 lets the chunk wrap it (videos/chunk-…/{key}/…). The key is whichever
    path segment is neither 'chunk-*' nor a file.
    """
    root = getattr(snap, "root", None)
    if root is None or not (Path(root) / "videos").is_dir():
        return {}
    out: dict[tuple[str, str], Path] = {}
    for path in (Path(root) / "videos").rglob("*.mp4"):
        rel = str(path.relative_to(Path(root) / "videos"))
        parts = rel.split("/")
        key = next((p for p in parts[:-1] if not p.startswith("chunk-")), "")
        out[(key, rel)] = path
    return out


def _diff_video_sizes(old: dict, new: dict, out: DiffReport) -> None:
    of, nf = _video_files(old["snap"]), _video_files(new["snap"])
    for key in sorted(set(of) & set(nf)):
        so, sn = of[key].stat().st_size, nf[key].stat().st_size
        if so and abs(sn - so) / so > VIDEO_SIZE_TOL:
            out.video_resized.append(f"{key[1]}: {so} → {sn} bytes")


def _diff_stats(old: dict, new: dict, out: DiffReport) -> None:
    import json

    def load(path: Path) -> dict | None:
        p = path / "meta" / "stats.json"
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    os_, ns_ = load(old["root"]), load(new["root"])
    if os_ is None and ns_ is None:
        return
    if os_ is None or ns_ is None:
        out.stats_changed.append("stats.json: " + ("added" if os_ is None else "removed"))
        return
    for col in sorted(set(os_) | set(ns_)):
        if col not in ns_:
            out.stats_changed.append(f"{col}: stats removed")
            continue
        if col not in os_:
            out.stats_changed.append(f"{col}: stats added")
            continue
        for stat in ("mean", "std"):
            om, nm = os_[col].get(stat), ns_[col].get(stat)
            if not om or not nm or len(om) != len(nm):
                if om != nm:
                    out.stats_changed.append(f"{col}.{stat}: shape changed")
                continue
            for i, (a, b) in enumerate(zip(om, nm, strict=True)):
                scale = max(abs(a), 1e-9)
                if abs(b - a) / scale > STATS_DRIFT_TOL:
                    out.stats_changed.append(f"{col}.{stat}[{i}]: {a:.4g} → {b:.4g}")


def diff_datasets(old_root: Path, new_root: Path) -> DiffReport:
    """Compare two dataset versions at meta level; no data files are parsed."""
    out = DiffReport(old_root=str(old_root), new_root=str(new_root))
    old, new = _summarize(old_root), _summarize(new_root)
    _diff_structural(old, new, out)
    _diff_features(old, new, out)
    _diff_episodes(old, new, out)
    _diff_video_sizes(old, new, out)
    _diff_stats(old, new, out)
    return out


def render_diff(report: DiffReport) -> str:
    """Markdown rendering of a diff report."""
    lines = [f"# Dataset diff: {report.old_root} → {report.new_root}", ""]
    if not report.has_differences():
        return "\n".join([*lines, "No differences found."])
    sections = [
        ("Structure", report.structural),
        ("Features", report.features),
        ("Episodes added", [f"ep {i}" for i in report.episodes_added]),
        ("Episodes removed", [f"ep {i}" for i in report.episodes_removed]),
        ("Episode lengths", report.length_changed),
        ("Video pointers", report.video_repointed),
        ("Video file sizes", report.video_resized),
        ("stats.json", report.stats_changed),
    ]
    for title, items in sections:
        if items:
            lines.append(f"## {title} ({len(items)})")
            lines.extend(f"- {item}" for item in items[:20])
            if len(items) > 20:
                lines.append(f"- … and {len(items) - 20} more")
            lines.append("")
    return "\n".join(lines)
