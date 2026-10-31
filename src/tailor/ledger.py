"""Provenance ledger (the Register stage): an append-only account of what
tailor did to a dataset, stored inside the dataset so it travels with the data.

One JSON line per event in meta/tailor_ledger.jsonl. Events are written by
filter (into the OUTPUT dataset, as its birth record) and clean (in place).
check never writes — acceptance is read-only.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from tailor.model import DatasetSnapshot

LEDGER_NAME = "tailor_ledger.jsonl"


def dataset_fingerprint(root: Path) -> str:
    """Cheap structural identity: episode count, total frames, file sizes.

    Not cryptographic content hashing — the goal is to notice 'this is a
    different dataset' in milliseconds, not to prove byte-level provenance.
    """
    h = hashlib.sha256()
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        h.update(info_path.read_bytes())
    for f in sorted((root / "data").rglob("*.parquet")) if (root / "data").is_dir() else []:
        st = f.stat()
        h.update(f"{f.relative_to(root)}:{st.st_size};".encode())
    return h.hexdigest()[:16]


def _ledger_path(root: Path) -> Path:
    return root / "meta" / LEDGER_NAME


def read_ledger(root: Path) -> list[dict]:
    p = _ledger_path(root)
    if not p.is_file():
        return []
    events = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def append_event(root: Path, event: dict) -> Path:
    """Append one event (a timestamp is added if missing). Never rewrites."""
    event.setdefault("ts", datetime.now(UTC).isoformat(timespec="seconds"))
    p = _ledger_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return p


def event_for_filter(
    source: Path, source_snap: DatasetSnapshot, drops: dict[int, list[str]], out: Path
) -> dict:
    return {
        "op": "filter",
        "source": {
            "name": source.name,
            "fingerprint": dataset_fingerprint(source),
            "episodes": len(source_snap.episodes or []),
        },
        "dropped": {str(k): v for k, v in sorted(drops.items())},
        "result": {"fingerprint": dataset_fingerprint(out)},
    }


def event_for_clean(report_stats: bool, report_ts_files: int, report_ts_rows: int) -> dict:
    ev: dict = {"op": "clean", "rewrote": []}
    if report_stats:
        ev["rewrote"].append("meta/stats.json")
    if report_ts_files:
        ev["rewrote"].append(
            {"files": report_ts_files, "rows": report_ts_rows, "what": "timestamp"}
        )
    return ev


def render_ledger(events: list[dict]) -> str:
    if not events:
        return "no tailor ledger — this dataset has not been filtered or cleaned by tailor"
    lines = []
    for ev in events:
        ts = ev.get("ts", "?")
        if ev.get("op") == "filter":
            src = ev.get("source", {})
            drops = ev.get("dropped", {})
            lines.append(
                f"{ts}  filter  from {src.get('name')} ({src.get('fingerprint')}, "
                f"{src.get('episodes')} eps) -> dropped {len(drops)}: "
                + ", ".join(f"ep{k} ({v[0]})" for k, v in list(drops.items())[:5])
                + ("…" if len(drops) > 5 else "")
            )
        elif ev.get("op") == "clean":
            what = ev.get("rewrote", [])
            lines.append(f"{ts}  clean   rewrote {what}")
        else:
            lines.append(f"{ts}  {ev.get('op', '?')}   {json.dumps(ev, ensure_ascii=False)[:120]}")
    return "\n".join(lines)
