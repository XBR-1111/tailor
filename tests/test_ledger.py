"""Ledger tests: fingerprints, append-only events, filter birth records."""

import json
from pathlib import Path

from tailor.ledger import (
    append_event,
    dataset_fingerprint,
    read_ledger,
    render_ledger,
)
from test_expectations import make_fixture


def test_fingerprint_stable_and_sensitive(tmp_path: Path) -> None:
    a = make_fixture(tmp_path / "a")
    b = make_fixture(tmp_path / "b")  # same generator, different directory
    fa, fb = dataset_fingerprint(a), dataset_fingerprint(b)
    # parquet bytes differ per run metadata; identity must at least distinguish
    # a dataset from a modified copy of itself
    import shutil

    c = tmp_path / "c"
    shutil.copytree(a, c)
    assert dataset_fingerprint(c) == fa
    (c / "data" / "chunk-000").mkdir(exist_ok=True)
    # touch a data file's size -> fingerprint changes
    f = next((c / "data" / "chunk-000").glob("*.parquet"))
    f.write_bytes(f.read_bytes() + b"\x00")
    assert dataset_fingerprint(c) != fa
    _ = fb  # both exist


def test_append_then_read_roundtrip(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path)
    append_event(ds, {"op": "clean", "rewrote": ["meta/stats.json"]})
    append_event(ds, {"op": "filter", "dropped": {"3": ["explicit"]}})
    events = read_ledger(ds)
    assert [e["op"] for e in events] == ["clean", "filter"]
    assert all("ts" in e for e in events)  # timestamps auto-added
    ledger_file = ds / "meta" / "tailor_ledger.jsonl"
    assert ledger_file.is_file()
    lines = ledger_file.read_text().strip().split("\n")
    assert len(lines) == 2  # one JSON object per line
    json.loads(lines[0])  # each line parses


def test_render_ledger_formats_events() -> None:
    text = render_ledger(
        [
            {
                "ts": "2026-10-31T00:00:00+00:00",
                "op": "filter",
                "source": {"name": "raw", "fingerprint": "abc123", "episodes": 38},
                "dropped": {"37": ["explicit --drop-episodes"]},
            },
            {"ts": "2026-10-31T00:05:00+00:00", "op": "clean", "rewrote": ["meta/stats.json"]},
        ]
    )
    assert "filter" in text and "raw" in text and "ep37" in text
    assert "clean" in text and "stats.json" in text


def test_render_empty_ledger(tmp_path: Path) -> None:
    text = render_ledger([])
    assert "no tailor ledger" in text
