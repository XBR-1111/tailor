"""Tests for the version-diff command and the datasheet generator."""

import subprocess
import sys
from pathlib import Path

from tailor.diff import diff_datasets, render_diff
from tailor.loader import load_snapshot


def make_fixture(tmp_path: Path, corrupt: str | None = None, fmt: str = "v2.1") -> Path:
    out = tmp_path / f"ds_{fmt}_{corrupt or 'clean'}"
    cmd = [sys.executable, "scripts/make_fixture.py", str(out), "--format", fmt]
    if corrupt:
        cmd += ["--corrupt", corrupt]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def test_diff_identical_versions_report_nothing(tmp_path: Path) -> None:
    a = make_fixture(tmp_path / "a")
    b = make_fixture(tmp_path / "b")
    report = diff_datasets(a, b)
    assert not report.has_differences()
    assert "No differences found" in render_diff(report)


def test_diff_detects_video_size_change(tmp_path: Path) -> None:
    a = make_fixture(tmp_path / "a")
    b = make_fixture(tmp_path / "b", "video-mismatch")
    report = diff_datasets(a, b)
    assert report.video_resized


def test_diff_detects_episode_length_change(tmp_path: Path) -> None:
    a = make_fixture(tmp_path / "a")
    b = make_fixture(tmp_path / "b", "short-episode")
    report = diff_datasets(a, b)
    assert report.length_changed


def test_diff_detects_feature_change(tmp_path: Path) -> None:
    """A feature dropped from the info.json declaration (data-level drops are
    schema_consistency's job — diff only sees the metadata layer)."""
    import json

    a = make_fixture(tmp_path / "a")
    b = make_fixture(tmp_path / "b")
    info_path = b / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["features"]["observation.state"]
    info_path.write_text(json.dumps(info))
    report = diff_datasets(a, b)
    assert report.features
    rendered = render_diff(report)
    assert "observation.state: removed" in rendered


def test_diff_detects_structural_change(tmp_path: Path) -> None:
    a = make_fixture(tmp_path / "a")
    b = make_fixture(tmp_path / "b", "bad-version")
    report = diff_datasets(a, b)
    assert any("codebase_version" in line for line in report.structural)


def test_datasheet_has_all_sections(tmp_path: Path) -> None:
    from tailor.datasheet import build_datasheet
    from tailor.expectations import run_all

    ds = make_fixture(tmp_path)
    snap = load_snapshot(ds)
    text = build_datasheet(ds, run_all(snap))
    for section in ("## Overview", "## Features", "## Acceptance", "## Lineage"):
        assert section in text
    assert "| episodes | 3 |" in text


def test_diff_cli_exit_codes(tmp_path: Path) -> None:
    a = make_fixture(tmp_path / "a")
    b_same = make_fixture(tmp_path / "b")
    b_diff = make_fixture(tmp_path / "c", "video-mismatch")
    same = subprocess.run([sys.executable, "-m", "tailor.cli", "diff", str(a), str(b_same)])
    assert same.returncode == 0
    differs = subprocess.run([sys.executable, "-m", "tailor.cli", "diff", str(a), str(b_diff)])
    assert differs.returncode == 1
