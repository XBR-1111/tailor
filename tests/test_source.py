"""Input resolution tests: repo-id detection, zip unpacking, dispatch.

No network: the download path is monkeypatched, zip cases use real fixtures.
"""

import zipfile
from pathlib import Path

import pytest

from tailor.source import looks_like_repo_id, resolve_input, unpack_zip
from test_expectations import make_fixture


def test_repo_id_detection() -> None:
    assert looks_like_repo_id("lerobot/unitreeh1_fold_clothes")
    assert not looks_like_repo_id("a-b_c/d.e-f")  # dots not allowed
    assert not looks_like_repo_id("some/local/path")  # two slashes
    assert not looks_like_repo_id("plainname")
    assert not looks_like_repo_id("/etc")  # leading slash, counts != 1


def test_existing_path_beats_repo_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a local dir literally named 'lerobot/foo' must load locally, never hit the hub
    local = tmp_path / "lerobot" / "foo"
    local.mkdir(parents=True)
    out = resolve_input(str(local))
    assert out == local


def test_zip_dataset_unwraps_single_dir(tmp_path: Path) -> None:
    ds = make_fixture(tmp_path / "src")
    inner = tmp_path / "wrap" / "dataset"
    inner.mkdir(parents=True)
    for f in ds.rglob("*"):
        if f.is_file():
            (inner / f.relative_to(ds)).parent.mkdir(parents=True, exist_ok=True)
            f.rename(inner / f.relative_to(ds))
    z = tmp_path / "ds.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in (tmp_path / "wrap").rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(tmp_path / "wrap"))
    out = unpack_zip(z, tmp_path / "out")
    assert (out / "meta" / "info.json").is_file()


def test_zip_rejected_without_meta(tmp_path: Path) -> None:
    z = tmp_path / "empty.zip"
    with zipfile.ZipFile(z, "w"):
        pass  # a valid zip with no entries
    with pytest.raises(NotADirectoryError):
        unpack_zip(z, tmp_path / "out2")


def test_remote_dispatch_calls_downloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_download(repo_id: str, dest: Path, full: bool = False) -> Path:
        calls.update(repo_id=repo_id, full=full)
        (dest / "meta").mkdir(parents=True, exist_ok=True)
        (dest / "meta" / "info.json").write_text("{}")
        return dest

    import tailor.source as src

    monkeypatch.setattr(src, "download_repo", fake_download)
    out = resolve_input("lerobot/some_dataset", download_dir=tmp_path, full=True)
    assert calls == {"repo_id": "lerobot/some_dataset", "full": True}
    assert out == tmp_path / "lerobot__some_dataset"


def test_meta_only_run_skips_file_checks(tmp_path: Path) -> None:
    # meta-only remote pre-check must SKIP file-presence checks, not fake-FAIL
    from tailor.expectations import run_all
    from tailor.loader import load_snapshot
    from tailor.model import Status

    ds = make_fixture(tmp_path / "src2")
    import shutil as _sh

    _sh.rmtree(ds / "data")
    _sh.rmtree(ds / "videos")
    st = {r.name: r.status for r in run_all(load_snapshot(ds))}
    assert st["episode_files"] == Status.SKIP
    assert st["video_files"] == Status.SKIP
    assert st["video_streams"] == Status.SKIP
