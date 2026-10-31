"""Input resolution: local path, zip archive, or Hugging Face repo id.

Everything lands as a local directory, then the existing loader runs
unchanged. Remote mode pulls only ``meta/*`` by default (a fast pre-check
against the metadata-level checks); ``--full`` snapshots the whole repo.
"""

import shutil
import tempfile
from pathlib import Path

_REPO_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/")


def looks_like_repo_id(value: str) -> bool:
    """'org/name' shape that is not an existing local path.

    A single '/' with sane characters, and the path does not exist locally —
    existing paths always win so a local dir named like a repo id keeps working.
    """
    if value.count("/") != 1:
        return False
    if not set(value) <= _REPO_ID_CHARS:
        return False
    org, name = value.split("/")
    return bool(org) and bool(name) and not Path(value).exists()


def unpack_zip(zip_path: Path, dest: Path) -> Path:
    """Extract a zip and return the dataset root (peels a single wrapper dir)."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(dest))
    root = _peel_wrapper(dest)
    if not (root / "meta" / "info.json").is_file():
        raise NotADirectoryError(f"no meta/info.json found inside {zip_path}")
    return root


def _peel_wrapper(root: Path) -> Path:
    """descend while a directory wraps exactly one child that holds the dataset"""
    while True:
        children = [c for c in root.iterdir() if c.name != "__MACOSX"]
        if len(children) == 1 and children[0].is_dir() and not (root / "meta").is_dir():
            root = children[0]
            continue
        return root


def download_repo(repo_id: str, dest: Path, full: bool = False) -> Path:
    """Snapshot a HF dataset repo. Meta-only by default, whole repo with full=True."""
    from huggingface_hub import snapshot_download

    patterns = None if full else ["meta/*"]
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=patterns,
    )
    return dest


def resolve_input(value: str, download_dir: Path | None = None, full: bool = False) -> Path:
    """Map a CLI argument to a local dataset directory."""
    path = Path(value)
    if path.exists():
        if path.is_dir():
            return path
        if path.suffix.lower() == ".zip":
            tmp = download_dir or Path(tempfile.mkdtemp(prefix="tailor-zip-"))
            return unpack_zip(path, tmp)
        raise NotADirectoryError(f"{path} is neither a dataset directory nor a .zip")
    if looks_like_repo_id(value):
        base = download_dir or (Path.home() / ".cache" / "tailor-hf")
        dest = base / value.replace("/", "__")
        return download_repo(value, dest, full=full)
    raise FileNotFoundError(f"not a local path, zip, or 'org/name' repo id: {value}")
