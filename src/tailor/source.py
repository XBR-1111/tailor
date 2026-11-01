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


def download_repo(
    repo_id: str, dest: Path, full: bool = False, revision: str | None = None
) -> Path:
    """Snapshot a HF dataset repo. Meta-only by default, whole repo with full=True.

    revision follows the HF `repo@ref` convention: LeRobot datasets keep each
    format version on its own ref (v2.1 / v3.0 tags, main = latest), so
    checking an older format means checking that ref.
    """
    from huggingface_hub import snapshot_download

    patterns = None if full else ["meta/*"]
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        allow_patterns=patterns,
    )
    return dest


def resolve_input(value: str, download_dir: Path | None = None, full: bool = False) -> Path:
    """Map a CLI argument to a local dataset directory.

    'org/name@ref' addresses a specific ref of an HF dataset (a branch or a
    tag — LeRobot datasets tag each format version, e.g. lerobot/pusht@v2.1).
    The ref is stripped before the local-path check so a directory literally
    named 'a@b' still resolves locally first.
    """
    # local paths win BEFORE '@' splitting: a cache dir literally named
    # 'repo__name@v2.1' (tailor's own cache slug) is a path, not a ref spec
    raw_path = Path(value)
    if raw_path.exists():
        if raw_path.is_dir():
            return raw_path
        if raw_path.suffix.lower() == ".zip":
            tmp = download_dir or Path(tempfile.mkdtemp(prefix="tailor-zip-"))
            return unpack_zip(raw_path, tmp)
        raise NotADirectoryError(f"{raw_path} is neither a dataset directory nor a .zip")
    repo_id, _, revision = value.partition("@")
    used_ref = revision or None
    if looks_like_repo_id(repo_id):
        base = download_dir or (Path.home() / ".cache" / "tailor-hf")
        # cache key must include the ref: v2.1 and v3.0 of one repo are
        # different datasets and must not overwrite each other
        slug = repo_id.replace("/", "__") + (f"@{revision}" if revision else "")
        dest = base / slug
        return download_repo(repo_id, dest, full=full, revision=used_ref)
    raise FileNotFoundError(f"not a local path, zip, or 'org/name[@ref]' repo id: {value}")
