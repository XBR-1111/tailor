"""tailor: acceptance, filtering, cleaning and provenance for robot datasets."""

try:  # single source of truth: pyproject.toml, via installed metadata
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("lerobot-tailor")
except Exception:  # bare source run (PYTHONPATH=src) with no install record
    __version__ = "unknown"
