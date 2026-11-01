"""Acceptance profiles: configurable dataset requirements.

The suite ships with sane defaults, but "what counts as acceptable" differs
per robot, task and training config. A profile (YAML) declares per-check
thresholds and severities; tailor is the engine that executes it.

    # my_requirements.yaml
    checks:
      episode_lengths:
        min: 10
      timestamps:
        spacing_tolerance: 0.2
      fps:
        allowed: [30, 50]

Unknown check names or keys are rejected loudly — a typo in a requirements
file must fail fast, not silently keep a default that was meant to change.
"""

from pathlib import Path

import yaml

from tailor.expectations import EXPECTATIONS
from tailor.model import Status

DEFAULTS: dict = {
    "codebase_version": {"allowed": ["v2.0", "v2.1", "v3.0"]},
    "fps": {"allowed": None},  # None = any positive fps
    "episode_lengths": {"min": 2},
    "timestamps": {"spacing_tolerance": 0.1},
    "stats_recompute": {"mean_tolerance": 0.05},
    "video_streams": {"frame_tolerance": 1},
    "zero_variance": {"eps": 1e-6},
    "training_readiness": {"chunk_size": None},
    "action_quality": {
        "frozen_frames": 10,
        "clip_run": 5,
        "jump_sigma": 8,
        "jump_ratio": 0.01,
    },
}

# severities a profile may remap; PASS is never remappable
_REMAPPABLE = {Status.WARN, Status.FAIL}


class ProfileError(Exception):
    """Raised when a profile is structurally invalid."""


def load_profile(path: Path) -> dict:
    """Parse and validate a profile file into a per-check config dict."""
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ProfileError(f"invalid YAML: {e}") from e
    if not isinstance(raw, dict) or "checks" not in raw:
        raise ProfileError("profile must be a mapping with a top-level 'checks' key")
    checks = raw["checks"]
    if not isinstance(checks, dict):
        raise ProfileError("'checks' must be a mapping")

    profile: dict = {}
    for name, cfg in checks.items():
        if name not in EXPECTATIONS:
            known = ", ".join(EXPECTATIONS)
            raise ProfileError(f"unknown check '{name}' (known: {known})")
        if cfg is None:
            continue
        if not isinstance(cfg, dict):
            raise ProfileError(f"checks.{name} must be a mapping")
        defaults = DEFAULTS.get(name, {})
        merged = dict(defaults)
        for key, value in cfg.items():
            if key == "severity":
                sev = str(value).upper()
                if sev not in ("WARN", "FAIL"):
                    raise ProfileError(
                        f"checks.{name}.severity must be WARN or FAIL, got {value!r}"
                    )
                merged["severity"] = sev
            elif key == "enabled":
                # orthogonal to severity: a disabled check is not run at all
                # (its SKIP row names the profile as the reason)
                if not isinstance(value, bool):
                    raise ProfileError(
                        f"checks.{name}.enabled must be true or false, got {value!r}"
                    )
                merged["enabled"] = value
            elif key not in defaults:
                raise ProfileError(f"checks.{name} has no configurable key '{key}'")
            else:
                merged[key] = value
        profile[name] = merged
    return profile


def default_profile() -> dict:
    """The built-in defaults, same shape as load_profile output."""
    return {name: dict(cfg) for name, cfg in DEFAULTS.items()}


def remap(result_status: Status, cfg: dict) -> Status:
    """Apply a profile's severity remap to one result."""
    target = cfg.get("severity")
    if target and result_status in _REMAPPABLE:
        return Status(target)
    return result_status
