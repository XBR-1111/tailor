"""Reproducible acceptance and filtering from one versioned YAML file."""

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from tailor import __version__
from tailor.expectations import run_all
from tailor.filtering import FilterPlan, collect_drops, rewrite_filtered_dataset
from tailor.ledger import append_event, event_for_filter
from tailor.loader import load_snapshot
from tailor.model import Status
from tailor.profile import load_profile, parse_profile
from tailor.source import looks_like_repo_id, resolve_input


@dataclass(frozen=True)
class Recipe:
    path: Path
    digest: str
    dataset: str
    output: Path
    profile: dict
    from_check: bool
    drop_warn: bool
    dedupe: bool
    drop_episodes: set[int]
    min_length: int | None


def _mapping(value: Any, name: str, keys: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) - keys:
        raise ValueError(f"{name} must be a mapping with keys from {', '.join(sorted(keys))}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _flag(rules: dict, key: str, default: bool = False) -> bool:
    value = rules.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"filter.{key} must be true or false")
    return value


def load_recipe(path: Path) -> Recipe:
    path = path.resolve()
    content = path.read_bytes()
    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid recipe YAML: {exc}") from exc
    recipe = _mapping(
        raw, "recipe", {"version", "dataset", "output", "profile", "checks", "filter"}
    )
    if type(recipe.get("version")) is not int or recipe["version"] != 1:
        raise ValueError("recipe.version must be 1")
    dataset = _text(recipe.get("dataset"), "dataset")
    output = path.parent / _text(recipe.get("output"), "output")
    if "profile" in recipe and "checks" in recipe:
        raise ValueError("choose either profile or inline checks")
    if "profile" in recipe:
        profile = load_profile(path.parent / _text(recipe["profile"], "profile"))
    else:
        profile = parse_profile({"checks": recipe.get("checks", {})})
    filters = _mapping(
        recipe.get("filter", {}),
        "filter",
        {"from_check", "drop_warn", "dedupe", "drop_episodes", "min_length"},
    )
    from_check = _flag(filters, "from_check", True)
    drop_warn = _flag(filters, "drop_warn")
    if drop_warn and not from_check:
        raise ValueError("filter.drop_warn requires filter.from_check")
    dedupe = _flag(filters, "dedupe")
    explicit = filters.get("drop_episodes", [])
    if not isinstance(explicit, list) or any(type(idx) is not int or idx < 0 for idx in explicit):
        raise ValueError("filter.drop_episodes must be a list of non-negative episode IDs")
    min_length = filters.get("min_length")
    if min_length is not None and (type(min_length) is not int or min_length < 1):
        raise ValueError("filter.min_length must be a positive integer")
    return Recipe(
        path,
        hashlib.sha256(content).hexdigest(),
        dataset,
        output,
        profile,
        from_check,
        drop_warn,
        dedupe,
        set(explicit),
        min_length,
    )


def _resolve_source(recipe: Recipe, zip_dir: Path) -> Path:
    local = recipe.path.parent / recipe.dataset
    if local.exists():
        return resolve_input(str(local), download_dir=zip_dir, full=True).resolve()
    repo, _, revision = recipe.dataset.partition("@")
    if looks_like_repo_id(repo):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(
                "remote recipes require a Hugging Face dataset commit SHA (org/name@SHA)"
            )
        return resolve_input(recipe.dataset, full=True).resolve()
    raise FileNotFoundError(f"dataset not found: {local}")


def run_recipe(recipe: Recipe, *, apply: bool = False) -> tuple[dict[str, Any], int]:
    with TemporaryDirectory(prefix="tailor-recipe-source-") as temp:
        source = _resolve_source(recipe, Path(temp))
        snap = load_snapshot(source)
        if not snap.readable or (snap.info or {}).get("codebase_version") != "v3.0":
            raise ValueError("recipe requires a readable LeRobot v3.0 dataset")
        findings = run_all(snap, profile=recipe.profile)
        decision = collect_drops(
            snap,
            explicit=recipe.drop_episodes,
            min_length=recipe.min_length,
            dedupe=recipe.dedupe,
            from_check=findings if recipe.from_check else None,
            drop_warn=recipe.drop_warn,
        )
        plan = FilterPlan(decision, findings)
        plan_json = plan.as_dict(source, recipe.output)
        plan_json["dataset"] = recipe.dataset
        profile_digest = hashlib.sha256(
            json.dumps(recipe.profile, sort_keys=True).encode()
        ).hexdigest()
        record = {
            "recipe_sha256": recipe.digest,
            "profile_sha256": profile_digest,
            "tailor_version": __version__,
            "dataset": recipe.dataset,
        }
        result = {
            "schema_version": 1,
            "status": "planned",
            "provenance": record,
            "plan": plan_json,
            "source_checks": [asdict(item) for item in findings],
            "output_checks": None,
            "error": plan.error,
        }
        if plan.error:
            result["status"] = "blocked"
            return result, 1
        if not apply:
            return result, 0

        target = recipe.output
        if target.resolve().is_relative_to(source):
            raise ValueError("output must be outside the source dataset")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".tailor-stage-", dir=target.parent) as temp_output:
            staged = Path(temp_output) / "dataset"
            if decision.n_drop:
                rewrite_filtered_dataset(source, staged, decision)
            else:
                shutil.copytree(
                    source, staged, symlinks=True, ignore=shutil.ignore_patterns(".cache")
                )
            output_snap = load_snapshot(staged)
            recheck = run_all(output_snap, profile=recipe.profile)
            result["output_checks"] = [asdict(item) for item in recheck]
            if not output_snap.readable or any(item.status == Status.FAIL for item in recheck):
                result["status"] = "rejected"
                result["error"] = "output failed acceptance checks"
                return result, 1
            if len(output_snap.episodes or []) != len(decision.keep):
                result["status"] = "rejected"
                result["error"] = "output episode count does not match the filter plan"
                return result, 1
            event = event_for_filter(source, snap, decision.drop, staged)
            event["recipe"] = record
            append_event(staged, event)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"output already exists: {target}")
            staged.rename(target)
        result["status"] = "accepted"
        return result, 0
