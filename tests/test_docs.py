"""Docs must match the code: a config-driven tool sells its reference docs.

Every assertion here failed at least once during the docs rollout — a check
key drifted from its output name (`episode_files_exist` vs `episode_files`),
and a tunable table can silently lag DEFAULTS. Keep them tests so the next
change cannot repeat either.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from tailor.expectations import EXPECTATIONS, run_all
from tailor.loader import load_snapshot
from tailor.profile import DEFAULTS, load_profile

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def _fixture(tmp_path: Path) -> Path:
    out = tmp_path / "ds"
    subprocess.run(
        [sys.executable, "scripts/make_fixture.py", str(out)],
        check=True,
        capture_output=True,
        cwd=ROOT,
    )
    return out


def test_check_names_match_output_names(tmp_path: Path) -> None:
    """A profile addresses checks by their registry key, so the key must equal
    the name the check reports — otherwise the check is unconfigurable."""
    names = {r.name for r in run_all(load_snapshot(_fixture(tmp_path)))}
    assert set(EXPECTATIONS) == names


def test_configuration_doc_lists_every_tunable() -> None:
    md = (DOCS / "configuration.md").read_text()
    documented = set(re.findall(r"\| `(\w+)` \| `(\w+)` \|", md))
    expected = {(check, key) for check, cfg in DEFAULTS.items() for key in cfg}
    assert documented == expected, {
        "undocumented": sorted(expected - documented),
        "stale": sorted(documented - expected),
    }


@pytest.mark.parametrize("doc", ["docs/configuration.md", "README.md", "README.zh-CN.md"])
def test_documented_acceptance_profiles_load(doc: str, tmp_path: Path) -> None:
    md = (ROOT / doc).read_text()
    profiles = re.findall(r"^```(?:yaml|yml)[ \t]*\n(.*?)^```[ \t]*$", md, re.M | re.S)
    if doc.startswith("README"):
        # The READMEs also show a GitHub Actions workflow, which is not a profile.
        profiles = [example for example in profiles if example.startswith("# my_requirements.yaml")]
    assert profiles, f"no YAML profile examples in {doc}"
    for index, example in enumerate(profiles, start=1):
        path = tmp_path / f"profile-{index}.yaml"
        path.write_text(example)
        load_profile(path)


def test_checks_doc_covers_every_check() -> None:
    md = (DOCS / "checks.md").read_text()
    missing = [name for name in EXPECTATIONS if f"`{name}`" not in md]
    assert not missing, missing
    assert f"The {len(EXPECTATIONS)} acceptance checks" in md


def test_checks_doc_corruptions_exist() -> None:
    md = (DOCS / "checks.md").read_text()
    fixture_src = (ROOT / "scripts" / "make_fixture.py").read_text()
    known = set(re.findall(r'^\s+"([\w-]+)",', fixture_src, re.M))
    cited = set(re.findall(r"--corrupt ([\w-]+)", md))
    assert cited <= known, cited - known


def test_cli_doc_flags_match_cli_source() -> None:
    md = (DOCS / "cli.md").read_text()
    src = (ROOT / "src" / "tailor" / "cli.py").read_text()
    code_flags = set(re.findall(r'"(--[\w-]+)"', src))
    doc_flags = set(re.findall(r"--[A-Za-z][\w-]*", md))
    assert code_flags <= doc_flags, f"undocumented: {code_flags - doc_flags}"
    assert doc_flags <= code_flags, f"invented: {doc_flags - code_flags}"
