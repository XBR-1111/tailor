# Contributing to tailor

Thanks for considering a contribution.

## Development setup

```bash
git clone https://github.com/XBR-1111/tailor
cd tailor
uv sync
uv run pytest
```

`uv sync` installs the dev group (ruff, pytest, mypy, coverage). Use
`uv run <cmd>` for everything so the project environment is always used.

The acceptance suite has **no lerobot dependency** — `tailor check` runs on
pyarrow/pandas/PyAV alone. Only `tailor filter` needs it:
`uv run --with 'lerobot[dataset]' tailor filter ...`.

## Making changes

1. Branch from `main`.
2. Add or change a check → extend `scripts/make_fixture.py` with an injected
   corruption and assert the check catches it, on **both** format versions
   (v2.1 and v3.0). A check without a corruption test does not exist.
3. Quality gates (CI enforces all of them):

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src              # types
uv run pytest --cov=tailor   # tests + coverage floor (75%)
```

4. Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:` …).
5. Docs: README.md and README.zh-CN.md are updated **together** — any user-facing
   change lands in both, in the same commit.
6. New checks must expose their tunables as profile keys (with defaults) —
   the check list is the configurable surface, that is the point of the project.

## Design rules

- Expectations receive a `(DatasetSnapshot, cfg)` and never crash: bad data
  degrades to a clean FAIL/SKIP, never a traceback.
- Version differences are normalized in `loader.py`; the expectation layer
  stays format-free.
- Unknown profile keys or check names are a hard error (typos must not
  silently keep defaults).
- `PASS` cannot be remapped by profiles — only WARN/FAIL severity and thresholds.

## Releasing

**Not every push is a release.** `release.yml` publishes only on tag push;
ordinary pushes run CI and stop there. Most merges should carry no tag.

When a release is warranted, the bump follows SemVer strictly — no judgement
calls at tag time:

| change in the release | bump | example |
| --- | --- | --- |
| new user-visible feature (new check, new flag, new profile key) | MINOR | `enabled: false` → 0.8.0 |
| bug fix, internal refactor, performance work, docs, tooling | PATCH | crash fix, bench speedup |
| breaking install/API change (0.x convention: breaking rides MINOR) | MINOR | PyPI rename → 0.6.0 |

Rule of thumb: if the CHANGELOG entry would convince a *user* to upgrade, it
is at least MINOR-worthy on its own; a batch of "won't change your life"
items is a PATCH. Performance work with no new behaviour is a PATCH even
when it took effort — effort is not a version signal.

Procedure: bump `pyproject.toml`, add the CHANGELOG section, then walk the
release checklist, then `git tag vX.Y.Z && git push origin vX.Y.Z` —
Trusted Publishing does the rest.

Release checklist (in order, before tagging):

1. For each CHANGELOG entry in this release, classify it against the table
   above — MINOR-worthy and PATCH-worthy items must not share a release.
   If they do, split the tag or defer the MINOR items to Unreleased.
2. Confirm the section heading version matches `pyproject.toml`.
3. Confirm tests/lint/mypy are green on the commit being tagged.
