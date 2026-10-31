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
