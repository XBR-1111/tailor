# tailor

**Refine distributed robot datasets into training-ready data.**

[![CI](https://github.com/XBR-1111/tailor/actions/workflows/ci.yml/badge.svg)](https://github.com/XBR-1111/tailor/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

English | [简体中文](README.zh-CN.md)

tailor is a data-refinement pipeline for robot learning datasets (LeRobot **v2.0 / v2.1 / v3.0**):

- **Accept** — verify incoming distributed datasets: format-version compliance, metadata-vs-data consistency ("does the package match the label"), transfer integrity, dims and stats vs your training config
- **Cut** — episode-level filtering (frame-level issues escalate to episode removal to preserve action-chunk coherence)
- **Sew** — deterministic cleaning only: timestamp rebuild, stats recompute, video re-encode. No imputation
- **Re-check** — cleaned output must pass the acceptance suite again before release
- **Register** — provenance ledger: source version, dropped episodes + why, recipe hash

Declarative recipes (YAML) in git; every dataset is reproducible and auditable.

*tailor-made datasets for your training.*

## Quickstart

```bash
pip install lerobot-tailor      # once published; until then:
pip install git+https://github.com/XBR-1111/tailor

tailor check /path/to/lerobot/dataset
tailor check dataset.zip                # zip archive (wrapper dirs peeled)
tailor check lerobot/unitreeh1_fold_clothes   # HF repo id: meta pre-check, no full download
tailor check lerobot/... --full               # download everything, all 22 checks
```

```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ check             ┃ status ┃ detail       ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━┩
│ info_json         │  PASS  │              │
│ codebase_version  │  PASS  │ v3.0         │
│ fps               │  PASS  │ 50           │
│ episodes_metadata │  PASS  │ 38 episodes  │
│ episode_files     │  PASS  │              │
│ total_frames      │  PASS  │ 19000        │
│ episode_lengths   │  PASS  │              │
│ task_indices      │  PASS  │ 1 task       │
│ state_action_dims │  PASS  │              │
│ nan_inf           │  PASS  │              │
│ timestamps        │  PASS  │              │
└───────────────────┴────────┴──────────────┘
ACCEPTED  ·  11 passed, 0 warn, 0 failed  ·  11 skipped (meta-only run)
```

Remote mode pulls only `meta/*` by default (seconds, a few MB) and runs every
metadata-level check — file-presence checks SKIP honestly instead of faking a
FAIL. `HF_ENDPOINT` is respected (mirrors work); downloads cache under
`~/.cache/tailor-hf/`.

Exit codes make it a CI gate: **0** accepted · **1** rejected · **2** unreadable
(`--fail-on-warn` makes WARN exit 1 too). `--json` emits machine-readable output,
`--markdown report.md` writes a PR/dataset-card friendly report, `--max-episodes N`
samples the first N episodes of huge datasets (a convenience sample —
tail-end episodes, a common corruption site, stay uncovered).

## Filter — the Cut stage

`tailor filter` rewrites a dataset without the episodes you don't want, using
lerobot's official `delete_episodes` engine (so the output is format-compatible
by construction — parquet, videos and stats are rebuilt consistently). The
result is **re-checked automatically**; its verdict is printed.

```bash
# drop explicit episodes (say, the ones a report flagged)
tailor filter /data/ds --drop-episodes 5,17,23 -o /data/ds_clean

# drop everything FAIL-level checks can pin to an episode (--drop-warn adds WARN)
tailor filter /data/ds --from-check --drop-warn

# rule-based
tailor filter /data/ds --min-length 100        # too short for your policy
tailor filter /data/ds --dedupe                # byte-identical trajectories, keep first
```

Requires `pip install 'lerobot-tailor[lerobot]'` (the acceptance suite itself stays
dependency-free). Currently v3.0 datasets. Every drop is printed with its
reason — no silent removals.

## Clean — deterministic repairs

`tailor clean` fixes what deterministic math can fix: `meta/stats.json`
recomputed from the parquet data (video entries preserved), timestamps
rebuilt as `frame_index / fps`. Dry-run by default; `--apply` writes, backing
up every replaced file to `<dataset>_pre_clean/` outside the dataset root, then
re-checks the dataset and prints the verdict. No lerobot dependency.

```bash
tailor clean /data/ds            # dry-run: what would change, nothing written
tailor clean /data/ds --apply    # repair + backup + automatic re-check
```

## Register — the provenance ledger

Every `filter` and `clean --apply` appends one event to
`meta/tailor_ledger.jsonl` **inside the dataset**, so the processing history
travels with the data: filter writes a birth record into the output (source
fingerprint, dropped episodes and why), clean appends what it rewrote.

```bash
tailor provenance /data/ds_filtered
# 2026-10-31T02:37:45+00:00  filter  from raw (b6b8be2337b9aae4, 38 eps) -> dropped 1: ep37 (explicit --drop-episodes)
# 2026-10-31T02:37:47+00:00  clean   rewrote ['meta/stats.json']
```

`check` never writes — acceptance is read-only. Fingerprints are structural
(episode metadata + data file sizes), for identity not cryptographic proof.

## Configurable acceptance profiles

What counts as acceptable differs per robot, task and training config. A
profile (YAML in git — your dataset SLA, versioned and auditable) overrides
per-check thresholds and severities; tailor is the engine that executes it:

```yaml
# my_requirements.yaml
checks:
  codebase_version:
    allowed: [v2.1]     # this org only accepts v2.1
  fps:
    allowed: [30, 50]   # hardware reality
  episode_lengths:
    min: 100            # policy window needs headroom
    severity: fail      # short episodes are a hard reject here
  timestamps:
    spacing_tolerance: 0.2
```

```bash
tailor check /data/dataset --profile my_requirements.yaml
```

Any check can be remapped between WARN and FAIL. Unknown check names or keys
in a profile fail fast with a clear error — a typo in your requirements must
not silently keep a default you meant to change.

## The 22 acceptance checks (v0.1)

| check | catches |
| --- | --- |
| info_json | missing / invalid meta/info.json |
| codebase_version | format version outside v2.0/v2.1/v3.0 |
| fps | non-positive or unusual fps |
| episodes_metadata | missing/invalid episode metadata |
| episode_files | data shard/parquet missing on disk (v2 files, v3 shard pointers) |
| total_frames | declared total ≠ sum of episode lengths |
| episode_lengths | zero/negative lengths; very short episodes that break chunked sampling |
| task_indices | task references out of range; declared task count mismatch |
| state_action_dims | state/action dims ≠ info.json features |
| nan_inf | non-finite values in state/action |
| timestamps | non-monotonic timestamps; frame spacing deviating >10% from 1/fps |
| stats_recompute | declared stats.json mean/std drifting from recomputed values |
| action_quality | frozen (stuck) action runs; one action dim pegged at its declared range edge; sudden jumps beyond k·sigma |
| video_files | declared video streams missing on disk (v2 per-episode files, v3 shard pointers) |
| video_streams | undecodable video; frame count vs episode lengths; resolution vs declared shape |
| zero_variance | dims that never move (std < eps): normalization would divide by zero |
| schema_consistency | column sets/dtypes drifting across episodes |
| frame_continuity | dropped frames (row count vs declared length, frame_index holes) |
| training_readiness | episodes shorter than your policy's chunk_size/horizon (configure to enable) |
| portability | symlinks (break on transfer), stray files in data/, unexpected top-level entries |
| duplicate_episodes | byte-identical state/action trajectories across episodes |
| splits | overlapping episode ranges across declared splits (train/eval leakage) |

Validated on a real 921 MB distributed dataset (`lerobot/unitreeh1_fold_clothes`,
38 episodes / 19k frames / stereo video) plus deterministic synthetic fixtures
with injected corruptions — every corruption is caught by a test, on both
format versions. The real dataset itself shipped two findings tailor caught
on first contact: an action dim pegged at its range edge for 29 frames
(gripper holding a folded shirt — WARN), and a genuine metadata defect where
`cam_left`'s episode-to-shard pointers all target `file-000.mp4` while 500 of
its frames actually live in an orphan `file-001.mp4` — verified with the
official reader: `LeRobotDataset` loads fine but every episode-37 frame
raises `IndexError` (video index capped at 18500), so training crashes the
moment a sampler touches the last episode.

## Performance

The suite converts each episode column to a numpy array exactly once and
shares it across checks (`DatasetSnapshot.col_array`), probes videos through
container metadata (`nb_frames`) in a thread pool, and vectorizes the
run-length scans. On a synthetic 150k-frame dataset (300 episodes x 500
frames): checks stage 0.79s -> 0.22s vs v0.5, ~11x on the pure check
computation — the end-to-end ratio is lower because loading PyAV (~0.16s,
one-time per process) dominates what's left. Reproduce with
`python scripts/bench.py`.

## Status

Acceptance suite, filter/clean and the provenance ledger shipped (v0.6).
Roadmap, in priority order:

**Accept depth (v0.2)** — complete.

**Pipeline closure (v0.3+) — Cut/Sew/Re-check/Register:**

- ~~`filter`~~ shipped (see above): explicit / rule-based / check-driven drops, dedupe, on official `delete_episodes`, auto re-check
- ~~`clean`~~ shipped (see above): stats recompute + timestamp rebuild, dry-run first
- ~~provenance ledger~~ shipped (see above): append-only, travels with the dataset
- YAML recipes (tracking upstream `recipe.py` direction rather than inventing a schema)

Reference points: lerobot-doctor (diagnostics, now in the official dataset visualizer), Calibra (episode disposition decisions). tailor's lane is the acceptance gate plus the executed pipeline — verdict + exit code as a CI door, then deterministic refinement with provenance, not another report.
