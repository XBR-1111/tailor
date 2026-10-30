# tailor

**Refine distributed robot datasets into training-ready data.**

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
pip install tailor            # once published; until then:
pip install git+https://github.com/XBR-1111/tailor

tailor check /path/to/lerobot/dataset
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
│ task_indices      │  PASS  │ 1 tasks      │
│ state_action_dims │  PASS  │              │
│ nan_inf           │  PASS  │              │
│ timestamps        │  PASS  │              │
└───────────────────┴────────┴──────────────┘
ACCEPTED  ·  11 passed, 0 warn, 0 failed
```

Exit codes make it a CI gate: **0** accepted · **1** rejected · **2** unreadable.
`--json` emits machine-readable output; `--max-episodes N` samples the first N
episodes of huge datasets.

## The 11 acceptance checks (v0.1)

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

Validated on a real 921 MB distributed dataset (`lerobot/unitreeh1_fold_clothes`,
38 episodes / 19k frames / stereo video) plus deterministic synthetic fixtures
with injected corruptions — every corruption is caught by a test, on both
format versions.

## Status

v0.1 acceptance suite shipped. Roadmap: filter/clean commands, YAML recipes,
video integrity checks, PyPI release. Issues track the plan.
