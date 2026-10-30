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

## The 15 acceptance checks (v0.1)

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

## Status

v0.1 acceptance suite shipped. Roadmap, in priority order:

**Accept depth (v0.2)** — close the check-category gap vs the ecosystem:

- zero-variance features
- cross-episode schema consistency (columns/dtype/shape drift)
- dropped frames, frame/index continuity
- action quality: clipped (pegged at min/max), frozen (stuck), sudden jumps
- training readiness: policy window compatibility, zero-std dims, delta_timestamps
- portability: absolute paths, broken symlinks, HF Hub compatibility
- duplicate episodes, train/eval split leakage
- remote acceptance from a HF repo id (check metadata before downloading), zip archives
- `--markdown` report for PRs/dataset cards, `--fail-on=warn`

**Pipeline closure (v0.3+)** — Cut/Sew/Re-check/Register:

- `filter` / `clean` commands built on lerobot's official `dataset_tools` (delete_episodes, compute_stats) behind a `[lerobot]` extra — outputs stay format-compatible by construction; acceptance itself stays dependency-free
- YAML recipes (tracking upstream `recipe.py` direction rather than inventing a schema)
- provenance ledger + registration

Reference points: lerobot-doctor (diagnostics, now in the official dataset visualizer), Calibra (episode disposition decisions). tailor's lane is the acceptance gate plus the executed pipeline — verdict + exit code as a CI door, then deterministic refinement with provenance, not another report.
