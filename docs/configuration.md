# Configuration reference

tailor ships with defaults that fit most datasets. A **profile** (YAML, kept
in git next to your training config) overrides per-check thresholds and
severities — your dataset SLA, versioned and auditable.

```bash
tailor check /data/dataset --profile my_requirements.yaml
```

## Profile file format

A profile is a mapping with one required top-level key, `checks`. Every entry
under it names a check and configures it:

```yaml
checks:
  codebase_version:
    allowed: [v2.1]     # tunable: only v2.1 is acceptable here
  episode_lengths:
    min: 100
    severity: fail      # WARN→FAIL remap: short episodes are a hard reject
  video_content:
    enabled: true       # the deep scan is off by default
  splits: null          # null / empty = keep built-in defaults
```

Three kinds of keys are accepted per check:

| key | type | meaning |
| --- | --- | --- |
| any tunable name | per-table below | overrides the built-in threshold |
| `severity` | `warn` \| `fail` | remap a check's WARN/FAIL verdict |
| `enabled` | `true` \| `false` | switch the check off entirely |

Validation is strict on purpose — a typo in a requirements file must fail
fast, not silently keep a default you meant to change:

- unknown top-level structure → `profile must be a mapping with a top-level 'checks' key`
- unknown check name → error listing every known check
- unknown key under a check → `checks.<name> has no configurable key '<key>'`
- bad severity → `checks.<name>.severity must be WARN or FAIL`

## Merge semantics

Each configured check starts from its built-in defaults and applies your keys
on top. Checks you don't mention keep their defaults. `checks.<name>: null`
and `checks.<name>: {}` both mean "defaults, no changes".

`severity` only remaps WARN and FAIL — PASS is never remappable. A check that
can only WARN natively (see the matrix in [checks.md](checks.md)) can be
promoted to FAIL; a check that can only FAIL can be softened to WARN.

`enabled: false` is orthogonal to severity: the check is not executed at all.
Its row reports SKIP naming the profile as the reason, and it contributes
nothing to `tailor filter --from-check`.

## Tunables, per check

Only the checks below have tunable parameters. Every other check
(25 total, see [checks.md](checks.md)) accepts just `severity` and `enabled`.

### Structure and metadata

| check | key | default | meaning | when to change |
| --- | --- | --- | --- | --- |
| `codebase_version` | `allowed` | `[v2.0, v2.1, v3.0]` | format versions accepted | pin one version: `[v2.1]` |
| `fps` | `allowed` | `null` (any positive) | permitted fps values | lock to hardware: `[30, 50]` |
| `episode_lengths` | `min` | `2` | shortest acceptable episode (frames) | raise to your policy's minimum; 100+ is common for chunked sampling |
| `timestamps` | `spacing_tolerance` | `0.1` | max relative deviation of frame spacing from 1/fps | loosen for jittery hardware clocks (0.2), tighten for strict pipelines (0.02) |

### Data (parquet columns)

| check | key | default | meaning | when to change |
| --- | --- | --- | --- | --- |
| `stats_recompute` | `mean_tolerance` | `0.05` | max relative drift between declared stats.json mean/std and recomputed values | datasets with heavy-tailed dims may need 0.1 |
| `zero_variance` | `eps` | `1e-6` | std below this counts as "never moves" | raise (1e-4) if normalization is float32-sensitive |
| `action_quality` | `frozen_frames` | `10` | consecutive identical actions that count as a stale run | lower for high-rate control (5 at 100 Hz) |
| `action_quality` | `clip_run` | `5` | consecutive frames pegged at the declared range edge | raise if your task legitimately saturates (gripper fully closed) |
| `action_quality` | `jump_sigma` | `8` | jump threshold in std units | loosen for impulsive tasks (12), tighten for smooth demos (5) |
| `action_quality` | `jump_ratio` | `0.01` | fraction of frames that may jump before WARN | scale with episode length |

### Video

| check | key | default | meaning | when to change |
| --- | --- | --- | --- | --- |
| `video_streams` | `frame_tolerance` | `1` | allowed frame-count difference vs episode length | leave at 1 unless your encoder drops boundary frames |
| `video_streams` | `fps_rel_tol` | `0.05` | container fps vs declared fps, relative tolerance | fixed for well-formed files; loosen only for exotic containers |
| `video_streams` | `duration_tol` | `0.02` | header duration vs decodable content, relative tolerance | truncated-recording detection; raise if your muxer writes pessimistic durations |
| `video_black` | `head_frames` | `24` | frames inspected at each file head | scale with fps (24 at 30 fps ≈ 0.8 s) |
| `video_black` | `tail_frames` | `24` | frames inspected at each file tail | same as above |
| `video_black` | `dark_share` | `0.9` | share of dark frames that makes an edge "black" | raise to 0.95 if your scenes are dim |
| `video_content` | `enabled` | `false` | **the deep scan is off by default** (full decode, minutes per GB) | turn on for delivery gates and investigation, not per-commit CI |
| `video_content` | `frozen_min_s` | `1.0` | shortest repeated-frame run that counts as frozen | lower (0.5) for fast cameras that would never stall that long |
| `video_content` | `black_onset` | `40.0` | minimum brightness just before a mid-file black span for it to count as a fault; darker lead-in means gradual occlusion (a wrist camera approaching an object), not a recorder failure | lower to flag occlusions too, raise to demand a lit→black cut |
| `video_content` | `flat_min_s` | `1.0` | shortest covered-lens / over-exposure span reported | raise to ignore brief occlusions |
| `video_content` | `blur_p99` | `40.0` | a frame counts as sharpness-free when its \|Laplacian\| p99 falls below this (64-px grayscale scale: normal cameras measure 76-128, a defocused lens 8-10) | raise to flag softer footage, lower to tolerate it |
| `video_content` | `blur_cover` | `0.95` | share of a segment's frames that must be sharpness-free to report defocus | lower to catch partial defocus at the cost of wrist-camera false positives |
| `video_sync` | `tolerance` | `0.05` | cross-camera from/to timestamp drift (seconds) per episode | raise to 0.1 for stereo rigs without hardware sync |

### Training

| check | key | default | meaning | when to change |
| --- | --- | --- | --- | --- |
| `training_readiness` | `chunk_size` | `null` (off) | episodes shorter than `chunk_size × horizon` fail | set to your policy's action chunk length; this check stays SKIP until you do |

`video_content` has one subtlety: the built-in default carries
`enabled: false`, and your profile merges on top of it — so an empty entry
(`video_content: {}`) stays **off**. The scan runs only when you write
`enabled: true` explicitly.

## Recipes

**Strict delivery gate** — pin the format, long episodes, deep video scan,
promote warnings to failures:

```yaml
checks:
  codebase_version: {allowed: [v3.0]}
  episode_lengths: {min: 100}
  video_content: {enabled: true, blur_rel: 0.2}
  video_sync: {tolerance: 0.02, severity: fail}
  duplicate_episodes: {severity: fail}
```

**Fast per-commit CI** — metadata-only, warnings tolerated:

```yaml
checks:
  video_content: {enabled: false}
  action_quality: {severity: fail}
  portability: {severity: fail}
```

**Exploration** — validate structure, skip expensive checks:

```yaml
checks:
  video_streams: {enabled: false}
  stats_recompute: {enabled: false}
  video_content: {enabled: false}
```

## Remote datasets

`--profile` works identically for local paths, zip archives and HF repo ids
(`lerobot/pusht`, `lerobot/pusht@v2.1`). Remote runs default to metadata-only;
checks that need file contents SKIP honestly and their tunables are simply not
exercised — pass `--full` to download everything and run the complete suite.
