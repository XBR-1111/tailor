# The 25 acceptance checks

Every check is independent, read-only and cheap by default: the whole suite
runs in about 3 s on 1.4 GB of real data. Each result is PASS, WARN, FAIL, or
SKIP (SKIP means "not applicable" — no file to look at, no config to judge
against — and never counts against the verdict).

Tunables for each check live in [configuration.md](configuration.md). Reproduce
any corruption locally with the fixture generator:

```bash
python scripts/make_fixture.py /tmp/probe --corrupt <name>   # v2.1 by default
python scripts/make_fixture.py /tmp/probe --corrupt <name> --format v3.0
tailor check /tmp/probe
```

## Verdict capability

Some checks can only produce WARN or only FAIL by design — severity remaps
work in either direction (see configuration.md).

| can WARN and FAIL | only WARN natively |
| --- | --- |
| `fps`, `episode_lengths`, `timestamps`, `portability` | `action_quality`, `zero_variance`, `duplicate_episodes`, `video_sync` |

All other checks fail closed (FAIL on any finding); promote the WARN-only
checks to hard rejects with `severity: fail`.

## Structure and metadata

**`info_json`** — `meta/info.json` exists, parses, and declares the required
keys (fps, features, shapes). Gate for everything else: most checks SKIP when
it fails.

**`codebase_version`** — format version inside v2.0 / v2.1 / v3.0 (pin one
version per org with `allowed`). Reproduce: `--corrupt bad-version` (v9.9).

**`fps`** — positive, plausible frame rate; unusual values WARN, a missing or
non-positive fps FAILs. Reproduce an off-spec fps with a profile
(`fps: {allowed: [50]}` on a 10 fps fixture).

**`episodes_metadata`** — episode metadata parses and carries the fields the
rest of the suite reads (episode_index, length; v3: the index pointers).

**`total_frames`** — declared `total_frames` equals the sum of episode
lengths. Reproduce: `--corrupt frame-mismatch` (declared count off by 7).

**`episode_lengths`** — no zero or negative lengths; episodes below `min`
WARN (and FAIL once you set `severity: fail`). Reproduce: `--corrupt
short-episode` (episode 2 declared as 1 frame).

**`task_indices`** — task references resolve; declared task count matches the
task table.

**`splits`** — declared train/eval splits don't overlap (leakage) and indexes
exist. Gate for anyone who trains and evaluates from one artifact.

## Data columns (parquet)

**`episode_files`** — every data shard the metadata points at exists on disk.
v2: one parquet per episode; v3: shared shards via `data/chunk_index` +
`data/file_index`. Reproduce: `--corrupt missing-file`.

**`state_action_dims`** — vector dims of `observation.state` / `action`
match the declared feature shapes.

**`nan_inf`** — no NaN or infinity in state/action. Reproduce: `--corrupt nan`
(NaN at frame 5, dim 2 of episode 1).

**`timestamps`** — monotonically increasing; frame spacing within
`spacing_tolerance` of 1/fps. Reproduce: `--corrupt dup-timestamp`
(episode 0 stamped all-zero).

**`stats_recompute`** — declared `meta/stats.json` mean/std match values
recomputed from the parquet, within `mean_tolerance`. Catches stats generated
for a different data revision. Reproduce: `--corrupt stats-mismatch`
(action mean shifted by 0.5).

**`action_quality`** — three trajectory defect classes, all WARN-level by
default: frozen runs (`frozen_frames` consecutive identical actions), range
edge pegging (`clip_run` frames at the declared min/max), sudden jumps beyond
`jump_sigma` std in more than `jump_ratio` of frames. Reproduce: `--corrupt
frozen-action` (12 identical actions in episode 1).

**`zero_variance`** — no dim whose std stays below `eps`; a dead sensor or
constant column would divide by zero in normalization. Reproduce:
`--corrupt zero-variance` (state dim 0 pinned at 0.5).

**`schema_consistency`** — column sets and dtypes identical across every
parquet in the dataset. A schema that drifts between episodes breaks
training-loop concatenation. Reproduce: `--corrupt schema-drop` (v2: episode 1
loses `observation.state`).

**`frame_continuity`** — row count matches declared length and `frame_index`
has no holes. Reproduce: `--corrupt dropped-frames` (frames 8–10 removed).

**`duplicate_episodes`** — no byte-identical state/action pairs across
episodes (WARN; says which episodes pair up). Reproduce: `--corrupt
duplicate-episode`.

**`training_readiness`** — episodes long enough for your policy's
`chunk_size × horizon`; SKIP until you configure `chunk_size`.

## Video

Video checks probe containers through PyAV metadata first and decode pixels
only where a metadata signal can't answer the question.

**`video_files`** — every declared video stream exists on disk. v2: one file
per episode; v3: shard pointers (`videos/<key>/chunk_index` + `file_index`).
Reproduce: `--corrupt missing-video`.

**`video_streams`** — five container-level sub-checks, one row each finding:

| sub-check | catches | reproduce |
| --- | --- | --- |
| decodability | file opens, stream present | truncate the file by hand |
| frame count | decoded `nb_frames` vs episode length (± `frame_tolerance`) | `--corrupt video-mismatch` (short by 5) |
| resolution | stream size vs declared feature shape | re-encode at another resolution |
| container fps | stream rate vs declared fps (± `fps_rel_tol`) | `--corrupt video-fps` (5 vs declared 10) |
| duration | header duration vs decodable content (± `duration_tol`) — truncated recordings | truncate by hand |

Rotation metadata and interlaced streams are also flagged: rotated drones or
sideways-mounted cameras produce datasets that train but never converge on.
Those two sub-checks have no fixture injection (PyAV cannot write the flags) —
they're unit-tested against synthesized stream objects.

**`video_black`** — decodes the first/last `head_frames`/`tail_frames` of each
file: a black opening (lens cap on, camera started late) or black closing
(recorder stopped early) fails when ≥`dark_share` of the window is dark.
Reproduce: `--corrupt video-blackout`.

**`video_content`** — the deep scan, **OFF by default** (full decode, minutes
per GB — enable with `video_content: {enabled: true}`). Five defect classes:

| class | signal |
| --- | --- |
| frozen | repeated frames: near-identical run (≤0.05% pixels differing by >2 grey levels) lasting `frozen_min_s`, with ≥25% of those bit-exact — a still scene has noise, a stalled camera has none |
| mid-file black | ≥98% dark pixels for ≥0.5 s, away from the file edges (edges belong to `video_black`) |
| covered lens | flat frame (`std < 8`) that is also darker than 70% of the camera's own lit median — blocked optic, not a dark scene |
| over-exposure | mean > 245 with std < 10, ≥1 s |
| blur | Laplacian variance below `blur_rel` × the dataset's own median, sustained ≥1 s, excluding high-motion frames (motion blur is physics, not a defect) |

v3 shards are decoded once; each episode is judged on its metadata timestamp
segment. Reproduce: `--corrupt video-freeze` (12 repeated frames mid-episode).

**`video_sync`** — v3 metadata: per-camera `from_timestamp`/`to_timestamp`
for one episode agree within `tolerance` seconds. Catches stereo rigs without
hardware sync before they reach training. Reproduce: `--corrupt video-desync`
(second camera's `to_timestamp` drifts 0.3 s in episode 1; the video file
itself stays well-formed so only this check fires).

## Portability

**`portability`** — the dataset survives a file transfer: no symlinks in
`data/` or `videos/` (they dangle after upload), no stray files where the
format expects data, no unexpected top-level entries. Reproduce: `--corrupt
portability-junk` (debug.log in data/, notes.txt at root).

## What tailor deliberately does not check

- **Licenses** — a curation concern, tracked upstream in LeRobot metadata.
- **Camera swap / cross-camera identity** — needs full decodes on both
  streams plus optical-flow voting; the failure is visible to the naked eye
  and batch-consistent, so it belongs to manual review, not a per-commit gate.
- **Content richness scoring** — a curation scorecard, not an acceptance
  verdict. tailor answers "is this dataset trainable", not "is this episode
  interesting".
