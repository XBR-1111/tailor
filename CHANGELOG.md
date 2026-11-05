# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.10.1] — 2026-11-05

### Fixed
- `video_content`'s blur detection was rebuilt after a real-data validation
  round (lerobot/utaustin_mutex@v2.1: 1500 episodes, 1.4 GB, two cameras):
  - `--max-episodes` now limits the deep scan. It decoded every episode
    regardless of the sample (a 100-episode run still took 3m15s).
  - blur no longer compares Laplacian variance against a dataset baseline —
    variance cannot separate a defocused lens from a legitimately flat
    close-up (a wrist camera against a plain surface sits at 5-15, the same
    range as sensor noise). Detection now uses the p99 of |Laplacian| over
    the whole segment (normal cameras measure 76-128, a defocused lens 8-10,
    threshold 40) and requires ≥95% of frames edge-free: a working camera
    sees something sharp at some point, only a broken or dirty lens is
    edge-free throughout. The previous per-span check produced 404 false
    positives on one real wrist camera; the rebuilt check produces zero on
    the full 1500-episode dataset.
  - tunables: `blur_rel` (removed) is replaced by `blur_p99` (40.0) and
    `blur_cover` (0.95). Existing profiles that set `blur_rel` must switch.
- Fixture generator: new `video-blur` corruption (box-blurred frames,
  p99 8-10) so the rebuilt check has a positive case on both formats.

## [0.10.0] — 2026-11-05

### Added
- `tailor diff OLD NEW`: meta-level comparison of two dataset versions —
  structure, feature-schema drift, episode set/length changes, video pointer
  and file-size changes (a re-encode signal), stats drift. No decoding, no
  parquet parsing. Exit 0 identical / 1 differs, so a re-delivery is gated
  on "did anything change".
- `tailor datasheet PATH`: delivery document from one check-suite run —
  overview (scale, footage hours), feature table, acceptance verdicts,
  ledger lineage. `-o` writes a file; feed it to a dataset card or a
  handoff ticket.
- `video_content` check (OFF by default): full-decode content scan — frozen
  frames (near-identical run with ≥25% bit-exact repeats, distinguishing a
  stalled camera from a still scene), mid-file black spans, covered lens,
  over-exposure, persistent blur relative to the dataset's own Laplacian
  baseline with motion frames excluded. Opt in via
  `video_content: {enabled: true}` in a profile; absent = SKIP, so default
  runs stay at the seconds budget. v3 shards are decoded once and episodes
  judged on their metadata segments.
- `video_black` check: black openings/closings per video file (lens cap left
  on, camera that never started). Decodes only the first/last few frames —
  edge failures that frame counts and metadata cannot see. Tunables:
  `head_frames` (24), `tail_frames` (24), `dark_share` (0.9).
- `video_streams` gained four header-level sub-checks at zero decode cost:
  container fps vs the fps declared in `info.json` features (`fps_rel_tol`,
  default 5% — catches silent profile mismatches before they corrupt
  timestamp reconstruction), header duration vs frames/fps (`duration_tol`,
  default 2% — truncated recordings), rotation metadata (players rotate,
  loaders don't), and interlaced stream flags.
- Reference docs under `docs/`: `configuration.md` (every tunable with
  defaults, merge semantics, validation rules, starter recipes),
  `checks.md` (per-check logic, verdict capability, corruption repro),
  `cli.md` (every command, flag, exit code, JSON payload).
- `tests/test_docs.py` pins the docs to the code: tunable tables must equal
  DEFAULTS, every check must appear with its exact registry name, cited
  corruptions must exist, CLI flags must match `cli.py` in both directions.
- Fixture videos now carry deterministic grey-noise content (they were solid
  black before — fine when only counts were checked, wrong once content is),
  and the fixture `info.json` declares per-video `fps` like real datasets.

### Fixed
- The `episode_files` check was registered as `episode_files_exist`, so a
  profile could not address it (`unknown check` error) while every doc and
  test called it `episode_files`. Renamed the checker to match its output
  name; `tests/test_docs.py` now asserts registry keys equal reported names
  for all 25 checks.
- The fixture generator wrote BOTH the v2 and v3 video layouts into every
  dataset (orphan files nobody read until `diff` started scanning disk).

## [0.9.0] — 2026-11-01

### Fixed — found by validating against real v2.x datasets
- **v2.x video layout was wrong**: checks expected `videos/{key}/chunk-XXX/…`
  but the official v2.0/v2.1 template (verified on lerobot/pusht@v2.1) is
  `videos/chunk-XXX/{key}/…` — every v2.1 dataset misreported ALL video files
  missing. The synthetic fixture carried the same wrong layout, which is why
  tests stayed green; the fixture now matches the official template.
- HF downloads tripped the portability check: huggingface_hub's `.cache`
  bookkeeping dir is now in the allowlist.
- (with the repo@ref feature) local paths win before '@' splitting — a cache
  dir literally named `repo__name@v2.1` is a path, not a ref spec.

### Added
- `repo@ref` addressing: `tailor check lerobot/pusht@v2.1` checks a specific
  format-version ref (LeRobot datasets tag each version; main is always the
  latest). Ref-aware cache dirs — v2.1 and v3.0 of one repo never collide.
- Reusable CI workflow `.github/workflows/tailor-check.yml`: one `uses:` line
  gates a dataset in any repository's pipeline (verdict -> job result,
  markdown report -> run artifacts). README documents it as "Use as a CI gate".
- v2 filter support: closed after investigation — lerobot's edit engine only
  writes v3.0 (`CODEBASE_VERSION`), so filtering a v2 dataset silently
  upgrades the format; the convert-first guidance in the error message stays.
- `video_sync` check: cross-camera from/to timestamp drift per episode
  (v3.0 metadata level; v2.x SKIPs — no per-camera timestamps to compare).
  `tolerance` configurable, default 50ms; WARN with episode attribution.
- `tailor.stats.RunningStats`: numerically stable streaming stats
  (Chan et al. merge). stats_recompute and clean's recompute now fold
  per-episode/per-file instead of concatenating whole columns — peak
  memory O(one episode/file) instead of O(dataset), removing the OOM
  ceiling on million-frame recomputes.
- `clean --apply` on a dataset without meta/stats.json now CREATES it from
  recomputation (was: crash on backing up a nonexistent file). Dry-run
  reports the missing file honestly instead of "no drift".

## [0.8.1] — 2026-11-01

### Fixed
- README claimed video re-encode as part of the clean stage; it was never
  implemented. The Sew description now says so and points at the roadmap.
- `--max-episodes N --sample last` selects the LAST N episodes — a
  convenience sample of the head never sees tail-end episodes, a common
  corruption site.
- Determinism regression test: two fresh loads + runs must agree on every
  result field (pins the column cache and parallel video probing).
- Dead code removed (unreachable branch in `_longest_true_run`, redundant
  ledger wrapper in the CLI).

## [0.8.0] — 2026-11-01

### Added
- `enabled: false` in a profile switches a check off entirely: it is not
  executed, reports SKIP naming the profile, and contributes no episodes to
  `--from-check`. Orthogonal to severity remapping; non-boolean values are
  rejected loudly like every other profile key.

## [0.7.0] — 2026-11-01

### Changed
- **Performance**: each episode column is converted to a numpy array once
  and shared by every check (`DatasetSnapshot.col_array` / `trajectory_digest`);
  video probing reads container `nb_frames` metadata (header read instead of
  full demux) and runs in a thread pool; run-length scans (frozen actions,
  range-edge pegging) are vectorized; timestamps/frame-continuity checks are
  numpy-native; v2 episode parquets load in parallel. 150k-frame benchmark:
  checks stage 0.79s -> 0.22s, ~11x on pure check computation
  (`scripts/bench.py` reproduces; `make_fixture.py` gained --episodes/--frames).

### Engineering
- ruff rule groups extended (SIM/C4/RET/RUF), mypy `disallow_untyped_defs`,
  `py.typed` ships in the wheel; coverage floor 75 -> 82; tests run under
  pytest-xdist in CI (37s -> ~11s locally); pre-commit config added.
- release workflow is Trusted-Publishing ready (no PyPI token in CI).

## [0.6.0] — 2026-10-31

### Fixed
- **clean dry-run no longer writes to disk.** The timestamp-rebuild path used
  the write-through implementation, so a dry-run silently rewrote parquet
  files while reporting "nothing written". Dry-run now goes through a
  detect-only pass; two regression tests pin parquet bytes unchanged.
- **clean --apply now backs up the originals.** Timestamps were written
  before the backup ran, so `<dataset>_pre_clean/` archived the already
  rewritten data. Order is detect -> backup -> write; a regression test
  asserts backups are byte-equal to the pre-clean files.
- check exits 2 (not a traceback) when a HF download fails — the exit-code
  contract now covers network errors.
- filter re-checks the output under the SAME profile that drove the drops.
- empty-but-readable episode parquet no longer crashes state_action_dims.
- clean dry-run stats wording states the 0.1% noise floor and how it differs
  from the stats_recompute check tolerance, instead of "already consistent".

### Changed
- PyPI name is `lerobot-tailor` (the `tailor` name is taken); the CLI stays
  `tailor`. requires-python lowered to 3.11.
- action_quality now attributes episodes in check details, so
  `filter --from-check --drop-warn` can drop exactly the flagged episodes.
- dataset fingerprints include video file sizes (video carries most of the
  bytes; clean/filter outputs were previously indistinguishable).
- report footers and JSON summaries count skipped checks — a meta-only run
  reads "11 passed, 11 skipped" instead of a bare "11 passed".
- clean ledger events use one structured shape (`{"what": ...}`); torn tail
  lines in a ledger no longer break `tailor provenance`.
- duplicate detection shares one trajectory digest between the check and
  --dedupe; nan/inf detection vectorized; several duplicate helpers folded.

## [0.5.0] — 2026-10-31

### Added
- Provenance ledger (the Register stage): filter writes a birth record into
  the output dataset (source fingerprint, dropped episodes and reasons),
  clean --apply appends what it rewrote; one JSON line per event in
  meta/tailor_ledger.jsonl — the processing history travels with the data.
- `tailor provenance <ds>` renders the timeline. check stays read-only.
- Structural fingerprints (episode metadata + data file sizes) identify the
  dataset across operations.

## [0.4.0] — 2026-10-31

### Added
- `tailor clean`: deterministic repairs — meta/stats.json recomputed from
  parquet (video entries preserved), timestamps rebuilt as frame_index/fps.
  Dry-run by default; --apply writes with per-file backups to
  <dataset>_pre_clean/ outside the root, then re-checks automatically.
  Same-process re-checks see fresh data (v2 frame cache invalidated on rewrite).
  73 tests; coverage floor holding (75% gate, 79% actual).

## [0.3.0] — 2026-10-31

### Added
- Remote acceptance: `tailor check org/name` pulls only `meta/*` from the
  Hugging Face Hub (respects `HF_ENDPOINT`) and runs all metadata-level checks;
  `--full` snapshots the whole repo for the complete run. File-presence checks
  SKIP honestly in meta-only mode instead of faking a FAIL.
- Zip support: `tailor check dataset.zip` extracts and peels wrapper
  directories automatically.
- huggingface-hub as a runtime dependency (light: requests/tqdm only).
- Engineering hardening: mypy (clean) in dev + CI, py.typed + public API
  exports, pytest-cov with a 75% floor (baseline 79%), CI matrix
  ubuntu/macos × py3.12/3.13, CHANGELOG / CONTRIBUTING / PR template /
  dependabot (actions), release workflow now also cuts a GitHub Release from
  the changelog.

## [0.2.0]

### Added
- `clean` commands (timestamp rebuild, stats recompute) on official `compute_stats`
- provenance ledger + registration

## [0.2.0] — 2026-10-27

### Added
- `filter` command (the Cut stage): explicit / rule-based (`--min-length`,
  `--dedupe`) / check-driven (`--from-check`, `--drop-warn`) episode removal on
  lerobot's official `delete_episodes` engine behind the `[lerobot]` extra;
  output is re-checked automatically, every drop printed with its reason.
- Nine new acceptance checks: `video_files`, `video_streams` (PyAV:
  decodability, frame counts vs episode lengths, resolution),
  `zero_variance`, `schema_consistency`, `frame_continuity`,
  `training_readiness` (profile-driven chunk_size), `portability`,
  `duplicate_episodes`, `splits` — 22 checks total.
- `--markdown report.md` (PR/dataset-card friendly output) and
  `--fail-on-warn` (strict CI gate).
- Configurable acceptance profiles (`--profile`): per-check thresholds and
  WARN/FAIL severity remapping in versioned YAML — the dataset SLA lives in git.
- `stats_recompute` (declared stats vs recomputed means) and `action_quality`
  (frozen runs, range-edge pegging, jump detection) checks.
- Real-dataset validation on `lerobot/unitreeh1_fold_clothes` (921 MB): tailor
  caught a genuine shard-pointer defect (episode 37 unreadable by the official
  reader) and a gripper pegged at its range edge for 29 frames; reported
  upstream with a reproducer.

### Changed
- `requires-python` >= 3.12 (aligned with lerobot).
- LeRobot v3.0 support in loader (shard pointers, per-episode metadata),
  format normalization keeps the expectation layer version-free.

## [0.1.0] — 2026-10-26

### Added
- Initial acceptance suite: 11 checks over LeRobot v2.0/v2.1 datasets,
  dual-format fixtures with 6 injected corruptions, `tailor check` CLI with
  `--json` / `--max-episodes` and 0/1/2 exit-code contract.
- CI (ruff + pytest), Trusted Publishing release workflow, bilingual README
  (English / 简体中文).
