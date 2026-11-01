# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
