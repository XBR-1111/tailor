# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
