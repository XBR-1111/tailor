# CLI reference

```
tailor check PATH        run the acceptance suite (read-only)
tailor filter PATH       drop episodes, auto re-check the output
tailor run RECIPE        preview or apply a versioned YAML recipe
tailor clean PATH        deterministic repairs (dry-run by default)
tailor provenance PATH   show the dataset's processing ledger
tailor diff OLD NEW      compare two dataset versions
tailor datasheet PATH    generate a delivery document
tailor version           print the installed version
```

PATH is any of: a local dataset directory, a `.zip` archive (wrapper
directories are peeled), or a Hugging Face repo id — `lerobot/pusht`
(latest format), `lerobot/pusht@v2.1` (pinned revision). Remote paths default
to a metadata-only pull (`meta/*`, seconds, cached under `~/.cache/tailor-hf/`);
`--full` downloads everything.

Every command exits **0** on success, **2** when the input is unreadable.

## check

```bash
tailor check PATH [--profile FILE] [--json] [--markdown FILE]
                  [--fail-on-warn] [--fail-on-skip] [--max-episodes N [--sample first|last]]
                  [--full]
```

| flag | effect |
| --- | --- |
| `--profile FILE` | apply a YAML acceptance profile (see [configuration.md](configuration.md)) |
| `--json` | emit machine-readable JSON on stdout |
| `--markdown FILE` | write a PR/dataset-card friendly report |
| `--fail-on-warn` | treat WARN as failure for the exit code (strict CI gate) |
| `--fail-on-skip` | exit **3** when any check was SKIPped — a remote meta-only run leaves file-dependent checks unjudged, and a green exit there is not a full pass |
| `--max-episodes N` | inspect only N episodes of a large dataset |
| `--sample first\|last` | which N: default `last` (since 0.11.0); tail-end episodes are a common corruption site, so the default looks there |
| `--full` | remote only: download everything, run all checks |

Exit codes: **0** accepted · **1** rejected (any FAIL, or any WARN with
`--fail-on-warn`/`--fail-on-skip`) · **2** unreadable input · **3** SKIPped checks with `--fail-on-skip`.

`--json` payload:

```json
{
  "dataset": "/data/ds",
  "verdict": "accepted",          // or "rejected"
  "summary": {"passed": 22, "warn": 1, "failed": 0, "skipped": 2},
  "checks": [
    {"name": "fps", "status": "PASS", "message": "30", "details": {...}}
  ]
}
```

`status` is one of `PASS`, `WARN`, `FAIL`, `SKIP`. A SKIP means the check was
not applicable (missing config, metadata-only run, or disabled by profile) —
it never counts against the verdict. Note that `verdict` follows FAILs only;
`--json` does not change that, and `--fail-on-warn` affects the exit code,
not the payload.

Remote metadata-only runs skip file checks. Local inputs and remote runs with
`--full` fail when required data or video files are missing.

## filter

```bash
tailor filter PATH [-o OUT] [--drop-episodes 3,17,23] [--min-length N]
                   [--dedupe] [--from-check [--drop-warn]] [-p PROFILE]
                   [--dry-run [--json]]
```

| flag | effect |
| --- | --- |
| `-o, --output` | output directory (default: `<path>_filtered`) |
| `--drop-episodes` | explicit indices to drop, comma-separated |
| `--min-length N` | drop episodes shorter than N frames |
| `--dedupe` | drop later members of byte-identical duplicate groups |
| `--from-check` | run the suite first and drop episodes that FAIL checks pin to an episode |
| `--drop-warn` | with `--from-check`: WARN-pinned episodes too |
| `-p, --profile` | acceptance profile for `--from-check` and for the output recheck |
| `--dry-run` | show the episode plan before writing the output |
| `--json` | emit the dry-run plan as JSON, including the full keep and drop lists and unassigned failures |

Rewrites via lerobot's official `delete_episodes` engine, so parquet, videos,
stats and metadata are rebuilt consistently. The output is re-checked
automatically and the verdict printed. Every drop is printed with its reason.
The command exits **1** if the acceptance verdict is rejected or the rules
select every episode for removal. An unknown explicit episode index exits **2**.
Dry-run exits **1** for a plan that would remove every episode or has no drops
and a rejected source. An applicable plan still needs to pass the output recheck.
Requires `pip install 'lerobot-tailor[lerobot]'`; v3.0 datasets only.

## run

```bash
tailor run RECIPE [--apply] [--json]
```

`run` loads a version 1 YAML recipe and runs acceptance on a complete LeRobot
v3.0 dataset. The default prints the episode plan. `--apply` writes to a
staging directory, rechecks the result with the same rules, and publishes the
dataset at `output` after acceptance. Rejected outputs are discarded. An
existing output is an error. The optional LeRobot extra is needed for a rewrite
that removes episodes. A clean source with no removals can be copied directly.

Paths are relative to the YAML file. The `dataset` field accepts a local
directory, a ZIP archive, or a Hugging Face dataset with an exact commit SHA.
Remote recipes fetch the complete snapshot, including when previewing.
Use either inline `checks` or `profile` with the path to a profile file.

```yaml
version: 1
dataset: ./raw-dataset
output: ./filtered-dataset
checks:
  episode_lengths: {min: 100, severity: fail}
filter:
  from_check: true
  drop_warn: false
  drop_episodes: []
```

`filter` also accepts `min_length` and `dedupe`. `--json` includes the plan,
source and output checks, and hashes of the recipe and effective profile.
The output ledger records those hashes, the source, and the planned removals.
Exit **1** means the plan cannot be applied or the output failed recheck.
An invalid recipe or an unavailable input exits **2**.

## clean

```bash
tailor clean PATH [--apply] [--no-stats] [--no-timestamps]
```

Dry-run by default: prints what would change, writes nothing. `--apply`
writes, backing up every replaced file to `<dataset>_pre_clean/` outside the
dataset root, then re-checks and prints the verdict. Repairs are deterministic
and rebuild timestamps as `frame_index / fps`. When stats recomputation is
enabled, it uses the repaired timestamps and preserves entries for features it
does not process. The command exits **1** if the re-check rejects the dataset.
No lerobot dependency.

## provenance

```bash
tailor provenance PATH
```

Prints every event in `meta/tailor_ledger.jsonl` — the ledger `filter` and
`clean --apply` append to, so the processing history travels with the data.
`check` never writes.

## diff

```bash
tailor diff OLD NEW [--json]
```

Meta-level comparison of two dataset versions, without decoding or parsing
parquet: structure (version, fps, counts, splits), feature-schema drift,
episode set and length changes, video pointer and file-size changes (a
re-encode signal), stats.json drift. Exit **0** when identical, **1** when
versions differ, **2** when either side is unreadable. `--json` emits the full
report (structural, feature, episode, video, stats sections).

## datasheet

```bash
tailor datasheet PATH [-o FILE] [-p PROFILE]
```

Emits a delivery document — overview (scale, footage hours), feature table,
acceptance verdicts, ledger lineage — from one run of the check suite. Default
stdout; `-o` writes a file. Feed it to a dataset card or a handoff ticket.
