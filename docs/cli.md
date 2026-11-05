# CLI reference

```
tailor check PATH        run the acceptance suite (read-only)
tailor filter PATH       drop episodes, auto re-check the output
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
                  [--fail-on-warn] [--max-episodes N [--sample first|last]]
                  [--full]
```

| flag | effect |
| --- | --- |
| `--profile FILE` | apply a YAML acceptance profile (see [configuration.md](configuration.md)) |
| `--json` | emit machine-readable JSON on stdout |
| `--markdown FILE` | write a PR/dataset-card friendly report |
| `--fail-on-warn` | treat WARN as failure for the exit code (strict CI gate) |
| `--max-episodes N` | inspect only N episodes of a large dataset |
| `--sample first\|last` | which N: default `first`; `last` catches tail-end corruption that head sampling never sees |
| `--full` | remote only: download everything, run all checks |

Exit codes: **0** accepted · **1** rejected (any FAIL, or any WARN with
`--fail-on-warn`) · **2** unreadable input.

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

SKIP reasons name their cause, so a metadata-only remote run reports
`videos/ not on disk (meta-only run? use --full)` instead of faking a FAIL.

## filter

```bash
tailor filter PATH [-o OUT] [--drop-episodes 3,17,23] [--min-length N]
                   [--dedupe] [--from-check [--drop-warn]] [-p PROFILE]
```

| flag | effect |
| --- | --- |
| `-o, --output` | output directory (default: `<path>_filtered`) |
| `--drop-episodes` | explicit indices to drop, comma-separated |
| `--min-length N` | drop episodes shorter than N frames |
| `--dedupe` | drop later members of byte-identical duplicate groups |
| `--from-check` | run the suite first and drop episodes that FAIL checks pin to an episode |
| `--drop-warn` | with `--from-check`: WARN-pinned episodes too |
| `-p, --profile` | acceptance profile for `--from-check` |

Rewrites via lerobot's official `delete_episodes` engine, so parquet, videos,
stats and metadata are rebuilt consistently. The output is re-checked
automatically and the verdict printed. Every drop is printed with its reason.
Requires `pip install 'lerobot-tailor[lerobot]'`; v3.0 datasets only.

## clean

```bash
tailor clean PATH [--apply] [--no-stats] [--no-timestamps]
```

Dry-run by default: prints what would change, writes nothing. `--apply`
writes, backing up every replaced file to `<dataset>_pre_clean/` outside the
dataset root, then re-checks and prints the verdict. Repairs are deterministic
only: `meta/stats.json` recomputed from parquet (video entries preserved),
timestamps rebuilt as `frame_index / fps`. No lerobot dependency.

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
