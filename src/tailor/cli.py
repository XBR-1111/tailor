"""CLI entry point: `tailor check <path>` runs the acceptance suite."""

import json
from pathlib import Path

import typer
from rich.console import Console

from tailor import __version__
from tailor.expectations import run_all
from tailor.ledger import dataset_fingerprint
from tailor.loader import load_snapshot
from tailor.model import Status
from tailor.report import render_json, render_terminal

app = typer.Typer(
    name="tailor",
    help="Refine distributed robot datasets into training-ready data.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed tailor version."""
    console.print(__version__)


@app.command()
def check(
    path: Path = typer.Argument(..., help="Path to a LeRobot-format dataset directory."),
    profile: Path | None = typer.Option(
        None, "--profile", "-p", help="YAML acceptance profile (per-check thresholds/severities)."
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help=(
            "Remote repo id (org/name or org/name@ref, e.g. lerobot/pusht@v2.1):"
            " download the whole dataset, not just meta."
        ),
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    markdown: Path | None = typer.Option(
        None,
        "--markdown",
        help="Also write a PR/dataset-card friendly Markdown report to this file.",
    ),
    fail_on_warn: bool = typer.Option(
        False, "--fail-on-warn", help="Exit 1 on WARN findings too (strict CI gate)."
    ),
    max_episodes: int | None = typer.Option(
        None, "--max-episodes", help="Only inspect N episodes (big datasets)."
    ),
    sample: str = typer.Option(
        "first",
        "--sample",
        help="With --max-episodes: keep 'first' (default) or 'last' N episodes. "
        "Tail-end episodes are a common corruption site.",
    ),
) -> None:
    """Run acceptance checks on a dataset.

    Exit codes: 0 accepted (no FAIL) · 1 rejected (>=1 FAIL) · 2 unreadable dataset.
    """
    from tailor.profile import ProfileError, load_profile
    from tailor.source import resolve_input

    try:
        root = resolve_input(str(path), full=full)
    except (FileNotFoundError, NotADirectoryError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e
    except Exception as e:  # HF/network errors are many and varied; keep the
        # exit-code contract (2 = unreadable input) instead of a raw traceback
        console.print(f"[red]input failed:[/red] {e}")
        raise typer.Exit(code=2) from e

    prof = None
    if profile is not None:
        try:
            prof = load_profile(profile)
        except ProfileError as e:
            console.print(f"[red]invalid profile:[/red] {e}")
            raise typer.Exit(code=2) from e
    if sample not in ("first", "last"):
        console.print(f"[red]--sample must be 'first' or 'last', got {sample!r}[/red]")
        raise typer.Exit(code=2)
    snap = load_snapshot(root, max_episodes=max_episodes, sample=sample)
    if not snap.readable:
        console.print(
            "[red]dataset unreadable:[/red] missing meta/info.json or meta/episodes.jsonl"
        )
        raise typer.Exit(code=2)
    results = run_all(snap, profile=prof)
    if json_output:
        typer.echo(render_json(results, str(path)))
    else:
        render_terminal(results, str(path))
        if profile is not None:
            console.print(f"[dim]profile: {profile}[/dim]")
    if markdown is not None:
        from tailor.report import render_markdown

        markdown.write_text(render_markdown(results, str(path)))
        console.print(f"[dim]markdown report: {markdown}[/dim]")
    fails = sum(r.status == Status.FAIL for r in results)
    warns = sum(r.status == Status.WARN for r in results)
    exit_code = 1 if fails or (fail_on_warn and warns) else 0
    raise typer.Exit(code=exit_code)


@app.command(name="filter")
def filter_(
    path: Path = typer.Argument(..., help="Path to a LeRobot-format dataset directory (v3.0)."),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output directory (default: <path>_filtered)."
    ),
    drop_episodes: str | None = typer.Option(
        None, "--drop-episodes", help="Comma-separated episode indices to drop, e.g. '3,17,23'."
    ),
    min_length: int | None = typer.Option(
        None, "--min-length", help="Drop episodes shorter than this many frames."
    ),
    dedupe: bool = typer.Option(
        False, "--dedupe", help="Drop later members of duplicate episode groups."
    ),
    from_check: bool = typer.Option(
        False, "--from-check", help="Drop episodes flagged by FAIL checks (run the suite first)."
    ),
    drop_warn: bool = typer.Option(
        False, "--drop-warn", help="With --from-check: WARN-flagged episodes too."
    ),
    profile: Path | None = typer.Option(
        None, "--profile", "-p", help="YAML acceptance profile for --from-check."
    ),
) -> None:
    """Filter episodes out of a dataset into a new dataset (the Cut stage).

    Requires the lerobot extra: pip install "tailor[lerobot]".
    The output is re-checked after the rewrite; its verdict is printed.
    """
    if not any([drop_episodes, min_length is not None, dedupe, from_check]):
        console.print(
            "[red]no filter rule given[/red] "
            "(see --drop-episodes / --min-length / --dedupe / --from-check)"
        )
        raise typer.Exit(code=2)
    from tailor.filtering import collect_drops, parse_episode_list, render_decision

    snap = load_snapshot(path)
    if not snap.readable:
        console.print("[red]dataset unreadable[/red]")
        raise typer.Exit(code=2)
    version = (snap.info or {}).get("codebase_version")
    if version != "v3.0":
        console.print(
            f"[red]filter currently supports v3.0 datasets (this is {version});[/red] "
            "convert first (lerobot's conversion scripts) or wait for v2 support"
        )
        raise typer.Exit(code=2)

    explicit = parse_episode_list(drop_episodes) if drop_episodes else None
    prof = None
    if profile is not None:
        from tailor.profile import ProfileError, load_profile

        try:
            prof = load_profile(profile)
        except ProfileError as e:
            console.print(f"[red]invalid profile:[/red] {e}")
            raise typer.Exit(code=2) from e
    results = None
    if from_check:
        results = run_all(snap, profile=prof)

    decision = collect_drops(
        snap,
        explicit=explicit,
        min_length=min_length,
        dedupe=dedupe,
        from_check=results,
        drop_warn=drop_warn,
    )
    console.print(render_decision(decision, len(snap.episodes or [])))
    if not decision.n_drop:
        console.print("[green]nothing to drop — dataset unchanged[/green]")
        raise typer.Exit(code=0)

    try:
        from lerobot.datasets.dataset_tools import delete_episodes as _del
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        console.print("[red]lerobot not installed:[/red] pip install 'tailor[lerobot]'")
        raise typer.Exit(code=2) from None

    out = output or path.parent / f"{path.name}_filtered"
    if out.exists():
        console.print(f"[red]output exists:[/red] {out}")
        raise typer.Exit(code=2)
    src = LeRobotDataset(path.name, root=str(path))
    _del(src, sorted(decision.drop), output_dir=str(out))
    console.print(f"[green]wrote[/green] {out}")

    # Re-check stage: the product must pass acceptance again — under the SAME
    # profile that drove the drops, or the verdict answers a different question
    new_snap = load_snapshot(out)
    recheck = run_all(new_snap, profile=prof)
    render_terminal(recheck, str(out))
    from tailor.ledger import append_event, event_for_filter

    append_event(out, event_for_filter(path, snap, decision.drop, out))
    raise typer.Exit(code=0)


@app.command()
def clean(
    path: Path = typer.Argument(..., help="Path to a LeRobot-format dataset directory."),
    apply: bool = typer.Option(
        False, "--apply", help="Write the repairs (default is a dry-run that only reports)."
    ),
    no_stats: bool = typer.Option(False, "--no-stats", help="Skip stats recompute."),
    no_timestamps: bool = typer.Option(False, "--no-timestamps", help="Skip timestamp rebuild."),
) -> None:
    """Deterministic repairs: recompute meta/stats.json, rebuild timestamps.

    Dry-run by default — shows what would change. With --apply, every replaced
    file is backed up to <dataset>_pre_clean/ outside the dataset root, then the
    dataset is re-checked and the verdict printed.
    """
    snap = load_snapshot(path)
    if not snap.readable:
        console.print("[red]dataset unreadable[/red]")
        raise typer.Exit(code=2)
    from tailor.cleaning import apply_clean, recompute_stats

    do_stats, do_ts = not no_stats, not no_timestamps
    if not (do_stats or do_ts):
        console.print("[red]nothing to do:[/red] both repairs disabled")
        raise typer.Exit(code=2)

    if not apply:
        fps = float((snap.info or {}).get("fps", 0))
        from tailor.cleaning import detect_timestamp_changes

        files, rows = detect_timestamp_changes(snap, fps) if (do_ts and fps > 0) else ([], 0)
        stats = snap.root / "meta" / "stats.json"
        drift = None
        stats_missing = do_stats and not stats.is_file()
        if do_stats and stats.is_file():
            fresh = recompute_stats(snap)
            old = json.loads(stats.read_text())
            import numpy as np

            cols = [c for c in fresh if c in old and c not in ("timestamp",)]
            diffs = []
            for c in cols:
                for stat in ("mean", "std"):
                    a, b = np.asarray(old[c][stat], float), np.asarray(fresh[c][stat], float)
                    scale = np.maximum(np.abs(a), 1e-9)
                    d = float((np.abs(a - b) / scale).max())
                    # 0.1% is float round-trip noise, not drift. Note this is
                    # the "would rewriting change anything" bar — the
                    # stats_recompute CHECK judges quality against its own
                    # mean_tolerance (default 5%), so a dataset can sit between
                    # the two: clean sees nothing worth rewriting, check flags
                    # drift. Different questions, different bars.
                    if d > 1e-3:
                        diffs.append((c, stat, d))
            drift = diffs
        console.print("[bold]dry-run — nothing written[/bold] (add --apply to repair)")
        if stats_missing:
            console.print(
                "  stats: [yellow]meta/stats.json missing — --apply would create it"
                " from recomputation[/yellow]"
            )
        elif do_stats:
            if drift:
                console.print(
                    f"  stats: [yellow]{len(drift)} drifted stat(s) would be rewritten[/yellow]"
                )
                for c, stat, d in drift[:8]:
                    console.print(f"    {c}.{stat}: {d:.1%} off")
            else:
                console.print(
                    "  stats: [green]no drift above 0.1%[/green] "
                    "(round-trip noise floor — rewriting would not change anything real)"
                )
        if do_ts:
            if files:
                console.print(
                    f"  timestamps: [yellow]{rows} row(s) in {len(files)} file(s) would be rebuilt"
                )
            else:
                console.print("  timestamps: [green]already consistent[/green]")
        raise typer.Exit(code=0)

    report = apply_clean(snap, stats=do_stats, timestamps=do_ts)
    from tailor.ledger import append_event, event_for_clean

    ev = event_for_clean(report.stats_rewritten, len(report.timestamp_files), report.timestamp_rows)
    ev["result_fingerprint"] = dataset_fingerprint(path)
    append_event(path, ev)
    if report.backup_dir:
        console.print(f"[dim]backups: {report.backup_dir}[/dim]")
    if report.stats_rewritten:
        console.print("[green]stats.json rewritten[/green] from recomputed values")
    if report.timestamp_files:
        console.print(
            f"[green]timestamps rebuilt[/green] in {len(report.timestamp_files)} file(s), "
            f"{report.timestamp_rows} row(s) changed"
        )

    new_snap = load_snapshot(path)
    recheck = run_all(new_snap)
    render_terminal(recheck, str(path))
    raise typer.Exit(code=0)


@app.command()
def provenance(path: Path = typer.Argument(..., help="Path to a dataset.")) -> None:
    """Show the tailor ledger: every filter/clean this dataset's lineage records."""
    from tailor.ledger import read_ledger, render_ledger

    snap = load_snapshot(path)
    if not snap.readable:
        console.print("[red]dataset unreadable[/red]")
        raise typer.Exit(code=2)
    console.print(render_ledger(read_ledger(path)))


if __name__ == "__main__":
    app()
