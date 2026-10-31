"""CLI entry point: `tailor check <path>` runs the acceptance suite."""

from pathlib import Path

import typer
from rich.console import Console

from tailor import __version__
from tailor.expectations import run_all
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
        None, "--max-episodes", help="Only inspect the first N episodes (big datasets)."
    ),
) -> None:
    """Run acceptance checks on a dataset.

    Exit codes: 0 accepted (no FAIL) · 1 rejected (>=1 FAIL) · 2 unreadable dataset.
    """
    if not path.is_dir():
        console.print(f"[red]not a directory:[/red] {path}")
        raise typer.Exit(code=2)
    from tailor.profile import ProfileError, load_profile

    prof = None
    if profile is not None:
        try:
            prof = load_profile(profile)
        except ProfileError as e:
            console.print(f"[red]invalid profile:[/red] {e}")
            raise typer.Exit(code=2) from e
    snap = load_snapshot(path, max_episodes=max_episodes)
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
    results = None
    if from_check:
        prof = None
        if profile is not None:
            from tailor.profile import ProfileError, load_profile

            try:
                prof = load_profile(profile)
            except ProfileError as e:
                console.print(f"[red]invalid profile:[/red] {e}")
                raise typer.Exit(code=2) from e
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

    # Re-check stage: the product must pass acceptance again
    new_snap = load_snapshot(out)
    recheck = run_all(new_snap)
    render_terminal(recheck, str(out))
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
