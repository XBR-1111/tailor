"""CLI entry point: `tailor check <path>` runs the acceptance suite."""

from pathlib import Path

import typer
from rich.console import Console

from tailor import __version__
from tailor.expectations import run_all
from tailor.loader import load_snapshot
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
    fails = sum(r.status == "FAIL" for r in results)
    warns = sum(r.status == "WARN" for r in results)
    exit_code = 1 if fails or (fail_on_warn and warns) else 0
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
