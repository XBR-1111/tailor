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
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
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
    snap = load_snapshot(path, max_episodes=max_episodes)
    if not snap.readable:
        console.print(
            "[red]dataset unreadable:[/red] missing meta/info.json or meta/episodes.jsonl"
        )
        raise typer.Exit(code=2)
    results = run_all(snap)
    if json_output:
        typer.echo(render_json(results, str(path)))
    else:
        render_terminal(results, str(path))
    fails = sum(r.status == "FAIL" for r in results)
    raise typer.Exit(code=1 if fails else 0)


if __name__ == "__main__":
    app()
