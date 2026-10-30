"""CLI entry point. `tailor check <path>` is the v0.1 target; other commands come later."""

import typer
from rich.console import Console

from tailor import __version__

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
    path: str = typer.Argument(..., help="Path to a LeRobot-format dataset directory."),
) -> None:
    """Run acceptance checks on a dataset (under development for v0.1.0)."""
    console.print("[yellow]check[/yellow] lands in v0.1.0 — see the roadmap in README")
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
