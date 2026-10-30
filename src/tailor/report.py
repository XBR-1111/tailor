"""Terminal and JSON report rendering for `tailor check`."""

import json

from rich.console import Console
from rich.table import Table

from tailor.model import CheckResult, Status

_STATUS_STYLE = {
    Status.PASS: "[green]PASS[/green]",
    Status.WARN: "[yellow]WARN[/yellow]",
    Status.FAIL: "[red]FAIL[/red]",
    Status.SKIP: "[dim]SKIP[/dim]",
}


def render_terminal(results: list[CheckResult], dataset: str) -> None:
    console = Console()
    fails, warns, passes = _counts(results)
    table = Table(title=f"tailor — acceptance report: {dataset}", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("status", justify="center")
    table.add_column("detail")
    for r in results:
        table.add_row(r.name, _STATUS_STYLE[r.status], r.message)
    console.print(table)
    verdict = "[red]REJECTED[/red]" if fails else "[green]ACCEPTED[/green]"
    console.print(f"{verdict}  ·  {passes} passed, {warns} warn, {fails} failed")


def render_json(results: list[CheckResult], dataset: str) -> str:
    fails, warns, passes = _counts(results)
    return json.dumps(
        {
            "dataset": dataset,
            "verdict": "rejected" if fails else "accepted",
            "summary": {"passed": passes, "warn": warns, "failed": fails},
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                }
                for r in results
            ],
        },
        indent=2,
        ensure_ascii=False,
    )


def _counts(results: list[CheckResult]) -> tuple[int, int, int]:
    fails = sum(r.status == Status.FAIL for r in results)
    warns = sum(r.status == Status.WARN for r in results)
    passes = sum(r.status == Status.PASS for r in results)
    return fails, warns, passes
