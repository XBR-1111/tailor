"""Terminal and JSON report rendering for `tailor check`."""

import json

from rich.console import Console
from rich.table import Table

from tailor.model import CheckResult, Status, summarize

_STATUS_STYLE = {
    Status.PASS: "[green]PASS[/green]",
    Status.WARN: "[yellow]WARN[/yellow]",
    Status.FAIL: "[red]FAIL[/red]",
    Status.SKIP: "[dim]SKIP[/dim]",
}


def _counts(results: list[CheckResult]) -> tuple[int, int, int, int]:
    """(fails, warns, passes, skips) — skips counted too: a meta-only run
    shows 11 passed + 11 skipped, and hiding the skips makes '22 checks'
    look like '11 checks'."""
    fails, warns, passes = summarize(results)
    skips = sum(r.status == Status.SKIP for r in results)
    return fails, warns, passes, skips


def render_terminal(results: list[CheckResult], dataset: str) -> None:
    console = Console()
    fails, warns, passes, skips = _counts(results)
    table = Table(title=f"tailor — acceptance report: {dataset}", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("status", justify="center")
    table.add_column("detail")
    for r in results:
        table.add_row(r.name, _STATUS_STYLE[r.status], r.message)
    console.print(table)
    verdict = "[red]REJECTED[/red]" if fails else "[green]ACCEPTED[/green]"
    footer = f"{verdict}  ·  {passes} passed, {warns} warn, {fails} failed"
    if skips:
        footer += f", {skips} skipped"
    console.print(footer)


def render_json(results: list[CheckResult], dataset: str) -> str:
    fails, warns, passes, skips = _counts(results)
    return json.dumps(
        {
            "dataset": dataset,
            "verdict": "rejected" if fails else "accepted",
            "summary": {
                "passed": passes,
                "warn": warns,
                "failed": fails,
                "skipped": skips,
            },
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


def render_markdown(results: list[CheckResult], dataset: str) -> str:
    """PR/dataset-card friendly report."""
    fails, warns, passes, skips = _counts(results)
    verdict = "REJECTED" if fails else ("ACCEPTED WITH WARNINGS" if warns else "ACCEPTED")
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "➖"}  # noqa: RUF001
    summary = f"{passes} passed, {warns} warn, {fails} failed"
    if skips:
        summary += f", {skips} skipped"
    lines = [
        "# tailor — acceptance report",
        "",
        f"**Dataset:** `{dataset}`",
        "",
        f"**Verdict:** {verdict} — {summary}",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for r in results:
        detail = r.message.replace("|", r"\|")
        lines.append(f"| `{r.name}` | {icon[r.status]} {r.status.value} | {detail} |")
    return "\n".join(lines) + "\n"
