# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Rich-based output formatters for submission records."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from ..runs.formatters import print_runs_table

__all__ = ["print_submissions_table", "print_submission_detail"]

_console = Console()


def print_submissions_table(submissions: list[dict[str, Any]]) -> None:
    """Print a summary table of submission records (SubmissionOut schema)."""
    if not submissions:
        _console.print("[dim]No submissions found.[/dim]")
        return

    table = Table(title="Submissions", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="yellow")
    table.add_column("Division")
    table.add_column("Availability")
    table.add_column("Runs", justify="right")
    table.add_column("Created At", style="dim")

    for sub in submissions:
        run_ids = sub.get("run_ids", [])
        table.add_row(
            str(sub.get("id", "")),
            str(sub.get("status", "—")),
            str(sub.get("division", "—")),
            str(sub.get("availability", "—")),
            str(len(run_ids)),
            _fmt_dt(sub.get("created_at")),
        )

    _console.print(table)


def print_submission_detail(submission: dict[str, Any]) -> None:
    """Print full details for a single submission (SubmissionWithRuns schema)."""
    table = Table(title=f"Submission {submission.get('id', '')}", show_lines=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")

    run_ids = submission.get("run_ids", [])
    rows = [
        ("ID", str(submission.get("id", ""))),
        ("User ID", str(submission.get("user_id", ""))),
        ("Status", str(submission.get("status", "—"))),
        ("Division", str(submission.get("division", "—"))),
        ("Availability", str(submission.get("availability", "—"))),
        ("Provisional", "Yes" if submission.get("early_publish") else "No"),
        ("Publication Cycle", str(submission.get("publication_cycle") or "—")),
        ("Target Availability Date", str(submission.get("target_availability_date") or "—")),
        ("Run IDs", "\n".join(run_ids) if run_ids else "—"),
        ("PR URL", str(submission.get("pr_url") or "—")),
        ("PR Number", str(submission.get("pr_number") or "—")),
        ("Archive URI", str(submission.get("archive_uri") or "—")),
        ("Created At", _fmt_dt(submission.get("created_at"))),
        ("Compliance Passed At", _fmt_dt(submission.get("compliance_passed_at")) or "—"),
        ("Finalized At", _fmt_dt(submission.get("finalized_at")) or "—"),
        ("Withdrawn At", _fmt_dt(submission.get("withdrawn_at")) or "—"),
    ]
    for field, value in rows:
        table.add_row(field, value)

    _console.print(table)

    # Embedded runs
    runs = submission.get("runs", [])
    if runs:
        _console.print(f"\n[bold]Runs ({len(runs)}):[/bold]")
        print_runs_table(runs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_dt(value: str | None) -> str:
    if not value:
        return ""
    # Truncate to seconds for readability
    return str(value).replace("T", " ").split(".")[0]
