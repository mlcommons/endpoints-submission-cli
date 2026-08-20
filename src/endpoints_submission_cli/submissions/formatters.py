# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Rich-based output formatters for submission records."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from ..formatting import DASH, fmt_bool, fmt_dt, fmt_int, fmt_str
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
    # no_wrap keeps the "*" test marker on the same line as the status instead of
    # doubling the row height; max_width stops the widened cell from starving the
    # other columns on a narrow terminal.
    table.add_column("Status", style="yellow", no_wrap=True, overflow="ellipsis", max_width=21)
    table.add_column("Division")
    table.add_column("Availability")
    table.add_column("Runs", justify="right")
    table.add_column("Created At", style="dim")

    for sub in submissions:
        run_ids = sub.get("run_ids", [])
        status = fmt_str(sub.get("status"))
        if sub.get("is_test"):
            # A one-char prefix rather than a column: a wider marker wraps the cell
            # onto a second line, and a trailing one is truncated away first.
            # Explained by the legend printed below the table.
            status = f"[yellow]*[/yellow] {status}"
        table.add_row(
            fmt_str(sub.get("id")),
            status,
            fmt_str(sub.get("division")),
            fmt_str(sub.get("availability")),
            str(len(run_ids)),
            fmt_dt(sub.get("created_at")),
        )

    _console.print(table)
    if any(s.get("is_test") for s in submissions):
        _console.print("[dim][yellow]*[/yellow] = test submission[/dim]")


def print_submission_detail(submission: dict[str, Any]) -> None:
    """Print full details for a single submission (SubmissionWithRuns schema)."""
    # No show_lines here: the detail view is ~27 rows, and rule lines between each
    # would push it past a single screen. The list views keep theirs.
    table = Table(title=f"Submission {submission.get('id', '')}")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")

    run_ids = submission.get("run_ids", [])
    rows = [
        ("ID", fmt_str(submission.get("id"))),
        ("User ID", fmt_str(submission.get("user_id"))),
        ("Status", fmt_str(submission.get("status"))),
        ("Division", fmt_str(submission.get("division"))),
        ("Scenario", fmt_str(submission.get("scenario"))),
        ("Availability", fmt_str(submission.get("availability"))),
        ("Early Publish", fmt_bool(submission.get("early_publish"))),
        ("Test Submission", fmt_bool(submission.get("is_test"))),
        ("Publication Cycle", fmt_str(submission.get("publication_cycle"))),
        ("Target Availability Date", fmt_str(submission.get("target_availability_date"))),
        ("Embargo Date", fmt_dt(submission.get("embargo_date"))),
        ("Reviewers Assigned", fmt_int(submission.get("reviewers_assigned"))),
        ("Checker Version", fmt_str(submission.get("submission_checker_version"))),
        ("API Version", fmt_str(submission.get("api_version"))),
        ("CLI Version", fmt_str(submission.get("cli_version"))),
        ("Run IDs", "\n".join(str(r) for r in run_ids) if run_ids else DASH),
        ("PR URL", fmt_str(submission.get("pr_url"))),
        ("PR Number", fmt_int(submission.get("pr_number"))),
        ("Archive URI", fmt_str(submission.get("archive_uri"))),
        ("Created At", fmt_dt(submission.get("created_at"))),
        ("Availability Qualified At", fmt_dt(submission.get("availability_qualified_at"))),
        ("Compliance Passed At", fmt_dt(submission.get("compliance_passed_at"))),
        ("Peer Review Started At", fmt_dt(submission.get("peer_review_started_at"))),
        # Abbreviated: the full label is 31 chars and would push the no_wrap Field
        # column wide enough to wrap the PR URL mid-string at 80 columns.
        ("Objection Res. Started At", fmt_dt(submission.get("objection_resolution_started_at"))),
        ("First Published At", fmt_dt(submission.get("first_published_at"))),
        ("Finalized At", fmt_dt(submission.get("finalized_at"))),
        ("Withdrawn At", fmt_dt(submission.get("withdrawn_at"))),
    ]
    for field, value in rows:
        table.add_row(field, value)

    _console.print(table)

    # Embedded runs
    runs = submission.get("runs", [])
    if runs:
        _console.print(f"\n[bold]Runs ({len(runs)}):[/bold]")
        print_runs_table(runs)
