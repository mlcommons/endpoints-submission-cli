# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Rich-based output formatters for runs and submissions."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.json import JSON
from rich.table import Table

__all__ = [
    "print_runs_table",
    "print_run_detail",
    "print_submissions_table",
    "print_submission_detail",
    "output_json",
]

_console = Console()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def output_json(data: Any) -> None:
    """Print *data* as syntax-highlighted JSON (plain when piped)."""
    _console.print(JSON(json.dumps(data, default=str)))


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def print_runs_table(runs: list[dict[str, Any]]) -> None:
    """Print a summary table of run records (RunSummary schema)."""
    if not runs:
        _console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Runs", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Concurrency", justify="right")
    table.add_column("Started At", style="dim")
    table.add_column("Finished At", style="dim")

    for run in runs:
        table.add_row(
            str(run.get("id", "")),
            str(run.get("model") or "—"),
            str(run.get("concurrency") or "—"),
            _fmt_dt(run.get("started_at")),
            _fmt_dt(run.get("finished_at")),
        )

    _console.print(table)


def print_run_detail(run: dict[str, Any]) -> None:
    """Print full details for a single run (RunOut schema)."""
    table = Table(title=f"Run {run.get('id', '')}", show_lines=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")

    rows = [
        ("ID", str(run.get("id", ""))),
        ("User ID", str(run.get("user_id", ""))),
        ("Benchmark Version", str(run.get("benchmark_version", ""))),
        ("Started At", _fmt_dt(run.get("started_at"))),
        ("Finished At", _fmt_dt(run.get("finished_at"))),
        ("Expires At", _fmt_dt(run.get("expires_at")) or "—"),
        ("Pinned", "Yes" if run.get("pinned") else "No"),
        ("Archive URI", str(run.get("archive_uri") or "—")),
    ]
    for field, value in rows:
        table.add_row(field, value)

    _console.print(table)

    # System info summary
    si = run.get("system_info", {})
    if si:
        si_table = Table(title="System Info", show_lines=True)
        si_table.add_column("Field", style="cyan")
        si_table.add_column("Value")
        for k, v in si.items():
            si_table.add_row(str(k), str(v))
        _console.print(si_table)


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------


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
        ("Early Publish", "Yes" if submission.get("early_publish") else "No"),
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
