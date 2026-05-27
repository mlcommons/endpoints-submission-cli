# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Rich-based output formatters for run records."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

__all__ = ["print_runs_table", "print_run_detail"]

_console = Console()


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
# Helpers
# ---------------------------------------------------------------------------


def _fmt_dt(value: str | None) -> str:
    if not value:
        return ""
    # Truncate to seconds for readability
    return str(value).replace("T", " ").split(".")[0]
