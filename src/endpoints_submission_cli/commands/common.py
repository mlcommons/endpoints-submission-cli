# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for all CLI commands."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.json import JSON
from rich.table import Table

from .. import _http
from ..exceptions import AuthError, SubmissionCheckError

__all__ = ["_console", "_get_token", "_SEVERITY_STYLE", "_run_submission_checker", "output_json"]

_console = Console(stderr=True)
_stdout_console = Console()

_SEVERITY_STYLE = {
    "error": "bold red",
    "warning": "yellow",
    "info": "dim",
}


def output_json(data: Any) -> None:
    """Print *data* as syntax-highlighted JSON (plain when piped)."""
    _stdout_console.print(JSON(json.dumps(data, default=str)))


def _get_token(token: str | None) -> str:
    try:
        return _http.get_token(token)
    except AuthError as exc:
        _console.print(f"[bold red]Auth error:[/bold red] {exc}")
        sys.exit(1)


def _run_submission_checker(submission_dir: Path) -> None:
    """Run the SubmissionChecker on *submission_dir*; print results table; raise on errors."""
    from submission_checker.checker import SubmissionChecker

    report = SubmissionChecker(submission_dir).run()

    # --- write Rich table to log file ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path.cwd() / f"submission_checker_{timestamp}.log"

    table = Table(show_lines=True, expand=False)
    table.add_column("Rule", style="cyan", no_wrap=True)
    table.add_column("§ Ref", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Message")
    table.add_column("Path")

    for r in report.results:
        sev = r.severity.value
        style = _SEVERITY_STYLE.get(sev, "")
        table.add_row(
            r.rule,
            r.spec_ref,
            f"[{style}]{sev}[/{style}]" if style else sev,
            r.message,
            str(r.path) if r.path else "",
        )

    with open(log_path, "w", encoding="utf-8") as fh:
        file_console = Console(file=fh, no_color=True, width=220)
        file_console.print(
            f"Submission Checker Report — {timestamp}\n"
            f"Directory : {submission_dir}\n"
            f"Results   : {len(report.results)} checks"
            f" ({len(report.errors)} error(s), {len(report.warnings)} warning(s))\n"
        )
        file_console.print(table)

    _console.print(f"[dim]Checker report written to {log_path}[/dim]")

    errors = report.errors
    if errors:
        msgs = "\n".join(f"  [{e.rule}] {e.message}" for e in errors)
        raise SubmissionCheckError(f"Submission checker found {len(errors)} error(s):\n{msgs}")
