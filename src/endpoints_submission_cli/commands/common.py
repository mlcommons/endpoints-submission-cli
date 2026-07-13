# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for all CLI commands."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .. import _http
from ..exceptions import AuthError, SubmissionCheckError

__all__ = [
    "_console",
    "_get_token",
    "_SEVERITY_STYLE",
    "_run_submission_checker",
    "_write_cli_metadata",
    "output_json",
]


def _write_cli_metadata(submission_dir: Path, command: str) -> None:
    """Write a cli_metadata.json marker at the bundle root.

    Records which CLI command and version assembled the submission so reviewers and
    the lifecycle can tell which bundled builder/checker schema produced it — the
    marker is what lets us reason about checker-format drift across CLI versions.
    """
    try:
        version = importlib.metadata.version("endpoints-submission-cli")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    meta = {
        "command": command,
        "cli_version": version,
        "created_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    }
    (submission_dir / "cli_metadata.json").write_text(json.dumps(meta, indent=2))


_console = Console(stderr=True)
_stdout_console = Console()

_SEVERITY_STYLE = {
    "error": "bold red",
    "warning": "yellow",
    "info": "dim",
}


def output_json(data: Any) -> None:
    """Print *data* as pretty-printed JSON."""
    print(json.dumps(data, indent=2, default=str))


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
