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

import click
from rich.console import Console
from rich.table import Table

from .. import _http
from ..exceptions import AuthError, SubmissionCheckError

__all__ = [
    "_confirm_provisional",
    "_console",
    "_check_publication_mode",
    "_normalize_embargo",
    "PUBLICATION_MODES",
    "_get_token",
    "_SEVERITY_STYLE",
    "_run_submission_checker",
    "_write_cli_metadata",
    "CLI_METADATA_FILENAME",
    "output_json",
]

_PROVISIONAL_WARNING = (
    "WARNING: This will make these results publicly viewable on the visualizer "
    "with a 'peer review pending' disclaimer, before peer review is complete, "
    "{when}."
)


# Submission Rules §6.2. Declared at creation; the API rejects any later change.
PUBLICATION_MODES = ("confidential", "confidential_embargoed", "provisional")


def _check_publication_mode(publication_mode: str, embargo_date: str | None) -> None:
    """Exit 1 unless the embargo fits the mode, as the API would require.

    confidential_embargoed needs an embargo, which is what holds its finalized result
    (§6.2.2); provisional may have one (§6.2.3); confidential must not. Checked before
    any download, so a mismatch costs nothing.
    """
    if publication_mode == "confidential" and embargo_date:
        _console.print(
            "[bold red]--embargo-date needs --publication-mode confidential_embargoed "
            "or provisional.[/bold red]"
        )
        sys.exit(1)
    if publication_mode == "confidential_embargoed" and not embargo_date:
        _console.print(
            "[bold red]--publication-mode confidential_embargoed requires --embargo-date."
            "[/bold red]"
        )
        sys.exit(1)


def _normalize_embargo(value: str, *, require_future: bool) -> str:
    """Validate an --embargo-date and return it with an explicit UTC offset; exit 1 if bad.

    A value without an offset is taken as UTC and sent with one, so the instant does
    not depend on the API server's local time zone. Called before any download, so a
    typo costs nothing.
    """
    text = value.strip()
    try:
        # fromisoformat only accepts a trailing Z from Python 3.11.
        parsed = datetime.datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    except ValueError:
        _console.print(
            f"[bold red]Invalid --embargo-date {value!r}:[/bold red] expected ISO 8601, "
            "e.g. 2026-12-01T16:00:00Z or 2026-12-01T08:00:00-08:00."
        )
        sys.exit(1)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        _console.print(
            f"[dim]--embargo-date has no UTC offset; taking it as UTC: {parsed.isoformat()}[/dim]"
        )
    if require_future and parsed <= datetime.datetime.now(datetime.timezone.utc):
        _console.print(f"[bold red]--embargo-date {value!r} is not in the future.[/bold red]")
        sys.exit(1)
    return parsed.isoformat()


def _confirm_provisional(assume_yes: bool, embargo_date: str | None = None) -> None:
    """Confirm provisional publication before submitting; exit 1 if declined.

    Opting in publishes unreviewed results, so the submitter has to say yes out loud.
    ``--yes`` is the escape hatch for non-interactive use, where there is no one to ask.
    """
    when = (
        f"from the embargo date, {embargo_date}"
        if embargo_date
        else "as soon as the automated compliance checks pass"
    )
    _console.print(f"[bold yellow]{_PROVISIONAL_WARNING.format(when=when)}[/bold yellow]")
    if assume_yes:
        _console.print("[dim]--yes given: continuing with provisional publication.[/dim]")
        return
    if not click.confirm("Would you like to continue?", default=False, err=True):
        _console.print("[yellow]Aborted — no submission created.[/yellow]")
        sys.exit(1)


CLI_METADATA_FILENAME = "cli_metadata.json"


def _cli_version() -> str:
    """The installed CLI version, or ``"unknown"`` outside an installed package."""
    try:
        return importlib.metadata.version("endpoints-submission-cli")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _write_cli_metadata(
    submission_dir: Path,
    command: str,
    submission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the cli_metadata.json marker inside a submission directory.

    Everything in the marker is derived from what the command already holds — the
    submission record it just received from the API, plus the version of the CLI
    running now. Nothing has to be read back out of the previous bundle.

    ``cli_version`` is the client that created the submission and stays fixed for its
    lifetime; ``most_recent_cli_used`` is the client running this command. On
    ``create`` the two are the same, because this *is* the creating command.

    Args:
        submission_dir: The ``<submission_id>/`` directory the marker describes. Pass
            the submission level, never the organisation directory above it: that one
            is shared by every submission from the org, so a marker there would claim
            to describe all of them and be overwritten by the next build.
        command: The CLI command that assembled the bundle.
        submission: The API's submission record, supplying the creating
            ``cli_version`` and ``created_at``. Omit it on ``create``, where the
            running CLI is the creating one and there is no earlier record.

    Returns:
        The metadata that was written.
    """
    current = _cli_version()
    record = submission or {}
    meta: dict[str, Any] = {
        "command": command,
        # Falls back to the running version: on create there is no prior record, and
        # an API too old to report cli_version should not blank the field out.
        "cli_version": record.get("cli_version") or current,
        "most_recent_cli_used": current,
        "created_at": record.get("created_at")
        or datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    }
    (submission_dir / CLI_METADATA_FILENAME).write_text(json.dumps(meta, indent=2))
    return meta


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
