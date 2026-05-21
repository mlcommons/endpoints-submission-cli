# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for managing benchmark runs."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

import click
from rich.console import Console

from .. import api_client
from ..exceptions import APIError, ArchiveError, AuthError, RunFolderError
from ..formatters import output_json, print_runs_table
from ..run_parser import build_archive, parse_run_folder

__all__ = ["runs"]

_console = Console(stderr=True)


def _get_token(token: str | None) -> str:
    try:
        return api_client.get_token(token)
    except AuthError as exc:
        _console.print(f"[bold red]Auth error:[/bold red] {exc}")
        sys.exit(1)


@click.group(name="runs")
def runs() -> None:
    """Manage benchmark runs."""


# ---------------------------------------------------------------------------
# runs list
# ---------------------------------------------------------------------------


@runs.command("list")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
@click.option("-j", "--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
def runs_list(token: str | None, as_json: bool) -> None:
    """List all runs for the authenticated user."""
    resolved_token = _get_token(token)
    try:
        run_list = api_client.list_runs(resolved_token)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    if as_json:
        output_json(run_list)
    else:
        print_runs_table(run_list)


# ---------------------------------------------------------------------------
# runs create
# ---------------------------------------------------------------------------


@runs.command("create")
@click.option(
    "--path",
    "path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to the local run folder.",
)
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the parsed payload as JSON and exit without calling the API.",
)
def runs_create(path: Path, token: str | None, dry_run: bool) -> None:
    """Create a run from a local benchmark result folder.

    Parses system_info.json, config.yaml, and result_summary.json from PATH,
    registers the run with the Submission API, and uploads the run folder as
    an archive.  If the archive upload fails the run record is deleted
    (rollback to clean state).
    """
    try:
        payload = parse_run_folder(path)
    except RunFolderError as exc:
        _console.print(f"[bold red]Run folder error:[/bold red] {exc}")
        sys.exit(1)

    if dry_run:
        print(json.dumps(payload, indent=2, default=str))
        return

    resolved_token = _get_token(token)

    try:
        run_out = api_client.create_run(resolved_token, payload)
    except APIError as exc:
        _console.print(f"[bold red]API error creating run:[/bold red] {exc}")
        sys.exit(1)

    run_id: str = run_out["id"]

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = build_archive(path, Path(tmp) / f"{path.name}.tar.gz")
        try:
            upload_result = api_client.upload_run_archive(resolved_token, run_id, archive_path)
        except (APIError, ArchiveError, OSError) as exc:
            _console.print(
                f"[bold red]Archive upload failed:[/bold red] {exc}\n"
                f"[yellow]Rolling back run {run_id}…[/yellow]"
            )
            with contextlib.suppress(APIError):
                api_client.delete_run_archive(resolved_token, run_id)
            try:
                api_client.delete_run(resolved_token, run_id)
                _console.print("[green]Rollback successful — run deleted.[/green]")
            except APIError as rb_exc:
                _console.print(
                    f"[bold red]Rollback also failed:[/bold red] {rb_exc}\n"
                    f"Orphaned run ID: {run_id}"
                )
            sys.exit(1)

    archive_uri = upload_result.get("archive_uri") if upload_result else None
    _console.print(f"[bold green]Run created:[/bold green] {run_id}")
    if archive_uri:
        _console.print(f"[dim]Archive:[/dim] {archive_uri}")


# ---------------------------------------------------------------------------
# runs get
# ---------------------------------------------------------------------------


@runs.command("get")
@click.option("--run-id", required=True, help="Run UUID.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
def runs_get(run_id: str, token: str | None) -> None:
    """Get full details of a specific run."""
    resolved_token = _get_token(token)
    try:
        run = api_client.get_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    output_json(run)


# ---------------------------------------------------------------------------
# runs delete
# ---------------------------------------------------------------------------


@runs.command("delete")
@click.option("--run-id", required=True, help="Run UUID.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
def runs_delete(run_id: str, token: str | None) -> None:
    """Delete a run and its stored archive.

    If the run is part of an active submission the API will reject the request.
    Withdraw the submission first in that case.

    The archive is removed from GCS first so that a DB-delete failure leaves
    nothing orphaned in storage.  A 404 on archive deletion means the run has
    no archive and is silently skipped.
    """
    resolved_token = _get_token(token)

    try:
        api_client.delete_run_archive(resolved_token, run_id)
    except APIError as exc:
        if "404" not in str(exc):
            _console.print(
                f"[yellow]Warning:[/yellow] Archive deletion failed: {exc}\n"
                "Continuing to delete the run record."
            )

    try:
        api_client.delete_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error deleting run:[/bold red] {exc}")
        sys.exit(1)

    _console.print(f"[bold green]Run deleted:[/bold green] {run_id}")


# ---------------------------------------------------------------------------
# runs pin / runs unpin
# ---------------------------------------------------------------------------


@runs.command("pin")
@click.option("--run-id", required=True, help="Run UUID.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
def runs_pin(run_id: str, token: str | None) -> None:
    """Pin a run to prevent expiry (sets expires_at = null)."""
    resolved_token = _get_token(token)
    try:
        api_client.pin_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    _console.print(f"[bold green]Run pinned:[/bold green] {run_id}")


@runs.command("unpin")
@click.option("--run-id", required=True, help="Run UUID.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
def runs_unpin(run_id: str, token: str | None) -> None:
    """Unpin a run to restore normal expiry behaviour."""
    resolved_token = _get_token(token)
    try:
        api_client.unpin_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    _console.print(f"[bold green]Run unpinned:[/bold green] {run_id}")
