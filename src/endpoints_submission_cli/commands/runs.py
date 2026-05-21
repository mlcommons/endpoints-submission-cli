# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for managing benchmark runs."""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from rich.console import Console

from .. import api_client
from ..exceptions import APIError, ArchiveError, AuthError, RunFolderError
from ..formatters import output_json, print_runs_table
from ..run_parser import build_archive, parse_run_folder

__all__ = ["runs"]

runs = App(name="runs", help="Manage benchmark runs.")
_console = Console(stderr=True)


def _get_token(token: str | None) -> str:
    try:
        return api_client.get_token(token)
    except AuthError as exc:
        _console.print(f"[bold red]Auth error:[/bold red] {exc}")
        raise SystemExit(1) from None


# ---------------------------------------------------------------------------
# runs list
# ---------------------------------------------------------------------------


@runs.command(name="list")
def runs_list(
    *,
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
    json: Annotated[
        bool,
        Parameter(name=["-j", "--json"], help="Output raw JSON."),
    ] = False,
) -> None:
    """List all runs for the authenticated user."""
    resolved_token = _get_token(token)
    try:
        run_list = api_client.list_runs(resolved_token)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    if json:
        output_json(run_list)
    else:
        print_runs_table(run_list)


# ---------------------------------------------------------------------------
# runs create
# ---------------------------------------------------------------------------


@runs.command(name="create")
def runs_create(
    *,
    path: Annotated[Path, Parameter(help="Path to the local run folder.")],
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Print the parsed payload as JSON and exit without calling the API.",
        ),
    ] = False,
) -> None:
    """Create a run from a local benchmark result folder.

    Parses system_info.json, config.yaml, and result_summary.json from PATH,
    registers the run with the Submission API, and uploads the run folder as
    an archive.  If the archive upload fails the run record is deleted
    (rollback to clean state).
    """
    # 1. Parse run folder
    try:
        payload = parse_run_folder(path)
    except RunFolderError as exc:
        _console.print(f"[bold red]Run folder error:[/bold red] {exc}")
        raise SystemExit(1) from None

    if dry_run:
        print(json.dumps(payload, indent=2, default=str))
        return

    resolved_token = _get_token(token)

    # 2. POST /runs → get run_id
    try:
        run_out = api_client.create_run(resolved_token, payload)
    except APIError as exc:
        _console.print(f"[bold red]API error creating run:[/bold red] {exc}")
        raise SystemExit(1) from None

    run_id: str = run_out["id"]

    # 3. Build .tar.gz archive and upload
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = build_archive(path, Path(tmp) / f"{path.name}.tar.gz")
        try:
            upload_result = api_client.upload_run_archive(resolved_token, run_id, archive_path)
        except (APIError, ArchiveError, OSError) as exc:
            _console.print(
                f"[bold red]Archive upload failed:[/bold red] {exc}\n"
                f"[yellow]Rolling back run {run_id}…[/yellow]"
            )
            # Best-effort: delete any partial GCS object before removing the DB record.
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
            raise SystemExit(1) from None

    archive_uri = upload_result.get("archive_uri") if upload_result else None
    _console.print(f"[bold green]Run created:[/bold green] {run_id}")
    if archive_uri:
        _console.print(f"[dim]Archive:[/dim] {archive_uri}")


# ---------------------------------------------------------------------------
# runs get
# ---------------------------------------------------------------------------


@runs.command(name="get")
def runs_get(
    *,
    run_id: Annotated[str, Parameter(name="--run-id", help="Run UUID.")],
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
) -> None:
    """Get full details of a specific run."""
    resolved_token = _get_token(token)
    try:
        run = api_client.get_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    output_json(run)


# ---------------------------------------------------------------------------
# runs delete
# ---------------------------------------------------------------------------


@runs.command(name="delete")
def runs_delete(
    *,
    run_id: Annotated[str, Parameter(name="--run-id", help="Run UUID.")],
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
) -> None:
    """Delete a run and its stored archive.

    If the run is part of an active submission the API will reject the request.
    Withdraw the submission first in that case.

    The archive is removed from GCS first so that a DB-delete failure leaves
    nothing orphaned in storage.  A 404 on archive deletion means the run has
    no archive and is silently skipped.
    """
    resolved_token = _get_token(token)

    # 1. Delete the GCS archive first (run record still exists, so auth works).
    #    404 = no archive uploaded — treat as a no-op.
    try:
        api_client.delete_run_archive(resolved_token, run_id)
    except APIError as exc:
        if "404" not in str(exc):
            _console.print(
                f"[yellow]Warning:[/yellow] Archive deletion failed: {exc}\n"
                "Continuing to delete the run record."
            )

    # 2. Delete the DB record.
    try:
        api_client.delete_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error deleting run:[/bold red] {exc}")
        raise SystemExit(1) from None

    _console.print(f"[bold green]Run deleted:[/bold green] {run_id}")


# ---------------------------------------------------------------------------
# runs pin / runs unpin
# ---------------------------------------------------------------------------


@runs.command(name="pin")
def runs_pin(
    *,
    run_id: Annotated[str, Parameter(name="--run-id", help="Run UUID.")],
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
) -> None:
    """Pin a run to prevent expiry (sets expires_at = null)."""
    resolved_token = _get_token(token)
    try:
        api_client.pin_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    _console.print(f"[bold green]Run pinned:[/bold green] {run_id}")


@runs.command(name="unpin")
def runs_unpin(
    *,
    run_id: Annotated[str, Parameter(name="--run-id", help="Run UUID.")],
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
) -> None:
    """Unpin a run to restore normal expiry behaviour."""
    resolved_token = _get_token(token)
    try:
        api_client.unpin_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    _console.print(f"[bold green]Run unpinned:[/bold green] {run_id}")
