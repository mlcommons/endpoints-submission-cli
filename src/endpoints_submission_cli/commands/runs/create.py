# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""runs create command."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

import click

from ...exceptions import APIError, ArchiveError, RunFolderError
from ...runs import api as runs_api
from ...runs.parser import build_archive, parse_run_folder
from ..common import _console, _get_token

__all__ = ["runs_create"]


@click.command("create")
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
    "--expires-at",
    default=None,
    help=(
        "Expiry datetime in ISO 8601 format (e.g. 2026-01-01T00:00:00)."
        " Defaults to server policy."
    ),
)
@click.option(
    "--pinned",
    is_flag=True,
    default=False,
    help="Pin the run immediately to prevent automatic expiry.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the parsed payload as JSON and exit without calling the API.",
)
def runs_create(
    path: Path, token: str | None, expires_at: str | None, pinned: bool, dry_run: bool
) -> None:
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

    if expires_at is not None:
        payload["expires_at"] = expires_at
    if pinned:
        payload["pinned"] = True

    if dry_run:
        print(json.dumps(payload, indent=2, default=str))
        return

    resolved_token = _get_token(token)

    try:
        run_out = runs_api.create_run(resolved_token, payload)
    except APIError as exc:
        _console.print(f"[bold red]API error creating run:[/bold red] {exc}")
        sys.exit(1)

    run_id: str = run_out["id"]

    # At this point the run record exists in the DB but has no archive yet.
    # If the upload fails we must roll back by deleting the run record so we
    # don't leave an orphaned entry with no associated data.
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = build_archive(path, Path(tmp) / f"{path.name}.tar.gz")
        try:
            upload_result = runs_api.upload_run_archive(resolved_token, run_id, archive_path)
        except (APIError, ArchiveError, OSError) as exc:
            _console.print(
                f"[bold red]Archive upload failed:[/bold red] {exc}\n"
                f"[yellow]Rolling back run {run_id}…[/yellow]"
            )
            # Step 1: attempt to delete the partial archive from storage (best-effort,
            # ignore failure — storage may not have received anything).
            with contextlib.suppress(APIError):
                runs_api.delete_run_archive(resolved_token, run_id)
            # Step 2: delete the run DB record to restore clean state.
            try:
                runs_api.delete_run(resolved_token, run_id)
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
