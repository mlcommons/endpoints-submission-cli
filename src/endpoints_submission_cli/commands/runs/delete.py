# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""runs delete command."""

from __future__ import annotations

import sys

import click

from ...exceptions import APIError
from ...runs import api as runs_api
from ..common import _console, _get_token

__all__ = ["runs_delete"]


@click.command("delete")
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
        runs_api.delete_run_archive(resolved_token, run_id)
    except APIError as exc:
        if "404" not in str(exc):
            _console.print(
                f"[yellow]Warning:[/yellow] Archive deletion failed: {exc}\n"
                "Continuing to delete the run record."
            )

    try:
        runs_api.delete_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error deleting run:[/bold red] {exc}")
        sys.exit(1)

    _console.print(f"[bold green]Run deleted:[/bold green] {run_id}")
