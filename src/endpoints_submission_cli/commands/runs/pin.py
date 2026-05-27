# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""runs pin command."""

from __future__ import annotations

import sys

import click

from ...exceptions import APIError
from ...runs import api as runs_api
from ..common import _console, _get_token

__all__ = ["runs_pin"]


@click.command("pin")
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
        runs_api.pin_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    _console.print(f"[bold green]Run pinned:[/bold green] {run_id}")
