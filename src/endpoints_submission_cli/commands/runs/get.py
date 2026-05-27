# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""runs get command."""

from __future__ import annotations

import sys

import click

from ...exceptions import APIError
from ...runs import api as runs_api
from ..common import _console, _get_token, output_json

__all__ = ["runs_get"]


@click.command("get")
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
        run = runs_api.get_run(resolved_token, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    output_json(run)
