# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""runs list command."""

from __future__ import annotations

import sys

import click

from ...exceptions import APIError
from ...runs import api as runs_api
from ...runs.formatters import print_runs_table
from ..common import _console, _get_token, output_json

__all__ = ["runs_list"]


@click.command("list")
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
        run_list = runs_api.list_runs(resolved_token)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    if as_json:
        output_json(run_list)
    else:
        print_runs_table(run_list)
