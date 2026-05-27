# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""submissions list command."""

from __future__ import annotations

import sys

import click

from ...exceptions import APIError
from ...submissions import api as subs_api
from ...submissions.formatters import print_submissions_table
from ..common import _console, _get_token, output_json

__all__ = ["submissions_list"]


@click.command("list")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
@click.option("-j", "--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
def submissions_list(token: str | None, as_json: bool) -> None:
    """List all submissions for the authenticated user."""
    resolved_token = _get_token(token)
    try:
        sub_list = subs_api.list_submissions(resolved_token)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    if as_json:
        output_json(sub_list)
    else:
        print_submissions_table(sub_list)
