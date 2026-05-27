# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""submissions get command."""

from __future__ import annotations

import sys

import click

from ...exceptions import APIError
from ...submissions import api as subs_api
from ...submissions.formatters import print_submission_detail
from ..common import _console, _get_token, output_json

__all__ = ["submissions_get"]


@click.command("get")
@click.option("--submission-id", required=True, help="Submission UUID.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
@click.option("-j", "--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
def submissions_get(submission_id: str, token: str | None, as_json: bool) -> None:
    """Get full submission details including embedded runs."""
    resolved_token = _get_token(token)
    try:
        sub = subs_api.get_submission(resolved_token, submission_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    if as_json:
        output_json(sub)
    else:
        print_submission_detail(sub)
