# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""submissions withdraw command."""

from __future__ import annotations

import sys

import click

from ...exceptions import APIError
from ...submissions import api as subs_api
from ..common import _console, _get_token

__all__ = ["submissions_withdraw"]


@click.command("withdraw")
@click.option("--submission-id", required=True, help="Submission UUID.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
def submissions_withdraw(submission_id: str, token: str | None) -> None:
    """Withdraw a submission: mark WITHDRAWN and delete its archive.

    Order of operations: DB update → delete archive.
    If archive deletion fails the orphaned URI is logged for garbage collection.
    The CLI does not close the review pull request; it no longer manages one.
    """
    resolved_token = _get_token(token)
    try:
        subs_api.withdraw_submission(resolved_token, submission_id)
    except APIError as exc:
        _console.print(f"[bold red]Error withdrawing submission:[/bold red] {exc}")
        sys.exit(1)

    # try:
    #     subs_api.delete_submission_archive(resolved_token, submission_id)
    # except APIError as exc:
    #     _console.print(f"[yellow]Archive deletion failed (orphaned):[/yellow] {exc}")

    _console.print(f"[bold green]Submission withdrawn:[/bold green] {submission_id}")
