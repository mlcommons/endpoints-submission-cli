# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Entry point for the endpoints-submission-cli tool."""

import click

from ._version_check import register_upgrade_notice
from .commands.check_submission import check_submission
from .commands.runs import runs
from .commands.submissions import submissions

__all__ = ["app", "main"]


@click.group()
@click.version_option(package_name="endpoints-submission-cli")
def app() -> None:
    """MLPerf rolling submission CLI — manage benchmark runs and submissions."""


app.add_command(runs)
app.add_command(submissions)
app.add_command(check_submission)


def main() -> None:
    """Entry point called by the ``endpoints-submission-cli`` script."""
    register_upgrade_notice()
    app()
