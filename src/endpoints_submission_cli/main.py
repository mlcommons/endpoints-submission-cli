# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Entry point for the endpoints-submission-cli tool."""

import click

from .commands.runs import runs
from .commands.submissions import submissions

__all__ = ["app", "main"]


@click.group()
@click.version_option(version="0.1.0")
def app() -> None:
    """MLPerf rolling submission CLI — manage benchmark runs and submissions."""


app.add_command(runs)
app.add_command(submissions)


def main() -> None:
    """Entry point called by the ``endpoints-submission-cli`` script."""
    app()
