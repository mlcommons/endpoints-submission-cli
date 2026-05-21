# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Entry point for the endpoints-submission-cli tool."""

from cyclopts import App

from .commands.runs import runs
from .commands.submissions import submissions

__all__ = ["app", "main"]

app = App(
    name="endpoints-submission-cli",
    help="MLPerf rolling submission CLI — manage benchmark runs and submissions.",
    version_flags=["--version"],
)
app.command(runs)
app.command(submissions)


def main() -> None:
    """Entry point called by the ``endpoints-submission-cli`` script."""
    app()
