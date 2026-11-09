# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Entry point for the endpoints-submission-cli tool."""

import logging

import click
from rich.console import Console
from rich.logging import RichHandler

from ._version_check import register_upgrade_notice
from .commands.check_submission import check_submission
from .commands.install_skill import install_skill
from .commands.runs import runs
from .commands.submissions import submissions

__all__ = ["app", "main"]


@click.group()
@click.version_option(package_name="endpoints-submission-cli")
@click.pass_context
def app(ctx: click.Context) -> None:
    """MLPerf rolling submission CLI — manage benchmark runs and submissions."""
    logger = logging.getLogger("endpoints_submission_cli")
    handler = RichHandler(console=Console(stderr=True), show_time=False, show_path=False)
    logger.addHandler(handler)
    ctx.call_on_close(lambda: logger.removeHandler(handler))


app.add_command(runs)
app.add_command(submissions)
app.add_command(check_submission)
app.add_command(install_skill)


def main() -> None:
    """Entry point called by the ``endpoints-submission-cli`` script."""
    register_upgrade_notice()
    app()
