# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""submissions command group."""

import click

from .add_run import submissions_add_run
from .create import submissions_create
from .create_local import submissions_create_local
from .get import submissions_get
from .list import submissions_list
from .remove_run import submissions_remove_run
from .update import submissions_update
from .withdraw import submissions_withdraw

__all__ = ["submissions"]


@click.group(name="submissions")
def submissions() -> None:
    """Manage MLPerf submissions."""


submissions.add_command(submissions_list)
submissions.add_command(submissions_create)
submissions.add_command(submissions_create_local)
submissions.add_command(submissions_get)
submissions.add_command(submissions_update)
submissions.add_command(submissions_withdraw)
submissions.add_command(submissions_add_run)
submissions.add_command(submissions_remove_run)
