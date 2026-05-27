# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""runs command group."""

import click

from .create import runs_create
from .delete import runs_delete
from .get import runs_get
from .list import runs_list
from .pin import runs_pin
from .unpin import runs_unpin

__all__ = ["runs"]


@click.group(name="runs")
def runs() -> None:
    """Manage benchmark runs."""


runs.add_command(runs_list)
runs.add_command(runs_create)
runs.add_command(runs_get)
runs.add_command(runs_delete)
runs.add_command(runs_pin)
runs.add_command(runs_unpin)
