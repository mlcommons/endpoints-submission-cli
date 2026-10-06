# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""``install-skill`` command — copy the bundled Claude skill where Claude Code finds it.

The skill ships inside the package so that it is versioned with the CLI it describes.
``pip install`` cannot place it in ``~/.claude/skills/``, so this command does, and is
re-run after an upgrade to pick up the matching version.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import click

__all__ = ["SKILL_NAME", "install_skill"]

#: The bundled skill's directory name, which is also its ``name`` in SKILL.md.
SKILL_NAME = "mlperf-submissions"


def _bundled_skill() -> str:
    return (files("endpoints_submission_cli") / "skills" / SKILL_NAME / "SKILL.md").read_text(
        encoding="utf-8"
    )


@click.command("install-skill")
@click.option(
    "--project",
    is_flag=True,
    help="Install into ./.claude/skills/ (this project) instead of ~/.claude/skills/.",
)
@click.option(
    "--dest",
    type=click.Path(file_okay=False, path_type=Path),
    help="Install into this skills directory instead.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace an installed copy that differs from this version's.",
)
def install_skill(project: bool, dest: Path | None, force: bool) -> None:
    """Install the mlperf-submissions skill for Claude Code."""
    if dest is not None and project:
        raise click.UsageError("--dest and --project are mutually exclusive.")
    if dest is None:
        dest = (Path.cwd() if project else Path.home()) / ".claude" / "skills"

    target = dest / SKILL_NAME / "SKILL.md"
    content = _bundled_skill()
    if target.exists():
        if target.read_text(encoding="utf-8") == content:
            click.echo(f"{target} is already up to date.")
            return
        if not force:
            raise click.ClickException(
                f"{target} exists and differs from this version's skill. Re-run with"
                " --force to replace it; any local edits to it will be lost."
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    click.echo(f"Installed the {SKILL_NAME} skill to {target}")
