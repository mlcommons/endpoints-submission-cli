# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``install-skill`` and the skill it installs."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from endpoints_submission_cli.commands.install_skill import SKILL_NAME
from endpoints_submission_cli.main import app

_BUNDLED = (files("endpoints_submission_cli") / "skills" / SKILL_NAME / "SKILL.md").read_text(
    encoding="utf-8"
)


def _installed(skills_dir: Path) -> Path:
    return skills_dir / SKILL_NAME / "SKILL.md"


@pytest.mark.unit
class TestBundledSkill:
    def test_frontmatter_names_the_skill(self) -> None:
        _, frontmatter, _ = _BUNDLED.split("---", 2)
        meta = yaml.safe_load(frontmatter)
        assert meta["name"] == SKILL_NAME
        assert meta["description"]

    def test_every_command_it_names_exists(self) -> None:
        """The skill documents the CLI; a renamed command must not leave it stale."""
        runner = CliRunner()
        for command in (
            ["check-submission"],
            ["runs", "list"],
            ["runs", "get"],
            ["runs", "create"],
            ["runs", "delete"],
            ["runs", "pin"],
            ["runs", "unpin"],
            ["submissions", "list"],
            ["submissions", "get"],
            ["submissions", "create"],
            ["submissions", "update"],
            ["submissions", "remove-run"],
            ["submissions", "withdraw"],
        ):
            assert " ".join(command) in _BUNDLED
            result = runner.invoke(app, [*command, "--help"])
            assert result.exit_code == 0, f"{command}: {result.output}"


@pytest.mark.unit
class TestInstallSkill:
    def test_installs_into_a_given_directory(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(app, ["install-skill", "--dest", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert _installed(tmp_path).read_text(encoding="utf-8") == _BUNDLED

    def test_defaults_to_the_home_directory(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        result = CliRunner().invoke(app, ["install-skill"])
        assert result.exit_code == 0, result.output
        assert _installed(tmp_path / ".claude" / "skills").exists()

    def test_project_installs_under_the_working_directory(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
            result = runner.invoke(app, ["install-skill", "--project"])
            assert result.exit_code == 0, result.output
            assert _installed(Path(cwd) / ".claude" / "skills").exists()

    def test_reinstalling_the_same_version_is_a_no_op(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(app, ["install-skill", "--dest", str(tmp_path)])
        result = runner.invoke(app, ["install-skill", "--dest", str(tmp_path)])
        assert result.exit_code == 0 and "already up to date" in result.output

    def test_a_different_copy_needs_force(self, tmp_path: Path) -> None:
        target = _installed(tmp_path)
        target.parent.mkdir(parents=True)
        target.write_text("locally edited", encoding="utf-8")
        runner = CliRunner()

        result = runner.invoke(app, ["install-skill", "--dest", str(tmp_path)])
        assert result.exit_code != 0 and "--force" in result.output
        assert target.read_text(encoding="utf-8") == "locally edited"

        result = runner.invoke(app, ["install-skill", "--dest", str(tmp_path), "--force"])
        assert result.exit_code == 0
        assert target.read_text(encoding="utf-8") == _BUNDLED

    def test_dest_and_project_are_exclusive(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(app, ["install-skill", "--dest", str(tmp_path), "--project"])
        assert result.exit_code != 0 and "mutually exclusive" in result.output
