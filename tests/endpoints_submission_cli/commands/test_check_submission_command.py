# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the top-level ``check-submission`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from endpoints_submission_cli.main import app

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_TEST_SUBMISSIONS = _REPO_ROOT / "test_submissions"
_VALID = _TEST_SUBMISSIONS / "valid_standardized"
_SUB_E = _TEST_SUBMISSIONS / "sub_e"  # missing low-throughput coverage -> error

_runner = CliRunner()

# Force a deterministic "outside CI" environment so annotation auto-detection
# (which keys off $GITHUB_ACTIONS) doesn't depend on where the tests run.
_NO_CI = {"GITHUB_ACTIONS": ""}


@pytest.mark.unit
class TestCheckSubmission:
    def test_valid_submission_passes(self) -> None:
        result = _runner.invoke(app, ["check-submission", str(_VALID)], env=_NO_CI)
        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_invalid_submission_fails(self) -> None:
        result = _runner.invoke(app, ["check-submission", str(_SUB_E)], env=_NO_CI)
        assert result.exit_code == 1
        assert "low-throughput-coverage" in result.output
        assert "FAILED" in result.output

    def test_missing_path_exits_one(self, tmp_path: Path) -> None:
        result = _runner.invoke(app, ["check-submission", str(tmp_path / "nope")], env=_NO_CI)
        assert result.exit_code == 1

    def test_quiet_hides_info(self) -> None:
        result = _runner.invoke(app, ["check-submission", "--quiet", str(_SUB_E)], env=_NO_CI)
        # The verdict line still reports counts, but no INFO rows should appear.
        assert "info" not in result.output.lower()

    def test_json_output_to_stdout(self) -> None:
        # No annotations (outside CI) -> stdout is pure JSON.
        result = _runner.invoke(app, ["check-submission", "--json", str(_VALID)], env=_NO_CI)
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["passed"] is True
        assert "results" in data

    def test_json_stdout_clean_even_with_annotations(self) -> None:
        """Annotations go to stderr, so --json stdout stays parseable inside CI."""
        result = _runner.invoke(
            app,
            ["check-submission", "--json", "--annotate", str(_SUB_E)],
            env=_NO_CI,
        )
        assert result.exit_code == 1
        data = json.loads(result.stdout)  # would raise if annotations leaked to stdout
        assert data["passed"] is False
        assert "::error " in result.stderr

    def test_output_flag_writes_json_file(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        result = _runner.invoke(
            app, ["check-submission", "--output", str(out), str(_VALID)], env=_NO_CI
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        assert data["passed"] is True

    def test_annotate_emits_github_commands_on_stderr(self) -> None:
        result = _runner.invoke(app, ["check-submission", "--annotate", str(_SUB_E)], env=_NO_CI)
        assert result.exit_code == 1
        assert "::error " in result.stderr
        assert "title=submission-checker:" in result.stderr

    def test_no_annotate_by_default_outside_ci(self) -> None:
        result = _runner.invoke(app, ["check-submission", str(_SUB_E)], env=_NO_CI)
        assert "::error " not in result.stderr
        assert "::error " not in result.stdout

    def test_github_env_enables_annotations(self) -> None:
        result = _runner.invoke(
            app,
            ["check-submission", str(_SUB_E)],
            env={"GITHUB_ACTIONS": "true"},
        )
        assert "::error " in result.stderr

    def test_step_summary_written(self, tmp_path: Path) -> None:
        summary = tmp_path / "summary.md"
        result = _runner.invoke(
            app,
            ["check-submission", str(_SUB_E)],
            env={"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(summary)},
        )
        assert result.exit_code == 1
        text = summary.read_text()
        assert "Submission Checker" in text
        assert "FAILED" in text

    def test_strict_treats_warnings_as_errors(self) -> None:
        """valid_standardized passes normally; under --strict any warning fails it."""
        plain = _runner.invoke(app, ["check-submission", str(_VALID)], env=_NO_CI)
        strict = _runner.invoke(app, ["check-submission", "--strict", str(_VALID)], env=_NO_CI)
        # If there were warnings, strict flips 0 -> 1; otherwise both pass.
        if "0 warning(s)" not in plain.output:
            assert strict.exit_code == 1
        else:
            assert strict.exit_code == 0
