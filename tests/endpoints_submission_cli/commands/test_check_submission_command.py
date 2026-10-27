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
# sub_g's points start at 64, so under v1.0's derived boundaries (C_min clamps to
# 32, Low Concurrency 33–45) it covers neither Ultra Low nor Low Concurrency.
_FAILING = _TEST_SUBMISSIONS / "sub_g"

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
        result = _runner.invoke(app, ["check-submission", str(_FAILING)], env=_NO_CI)
        assert result.exit_code == 1
        assert "low-concurrency-coverage" in result.output
        assert "FAILED" in result.output

    def test_missing_path_exits_one(self, tmp_path: Path) -> None:
        result = _runner.invoke(app, ["check-submission", str(tmp_path / "nope")], env=_NO_CI)
        assert result.exit_code == 1

    def test_quiet_hides_info(self) -> None:
        result = _runner.invoke(app, ["check-submission", "--quiet", str(_FAILING)], env=_NO_CI)
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
            ["check-submission", "--json", "--annotate", str(_FAILING)],
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
        result = _runner.invoke(app, ["check-submission", "--annotate", str(_FAILING)], env=_NO_CI)
        assert result.exit_code == 1
        assert "::error " in result.stderr
        # Titled with the rule's title and spec section, not its id twice.
        assert "title=Low Concurrency coverage (§5.4)::" in result.stderr
        assert "[low-concurrency-coverage]" not in result.stderr

    def test_no_annotate_by_default_outside_ci(self) -> None:
        result = _runner.invoke(app, ["check-submission", str(_FAILING)], env=_NO_CI)
        assert "::error " not in result.stderr
        assert "::error " not in result.stdout

    def test_github_env_enables_annotations(self) -> None:
        result = _runner.invoke(
            app,
            ["check-submission", str(_FAILING)],
            env={"GITHUB_ACTIONS": "true"},
        )
        assert "::error " in result.stderr

    def test_step_summary_written(self, tmp_path: Path) -> None:
        summary = tmp_path / "summary.md"
        result = _runner.invoke(
            app,
            ["check-submission", str(_FAILING)],
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


_GITHUB = {"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": ""}


def _first_index(text: str, needle: str) -> int:
    index = text.find(needle)
    assert index >= 0, f"{needle!r} not in output"
    return index


@pytest.mark.unit
class TestCheckSubmissionOrdering:
    """Errors come first and stand out; warnings and info are folded away in CI."""

    def test_errors_listed_before_warnings_and_info(self) -> None:
        # sub_g's checker order interleaves severities; the table must not.
        result = _runner.invoke(app, ["check-submission", str(_FAILING)], env=_NO_CI)
        severities = [
            word
            for line in result.output.splitlines()
            for word in ("error", "warning", "info")
            if f"│ {word} " in line
        ]
        assert severities == sorted(severities, key=["error", "warning", "info"].index)
        assert severities[0] == "error"

    def test_github_log_is_coloured_and_wide(self) -> None:
        result = _runner.invoke(app, ["check-submission", str(_FAILING)], env=_GITHUB)
        assert "\x1b[" in result.stdout  # ANSI colour despite stdout not being a TTY
        # Messages are not cut to an 80-column terminal.
        assert "low-concurrency-coverage" in result.stdout

    def test_github_log_folds_warnings_and_info_but_not_errors(self) -> None:
        result = _runner.invoke(app, ["check-submission", str(_FAILING)], env=_GITHUB)
        out = result.stdout
        errors_table = _first_index(out, ": errors")
        warnings_group = _first_index(out, "::group::")
        assert errors_table < warnings_group  # errors print outside any group
        assert "warning(s)" in out[warnings_group : out.index("\n", warnings_group)]
        assert out.count("::group::") == out.count("::endgroup::") == 2  # warnings, info

    def test_github_quiet_drops_the_info_group(self) -> None:
        result = _runner.invoke(app, ["check-submission", "--quiet", str(_FAILING)], env=_GITHUB)
        assert result.stdout.count("::group::") == 1
        assert "info result(s)" not in result.stdout

    def test_passing_submission_has_no_error_table(self) -> None:
        result = _runner.invoke(app, ["check-submission", str(_VALID)], env=_GITHUB)
        assert result.exit_code == 0
        assert ": errors" not in result.stdout

    def test_step_summary_shows_errors_and_collapses_warnings(self, tmp_path: Path) -> None:
        summary = tmp_path / "summary.md"
        _runner.invoke(
            app,
            ["check-submission", str(_FAILING)],
            env={"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(summary)},
        )
        text = summary.read_text()
        errors = _first_index(text, "### 🔴 Errors")
        details = _first_index(text, "<details>")
        assert errors < details
        assert "| 🔴 error |" in text[errors:details]
        assert "🟡 warning" not in text[errors:details]
        assert "| 🟡 warning |" in text[details : _first_index(text, "</details>")]


@pytest.mark.unit
class TestCheckSubmissionWording:
    """Titles, fixes and grouping come from the message catalog."""

    def test_annotations_carry_the_fix(self) -> None:
        result = _runner.invoke(app, ["check-submission", "--annotate", str(_FAILING)], env=_NO_CI)
        coverage = next(
            line for line in result.stderr.splitlines() if "title=Low Concurrency coverage" in line
        )
        assert "%0A%0AFix: Add a point with concurrency in" in coverage

    def test_table_shows_titles_and_fixes(self) -> None:
        result = _runner.invoke(app, ["check-submission", str(_FAILING)], env=_GITHUB)
        out = result.stdout
        assert "Benchmark model name" in out
        assert "Fix: Set model_name in every point.yaml" in out

    def test_missing_thresholds_are_reported_once_per_curve(self) -> None:
        result = _runner.invoke(app, ["check-submission", "--json", str(_FAILING)], env=_NO_CI)
        gate = [r for r in json.loads(result.stdout)["results"] if r["key"] == "no-thresholds"]
        assert len(gate) == 1

    def test_step_summary_has_titles_and_fixes(self, tmp_path: Path) -> None:
        summary = tmp_path / "summary.md"
        _runner.invoke(
            app,
            ["check-submission", str(_FAILING)],
            env={"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(summary)},
        )
        text = summary.read_text()
        assert "| Severity | Check | Message | Path |" in text
        assert "Benchmark model name (§3.2)<br><sub>`model-name-valid`</sub>" in text
        assert "<br>**Fix:** Set model_name in every point.yaml" in text


@pytest.mark.unit
class TestFindingGrouping:
    def test_identical_results_become_one_row_with_a_count(self) -> None:
        from endpoints_submission_cli.commands.check_submission import _findings
        from submission_checker.models import err

        results = [
            err("point-dirs", "fail-2", Path(f"/s/r{c}"), rel=f"results/x/r{c}") for c in (1, 2)
        ] + [err("src-dir", "fail", Path(f"/s/{n}")) for n in ("a", "b", "c")]

        (first, second, third) = _findings(results)
        assert (first.count, second.count, third.count) == (1, 1, 3)
        assert third.message().endswith("(×3)")
