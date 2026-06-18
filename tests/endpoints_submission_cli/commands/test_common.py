# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for endpoints_submission_cli.commands.common._run_submission_checker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from endpoints_submission_cli.commands.common import _run_submission_checker
from endpoints_submission_cli.exceptions import SubmissionCheckError
from submission_checker.models.results import CheckResult, Report, Severity


def _report(tmp_path: Path, *results: CheckResult) -> Report:
    return Report(submission_path=tmp_path, results=list(results))


def _err(rule: str = "test-rule", message: str = "something broke") -> CheckResult:
    return CheckResult(rule=rule, message=message, severity=Severity.ERROR)


def _ok(rule: str = "test-rule") -> CheckResult:
    return CheckResult(rule=rule, message="all good", severity=Severity.INFO)


@pytest.mark.unit
class TestRunSubmissionChecker:
    def test_raises_submission_check_error_when_errors(self, tmp_path: Path) -> None:
        report = _report(tmp_path, _err("path-exists", "Submission path does not exist"))
        with patch.object(Path, "cwd", return_value=tmp_path), patch(
            "submission_checker.checker.SubmissionChecker"
        ) as mock_cls:
            mock_cls.return_value.run.return_value = report
            with pytest.raises(SubmissionCheckError, match="1 error"):
                _run_submission_checker(tmp_path)

    def test_no_raise_when_report_is_clean(self, tmp_path: Path) -> None:
        report = _report(tmp_path, _ok("path-exists"), _ok("system-description-valid"))
        with patch.object(Path, "cwd", return_value=tmp_path), patch(
            "submission_checker.checker.SubmissionChecker"
        ) as mock_cls:
            mock_cls.return_value.run.return_value = report
            _run_submission_checker(tmp_path)  # must not raise

    def test_writes_log_file_to_cwd(self, tmp_path: Path) -> None:
        report = _report(tmp_path)
        with patch.object(Path, "cwd", return_value=tmp_path), patch(
            "submission_checker.checker.SubmissionChecker"
        ) as mock_cls:
            mock_cls.return_value.run.return_value = report
            _run_submission_checker(tmp_path)
        log_files = list(tmp_path.glob("submission_checker_*.log"))
        assert len(log_files) == 1

    def test_error_message_includes_rule_and_text(self, tmp_path: Path) -> None:
        report = _report(tmp_path, _err("low-throughput-coverage", "Region not covered"))
        with patch.object(Path, "cwd", return_value=tmp_path), patch(
            "submission_checker.checker.SubmissionChecker"
        ) as mock_cls:
            mock_cls.return_value.run.return_value = report
            with pytest.raises(SubmissionCheckError) as exc_info:
                _run_submission_checker(tmp_path)
        msg = str(exc_info.value)
        assert "low-throughput-coverage" in msg
        assert "Region not covered" in msg

    def test_multiple_errors_all_reported(self, tmp_path: Path) -> None:
        report = _report(tmp_path, _err("rule-a", "first"), _err("rule-b", "second"))
        with patch.object(Path, "cwd", return_value=tmp_path), patch(
            "submission_checker.checker.SubmissionChecker"
        ) as mock_cls:
            mock_cls.return_value.run.return_value = report
            with pytest.raises(SubmissionCheckError, match="2 error"):
                _run_submission_checker(tmp_path)

    def test_checker_called_with_submission_dir(self, tmp_path: Path) -> None:
        report = _report(tmp_path)
        with patch.object(Path, "cwd", return_value=tmp_path), patch(
            "submission_checker.checker.SubmissionChecker"
        ) as mock_cls:
            mock_cls.return_value.run.return_value = report
            _run_submission_checker(tmp_path / "my_sub")
        mock_cls.assert_called_once_with(tmp_path / "my_sub")
