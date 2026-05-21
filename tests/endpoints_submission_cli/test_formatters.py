# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for formatters module."""

from __future__ import annotations

import json

import pytest

from endpoints_submission_cli.formatters import (
    output_json,
    print_run_detail,
    print_runs_table,
    print_submission_detail,
    print_submissions_table,
)
from tests.endpoints_submission_cli.conftest import (
    RUN_ID,
    RUN_OUT,
    RUN_SUMMARY,
    SUBMISSION_ID,
    SUBMISSION_OUT,
)


@pytest.mark.unit
class TestOutputJson:
    def test_pretty_prints(self, capsys: pytest.CaptureFixture) -> None:
        output_json({"key": "value"})
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed == {"key": "value"}

    def test_list(self, capsys: pytest.CaptureFixture) -> None:
        output_json([1, 2, 3])
        out = capsys.readouterr().out
        assert json.loads(out) == [1, 2, 3]

    def test_datetime_serialized(self, capsys: pytest.CaptureFixture) -> None:
        from datetime import datetime

        output_json({"ts": datetime(2025, 1, 1)})
        out = capsys.readouterr().out
        assert "2025" in out


@pytest.mark.unit
class TestPrintRunsTable:
    def test_empty_list_no_table(self, capsys: pytest.CaptureFixture) -> None:
        print_runs_table([])
        # Rich prints to stderr by default in tests; just ensure no crash
        # (actual output goes to rich console stderr)

    def test_runs_rendered(self) -> None:
        # Just verify no exception is raised
        print_runs_table([RUN_SUMMARY])

    def test_multiple_runs(self) -> None:
        runs = [RUN_SUMMARY, {**RUN_SUMMARY, "id": "other-id"}]
        print_runs_table(runs)  # no exception


@pytest.mark.unit
class TestPrintRunDetail:
    def test_renders_without_error(self) -> None:
        print_run_detail(RUN_OUT)

    def test_renders_minimal_run(self) -> None:
        print_run_detail({"id": RUN_ID, "system_info": {}})


@pytest.mark.unit
class TestPrintSubmissionsTable:
    def test_empty_list(self) -> None:
        print_submissions_table([])

    def test_renders_submission(self) -> None:
        print_submissions_table([SUBMISSION_OUT])


@pytest.mark.unit
class TestPrintSubmissionDetail:
    def test_renders_without_error(self) -> None:
        print_submission_detail(SUBMISSION_OUT)

    def test_renders_with_runs(self) -> None:
        sub = {**SUBMISSION_OUT, "runs": [RUN_OUT]}
        print_submission_detail(sub)

    def test_renders_minimal(self) -> None:
        print_submission_detail({"id": SUBMISSION_ID})
