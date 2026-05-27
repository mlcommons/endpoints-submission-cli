# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runs.formatters module."""

from __future__ import annotations

import pytest

from endpoints_submission_cli.runs.formatters import print_run_detail, print_runs_table
from tests.endpoints_submission_cli.conftest import RUN_ID, RUN_OUT, RUN_SUMMARY


@pytest.mark.unit
class TestPrintRunsTable:
    def test_empty_list_no_table(self) -> None:
        print_runs_table([])

    def test_runs_rendered(self) -> None:
        print_runs_table([RUN_SUMMARY])

    def test_multiple_runs(self) -> None:
        runs = [RUN_SUMMARY, {**RUN_SUMMARY, "id": "other-id"}]
        print_runs_table(runs)


@pytest.mark.unit
class TestPrintRunDetail:
    def test_renders_without_error(self) -> None:
        print_run_detail(RUN_OUT)

    def test_renders_minimal_run(self) -> None:
        print_run_detail({"id": RUN_ID, "system_info": {}})
