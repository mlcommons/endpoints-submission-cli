# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared Rich cell formatters."""

from __future__ import annotations

import pytest

from endpoints_submission_cli.formatting import DASH, fmt_bool, fmt_dt, fmt_int, fmt_str


class TestFmtInt:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # 0 is a real count, not "absent" — this is the whole reason fmt_int exists.
            (0, "0"),
            (42, "42"),
            (None, DASH),
        ],
    )
    def test_renders(self, value: int | None, expected: str) -> None:
        assert fmt_int(value) == expected


class TestFmtBool:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, "Yes"),
            # False is a real answer; only None means "the server didn't send it".
            (False, "No"),
            (None, DASH),
        ],
    )
    def test_renders(self, value: bool | None, expected: str) -> None:
        assert fmt_bool(value) == expected


class TestFmtDt:
    def test_truncates_to_seconds(self) -> None:
        assert fmt_dt("2025-04-28T09:15:00.123456") == "2025-04-28 09:15:00"

    def test_without_fractional_seconds(self) -> None:
        assert fmt_dt("2025-04-28T09:15:00") == "2025-04-28 09:15:00"

    @pytest.mark.parametrize("value", [None, ""])
    def test_missing_renders_dash(self, value: str | None) -> None:
        assert fmt_dt(value) == DASH


class TestFmtStr:
    def test_stringifies(self) -> None:
        assert fmt_str(42) == "42"

    @pytest.mark.parametrize("value", [None, ""])
    def test_missing_renders_dash(self, value: str | None) -> None:
        assert fmt_str(value) == DASH
