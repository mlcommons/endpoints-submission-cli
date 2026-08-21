# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pre-submission run-set checks."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from endpoints_submission_cli.exceptions import MixedTestRunsError
from endpoints_submission_cli.submissions.validation import resolve_test_flag

TOKEN = "mlc_test"
_A, _B, _C = "run-a", "run-b", "run-c"


def _runs(**flags: bool | None):
    """Patch get_run so each id maps to a run carrying the given is_test."""

    def _get(_token: str, run_id: str) -> dict:
        value = flags[run_id]
        return {"id": run_id} if value is None else {"id": run_id, "is_test": value}

    return patch("endpoints_submission_cli.runs.api.get_run", side_effect=_get)


@pytest.mark.unit
class TestResolveTestFlag:
    def test_all_real_runs(self) -> None:
        with _runs(**{_A: False, _B: False}):
            assert resolve_test_flag(TOKEN, [_A, _B]) is False

    def test_all_test_runs(self) -> None:
        with _runs(**{_A: True, _B: True}):
            assert resolve_test_flag(TOKEN, [_A, _B]) is True

    def test_mixed_runs_rejected(self) -> None:
        with _runs(**{_A: True, _B: False}):
            with pytest.raises(MixedTestRunsError, match="cannot mix test runs"):
                resolve_test_flag(TOKEN, [_A, _B])

    def test_mixed_error_names_both_sides(self) -> None:
        with _runs(**{_A: True, _B: False, _C: False}):
            with pytest.raises(MixedTestRunsError) as exc:
                resolve_test_flag(TOKEN, [_A, _B, _C])
        msg = str(exc.value)
        assert "1 test run(s): run-a" in msg
        assert "2 non-test run(s): run-b, run-c" in msg

    def test_duplicate_ids_fetched_once(self) -> None:
        with _runs(**{_A: False}) as mock_get:
            assert resolve_test_flag(TOKEN, [_A, _A, _A]) is False
        assert mock_get.call_count == 1

    def test_empty_run_set_is_unknown(self) -> None:
        with _runs():
            assert resolve_test_flag(TOKEN, []) is None

    # ── agreement with the submission's own flag ──────────────────────────

    def test_test_runs_match_test_submission(self) -> None:
        with _runs(**{_A: True}):
            assert resolve_test_flag(TOKEN, [_A], expected=True) is True

    def test_test_runs_rejected_for_real_submission(self) -> None:
        """Test runs + real submission — the fix is to make the submission a test one."""
        with _runs(**{_A: True}):
            with pytest.raises(MixedTestRunsError, match=r"Pass --test"):
                resolve_test_flag(TOKEN, [_A], expected=False)

    def test_real_runs_rejected_for_test_submission(self) -> None:
        """Real runs + test submission — the fix is to drop the flag."""
        with _runs(**{_A: False}):
            with pytest.raises(MixedTestRunsError, match=r"Drop --test"):
                resolve_test_flag(TOKEN, [_A], expected=True)

    def test_expected_none_skips_agreement_check(self) -> None:
        with _runs(**{_A: True}):
            assert resolve_test_flag(TOKEN, [_A], expected=None) is True

    # ── tolerance for an API that predates is_test ────────────────────────

    def test_server_without_is_test_skips_check(self) -> None:
        """No run reports the field — skip rather than guessing False."""
        with _runs(**{_A: None, _B: None}):
            assert resolve_test_flag(TOKEN, [_A, _B], expected=True) is None

    def test_partial_reporting_uses_only_known_runs(self) -> None:
        with _runs(**{_A: True, _B: None}):
            assert resolve_test_flag(TOKEN, [_A, _B]) is True
