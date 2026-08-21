# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Cross-run checks applied before a submission's run set is accepted."""

from __future__ import annotations

from collections.abc import Sequence

from ..exceptions import MixedTestRunsError
from ..runs import api as runs_api

__all__ = ["resolve_test_flag"]


def resolve_test_flag(
    token: str,
    run_ids: Sequence[str],
    *,
    expected: bool | None = None,
    expected_label: str = "submission",
) -> bool | None:
    """Return the ``is_test`` value shared by *run_ids*, or None if unknowable.

    A submission is either a test entry or a real results entry; it cannot be half
    of each. Mixing the two would let a real run ride along inside a submission
    excluded from reporting, or strand a test run inside a published one.

    Args:
        token: PRISM API key.
        run_ids: Runs that would make up the submission.
        expected: If given, the flag the run set must also agree with — the
            submission's own ``is_test``.
        expected_label: What *expected* describes, used in the error message.

    Returns:
        The shared ``is_test`` of the run set, or ``None`` when the API does not
        report the field (a server older than ``is_test``), in which case no
        check is performed.

    Raises:
        MixedTestRunsError: If the runs disagree with each other or with *expected*.
        APIError: If a run cannot be fetched.
    """
    flags: dict[str, bool] = {}
    for run_id in dict.fromkeys(run_ids):
        run = runs_api.get_run(token, run_id)
        value = run.get("is_test")
        # A server predating is_test omits the key entirely; skip rather than
        # guessing False and rejecting a legitimate set.
        if value is not None:
            flags[run_id] = bool(value)

    if not flags:
        return None

    test_runs = sorted(rid for rid, v in flags.items() if v)
    real_runs = sorted(rid for rid, v in flags.items() if not v)

    if test_runs and real_runs:
        raise MixedTestRunsError(
            "a submission cannot mix test runs with real ones — "
            f"{len(test_runs)} test run(s): {', '.join(test_runs)}; "
            f"{len(real_runs)} non-test run(s): {', '.join(real_runs)}. "
            "Re-register the odd ones out with a matching --test setting, "
            "or submit them separately."
        )

    actual = bool(test_runs)
    if expected is not None and actual != expected:
        run_kind = "test runs" if actual else "non-test runs"
        want = "a test submission" if expected else "a real results submission"
        raise MixedTestRunsError(
            f"the run set contains only {run_kind}, which does not match "
            f"{want} ({expected_label} is_test={expected}). "
            f"{'Pass --test' if actual else 'Drop --test'} so the two agree."
        )
    return actual
