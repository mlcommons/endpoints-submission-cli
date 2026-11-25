# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Run every registered check over one submission."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from .checks import ALL_CHECKS, Check
from .context import (
    INPUT_BASE,
    INPUT_LOGS,
    INPUT_POINT_CONFIGS,
    INPUT_SUMMARIES,
    INPUT_SYSTEM_DESCRIPTIONS,
    ReviewContext,
    build_context,
)
from .findings import CheckOutcome, Finding, ReviewReport
from .thresholds import Thresholds, load_thresholds

__all__ = ["review"]

_log = logging.getLogger(__name__)

#: Why a check is skipped when one of its inputs is missing.
_MISSING_INPUT: dict[str, str] = {
    INPUT_SUMMARIES: "no result_summary.json could be read",
    INPUT_POINT_CONFIGS: "no point.yaml could be read",
    INPUT_SYSTEM_DESCRIPTIONS: "no system_desc.json could be read",
    INPUT_BASE: "needs the original submission to compare with (--base-ref); "
    "the review workflow passes it once the PR has updates",
    INPUT_LOGS: "needs the per-request logs (events.jsonl), which are not in the submission "
    "(pass --logs-dir)",
}


#: Point names listed in a coverage gap before the rest are counted.
_MAX_NAMES = 6


def _names(point_dirs: set[Path]) -> str:
    names = sorted({d.name for d in point_dirs}, key=lambda n: (len(n), n))
    shown = ", ".join(names[:_MAX_NAMES])
    return shown + (f" and {len(names) - _MAX_NAMES} more" if len(names) > _MAX_NAMES else "")


def _ran(check: Check, ctx: ReviewContext) -> CheckOutcome:
    """The outcome of a check that ran, accounting for the logs it could not use.

    A log check covers the points it asked a log for. With none asked for, or no usable
    log among them, it evaluated nothing and is skipped; with some missing or unreadable,
    it covered the focus area in part, and the reason names the points left out.
    """
    if INPUT_LOGS not in check.requires:
        return CheckOutcome(check=check.name, focus_area=check.focus_area, status="ran")
    requested = ctx.logs.requested
    if not requested:
        return CheckOutcome(
            check=check.name,
            focus_area=check.focus_area,
            status="skipped",
            reason="no point it applies to",
        )
    missing = ctx.logs.missing(requested)
    unreadable = ctx.logs.unreadable(requested)
    gaps = []
    if missing:
        gaps.append(f"no log for {_names(missing)}")
    if unreadable:
        gaps.append(f"unreadable log for {_names(unreadable)} (see the notes)")
    usable = len(requested) - len(missing) - len(unreadable)
    if not gaps:
        return CheckOutcome(check=check.name, focus_area=check.focus_area, status="ran")
    if usable == 0:
        return CheckOutcome(
            check=check.name, focus_area=check.focus_area, status="skipped", reason="; ".join(gaps)
        )
    return CheckOutcome(
        check=check.name,
        focus_area=check.focus_area,
        status="ran",
        reason=f"{usable} of {len(requested)} points evaluated; " + "; ".join(gaps),
    )


def review(
    root: Path,
    thresholds: Thresholds | None = None,
    label: str | None = None,
    checks: Sequence[Check] = ALL_CHECKS,
    base_ref: str | None = None,
    logs_dir: Path | None = None,
) -> ReviewReport:
    """Review the submission at *root* and return what every check found.

    A check whose inputs are missing is recorded as skipped. A check that raises is
    recorded as an error and the others still run: one broken heuristic must not hide
    the rest of the review.

    Args:
        root: A ``<submission_id>/`` directory, holding ``results/``.
        thresholds: Tolerances; the packaged defaults when omitted.
        label: How to name the submission in the report.
        checks: The checks to run; every registered check by default.
        base_ref: A git revision holding the original submission. The checks that
            compare an update with it are skipped without one.
        logs_dir: Where to find per-request logs a submission does not ship; see
            :func:`~submission_reviewer.context.build_context`.
    """
    ctx = build_context(root, thresholds or load_thresholds(), label, base_ref, logs_dir)
    findings: list[Finding] = []
    outcomes: list[CheckOutcome] = []
    for check in checks:
        missing = sorted(check.requires - ctx.inputs)
        if missing:
            reason = "; ".join(_MISSING_INPUT.get(m, f"{m} unavailable") for m in missing)
            outcomes.append(
                CheckOutcome(
                    check=check.name, focus_area=check.focus_area, status="skipped", reason=reason
                )
            )
            continue
        ctx.logs.track()
        try:
            found = check.run(ctx)
        except Exception as exc:  # isolate one check's failure from the rest
            _log.exception("review check %s failed", check.name)
            outcomes.append(
                CheckOutcome(
                    check=check.name,
                    focus_area=check.focus_area,
                    status="error",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        findings.extend(found)
        outcomes.append(_ran(check, ctx))
    return ReviewReport(
        submission=ctx.label,
        findings=findings,
        outcomes=outcomes,
        notes=[*ctx.notes, *ctx.logs.problems],
    )
