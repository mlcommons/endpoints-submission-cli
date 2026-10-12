# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding drafters (§2.9.4) — approval and lead time.

§9.1 adds two checks, both per point but both needing the submission's cohort and
the curve's benchmark, so they live here rather than on :class:`PointConfig`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ...cohorts import Cohort
from ...drafters import DRAFTER_APPROVAL_LEAD_COHORTS, ApprovedDrafter
from ..file.point_config import PointConfig
from ..results import CheckResult, err, ok, warn

__all__ = ["DrafterBinding"]


class DrafterBinding(BaseModel):
    """Validates a curve's speculative-decoding disclosures against the approved list."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    #: ``(point.yaml path, parsed config)`` for every point whose config loaded.
    points: list[tuple[Path, PointConfig]]
    #: Approved drafters for this curve's benchmark (§2.9.4). Empty means none are.
    approved: list[ApprovedDrafter]
    #: The benchmark model this curve measures, for messages.
    benchmark: str
    #: The curve directory, used as the path for curve-wide findings.
    model_dir: Path
    #: The first cohort date on or after the submission's receipt — §2.9.4's "review
    #: eligibility date" read as a cohort.
    submission_cohort: Cohort

    @property
    def speculative_points(self) -> list[tuple[Path, PointConfig]]:
        """Points declaring speculative decoding."""
        return [(p, c) for p, c in self.points if c.speculative_decoding]

    @model_validator(mode="after")
    def _check_approved_drafter(self) -> DrafterBinding:
        """§9.1: a drafter used must be on its benchmark's published list."""
        for yaml_path, config in self.speculative_points:
            declared = config.speculative_decoding or {}
            if not self.approved:
                self._check_results.append(
                    err(
                        "approved-drafter",
                        "fail",
                        yaml_path,
                        concurrency=config.concurrency,
                        benchmark=self.benchmark,
                    )
                )
                continue
            if any(entry.matches(declared) for entry in self.approved):
                self._check_results.append(
                    ok("approved-drafter", "pass", yaml_path, concurrency=config.concurrency)
                )
            else:
                self._check_results.append(
                    err(
                        "approved-drafter",
                        "fail-2",
                        yaml_path,
                        concurrency=config.concurrency,
                        benchmark=self.benchmark,
                    )
                )
        return self

    @model_validator(mode="after")
    def _check_approval_lead_time(self) -> DrafterBinding:
        """§2.9.4: the submission's cohort is at least two cohorts after the drafter's approval.

        §9.1 still words this against a declared ``target_cohort``; under the rolling
        model the anchor is the submission's own cohort, so a submitter cannot name an
        earlier one to use a drafter before it is eligible.
        """
        for yaml_path, config in self.speculative_points:
            declared = config.speculative_decoding or {}
            matched = next((e for e in self.approved if e.matches(declared)), None)
            if matched is None:
                continue  # already reported by approved-drafter
            target = self.submission_cohort
            earliest = matched.earliest_target_cohort()
            if earliest is None:
                self._check_results.append(
                    warn(
                        "drafter-approval-lead-time",
                        "warn",
                        yaml_path,
                        concurrency=config.concurrency,
                        approved_cohort=matched.approved_cohort,
                    )
                )
                continue
            if target < earliest:
                self._check_results.append(
                    err(
                        "drafter-approval-lead-time",
                        "fail",
                        yaml_path,
                        concurrency=config.concurrency,
                        approved_cohort=matched.approved_cohort,
                        earliest=earliest,
                        drafter_approval_lead_cohorts=DRAFTER_APPROVAL_LEAD_COHORTS,
                        target=target,
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "drafter-approval-lead-time",
                        "pass",
                        yaml_path,
                        concurrency=config.concurrency,
                        target=target,
                        earliest=earliest,
                    )
                )
        return self
