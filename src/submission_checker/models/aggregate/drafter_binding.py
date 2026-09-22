# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding drafters (§2.9.4) — approval and lead time.

§9.1 adds two checks, both per point but both needing the curve's ``target_cohort``
and benchmark, so they live here rather than on :class:`PointConfig`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ...cohorts import Cohort
from ...drafters import DRAFTER_APPROVAL_LEAD_COHORTS, ApprovedDrafter
from ..file.point_config import PointConfig
from ..results import CheckResult, err, ok, warn

__all__ = ["DrafterBinding"]

_SPEC_REF = "#2.9.4"


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
                        f"Point {config.concurrency} uses speculative decoding, but no drafter"
                        f" is approved for {self.benchmark!r}. §2.9.4 disallows speculative"
                        " decoding entirely for a benchmark with no approved drafter"
                        " (--approved-drafters points at a newer published list)",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
                continue
            if any(entry.matches(declared) for entry in self.approved):
                self._check_results.append(
                    ok(
                        "approved-drafter",
                        f"Point {config.concurrency}: drafter matches an approved entry",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
            else:
                self._check_results.append(
                    err(
                        "approved-drafter",
                        f"Point {config.concurrency}: declared drafter matches no approved entry"
                        f" for {self.benchmark!r}. §2.9.4 identifies an entry by weight"
                        " checksum, or by target checksum plus configuration",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
        return self

    @model_validator(mode="after")
    def _check_approval_lead_time(self) -> DrafterBinding:
        """§9.1: the drafter was approved at least two cohorts before `target_cohort`."""
        for yaml_path, config in self.speculative_points:
            declared = config.speculative_decoding or {}
            matched = next((e for e in self.approved if e.matches(declared)), None)
            if matched is None:
                continue  # already reported by approved-drafter
            target = Cohort.parse(config.target_cohort or "")
            earliest = matched.earliest_target_cohort()
            if target is None or earliest is None:
                self._check_results.append(
                    warn(
                        "drafter-approval-lead-time",
                        f"Point {config.concurrency}: lead time unevaluable —"
                        f" target_cohort={config.target_cohort!r},"
                        f" approved_cohort={matched.approved_cohort!r}",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
                continue
            if target < earliest:
                self._check_results.append(
                    err(
                        "drafter-approval-lead-time",
                        f"Point {config.concurrency}: drafter approved in"
                        f" {matched.approved_cohort}, so the earliest usable target_cohort is"
                        f" {earliest} ({DRAFTER_APPROVAL_LEAD_COHORTS} cohorts later);"
                        f" this submission targets {target}",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "drafter-approval-lead-time",
                        f"Point {config.concurrency}: target_cohort {target} ≥ earliest"
                        f" usable {earliest}",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
        return self
