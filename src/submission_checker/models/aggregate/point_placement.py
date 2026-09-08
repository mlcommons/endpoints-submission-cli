"""Where a point sits in the concurrency space — §9.1 "Concurrency in range".

These rules used to live on :class:`~submission_checker.models.file.PointConfig`,
but v1.0 derives ``C_min`` from the submission's points (§5.4), so the region
boundaries are a property of the whole curve rather than of any single file. That
makes this an aggregate check, which is what this package is for.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ..file.point_config import PointConfig
from ..regions import SUBMITTERS_CHOICE, Regions, classify_concurrency, covered_region
from ..results import CheckResult, err, ok, warn

__all__ = ["RegionPlacement"]


class RegionPlacement(BaseModel):
    """Validates one point's concurrency against its curve's region boundaries."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    config: PointConfig
    regions: Regions
    yaml_path: Path

    @model_validator(mode="after")
    def _check_concurrency_range(self) -> RegionPlacement:
        """§9.1: the concurrency must fall inside a valid region, margin included."""
        concurrency = self.config.concurrency
        region = classify_concurrency(concurrency, self.regions)
        if region is None:
            self._check_results.append(
                err(
                    "concurrency-in-range",
                    f"Concurrency {concurrency} exceeds the maximum valid range "
                    f"(including the 10% margin: {self.regions.margin.end})",
                    self.yaml_path,
                    "#9",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "concurrency-in-range",
                    f"Concurrency {concurrency} valid ({region})",
                    self.yaml_path,
                    "#9",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_region_placement(self) -> RegionPlacement:
        """§8.3: a declared region should match the one computed for its concurrency."""
        declared = self.config.region
        if declared is None or declared == SUBMITTERS_CHOICE:
            return self
        computed = classify_concurrency(self.config.concurrency, self.regions)
        if computed is None:
            return self  # already reported by concurrency-in-range
        if computed != declared:
            self._check_results.append(
                warn(
                    "region-placement",
                    f"Declared region '{declared}' ≠ computed region '{computed}' for "
                    f"concurrency {self.config.concurrency}",
                    self.yaml_path,
                    "#8.3",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "region-placement",
                    f"Declared region '{declared}' matches concurrency {self.config.concurrency}",
                    self.yaml_path,
                    "#8.3",
                )
            )
        return self

    @property
    def covered_region(self) -> str | None:
        """The region this point counts towards for §9.1 coverage, if any."""
        return covered_region(self.config.concurrency, self.regions)
