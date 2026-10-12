"""Seed binding — §4.6 seed rotation, checked across one Pareto curve.

§9.1's "Seed-set validity" row makes this a property of the whole submission, not of
one file: every point must record the same seed set, and that set must be adoptable
when the submission first appears (Submission Rules §4.6). §9.1 still words the second
half against a declared ``target_cohort``, but under the rolling model (§4.1) a
submission is timestamped on receipt and targets nothing, so the window is tested
against the first cohort date on or after receipt — see :attr:`submission_cohort`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ...cohorts import Cohort
from ...seed_sets import SeedSet
from ..file.point_config import PointConfig
from ..results import CheckResult, err, ok, warn

__all__ = ["SeedBinding"]


class SeedBinding(BaseModel):
    """Validates one curve's seed declarations against the published seed sets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    #: ``(point.yaml path, parsed config)`` for every point whose config loaded.
    points: list[tuple[Path, PointConfig]]
    #: The published sets, keyed by id — see :func:`submission_checker.seed_sets.load_seed_sets`.
    registry: dict[str, SeedSet]
    #: The curve directory, used as the path for curve-wide findings.
    model_dir: Path
    #: The first cohort date on or after the submission's receipt. An amendment keeps
    #: its original submission's, so §4.6's binding outlives the adoption window.
    submission_cohort: Cohort

    @property
    def adoption_checkable(self) -> bool:
        """True once at least one published set names the cohorts it was published for."""
        return any(s.cohorts for s in self.registry.values())

    @model_validator(mode="after")
    def _check_seed_set_consistency(self) -> SeedBinding:
        """§9.1: every point of a submission must record the same seed set."""
        if not self.points:
            return self
        declared = {config.seed_set for _, config in self.points if config.seed_set is not None}
        if not declared:
            return self  # absence is reported by point-disclosure-complete
        if len(declared) > 1:
            self._check_results.append(
                err(
                    "seed-set-consistency",
                    "fail",
                    self.model_dir,
                    declared_count=len(declared),
                    declared=", ".join(sorted(declared)),
                )
            )
        else:
            self._check_results.append(
                ok("seed-set-consistency", "pass", self.model_dir, next=next(iter(declared)))
            )
        return self

    @model_validator(mode="after")
    def _check_seed_set_membership(self) -> SeedBinding:
        """§9.1: the bound set must be one MLCommons published."""
        for yaml_path, config in self.points:
            set_id = config.seed_set
            if set_id is None:
                continue
            if set_id in self.registry:
                self._check_results.append(
                    ok("seed-set-membership", "pass", yaml_path, set_id=set_id)
                )
            else:
                self._check_results.append(
                    err(
                        "seed-set-membership",
                        "fail",
                        yaml_path,
                        set_id=set_id,
                        value=", ".join(sorted(self.registry)) or "none",
                    )
                )
        return self

    @model_validator(mode="after")
    def _check_seed_runtime_match(self) -> SeedBinding:
        """§2.1.1: the client's RNGs must actually be seeded from the bound set."""
        for yaml_path, config in self.points:
            published = self.registry.get(config.seed_set or "")
            if published is None:
                continue  # unknown set: already reported by seed-set-membership
            runtime = config.runtime_settings.runtime
            mismatched = [
                f"{name}={getattr(runtime, name)!r} (expected {expected})"
                for name, expected in published.seeds.items()
                if getattr(runtime, name) != expected
            ]
            if mismatched:
                self._check_results.append(
                    err(
                        "seed-runtime-match",
                        "fail",
                        yaml_path,
                        published_id=published.id,
                        mismatched="; ".join(mismatched),
                    )
                )
            else:
                self._check_results.append(
                    ok("seed-runtime-match", "pass", yaml_path, published_id=published.id)
                )
            if runtime.uses_legacy_names:
                self._check_results.append(warn("seed-runtime-match", "warn", yaml_path))
        return self

    @model_validator(mode="after")
    def _check_seed_set_adoption(self) -> SeedBinding:
        """§4.6: the bound set must be adoptable at the submission's cohort.

        Reported as INFO while the published registry carries no cohort keys: with no
        cohort attached to any set, every submission would either pass vacuously or
        fail universally, and neither says anything true about the submission.
        """
        if not self.points:
            return self
        target = str(self.submission_cohort)
        if not self.adoption_checkable:
            self._check_results.append(
                ok("seed-set-adoption", "skipped-cohorts", self.model_dir, cohorts=target)
            )
            return self

        for yaml_path, config in self.points:
            published = self.registry.get(config.seed_set or "")
            if published is None:
                continue
            if target in published.cohorts:
                self._check_results.append(
                    ok(
                        "seed-set-adoption",
                        "pass",
                        yaml_path,
                        published_id=published.id,
                        target=target,
                    )
                )
            else:
                self._check_results.append(
                    err(
                        "seed-set-adoption",
                        "fail",
                        yaml_path,
                        published_id=published.id,
                        target=target,
                        value=", ".join(published.cohorts) or "no cohort",
                    )
                )
        return self
