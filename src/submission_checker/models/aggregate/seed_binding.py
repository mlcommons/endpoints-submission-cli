"""Seed binding — §4.6 seed rotation, checked across one Pareto curve.

§9.1's "Seed-set validity" row makes this a property of the whole submission, not of
one file: "every point must record the same seed set, and that set must have been
published for ``target_cohort`` or one of the three immediately preceding cohorts."
The first half is checkable today; the second is not, because the published registry
carries no cohort keys yet — see :attr:`SeedBinding.adoption_checkable`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ...seed_sets import SeedSet
from ..file.point_config import PointConfig
from ..results import CheckResult, err, ok, warn

__all__ = ["SeedBinding"]

_SPEC_REF = "#4.6"


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
                    f"Points declare {len(declared)} different seed sets:"
                    f" {', '.join(sorted(declared))} — a submission binds exactly one",
                    self.model_dir,
                    _SPEC_REF,
                )
            )
        else:
            self._check_results.append(
                ok(
                    "seed-set-consistency",
                    f"All points bind seed set {next(iter(declared))!r}",
                    self.model_dir,
                    _SPEC_REF,
                )
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
                    ok(
                        "seed-set-membership",
                        f"seed_set {set_id!r} is a published set",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
            else:
                self._check_results.append(
                    err(
                        "seed-set-membership",
                        f"seed_set {set_id!r} is not a published set"
                        f" (known: {', '.join(sorted(self.registry)) or 'none'})",
                        yaml_path,
                        _SPEC_REF,
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
                        f"runtime_settings.runtime does not match seed set {published.id!r}:"
                        f" {'; '.join(mismatched)}",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "seed-runtime-match",
                        f"RNG seeds match seed set {published.id!r}",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
            if runtime.uses_legacy_names:
                self._check_results.append(
                    warn(
                        "seed-runtime-match",
                        "runtime_settings.runtime uses the v0.7 seed names"
                        " (scheduler_random_seed / dataloader_random_seed); v1.0 names them"
                        " scheduler_rng_seed / sample_index_rng_seed and adds model_seed",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
        return self

    @model_validator(mode="after")
    def _check_seed_set_adoption(self) -> SeedBinding:
        """§4.6: the bound set must be published for ``target_cohort`` or the three before it.

        Reported as INFO while the published registry carries no cohort keys: with no
        cohort attached to any set, every submission would either pass vacuously or
        fail universally, and neither says anything true about the submission.
        """
        if not self.points:
            return self
        cohorts = {
            config.target_cohort for _, config in self.points if config.target_cohort is not None
        }
        if not self.adoption_checkable:
            self._check_results.append(
                ok(
                    "seed-set-adoption",
                    "SKIPPED: the published seed-set registry carries no cohort keys, so §4.6's"
                    " four-cohort adoption window cannot be evaluated"
                    + (f" (points target {', '.join(sorted(cohorts))})" if cohorts else ""),
                    self.model_dir,
                    _SPEC_REF,
                )
            )
            return self

        for yaml_path, config in self.points:
            published = self.registry.get(config.seed_set or "")
            target = config.target_cohort
            if published is None or target is None:
                continue
            if target in published.cohorts:
                self._check_results.append(
                    ok(
                        "seed-set-adoption",
                        f"Seed set {published.id!r} was published for cohort {target}",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
            else:
                self._check_results.append(
                    err(
                        "seed-set-adoption",
                        f"Seed set {published.id!r} was not published for cohort {target}"
                        f" (published for: {', '.join(published.cohorts) or 'no cohort'})",
                        yaml_path,
                        _SPEC_REF,
                    )
                )
        return self
