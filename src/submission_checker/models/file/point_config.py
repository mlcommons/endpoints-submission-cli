"""Point configuration model — §8.3 measurement point YAML schema and per-point checks."""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["PointConfig", "RuntimeSettings", "WarmupSpec"]

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    model_validator,
)

from ..regions import REGION_NAMES, SUBMITTERS_CHOICE
from ..results import CheckResult, err, ok, warn

# §8.3's own table still lists the v0.7 *_throughput names, but §5.5's reference
# algorithm and §9.1's check names both use *_concurrency. The algorithm wins.
_VALID_REGIONS = frozenset({*REGION_NAMES, SUBMITTERS_CHOICE})

#: Cohort identifier (§4.6): calendar year-month plus the cohort index within it.
_COHORT_RE = re.compile(r"\d{4}-\d{2}-C[01]")


class WarmupSpec(BaseModel):
    """Warmup procedure declaration required by §6.3.3.

    Attributes:
        duration_s: Total warmup duration in seconds.
        requests_issued: Number of requests sent during warmup.
        requests_completed: Number of requests that completed successfully.
        data_source: Description of the warmup data and its origin.
        concurrency: Concurrency level used during warmup.
        initialization_steps: Platform-specific setup steps completed before TEST_STARTED.
        logs_retained: Whether the warmup request logs are kept for reviewer inspection
            (§6.3.2). §9.1 flags rather than rejects when this is absent or false.
        link_logs: Where a reviewer can find the retained warmup logs.
    """

    model_config = ConfigDict(extra="allow")

    duration_s: float = Field(ge=0, le=86400)
    requests_issued: int = Field(ge=0)
    requests_completed: int = Field(ge=0)
    data_source: str = Field(min_length=1)
    concurrency: int = Field(gt=0)
    initialization_steps: list[str] = Field(default_factory=list)
    logs_retained: bool | None = None
    link_logs: str | None = None

    @model_validator(mode="after")
    def _check_completed_le_issued(self) -> WarmupSpec:
        if self.requests_completed > self.requests_issued:
            raise ValueError(
                f"requests_completed ({self.requests_completed})"
                f" > requests_issued ({self.requests_issued})"
            )
        return self


class RuntimeSettings(BaseModel):
    """``runtime_settings`` block from ``points/point_<N>.yaml`` (§8.3).

    Attributes:
        load_pattern: Load pattern type — must be ``"concurrency"`` for submissions (§6.1).
        min_duration_ms: Minimum steady-state duration in milliseconds (§6.2).
        min_sample_count: Minimum completed queries required (§6.4). ``None`` = no override.
        stream_all_chunks: Must be ``True`` for all performance runs to enable per-token timing
            (§6.5).
    """

    model_config = ConfigDict(extra="allow")

    load_pattern: str = "concurrency"
    min_duration_ms: int = 600_000
    min_sample_count: int | None = None
    stream_all_chunks: bool = True

    class Runtime(BaseModel):
        """RNG seed configuration, bound to a published seed set (§4.6).

        v1.0 renamed every seed and stopped fixing them at 42: `seedset.yaml` names
        them ``scheduler_rng_seed``, ``sample_index_rng_seed`` and ``model_seed``. The
        v0.7 spellings are still read as aliases so a v0.7 run folder parses and is
        told what changed, rather than failing with an opaque schema error.

        All four are optional at the schema level; whether the right ones are present
        and match the bound set is
        :class:`~submission_checker.models.aggregate.SeedBinding`'s job.
        """

        model_config = ConfigDict(extra="allow")

        scheduler_rng_seed: int | None = None
        sample_index_rng_seed: int | None = None
        model_seed: int | None = None

        # v0.7 spellings, deprecated. Read only as a fallback for the two that were
        # renamed rather than added; see _lift_legacy_seed_names.
        scheduler_random_seed: int | None = None
        dataloader_random_seed: int | None = None

        @model_validator(mode="after")
        def _lift_legacy_seed_names(self) -> RuntimeSettings.Runtime:
            """Fill the v1.0 seed names from their v0.7 predecessors when only those exist."""
            if self.scheduler_rng_seed is None and self.scheduler_random_seed is not None:
                object.__setattr__(self, "scheduler_rng_seed", self.scheduler_random_seed)
            if self.sample_index_rng_seed is None and self.dataloader_random_seed is not None:
                object.__setattr__(self, "sample_index_rng_seed", self.dataloader_random_seed)
            return self

        @property
        def uses_legacy_names(self) -> bool:
            """True when the point declares only the v0.7 seed spellings."""
            declares_legacy = (
                self.scheduler_random_seed is not None or self.dataloader_random_seed is not None
            )
            return declares_legacy and self.model_seed is None

    class WarmupLoadgen(BaseModel):
        """Loadgen warmup options."""

        model_config = ConfigDict(extra="allow")

        salt: bool | None = None

    runtime: Runtime
    warmup: WarmupLoadgen | None = None


#: §8.3 disclosure fields, required but landed as optional so a bundle missing one
#: still validates far enough to report every other defect. Absence is reported by
#: ``point-disclosure-complete`` rather than by a schema error, which would abort the
#: whole file and hide the rest.
_REQUIRED_DISCLOSURE_FIELDS = (
    "division",
    "max_supported_concurrency",
    "model_name",
    "model_precision",
    "link_to_model",
    "dataset_name",
    "dataset_type",
    "dataset_link",
)

#: Required by §8.1's tree and §9.1's "Shared path resolution" and "Seed-set validity"
#: rows, but absent from §8.3's own table — PR #90 deleted them from it while the prose
#: and the check table kept depending on them. Treated as required; raised with the WG.
_REQUIRED_UNDOCUMENTED_FIELDS = (
    "shared_src",
    "shared_docs",
    "seed_set",
    "target_cohort",
)


class PointConfig(BaseModel):
    """Parsed contents of a Pareto point's ``point.yaml`` (§8.3).

    Attributes:
        concurrency: Target concurrency level for this measurement point.
        region: Spec region this point is claimed to satisfy (optional submitter hint).
        dataset: Dataset name used for this measurement point.
        runtime_settings: Load pattern, duration, and client settings.
        warmup: §6.3.3 warmup declaration.
        division: Standardized, Serviced, or RDI.
        max_supported_concurrency: The curve's declared ``C_max``.
        model_name: Display name of the benchmark model.
        model_precision: Lowest precision numerical format used for the weights.
        link_to_model: Link to the submitted model.
        link_to_model_transformation: Link to the calibration/quantization write-up.
        model_notes: Freeform submitter notes about the model.
        dataset_name: Display name of the dataset.
        dataset_type: "Accuracy", "Performance", or "Accuracy + Performance".
        dataset_link: Link to the data used for the submission.
        shared_src: Root-relative path to the implementation directory under ``src/``.
        shared_docs: Root-relative path to the shared ``docs/`` directory.
        seed_set: Identifier of the bound seed set (§4.6).
        target_cohort: Cohort this submission targets, ``YYYY-MM-C0`` or ``…-C1``.
    """

    model_config = ConfigDict(extra="allow")
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    concurrency: int
    region: str | None = None
    dataset: str = ""
    runtime_settings: RuntimeSettings
    warmup: WarmupSpec | None = None

    # §8.3 disclosure
    division: str | None = None
    max_supported_concurrency: int | None = None
    model_name: str | None = None
    model_precision: str | None = None
    link_to_model: str | None = None
    link_to_model_transformation: str | None = None
    model_notes: str | None = None
    dataset_name: str | None = None
    dataset_type: str | None = None
    dataset_link: str | None = None

    # §8.1 / §9.1, absent from §8.3's table — see _REQUIRED_UNDOCUMENTED_FIELDS.
    shared_src: str | None = None
    shared_docs: str | None = None
    seed_set: str | None = None
    target_cohort: str | None = None

    @model_validator(mode="after")
    def _check_disclosure_complete(self, info: ValidationInfo) -> PointConfig:
        """§8.3 + §9.1 "Submission completeness": every disclosure field must be present."""
        path: Path | None = (info.context or {}).get("yaml_path")
        missing = [
            name
            for name in (*_REQUIRED_DISCLOSURE_FIELDS, *_REQUIRED_UNDOCUMENTED_FIELDS)
            if getattr(self, name) in (None, "")
        ]
        if missing:
            self._check_results.append(
                err(
                    "point-disclosure-complete",
                    f"point.yaml is missing required §8.3 field(s): {', '.join(missing)}",
                    path,
                    "#8.3",
                )
            )
        else:
            self._check_results.append(
                ok("point-disclosure-complete", "All §8.3 disclosure fields present", path, "#8.3")
            )
        return self

    @model_validator(mode="after")
    def _check_target_cohort(self, info: ValidationInfo) -> PointConfig:
        """§4.6: ``target_cohort`` names a cohort as ``YYYY-MM-C0`` or ``YYYY-MM-C1``."""
        path: Path | None = (info.context or {}).get("yaml_path")
        if self.target_cohort is None:
            return self  # absence is reported by point-disclosure-complete
        if _COHORT_RE.fullmatch(self.target_cohort):
            self._check_results.append(
                ok("target-cohort", f"target_cohort='{self.target_cohort}'", path, "#4.6")
            )
        else:
            self._check_results.append(
                err(
                    "target-cohort",
                    f"target_cohort '{self.target_cohort}' is not of the form YYYY-MM-C0/C1",
                    path,
                    "#4.6",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_seeds_legacy(self, info: ValidationInfo) -> PointConfig:
        """v0.7 fallback: seeds must equal 42 when the point binds no seed set.

        v0.7 fixed every seed at 42. v1.0 rotates them (§4.6), so the constant is now
        wrong for any point that names a ``seed_set`` — those are checked against the
        published registry by
        :class:`~submission_checker.models.aggregate.SeedBinding` instead. This rule
        survives only to give a v0.7 bundle a comprehensible message.
        """
        path: Path | None = (info.context or {}).get("yaml_path")
        if self.seed_set is not None:
            return self
        rt = self.runtime_settings.runtime
        for field_name, val in (
            ("runtime_settings.runtime.scheduler_rng_seed", rt.scheduler_rng_seed),
            ("runtime_settings.runtime.sample_index_rng_seed", rt.sample_index_rng_seed),
        ):
            if val != 42:
                self._check_results.append(
                    err(
                        "seed-config-legacy",
                        f"Point {self.concurrency}: {field_name} = {val!r}; v0.7 required 42"
                        " and v1.0 requires a declared seed_set (§4.6) — neither is satisfied",
                        path,
                        "#4.6",
                    )
                )
            else:
                self._check_results.append(
                    warn(
                        "seed-config-legacy",
                        f"Point {self.concurrency}: {field_name} = 42 with no seed_set declared;"
                        " v1.0 binds seeds to a published set (§4.6)",
                        path,
                        "#4.6",
                    )
                )
        return self

    @model_validator(mode="after")
    def _check_warmup(self, info: ValidationInfo) -> PointConfig:
        """§6.3.3: warmup declaration is required for every measurement point."""
        path: Path | None = (info.context or {}).get("yaml_path")
        if self.warmup is None:
            self._check_results.append(
                err(
                    "warmup-present",
                    f"Point {self.concurrency}: missing warmup declaration (§6.3.3)",
                    path,
                    "#6.3.3",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "warmup-present",
                    f"Point {self.concurrency}: warmup declaration present",
                    path,
                    "#6.3.3",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_warmup_logs_retained(self, info: ValidationInfo) -> PointConfig:
        """§6.3.2 / §9.1: warmup request logs must be retained for reviewer inspection.

        WARN rather than ERROR because §9.1's failure action for this row is "Flag
        non-compliant points", not "Reject submission" — the checker cannot see the
        submitter's log archive, so only the declaration is verifiable here.
        """
        path: Path | None = (info.context or {}).get("yaml_path")
        if self.warmup is None:
            return self  # absence is reported by warmup-present
        retained = self.warmup.logs_retained
        if retained is True:
            self._check_results.append(
                ok(
                    "warmup-logs-retained",
                    f"Point {self.concurrency}: warmup logs declared retained"
                    + (f" ({self.warmup.link_logs})" if self.warmup.link_logs else ""),
                    path,
                    "#6.3.2",
                )
            )
        elif retained is False:
            self._check_results.append(
                warn(
                    "warmup-logs-retained",
                    f"Point {self.concurrency}: warmup logs declared NOT retained;"
                    " §6.3.2 requires them to remain available for reviewer inspection",
                    path,
                    "#6.3.2",
                )
            )
        else:
            self._check_results.append(
                warn(
                    "warmup-logs-retained",
                    f"Point {self.concurrency}: warmup block does not declare logs_retained,"
                    " so §6.3.2 log retention cannot be confirmed",
                    path,
                    "#6.3.2",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_warmup_salt(self, info: ValidationInfo) -> PointConfig:
        """Warn when warmup_loadgen.warmup.salt is True."""
        path: Path | None = (info.context or {}).get("yaml_path")
        warmup_block = self.runtime_settings.warmup
        if warmup_block is not None and warmup_block.salt is True:
            self._check_results.append(
                warn(
                    "warmup-salt",
                    f"Point {self.concurrency}: warmup salt is enabled",
                    path,
                    "#6.3.3",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_load_pattern(self, info: ValidationInfo) -> PointConfig:
        """§10: load_pattern must be 'concurrency' with a positive concurrency level."""
        path: Path | None = (info.context or {}).get("yaml_path")
        lp = self.runtime_settings.load_pattern
        if lp != "concurrency":
            self._check_results.append(
                err(
                    "load-pattern",
                    f"Point {self.concurrency}: load_pattern '{lp}' ≠ 'concurrency'",
                    path,
                    "#10",
                )
            )
        elif self.concurrency <= 0:
            self._check_results.append(
                err(
                    "load-pattern",
                    f"concurrency must be positive, got {self.concurrency}",
                    path,
                    "#10",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "load-pattern",
                    f"Point {self.concurrency}: load pattern OK (concurrency)",
                    path,
                    "#10",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_streaming(self, info: ValidationInfo) -> PointConfig:
        """§6.5: stream_all_chunks must be True for all performance runs."""
        path: Path | None = (info.context or {}).get("yaml_path")
        if not self.runtime_settings.stream_all_chunks:
            self._check_results.append(
                err(
                    "streaming-config",
                    f"Point {self.concurrency}: stream_all_chunks must be True",
                    path,
                    "#6.5",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "streaming-config",
                    f"Point {self.concurrency}: stream_all_chunks=True",
                    path,
                    "#6.5",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_region_declared(self, info: ValidationInfo) -> PointConfig:
        """§8.3: the declared region must be a value the spec defines.

        Only the vocabulary is checked here. Whether the declared region matches the
        one computed for this concurrency is a property of the whole curve — the
        boundaries depend on ``C_min``, derived from every point — so that lives in
        :class:`~submission_checker.models.aggregate.RegionPlacement`.
        """
        path: Path | None = (info.context or {}).get("yaml_path")
        region = self.region
        if region is None:
            return self  # optional field — absence is not an error
        if region not in _VALID_REGIONS:
            self._check_results.append(
                err(
                    "region-declared",
                    f"Invalid region '{region}': must be one of {sorted(_VALID_REGIONS)}",
                    path,
                    "#8.3",
                )
            )
            return self
        self._check_results.append(ok("region-declared", f"region='{region}'", path, "#8.3"))
        return self
