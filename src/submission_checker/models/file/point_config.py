"""Point configuration model — §8.3 measurement point YAML schema and per-point checks."""

from __future__ import annotations

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

from ..regions import classify_concurrency
from ..results import CheckResult, err, ok, warn

_VALID_REGIONS = frozenset(
    {"low_latency", "low_throughput", "med_throughput", "high_throughput", "submitters_choice"}
)


class WarmupSpec(BaseModel):
    """Warmup procedure declaration required by §6.3.3.

    Attributes:
        duration_s: Total warmup duration in seconds.
        requests_issued: Number of requests sent during warmup.
        requests_completed: Number of requests that completed successfully.
        data_source: Description of the warmup data and its origin.
        concurrency: Concurrency level used during warmup.
        initialization_steps: Platform-specific setup steps completed before TEST_STARTED.
    """

    model_config = ConfigDict(extra="allow")

    duration_s: float = Field(ge=0, le=86400)
    requests_issued: int = Field(ge=0)
    requests_completed: int = Field(ge=0)
    data_source: str = Field(min_length=1)
    concurrency: int = Field(gt=0)
    initialization_steps: list[str] = Field(default_factory=list)

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
        """Random seed configuration for the scheduler and dataloader."""

        model_config = ConfigDict(extra="allow")

        scheduler_random_seed: int
        dataloader_random_seed: int

    class WarmupLoadgen(BaseModel):
        """Loadgen warmup options."""

        model_config = ConfigDict(extra="allow")

        salt: bool | None = None

    runtime: Runtime
    warmup: WarmupLoadgen | None = None


class PointConfig(BaseModel):
    """Parsed contents of ``points/point_<N>.yaml`` (§8.3).

    Attributes:
        concurrency: Target concurrency level for this measurement point.
        region: Spec region this point is claimed to satisfy (optional submitter hint).
        dataset: Dataset name used for this measurement point.
        runtime_settings: Load pattern, duration, and client settings.
    """

    model_config = ConfigDict(extra="allow")
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    concurrency: int
    region: str | None = None
    dataset: str = ""
    runtime_settings: RuntimeSettings
    warmup: WarmupSpec | None = None

    @model_validator(mode="after")
    def _check_seeds(self, info: ValidationInfo) -> PointConfig:
        """Random seeds in runtime_settings.runtime must equal 42."""
        path: Path | None = (info.context or {}).get("yaml_path")
        rt = self.runtime_settings.runtime
        for field_name, val in (
            ("runtime_settings.runtime.scheduler_random_seed", rt.scheduler_random_seed),
            ("runtime_settings.runtime.dataloader_random_seed", rt.dataloader_random_seed),
        ):
            if val != 42:
                self._check_results.append(
                    err(
                        "seed-config",
                        f"Point {self.concurrency}: {field_name} = {val!r}, expected 42",
                        path,
                        "#8.1",
                    )
                )
            else:
                self._check_results.append(
                    ok("seed-config", f"Point {self.concurrency}: {field_name} = 42", path, "#8.1")
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
    def _check_concurrency_range(self, info: ValidationInfo) -> PointConfig:
        """§9: concurrency must not exceed the high_throughput upper bound (incl. 10% margin)."""
        path: Path | None = (info.context or {}).get("yaml_path")
        regions = (info.context or {}).get("regions")
        if regions is None:
            return self
        actual_region = classify_concurrency(self.concurrency, regions)
        if actual_region is None:
            self._check_results.append(
                err(
                    "concurrency-in-range",
                    f"Concurrency {self.concurrency} exceeds max valid range"
                    f" (max including 10% margin: {regions.high_throughput.end})",
                    path,
                    "#9",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "concurrency-in-range",
                    f"Concurrency {self.concurrency} valid ({actual_region})",
                    path,
                    "#9",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_region_declared(self, info: ValidationInfo) -> PointConfig:
        """§8.3: region must be a valid value; if declared, must match the computed region."""
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
        regions = (info.context or {}).get("regions")
        if regions is not None and region != "submitters_choice":
            computed = classify_concurrency(self.concurrency, regions)
            if computed is not None and computed != region:
                self._check_results.append(
                    warn(
                        "region-declared",
                        f"Declared region '{region}' ≠ computed region '{computed}'"
                        f" for concurrency {self.concurrency}",
                        path,
                        "#8.3",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "region-declared",
                        f"Declared region '{region}' consistent with"
                        f" concurrency {self.concurrency}",
                        path,
                        "#8.3",
                    )
                )
        else:
            self._check_results.append(ok("region-declared", f"region='{region}'", path, "#8.3"))
        return self
