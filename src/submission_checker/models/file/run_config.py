"""Run configuration model — ``results/point_<N>/config.yaml`` (§8.4).

Only the fields needed for compliance checks are parsed; all other fields are
accepted via ``extra="allow"`` so that new endpoint tool versions don't break
older checker versions.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["RunConfig"]

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationInfo, model_validator

from ..results import CheckResult, err, ok


class _WarmupConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    salt: bool = False


class _RunSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    warmup: _WarmupConfig = Field(default_factory=_WarmupConfig)


class RunConfig(BaseModel):
    """Parsed ``results/point_<N>/config.yaml`` — checked for warmup salt compliance."""

    model_config = ConfigDict(extra="allow")
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    settings: _RunSettings = Field(default_factory=_RunSettings)

    @model_validator(mode="after")
    def _check_warmup_salted(self, info: ValidationInfo) -> RunConfig:
        """§6.3: warmup prompts must be salted to prevent KV-cache priming of the perf run."""
        path: Path | None = (info.context or {}).get("config_path")
        warmup = self.settings.warmup
        if not warmup.enabled:
            return self
        if not warmup.salt:
            self._check_results.append(
                err(
                    "warmup-salt",
                    "Warmup is enabled but salt=false; unsalted prompts may prime the KV cache"
                    " before the performance phase",
                    path,
                    "#6.3",
                )
            )
        else:
            self._check_results.append(ok("warmup-salt", "Warmup salt enabled", path, "#6.3"))
        return self
