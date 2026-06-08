"""Accuracy result model — §4.3, §6.6 accuracy_result.json schema."""

from __future__ import annotations

from pathlib import Path

__all__ = ["AccuracyResult"]

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationInfo, field_validator, model_validator

from ..results import CheckResult, err, ok


class AccuracyResult(BaseModel):
    """Parsed contents of ``accuracy/accuracy_result.json`` (§4.3, §6.6).

    Attributes:
        metric: Name of the accuracy metric (e.g., ``rouge1``).
        score: Achieved score on the accuracy dataset.
        quality_target: Minimum acceptable score defined by the benchmark.
        passed: Whether the score meets the quality target.
    """

    model_config = ConfigDict(extra="allow")
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    metric: str
    score: float | dict
    quality_target: float
    passed: bool

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score_strings(cls, v: object) -> object:
        if isinstance(v, dict):
            coerced = {}
            for k, val in v.items():
                try:
                    coerced[k] = float(val)
                except (TypeError, ValueError):
                    coerced[k] = val
            return coerced
        return v

    @model_validator(mode="after")
    def _check_score_consistency(self, info: ValidationInfo) -> AccuracyResult:
        """§15: passed flag must be consistent with score >= quality_target."""
        path: Path | None = (info.context or {}).get("json_path")
        if isinstance(self.score, dict):
            score_repr = ", ".join(f"{k}={v}" for k, v in self.score.items())
            meets = all(v >= self.quality_target for v in self.score.values())
            if self.passed != meets:
                self._check_results.append(
                    err(
                        "accuracy-consistency",
                        f"passed={self.passed} but {self.metric} = {score_repr}"
                        f" {'all ≥' if meets else 'some <'} quality_target {self.quality_target:.4f}",
                        path,
                        "#15",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "accuracy-consistency",
                        f"{self.metric} = {score_repr}, quality_target = {self.quality_target:.4f},"
                        f" passed = {self.passed}",
                        path,
                        "#15",
                    )
                )
        else:
            meets = self.score >= self.quality_target
            if self.passed != meets:
                self._check_results.append(
                    err(
                        "accuracy-consistency",
                        f"passed={self.passed} but {self.metric} score {self.score:.4f}"
                        f" {'≥' if meets else '<'} quality_target {self.quality_target:.4f}",
                        path,
                        "#15",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "accuracy-consistency",
                        f"{self.metric} = {self.score:.4f}, quality_target = {self.quality_target:.4f},"
                        f" passed = {self.passed}",
                        path,
                        "#15",
                    )
                )
        return self
