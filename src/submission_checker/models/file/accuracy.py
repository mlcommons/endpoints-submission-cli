"""Accuracy result model — §4.3, §6.6 accuracy_result.json schema."""

from __future__ import annotations

from typing import Any

__all__ = ["AccuracyResult"]

from pydantic import RootModel, model_validator, PrivateAttr

from ..results import CheckResult, err


class AccuracyResult(RootModel[dict[str, dict[str, Any]]]):
    """Parsed ``accuracy/accuracy_result.json``.

    Format: one entry per evaluated dataset, keyed by dataset name:
        {
          "cnn_dailymail::llama3_8b": {
            "dataset_name": "cnn_dailymail::llama3_8b",
            "num_samples": 13368,
            "score": {"rouge1": "38.7287", "rouge2": "16.0968", ...},
            ...
          }
        }
    """

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    metric: str
    score: float | dict
    quality_target: float
    passed: bool

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

    def metric_scores(self) -> dict[str, dict[str, float]]:
        """Return ``{dataset_name: {metric: float}}`` for all datasets.

        The ``score`` field may be a dict of sub-metrics (rouge breakdown) or a
        scalar.  Numeric strings are coerced; non-numeric values are dropped.
        """
        result: dict[str, dict[str, float]] = {}
        for ds_name, entry in self.root.items():
            raw_score = entry.get("score")
            if isinstance(raw_score, dict):
                scores: dict[str, float] = {}
                for k, v in raw_score.items():
                    try:
                        scores[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                if scores:
                    result[ds_name] = scores
            elif raw_score is not None:
                try:
                    result[ds_name] = {"score": float(raw_score)}
                except (TypeError, ValueError):
                    pass
        return result
