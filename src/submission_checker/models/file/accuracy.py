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

    @model_validator(mode="after")
    def _check_not_empty(self) -> AccuracyResult:
        if not self.root:
            self._check_results.append(
                err("accuracy-valid", "accuracy_result.json is empty", None, "#4.3")
            )
        return self

    def metric_scores(self) -> dict[str, dict[str, float]]:
        """Return ``{dataset_name: {metric: float}}`` for all datasets."""
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
