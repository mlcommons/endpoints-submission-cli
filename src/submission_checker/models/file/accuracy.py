"""Accuracy result model — §4.3, §6.6 accuracy_result.json schema."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import PrivateAttr, RootModel, model_validator

from ..results import CheckResult, err

__all__ = ["AccuracyResult"]

_log = logging.getLogger(__name__)


class AccuracyResult(RootModel[dict[str, dict[str, Any]]]):
    """Parsed ``accuracy/accuracy_result.json``.

    Format: one entry per evaluated dataset, keyed by dataset name::

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

    @model_validator(mode="before")
    @classmethod
    def _read_native_report(cls, data: Any) -> Any:
        """Accept native reports and index their entries without changing the file."""
        if isinstance(data, dict) and "accuracy_scores" in data:
            data = data["accuracy_scores"]
            if not isinstance(data, (dict, list)):
                raise ValueError("accuracy_scores must be a dataset mapping or list")
        if not isinstance(data, list):
            return data
        indexed: dict[str, dict[str, Any]] = {}
        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError("Each accuracy_scores entry must be a dictionary")
            name = entry.get("dataset_name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each accuracy_scores entry must have a non-empty dataset_name")
            if name in indexed:
                raise ValueError(f"Duplicate accuracy dataset_name: {name!r}")
            if "score" not in entry:
                raise ValueError(f"Native accuracy entry {name!r} is missing score")
            normalized = dict(entry)
            # Native scorers name these fields differently. Keep the originals
            # and expose aliases used by the existing sample-count/weight gates.
            for native, canonical in (
                ("unit_samples", "num_samples"),
                ("num_repeats", "n_repeats"),
            ):
                if native in entry:
                    if canonical in entry and entry[canonical] != entry[native]:
                        raise ValueError(f"Conflicting {native} and {canonical} for {name!r}")
                    normalized[canonical] = entry[native]
            indexed[name] = normalized
        return indexed

    @model_validator(mode="after")
    def _check_not_empty(self) -> AccuracyResult:
        if not self.root:
            self._check_results.append(
                err("accuracy-valid", "accuracy_result.json is empty", None, "#4.3")
            )
        return self

    #: Per-entry bookkeeping keys that are not accuracy metrics.
    _META_KEYS = frozenset(
        {
            "dataset_name",
            "num_samples",
            "status",
            "extractor",
            "ground_truth_column",
            "n_repeats",
            "complete",
            "unit_samples",
            "total_samples",
            "num_repeats",
            "duration_s",
            "dataset_type",
            "response_counts",
        }
    )

    def metric_scores(self) -> dict[str, dict[str, float]]:
        """Return ``{dataset_name: {metric: float}}`` for all datasets.

        Supports three on-disk shapes per dataset entry:
          * ``"score": {metric: value, ...}``  — named metrics under ``score``
          * ``"score": value``                 — a single unnamed scalar (kept as ``score``)
          * no ``score`` key                    — metrics sit directly on the entry
            (e.g. ``{"exact_match": 81.3, "tokens_per_sample": 3886, "num_samples": ...}``),
            so every non-bookkeeping numeric key is treated as a named metric.
        """
        result: dict[str, dict[str, float]] = {}
        for ds_name, entry in self.root.items():
            raw_score = entry.get("score")
            scores: dict[str, float] = {}
            if isinstance(raw_score, dict):
                items = raw_score.items()
            elif raw_score is not None:
                items = {"score": raw_score}.items()
            else:
                # No `score` key — metrics are direct entry keys.
                items = {k: v for k, v in entry.items() if k not in self._META_KEYS}.items()
            for k, v in items:
                try:
                    scores[k] = float(v)
                except (TypeError, ValueError):
                    _log.warning(
                        "accuracy_result.json: cannot convert %r=%r to float; skipping", k, v
                    )
            if scores:
                result[ds_name] = scores
        return result
