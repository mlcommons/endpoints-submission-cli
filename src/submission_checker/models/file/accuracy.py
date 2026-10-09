"""Accuracy result model — §4.3, §6.6 accuracy_result.json schema."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import PrivateAttr, RootModel, model_validator

from ...messages import Invalid
from ..results import CheckResult, err

__all__ = ["PERFORMANCE_DATASET_TYPE", "AccuracyResult"]

_log = logging.getLogger(__name__)

#: ``dataset_type`` of the entry the client scores inline on the performance run
#: (``accuracy_config`` on the performance dataset). The client also names that entry
#: ``performance``, whatever the performance dataset is called in ``config.yaml``.
PERFORMANCE_DATASET_TYPE = "performance"


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
                raise Invalid("accuracy-valid", "scores-not-collection")
        if not isinstance(data, list):
            return data
        indexed: dict[str, dict[str, Any]] = {}
        for entry in data:
            if not isinstance(entry, dict):
                raise Invalid("accuracy-valid", "entry-not-mapping")
            name = entry.get("dataset_name")
            if not isinstance(name, str) or not name.strip():
                raise Invalid("accuracy-valid", "entry-unnamed")
            if name in indexed:
                raise Invalid("accuracy-valid", "dataset-duplicate", name=name)
            if "score" not in entry:
                raise Invalid("accuracy-valid", "score-missing", name=name)
            normalized = dict(entry)
            # Native scorers name these fields differently. Keep the originals
            # and expose aliases used by the existing sample-count/weight gates.
            for native, canonical in (
                ("unit_samples", "num_samples"),
                ("num_repeats", "n_repeats"),
            ):
                if native in entry:
                    if canonical in entry and entry[canonical] != entry[native]:
                        raise Invalid(
                            "accuracy-valid",
                            "alias-conflict",
                            name=name,
                            native=native,
                            canonical=canonical,
                        )
                    normalized[canonical] = entry[native]
            indexed[name] = normalized
        return indexed

    @model_validator(mode="after")
    def _check_not_empty(self) -> AccuracyResult:
        if not self.root:
            self._check_results.append(err("accuracy-valid", "fail", None))
        return self

    @model_validator(mode="after")
    def _check_one_performance_entry(self) -> AccuracyResult:
        """The client scores the performance run once, so it writes one such entry."""
        typed = [
            name
            for name, entry in self.root.items()
            if entry.get("dataset_type") == PERFORMANCE_DATASET_TYPE
        ]
        if len(typed) > 1:
            raise Invalid("accuracy-valid", "performance-duplicate", names=", ".join(typed))
        return self

    def performance_dataset(self) -> str | None:
        """Name of the entry scored inline on the performance run, or ``None``.

        Found by ``dataset_type``, as the client asks of consumers: an accuracy dataset
        may itself be named ``performance``. Clients that predate ``dataset_type``
        wrote the name alone, so an untyped ``performance`` entry still counts.
        """
        for name, entry in self.root.items():
            if entry.get("dataset_type") == PERFORMANCE_DATASET_TYPE:
                return name
        legacy = self.root.get(PERFORMANCE_DATASET_TYPE)
        if legacy is not None and "dataset_type" not in legacy:
            return PERFORMANCE_DATASET_TYPE
        return None

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
