"""Model context — aggregate model-level validation (point count, coverage, consistency).

Handles accuracy and overall compliance checks.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["ModelContext"]

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ...accuracy_targets import get_thresholds
from ..file.accuracy import AccuracyResult
from ..file.point_config import PointConfig
from ..file.point_summary import PointSummary
from ..file.system import SystemDescription
from ..regions import Regions
from ..results import CheckResult, err, ok, warn

_MIN_POINTS = 7
_MAX_POINTS = 32


class ModelContext(BaseModel):
    """Aggregated data for one benchmark-model directory — carries all model-level validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    system_id: str
    system_desc: SystemDescription
    model_dir: Path
    regions: Regions
    points_dir: Path
    accuracy_dir: Path | None = None
    all_point_count: int
    valid_points: list[tuple[Path, PointConfig]]
    loaded_points: list[tuple[PointConfig, PointSummary]]
    accuracy_result: AccuracyResult | None = None

    @model_validator(mode="after")
    def _check_point_count(self) -> ModelContext:
        """§2, §8: submission must have 7–32 measurement points."""
        n = self.all_point_count
        if n < _MIN_POINTS:
            self._check_results.append(
                err(
                    "point-count",
                    f"Only {n} measurement point(s) — minimum {_MIN_POINTS} required",
                    self.points_dir,
                    "#2, #8",
                )
            )
        else:
            self._check_results.append(
                ok("point-count", f"Point count OK: {n}", self.points_dir, "#2, #8")
            )
        if n > _MAX_POINTS:
            self._check_results.append(
                err(
                    "point-cap",
                    f"{n} points exceed the {_MAX_POINTS}-point cap",
                    self.points_dir,
                    "#2, #8",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_regional_coverage(self) -> ModelContext:
        """§3–6: at least one valid point must fall in each of the four concurrency regions."""
        concurrencies = [config.concurrency for _, config in self.valid_points]
        r = self.regions
        coverage_checks = [
            ("low-latency-coverage", "Low Latency", r.low_latency),
            ("low-throughput-coverage", "Low Throughput", r.low_throughput),
            ("med-throughput-coverage", "Medium Throughput", r.med_throughput),
            ("high-throughput-coverage", "High Throughput", r.high_throughput),
        ]
        for rule, label, bounds in coverage_checks:
            matching = [c for c in concurrencies if bounds.contains(c)]
            if matching:
                self._check_results.append(
                    ok(
                        rule,
                        f"{label} region covered: {sorted(matching)} (range {bounds})",
                        self.points_dir,
                        "#3–6",
                    )
                )
            else:
                self._check_results.append(
                    err(
                        rule,
                        f"No point in {label} region (concurrency {bounds})",
                        self.points_dir,
                        "#3–6",
                    )
                )
        return self

    @model_validator(mode="after")
    def _check_model_name_consistency(self) -> ModelContext:
        """§16: model name in system_desc must match the model directory name.

        The model directory name is derived from config.yaml's model_params.name
        (last path component, slugified). system_desc.model_id is the authoritative
        source; system_desc.model_name is the fallback. Both may be in HuggingFace
        format (e.g. "meta-llama/Llama-3.1-8B-Instruct") so we take the last "/"
        component before comparing.

        - system_desc has no model id/name  → warning (submitter hasn't filled it in)
        - system_desc model normalizes to a different name than the directory → error
        - they match → ok
        """
        # Strips HuggingFace org prefix, lowercases, and replaces non-word chars with
        # underscores so "meta-llama/Llama-3.1-8B" compares equal to "Llama-3.1-8B".
        def _normalize(name: str) -> str:
            part = name.split("/")[-1].strip()
            slug = re.sub(r"[^\w\-]", "_", part)
            slug = re.sub(r"_+", "_", slug).strip("_")
            return slug[:64]

        sd_raw = (self.system_desc.model_id or self.system_desc.model_name or "").strip()

        if not sd_raw:
            self._check_results.append(
                warn(
                    "model-name-consistency",
                    f"system_desc has no model_id or model_name; "
                    f"model directory is '{self.model_dir.name}'",
                    self.model_dir,
                    "#16",
                )
            )
        else:
            sd_normalized = _normalize(sd_raw)
            dir_name = self.model_dir.name
            if sd_normalized != dir_name:
                self._check_results.append(
                    err(
                        "model-name-consistency",
                        f"system_desc model '{sd_raw}' (normalized: '{sd_normalized}')"
                        f" does not match model directory '{dir_name}'",
                        self.model_dir,
                        "#16",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "model-name-consistency",
                        f"Model name consistent: {dir_name}",
                        self.model_dir,
                        "#16",
                    )
                )
        return self

    @model_validator(mode="after")
    def _check_config_consistency(self) -> ModelContext:
        """§16: all points must use the same dataset."""
        if not self.loaded_points:
            return self
        datasets = {config.dataset for config, _ in self.loaded_points}
        if len(datasets) > 1:
            self._check_results.append(
                err(
                    "config-consistency-dataset",
                    f"Inconsistent datasets across points: {datasets}",
                    self.model_dir,
                    "#16",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "config-consistency-dataset",
                    f"Dataset consistent: {next(iter(datasets))}",
                    self.model_dir,
                    "#16",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_accuracy(self) -> ModelContext:
        """§15: every accuracy metric must meet its quality threshold."""
        if self.accuracy_result is None:
            return self  # file missing/invalid already reported by checker.py

        json_path = (self.accuracy_dir / "results.json") if self.accuracy_dir else (self.model_dir / "results.json")
        target = get_thresholds(self.model_dir.name)

        if target is None:
            self._check_results.append(
                warn(
                    "accuracy-gate",
                    f"No accuracy thresholds defined for model '{self.model_dir.name}'"
                    " — skipping gate check",
                    json_path,
                    "#15",
                )
            )
            return self

        thresholds, min_queries = target

        # Check per-dataset sample counts
        for ds_name, entry in self.accuracy_result.root.items():
            num_samples = entry.get("num_samples")
            if num_samples is None:
                continue
            try:
                n = int(num_samples)
            except (TypeError, ValueError):
                continue
            if n < min_queries:
                self._check_results.append(
                    err(
                        "accuracy-sample-count",
                        f"{ds_name}: {n} samples < required {min_queries}",
                        json_path,
                        "#15",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "accuracy-sample-count",
                        f"{ds_name}: {n} samples ≥ required {min_queries}",
                        json_path,
                        "#15",
                    )
                )

        all_scores = self.accuracy_result.metric_scores()  # {ds: {metric: float}}
        # Flatten all per-dataset metric scores into one dict for threshold matching
        flat_scores: dict[str, float] = {}
        for ds_scores in all_scores.values():
            flat_scores.update(ds_scores)

        for threshold_key, (lower, upper) in thresholds.items():
            # Match score key case-insensitively
            score: float | None = None
            matched_key: str = threshold_key
            for k, v in flat_scores.items():
                if k.lower() == threshold_key:
                    score = v
                    matched_key = k
                    break
            if score is None:
                continue  # metric not present in this run's results

            if score < lower:
                self._check_results.append(
                    err(
                        "accuracy-gate",
                        f"{matched_key} = {score:.4f} < min {lower:.4f}",
                        json_path,
                        "#15",
                    )
                )
            elif upper is not None and score > upper:
                self._check_results.append(
                    err(
                        "accuracy-gate",
                        f"{matched_key} = {score:.4f} > max {upper:.4f}",
                        json_path,
                        "#15",
                    )
                )
            else:
                bound = f"[{lower:.4f}, {upper:.4f}]" if upper is not None else f"≥ {lower:.4f}"
                self._check_results.append(
                    ok(
                        "accuracy-gate",
                        f"{matched_key} = {score:.4f} PASSED (target {bound})",
                        json_path,
                        "#15",
                    )
                )
        return self
