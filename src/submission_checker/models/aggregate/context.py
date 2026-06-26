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


def _weighted_mean(pairs: list[tuple[float, float | None]]) -> float:
    """Sample-weighted mean of ``(value, weight)`` pairs.

    Falls back to a plain mean when any weight is missing or the weights sum to zero.
    """
    weights = [w for _, w in pairs]
    if all(w is not None for w in weights):
        total = sum(w for w in weights if w is not None)
        if total > 0:
            return sum(v * w for v, w in pairs if w is not None) / total
    return sum(v for v, _ in pairs) / len(pairs)


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

        json_path = (
            (self.accuracy_dir / "results.json")
            if self.accuracy_dir
            else (self.model_dir / "results.json")
        )
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
        root = self.accuracy_result.root

        # MLPerf inference gates accuracy as a single aggregate over the whole dataset:
        # one total sample count and one sample-weighted score per metric. Endpoints
        # results may instead report per-subset entries, so aggregate them the same way.

        # Total = issued sample count = Σ(num_samples × n_repeats) across subset entries.
        # MLPerf's accuracy-sample-count is the *issued* total (datasets are run with
        # repeats, e.g. gpt-oss aime×8/gpqa×5/lcb×3), so compare in issued units, not
        # unique. n_repeats defaults to 1 (e.g. deepseek), where issued == unique.
        sample_counts: list[int] = []
        for entry in root.values():
            raw = entry.get("num_samples")
            if raw is None:
                continue
            try:
                n = int(raw)
            except (TypeError, ValueError):
                continue
            try:
                repeats = int(entry.get("n_repeats", 1) or 1)
            except (TypeError, ValueError):
                repeats = 1
            sample_counts.append(n * max(repeats, 1))
        if sample_counts:
            total = sum(sample_counts)
            n_ds = len(sample_counts)
            suffix = f" (issued, across {n_ds} datasets)" if n_ds > 1 else " (issued)"
            if total < min_queries:
                self._check_results.append(
                    err(
                        "accuracy-sample-count",
                        f"{total} samples{suffix} < required {min_queries}",
                        json_path,
                        "#15",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "accuracy-sample-count",
                        f"{total} samples{suffix} ≥ required {min_queries}",
                        json_path,
                        "#15",
                    )
                )

        # Sample-weighted mean per metric across subsets (the aggregate accuracy).
        per_ds = self.accuracy_result.metric_scores()  # {ds: {metric: float}}
        weighted: dict[str, list[tuple[float, float | None]]] = {}
        for ds_name, scores in per_ds.items():
            raw = root.get(ds_name, {}).get("num_samples")
            weight: float | None
            try:
                weight = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                weight = None
            for metric, value in scores.items():
                weighted.setdefault(metric, []).append((value, weight))
        agg_scores: dict[str, float] = {m: _weighted_mean(pairs) for m, pairs in weighted.items()}

        # Endpoints scorers (e.g. DeepSeekR1Scorer) write a single *unnamed* scalar
        # `score` per dataset into results.json — the scorer's primary metric, with its
        # identity dropped. When the model declares exactly one accuracy metric, gate
        # that scalar against it; but WARN, because we cannot verify the scalar's
        # identity, and any secondary metrics (e.g. tokens_per_sample) are absent from
        # results.json and are therefore NOT checked.
        if list(agg_scores) == ["score"] and len(thresholds) == 1:
            only_metric = next(iter(thresholds))
            self._check_results.append(
                warn(
                    "accuracy-gate",
                    f"results.json exposes only an unnamed scalar accuracy score; "
                    f"gating it as '{only_metric}'. Secondary metrics (if any) are not "
                    f"present in results.json and are not checked.",
                    json_path,
                    "#15",
                )
            )
            agg_scores = {only_metric: agg_scores["score"]}

        for threshold_key, (lower, upper) in thresholds.items():
            # Match score key case-insensitively
            score: float | None = None
            matched_key: str = threshold_key
            for k, v in agg_scores.items():
                if k.lower() == threshold_key:
                    score = v
                    matched_key = k
                    break
            if score is None:
                continue  # metric not present in this run's results

            # Endpoints scorers report fractions (0–1); targets are on a 0–100 scale.
            # Rescale to a percentage so the comparison is apples-to-apples.
            if 0.0 <= score <= 1.0 and lower > 1.0:
                score *= 100.0

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
