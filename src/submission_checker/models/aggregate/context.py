"""Model context — aggregate model-level validation (point count, coverage, consistency).

Handles accuracy and overall compliance checks.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["ModelContext"]

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ...accuracy_targets import get_thresholds
from ..file.accuracy import AccuracyResult
from ..file.point_config import OFFLINE_DEDICATED, OFFLINE_ELECTED, PointConfig
from ..file.point_summary import PointSummary
from ..file.system import SystemDescription
from ..regions import ULTRA_LOW_CONCURRENCY_MAX, Regions, covered_region
from ..results import CheckResult, err, ok, warn

#: §5.3 minimum point counts. A non-agentic submission is 1 + 3 + 3 + 1 — the fourth
#: group being the Offline point. Where the submitter *elects* the C_max point as the
#: Offline result no separate run happens, so the minimum falls back to 7; agentic
#: benchmarks are 7 because §5.7 does not apply to them.
_MIN_POINTS_WITH_DEDICATED_OFFLINE = 8

#: §5.7.2's throughput tolerance between a dedicated Offline run and the C_max point.
#: A tolerance for run-to-run variation, not a target.
_OFFLINE_TPS_MARGIN = 0.98
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
    #: ``None`` when no point parsed, so no ``C_min`` basis exists (§5.4).
    regions: Regions | None
    points_dir: Path
    accuracy_dir: Path | None = None
    all_point_count: int
    valid_points: list[tuple[Path, PointConfig]]
    loaded_points: list[tuple[PointConfig, PointSummary]]
    #: Accuracy results keyed by the concurrency of the point carrying them (§5.3).
    accuracy_by_point: dict[int, AccuracyResult] = Field(default_factory=dict)

    @property
    def offline_points(self) -> list[tuple[Path, PointConfig]]:
        """Points carrying an Offline declaration (§5.7)."""
        return [(path, c) for path, c in self.valid_points if c.is_offline]

    @property
    def min_points(self) -> int:
        """§5.3's minimum for this curve, which depends on how Offline is satisfied.

        A dedicated Offline run is an extra point on top of the seven, so the minimum
        is 8. Electing the C_max point adds no run, so it stays at 7 — as does an
        agentic benchmark, where §5.7 does not apply.
        """
        declared = {c.offline for _, c in self.offline_points}
        if OFFLINE_DEDICATED in declared:
            return _MIN_POINTS_WITH_DEDICATED_OFFLINE
        return _MIN_POINTS

    @model_validator(mode="after")
    def _check_point_count(self) -> ModelContext:
        """§5.3: 7–32 measurement points, or 8 with a dedicated Offline run."""
        n = self.all_point_count
        minimum = self.min_points
        if n < minimum:
            self._check_results.append(
                err(
                    "point-count",
                    f"Only {n} measurement point(s) — minimum {minimum} required",
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
    def _check_offline_point_present(self) -> ModelContext:
        """§9.1: exactly one point carries an Offline declaration (§5.7).

        Skipped entirely for agentic benchmarks, where §5.7 does not apply and §9.1
        requires that *none* is present. This checker has no way to tell an agentic
        curve from a single-turn one yet — §3 does not surface the distinction in any
        file it reads — so the absent case is reported as a WARNING rather than an
        error, and says why. Once the benchmark definition is machine-readable this
        becomes the ERROR §9.1 specifies for non-agentic submissions.
        """
        declared = self.offline_points
        if len(declared) > 1:
            listed = ", ".join(f"r{c.concurrency}" for _, c in declared)
            self._check_results.append(
                err(
                    "offline-point-present",
                    f"{len(declared)} points declare `offline` ({listed}); §5.7 allows exactly one",
                    self.points_dir,
                    "#5.7",
                )
            )
            return self
        if not declared:
            self._check_results.append(
                warn(
                    "offline-point-present",
                    "No point declares `offline`. §5.3 requires one for every non-agentic"
                    " submission; agentic benchmarks must not have one, and the benchmark"
                    " type is not yet machine-readable here",
                    self.points_dir,
                    "#5.7",
                )
            )
            return self

        _path, config = declared[0]
        if config.offline == OFFLINE_ELECTED:
            c_max = self.system_desc.max_supported_concurrency
            if config.concurrency != c_max:
                self._check_results.append(
                    err(
                        "offline-point-present",
                        f"`offline: elected` is declared at concurrency {config.concurrency},"
                        f" but §5.7.2 elects the C_max point and C_max = {c_max}",
                        self.points_dir,
                        "#5.7.2",
                    )
                )
                return self
        self._check_results.append(
            ok(
                "offline-point-present",
                f"Offline point: r{config.concurrency} ({config.offline})",
                self.points_dir,
                "#5.7",
            )
        )
        return self

    @model_validator(mode="after")
    def _check_offline_ordering(self) -> ModelContext:
        """§5.7.2: a dedicated Offline run must beat the C_max point on both axes.

        ``system_tps(Offline) ≥ 0.98 × system_tps(C_max)`` and
        ``concurrency(Offline) ≥ C_max``. The 2 % is a tolerance for run-to-run
        variation between two runs of the same system, not a target — §5.7.2 says so
        explicitly — and it is tighter than the 5 % same-system reproducibility margin
        because both runs come from one submission.

        WARN, not ERROR: §9.1's failure action for this row is "Flag non-compliant
        submission". Not applicable to an elected point, which *is* the C_max point.
        """
        dedicated = [(path, c) for path, c in self.offline_points if c.offline == OFFLINE_DEDICATED]
        if not dedicated or not self.loaded_points:
            return self
        _path, offline_config = dedicated[0]

        tps_by_concurrency = {
            config.concurrency: summary.system_tps for config, summary in self.loaded_points
        }
        offline_tps = tps_by_concurrency.get(offline_config.concurrency)
        c_max = self.system_desc.max_supported_concurrency
        c_max_tps = tps_by_concurrency.get(c_max)

        if offline_config.concurrency < c_max:
            self._check_results.append(
                warn(
                    "offline-ordering",
                    f"Offline concurrency {offline_config.concurrency} < C_max {c_max} (§5.7.2)",
                    self.points_dir,
                    "#5.7.2",
                )
            )
        if offline_tps is None or c_max_tps is None:
            # One of the two summaries did not load; the missing-file rules report that.
            return self
        floor = _OFFLINE_TPS_MARGIN * c_max_tps
        if offline_tps < floor:
            self._check_results.append(
                warn(
                    "offline-ordering",
                    f"Offline system_tps {offline_tps:.3f} < {_OFFLINE_TPS_MARGIN:.2f} ×"
                    f" C_max system_tps {c_max_tps:.3f} = {floor:.3f} (§5.7.2)",
                    self.points_dir,
                    "#5.7.2",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "offline-ordering",
                    f"Offline system_tps {offline_tps:.3f} ≥ {floor:.3f}"
                    f" and concurrency {offline_config.concurrency} ≥ C_max {c_max}",
                    self.points_dir,
                    "#5.7.2",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_ultra_low_concurrency_coverage(self) -> ModelContext:
        """§5.4: at least one point must sit in the Ultra Low Concurrency band (1–32).

        Checked against the *fixed* 1–32 window rather than the derived ``low_latency``
        region. ``C_min`` is the lowest submitted concurrency, so ``low_latency`` is
        ``1–C_min`` and always contains that point — testing it would be vacuous. The
        real requirement is that the curve reaches down into the band at all.
        """
        concurrencies = [config.concurrency for _, config in self.valid_points]
        ultra_low = [c for c in concurrencies if c <= ULTRA_LOW_CONCURRENCY_MAX]
        if ultra_low:
            self._check_results.append(
                ok(
                    "ultra-low-concurrency-coverage",
                    f"Ultra Low Concurrency covered: {sorted(ultra_low)}"
                    f" (≤ {ULTRA_LOW_CONCURRENCY_MAX})",
                    self.points_dir,
                    "#5.4",
                )
            )
        else:
            lowest = min(concurrencies) if concurrencies else None
            detail = f"lowest point is {lowest}" if lowest is not None else "no valid points"
            self._check_results.append(
                err(
                    "ultra-low-concurrency-coverage",
                    f"No point at concurrency ≤ {ULTRA_LOW_CONCURRENCY_MAX} ({detail})",
                    self.points_dir,
                    "#5.4",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_regional_coverage(self) -> ModelContext:
        """§3–6: at least one valid point must fall in each of the three concurrency regions.

        Attribution goes through
        :func:`~submission_checker.models.regions.covered_region`, so a point in the
        10 % margin does not stand in for a High Concurrency point.
        """
        if self.regions is None:
            return self  # no C_min basis; region-basis already reported it
        r = self.regions
        attributed: dict[str, list[int]] = {}
        for _, config in self.valid_points:
            region = covered_region(config.concurrency, r)
            if region is not None:
                attributed.setdefault(region, []).append(config.concurrency)

        coverage_checks = [
            ("low-concurrency-coverage", "Low Concurrency", "low_concurrency", r.low_concurrency),
            (
                "med-concurrency-coverage",
                "Medium Concurrency",
                "med_concurrency",
                r.med_concurrency,
            ),
            (
                "high-concurrency-coverage",
                "High Concurrency",
                "high_concurrency",
                r.high_concurrency,
            ),
        ]
        for rule, label, key, bounds in coverage_checks:
            matching = attributed.get(key, [])
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

        The model directory name comes from point.yaml's §8.3 ``model_name`` (last path
        component, slugified), which is what §8.1's ``results/<system>/<model_name>/``
        asks for. system_desc.model_id is the authoritative source here;
        system_desc.model_name is the fallback. Both may be in HuggingFace
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
    def _check_accuracy_coverage(self) -> ModelContext:
        """§5.3: accuracy is required at N points, not once per submission.

        The four mandatory concurrency points — one in Ultra Low Concurrency and one
        in each of Low, Medium and High — plus the Offline point. N is 5 for a
        non-agentic benchmark and 4 for an agentic one.

        Reported per band rather than as a bare count, because "5 accuracy runs" all
        clustered in one region satisfies a count and not the requirement.
        """
        if self.regions is None or not self.valid_points:
            return self

        covered: set[str] = set()
        for _path, config in self.valid_points:
            if config.concurrency not in self.accuracy_by_point:
                continue
            if config.concurrency <= ULTRA_LOW_CONCURRENCY_MAX:
                covered.add("ultra_low_concurrency")
            band = covered_region(config.concurrency, self.regions)
            if band is not None and band != "low_latency":
                covered.add(band)

        required = (
            "ultra_low_concurrency",
            "low_concurrency",
            "med_concurrency",
            "high_concurrency",
        )
        missing = [band for band in required if band not in covered]
        if missing:
            self._check_results.append(
                err(
                    "accuracy-coverage",
                    "No accuracy results at a point in: "
                    + ", ".join(b.replace("_", " ") for b in missing)
                    + " (§5.3 requires accuracy at each of the four mandatory points)",
                    self.points_dir,
                    "#5.3",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "accuracy-coverage",
                    f"Accuracy present in all four mandatory bands ({len(self.accuracy_by_point)}"
                    " point(s) carry results)",
                    self.points_dir,
                    "#5.3",
                )
            )

        for _path, config in self.offline_points:
            if config.concurrency in self.accuracy_by_point:
                self._check_results.append(
                    ok(
                        "accuracy-coverage",
                        f"Offline point r{config.concurrency} carries accuracy results",
                        self.points_dir,
                        "#5.3",
                    )
                )
            else:
                self._check_results.append(
                    err(
                        "accuracy-coverage",
                        f"Offline point r{config.concurrency} has no accuracy results;"
                        " §5.3 counts it among the N required points",
                        self.points_dir,
                        "#5.3",
                    )
                )
        return self

    @model_validator(mode="after")
    def _check_accuracy(self) -> ModelContext:
        """§15: every accuracy metric must meet its quality threshold.

        Runs over every point that carries results — §5.3 requires N of them, and a
        failing gate at any one is a failing submission.
        """
        if not self.accuracy_by_point:
            return self  # file missing/invalid already reported by checker.py
        for concurrency in sorted(self.accuracy_by_point):
            self._gate_accuracy(self.accuracy_by_point[concurrency])
        return self

    def _gate_accuracy(self, accuracy_result: AccuracyResult) -> None:
        """Gate one point's accuracy results against the model's thresholds (§15)."""
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
            return

        thresholds, min_queries = target
        root = accuracy_result.root

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
        per_ds = accuracy_result.metric_scores()  # {ds: {metric: float}}
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
        return
