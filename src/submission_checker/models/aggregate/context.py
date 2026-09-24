"""Model context — aggregate model-level validation (point count, coverage, consistency).

Handles accuracy and overall compliance checks.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["ModelContext"]

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ...accuracy_targets import get_thresholds
from ...agentic_targets import (
    INLINE_DATASET,
    OSL_FULL_RUN_FIELD,
    SWEBENCH_DATASET,
    SWEBENCH_MEAN_OF_N,
    AgenticTargets,
    get_agentic_targets,
)
from ..file.accuracy import AccuracyResult
from ..file.point_config import (
    LOAD_PATTERN_AGENTIC,
    OFFLINE_DEDICATED,
    OFFLINE_ELECTED,
    PointConfig,
)
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
    def is_agentic(self) -> bool:
        """Whether this curve measures an agentic benchmark (§6.1's load pattern).

        The benchmark type governs four §9.1 rows but §8.3 has no field for it, so it
        is read from ``runtime_settings.load_pattern``: the reference implementation
        names its fixed-concurrency agentic scheduler ``agentic_inference``, and that
        name is the only agentic signal in any file this checker reads.

        A curve is one benchmark (§8.5: "one system, one benchmark model, one
        dataset"), so this requires unanimity. Points that disagree are reported by
        ``benchmark-type-consistency`` and the curve is treated as non-agentic — the
        stricter reading, since it keeps the Offline requirement in force rather than
        letting one mislabelled point switch it off.
        """
        patterns = {c.runtime_settings.load_pattern for _, c in self.valid_points}
        return patterns == {LOAD_PATTERN_AGENTIC}

    @model_validator(mode="after")
    def _check_benchmark_type_consistency(self) -> ModelContext:
        """§8.5: one curve is one benchmark, so its points must agree on the type."""
        if not self.valid_points:
            return self
        patterns = sorted({c.runtime_settings.load_pattern for _, c in self.valid_points})
        if len(patterns) > 1:
            self._check_results.append(
                err(
                    "benchmark-type-consistency",
                    "Points disagree on `runtime_settings.load_pattern` ("
                    + ", ".join(repr(lp) for lp in patterns)
                    + "); §8.5 defines one result as a single benchmark, and §5.3/§5.7"
                    " apply differently to agentic and single-turn benchmarks",
                    self.points_dir,
                    "#6.1, #8.5",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "benchmark-type-consistency",
                    f"Benchmark type consistent: {patterns[0]!r}"
                    f" ({'agentic' if self.is_agentic else 'single-turn'})",
                    self.points_dir,
                    "#6.1, #8.5",
                )
            )
        return self

    @property
    def min_points(self) -> int:
        """§5.3's minimum for this curve, which depends on how Offline is satisfied.

        A dedicated Offline run is an extra point on top of the seven, so the minimum
        is 8. Electing the C_max point adds no run, so it stays at 7 — as does an
        agentic benchmark, where §5.7 does not apply and no Offline point may be
        present at all.
        """
        if self.is_agentic:
            return _MIN_POINTS
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

        The requirement inverts for agentic benchmarks. §5.7: "An agentic submission
        neither requires nor may include an Offline point", and §9.1 asks for "exactly
        one … for non-agentic benchmarks; none is present for agentic benchmarks". So
        an agentic curve is checked for *absence*, and a declaration on one is an
        error rather than the thing being required.

        Which branch applies comes from :attr:`is_agentic`.
        """
        declared = self.offline_points
        if self.is_agentic:
            if declared:
                listed = ", ".join(f"r{c.concurrency}" for _, c in declared)
                self._check_results.append(
                    err(
                        "offline-point-present",
                        f"Agentic benchmark declares an Offline point ({listed}); §5.7 says"
                        " an agentic submission “neither requires nor may include” one",
                        self.points_dir,
                        "#5.7",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "offline-point-present",
                        "Agentic benchmark: no Offline point, as §5.7 requires",
                        self.points_dir,
                        "#5.7",
                    )
                )
            return self
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
                err(
                    "offline-point-present",
                    "No point declares `offline`; §5.3 requires one for every non-agentic"
                    " submission. An agentic submission declares"
                    f" `runtime_settings.load_pattern: {LOAD_PATTERN_AGENTIC}`, which this"
                    " curve does not",
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

        The four mandatory bands are checked the same way either way; N differs only
        because the Offline row below iterates ``offline_points``, which an agentic
        curve has none of. The pass message names the applicable N so a reviewer can
        see which reading was applied.
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
                    "Accuracy present in all four mandatory bands"
                    f" ({len(self.accuracy_by_point)} point(s) carry results;"
                    f" §5.3 N={4 if self.is_agentic else 5})",
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

    def _dataset_score(self, result: AccuracyResult, dataset: str) -> float | None:
        """One dataset's scalar score from a point's accuracy results, rescaled to 0–100.

        Agentic scorers report a fraction; the README's thresholds are percentages.
        Only a value that cannot already be a percentage is rescaled, so a scorer that
        reports 58.9 and one that reports 0.589 both gate correctly.
        """
        scores = result.metric_scores().get(dataset)
        if not scores:
            return None
        value = scores.get("score")
        if value is None:
            value = next(iter(scores.values()), None)
        if value is None:
            return None
        return value * 100.0 if 0.0 <= value <= 1.0 else value

    @model_validator(mode="after")
    def _check_agentic_accuracy(self) -> ModelContext:
        """The agentic accuracy gates (§3.2, §4.3), which do not reduce to §15's.

        Three quantities, aggregated three different ways — see
        :mod:`submission_checker.agentic_targets` for why they cannot share the
        single-turn gate. Runs only for an agentic curve whose model the reference
        implementation names.
        """
        if not self.is_agentic:
            return self
        targets = get_agentic_targets(self.model_dir.name)
        if targets is None:
            self._check_results.append(
                warn(
                    "agentic-accuracy",
                    f"Agentic curve for model '{self.model_dir.name}', which is not one of"
                    " the agentic benchmarks the reference implementation publishes"
                    " thresholds for — no agentic accuracy gate applied",
                    self.model_dir,
                    "#3.2",
                )
            )
            return self
        if not targets.published:
            self._check_results.append(
                warn(
                    "agentic-accuracy",
                    f"{targets.name}: the reference implementation records its accuracy"
                    " thresholds and SWE-bench evaluation policy as TBD, so no agentic"
                    " accuracy gate can be applied to this submission",
                    self.model_dir,
                    "#3.2",
                )
            )
            return self
        self._gate_agentic_inline(targets)
        self._gate_agentic_swebench(targets)
        self._gate_agentic_osl(targets)
        return self

    def _gate_agentic_inline(self, targets: AgenticTargets) -> None:
        """Inline accuracy, per point: "Every … submitted Pareto point must satisfy"."""
        assert targets.inline_min is not None
        seen = False
        for concurrency in sorted(self.accuracy_by_point):
            score = self._dataset_score(self.accuracy_by_point[concurrency], INLINE_DATASET)
            if score is None:
                continue
            seen = True
            if score < targets.inline_min:
                self._check_results.append(
                    err(
                        "agentic-accuracy-inline",
                        f"r{concurrency}: inline accuracy {score:.2f} <"
                        f" {targets.inline_min} required for {targets.name}",
                        self.points_dir,
                        "#4.3",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "agentic-accuracy-inline",
                        f"r{concurrency}: inline accuracy {score:.2f} ≥ {targets.inline_min}",
                        self.points_dir,
                        "#4.3",
                    )
                )
        if not seen:
            self._check_results.append(
                warn(
                    "agentic-accuracy-inline",
                    f"No point reports a `{INLINE_DATASET}` accuracy score; official"
                    " agentic submissions set `accuracy_config.eval_method:"
                    " agentic_inference_inline` on the performance dataset",
                    self.points_dir,
                    "#4.3",
                )
            )

    def _gate_agentic_swebench(self, targets: AgenticTargets) -> None:
        """SWE-bench, mean-of-N across points — §4.3's multi-turn branch.

        "The arithmetic mean of the N required accuracy results MUST meet the quality
        threshold; individual results need not." A short set is still gated, on the
        mean of what is present, and said to be short: the missing results are
        ``accuracy-coverage``'s report, and staying silent here would let a submission
        with three strong points pass unremarked.
        """
        assert targets.swebench_min is not None
        scores = [
            (c, score)
            for c in sorted(self.accuracy_by_point)
            if (score := self._dataset_score(self.accuracy_by_point[c], SWEBENCH_DATASET))
            is not None
        ]
        if not scores:
            self._check_results.append(
                warn(
                    "agentic-accuracy-swebench",
                    f"No point reports a `{SWEBENCH_DATASET}` accuracy score; official"
                    " agentic submissions must enable SWE-bench accuracy",
                    self.points_dir,
                    "#4.3",
                )
            )
            return
        mean = sum(score for _, score in scores) / len(scores)
        n = len(scores)
        basis = f"mean of {n}" + (
            f" (§4.3 requires {SWEBENCH_MEAN_OF_N})" if n != SWEBENCH_MEAN_OF_N else ""
        )
        if mean < targets.swebench_min:
            self._check_results.append(
                err(
                    "agentic-accuracy-swebench",
                    f"SWE-bench {basis} = {mean:.2f} < {targets.swebench_min} required for"
                    f" {targets.name}",
                    self.points_dir,
                    "#4.3",
                )
            )
        elif n != SWEBENCH_MEAN_OF_N:
            self._check_results.append(
                warn(
                    "agentic-accuracy-swebench",
                    f"SWE-bench {basis} = {mean:.2f} ≥ {targets.swebench_min}, but §4.3"
                    f" averages one result from each of the {SWEBENCH_MEAN_OF_N} mandatory"
                    " regions",
                    self.points_dir,
                    "#4.3",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "agentic-accuracy-swebench",
                    f"SWE-bench {basis} = {mean:.2f} ≥ {targets.swebench_min}",
                    self.points_dir,
                    "#4.3",
                )
            )

    def _gate_agentic_osl(self, targets: AgenticTargets) -> None:
        """OSL per-turn mean, per point, against an inclusive range.

        Read from ``output_sequence_lengths_full_run.output_sequence_lengths.avg`` —
        the full-run, all-turns mean. The windowed ``output_sequence_lengths`` block
        has the same shape and a different value, so the field is resolved explicitly
        rather than falling back to it: a silent fallback would gate the wrong number.
        """
        assert targets.osl_range is not None
        low, high = targets.osl_range
        missing: list[int] = []
        for config, summary in self.loaded_points:
            full_run = getattr(summary, OSL_FULL_RUN_FIELD, None)
            avg = None
            if isinstance(full_run, dict):
                inner = full_run.get("output_sequence_lengths")
                if isinstance(inner, dict):
                    avg = inner.get("avg")
            if not isinstance(avg, (int, float)):
                missing.append(config.concurrency)
                continue
            if low <= avg <= high:
                self._check_results.append(
                    ok(
                        "agentic-osl-range",
                        f"r{config.concurrency}: full-run OSL per-turn mean {avg:.1f} within"
                        f" {low:g}–{high:g} tokens",
                        self.points_dir,
                        "#4.3",
                    )
                )
            else:
                self._check_results.append(
                    err(
                        "agentic-osl-range",
                        f"r{config.concurrency}: full-run OSL per-turn mean {avg:.1f} outside"
                        f" {low:g}–{high:g} tokens required for {targets.name}",
                        self.points_dir,
                        "#4.3",
                    )
                )
        if missing:
            listed = ", ".join(f"r{c}" for c in sorted(missing))
            self._check_results.append(
                warn(
                    "agentic-osl-range",
                    f"No `{OSL_FULL_RUN_FIELD}.output_sequence_lengths.avg` in"
                    f" result_summary.json for {listed}; the windowed"
                    " `output_sequence_lengths` is explicitly not the field to gate",
                    self.points_dir,
                    "#4.3",
                )
            )

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
        if self.is_agentic and get_agentic_targets(self.model_dir.name) is not None:
            return  # gated by _check_agentic_accuracy, which aggregates differently
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
