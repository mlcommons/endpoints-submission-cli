"""Model context — aggregate model-level validation (point count, coverage, consistency).

Handles accuracy and overall compliance checks.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ModelContext"]

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ...accuracy_targets import get_thresholds
from ...agentic_targets import (
    OSL_FULL_RUN_FIELD,
    SWEBENCH_DATASET,
    SWEBENCH_MEAN_OF_N,
    AgenticTargets,
    get_agentic_targets,
)
from ..file.accuracy import PERFORMANCE_DATASET_TYPE, AccuracyResult
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

#: §5.3's four mandatory concurrency points, one per band, each requiring accuracy.
_MANDATORY_BANDS = (
    "ultra_low_concurrency",
    "low_concurrency",
    "med_concurrency",
    "high_concurrency",
)
#: How each mandatory band is named to a submitter (§3–6's region names).
_BAND_LABELS = {
    "ultra_low_concurrency": "Ultra Low Concurrency",
    "low_concurrency": "Low Concurrency",
    "med_concurrency": "Medium Concurrency",
    "high_concurrency": "High Concurrency",
}
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
    #: ``None`` when the curve's ``system_desc.json`` did not validate. The curve is
    #: still checked; only the rules that compare against ``C_max`` need it, and they
    #: fall back to :attr:`raw_c_max`.
    system_desc: SystemDescription | None
    #: ``max_supported_concurrency`` read from the raw JSON when :attr:`system_desc`
    #: is ``None`` but that one field was still readable.
    raw_c_max: int | None = None
    model_dir: Path
    #: ``None`` when no point parsed, so no ``C_min`` basis exists (§5.4).
    regions: Regions | None
    points_dir: Path
    accuracy_dir: Path | None = None
    all_point_count: int
    valid_points: list[tuple[Path, PointConfig]]
    loaded_points: list[tuple[PointConfig, PointSummary]]
    #: Concurrencies of points that exist but whose ``point.yaml`` is missing or did
    #: not validate. They count toward coverage: the point is there, and its own
    #: error is reported once by the loader. Leaving it out would report the region
    #: it covers as empty as well.
    unparsed_concurrencies: list[int] = Field(default_factory=list)
    #: Accuracy results keyed by the concurrency of the point carrying them (§5.3).
    accuracy_by_point: dict[int, AccuracyResult] = Field(default_factory=dict)

    @property
    def c_max(self) -> int | None:
        """``C_max`` from the validated description, or the raw fallback, or ``None``."""
        if self.system_desc is not None:
            return self.system_desc.max_supported_concurrency
        return self.raw_c_max

    @property
    def _present_concurrencies(self) -> list[int]:
        """Every point's concurrency, parsed or not — what coverage is judged on."""
        return [c.concurrency for _, c in self.valid_points] + self.unparsed_concurrencies

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

        A dedicated Offline run is left out of the vote (see
        :attr:`_type_voting_points`), so an agentic curve that wrongly includes one is
        still read as agentic, and ``offline-point-present`` rejects it under §5.7.
        """
        patterns = {c.runtime_settings.load_pattern for _, c in self._type_voting_points}
        return patterns == {LOAD_PATTERN_AGENTIC}

    @property
    def _type_voting_points(self) -> list[tuple[Path, PointConfig]]:
        """The points whose load pattern declares the benchmark type.

        §6.1 gives the Offline point its own pattern — "It is the only point for which
        the fixed-concurrency pattern is not used" — and the reference implementation
        names it ``max_throughput``. A dedicated Offline run therefore disagrees with
        the rest of a correct single-turn curve by design, and says nothing about the
        benchmark type. An *elected* point is an ordinary fixed-concurrency run, so it
        still votes.
        """
        return [(p, c) for p, c in self.valid_points if c.offline != OFFLINE_DEDICATED]

    @model_validator(mode="after")
    def _check_benchmark_type_consistency(self) -> ModelContext:
        """§8.5: one curve is one benchmark, so its points must agree on the type.

        Only :attr:`_type_voting_points` are compared: §6.1 expects a dedicated Offline
        run to use a different pattern from the rest of its curve.
        """
        if not self._type_voting_points:
            return self
        patterns = sorted({c.runtime_settings.load_pattern for _, c in self._type_voting_points})
        if len(patterns) > 1:
            self._check_results.append(
                err(
                    "benchmark-type-consistency",
                    "fail",
                    self.points_dir,
                    patterns=", ".join(repr(lp) for lp in patterns),
                )
            )
        else:
            self._check_results.append(
                ok(
                    "benchmark-type-consistency",
                    "pass",
                    self.points_dir,
                    patterns=patterns[0],
                    value="agentic" if self.is_agentic else "single-turn",
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
                err("point-count", "fail", self.points_dir, n=n, minimum=minimum)
            )
        else:
            self._check_results.append(ok("point-count", "pass", self.points_dir, n=n))
        if n > _MAX_POINTS:
            self._check_results.append(
                err("point-cap", "fail", self.points_dir, n=n, max_points=_MAX_POINTS)
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
                    err("offline-point-present", "fail-3", self.points_dir, listed=listed)
                )
            else:
                self._check_results.append(ok("offline-point-present", "pass-2", self.points_dir))
            return self
        if len(declared) > 1:
            listed = ", ".join(f"r{c.concurrency}" for _, c in declared)
            self._check_results.append(
                err(
                    "offline-point-present",
                    "fail",
                    self.points_dir,
                    declared_count=len(declared),
                    listed=listed,
                )
            )
            return self
        if not declared:
            self._check_results.append(
                err(
                    "offline-point-present",
                    "fail-2",
                    self.points_dir,
                    load_pattern_agentic=LOAD_PATTERN_AGENTIC,
                )
            )
            return self

        _path, config = declared[0]
        if config.offline == OFFLINE_ELECTED:
            c_max = self.c_max
            if c_max is not None and config.concurrency != c_max:
                self._check_results.append(
                    err(
                        "offline-point-present",
                        "fail-4",
                        self.points_dir,
                        concurrency=config.concurrency,
                        c_max=c_max,
                    )
                )
                return self
        self._check_results.append(
            ok(
                "offline-point-present",
                "pass",
                self.points_dir,
                concurrency=config.concurrency,
                offline=config.offline,
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
        c_max = self.c_max
        if c_max is None:
            return self  # no C_max to compare against; system-description-valid says why
        c_max_tps = tps_by_concurrency.get(c_max)

        if offline_config.concurrency < c_max:
            self._check_results.append(
                warn(
                    "offline-ordering",
                    "warn",
                    self.points_dir,
                    concurrency=offline_config.concurrency,
                    c_max=c_max,
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
                    "warn-2",
                    self.points_dir,
                    offline_tps=offline_tps,
                    offline_tps_margin=_OFFLINE_TPS_MARGIN,
                    c_max_tps=c_max_tps,
                    floor=floor,
                )
            )
        else:
            self._check_results.append(
                ok(
                    "offline-ordering",
                    "pass",
                    self.points_dir,
                    offline_tps=offline_tps,
                    floor=floor,
                    concurrency=offline_config.concurrency,
                    c_max=c_max,
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
        concurrencies = self._present_concurrencies
        ultra_low = [c for c in concurrencies if c <= ULTRA_LOW_CONCURRENCY_MAX]
        if ultra_low:
            self._check_results.append(
                ok(
                    "ultra-low-concurrency-coverage",
                    "pass",
                    self.points_dir,
                    ultra_low=sorted(ultra_low),
                    ultra_low_concurrency_max=ULTRA_LOW_CONCURRENCY_MAX,
                )
            )
        else:
            lowest = min(concurrencies) if concurrencies else None
            detail = f"lowest point is {lowest}" if lowest is not None else "no valid points"
            self._check_results.append(
                err(
                    "ultra-low-concurrency-coverage",
                    "fail",
                    self.points_dir,
                    ultra_low_concurrency_max=ULTRA_LOW_CONCURRENCY_MAX,
                    detail=detail,
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
        for concurrency in self._present_concurrencies:
            region = covered_region(concurrency, r)
            if region is not None:
                attributed.setdefault(region, []).append(concurrency)

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
                        "pass",
                        self.points_dir,
                        label=label,
                        covered=sorted(matching),
                        bounds=bounds,
                    )
                )
            else:
                self._check_results.append(
                    err(rule, "fail", self.points_dir, label=label, bounds=bounds)
                )
        return self

    @model_validator(mode="after")
    def _check_model_name_consistency(self) -> ModelContext:
        """§8.1: the model directory must be named after the points' ``model_name``.

        §8.1 names it ``results/<system>/<model_name>/`` and §8.5 sources that name from
        ``point.yaml`` (§8.3), so the disclosure is the only source read. The two are
        compared exactly: ``model-name-valid`` already requires the canonical spelling,
        which is the directory name the builder writes, so rewriting the declared name
        here would only let this check pass a name that one fails.

        ``system_desc.json`` is not consulted. It stopped defining ``model_name`` when
        policies PR #130 removed the field, and v1.0 does not read a field the rules no
        longer define — a bundle whose ``point.yaml`` omits the name is incomplete, and
        ``point-disclosure-complete`` says so, rather than being quietly rescued by a
        value from a file that is no longer authoritative.

        - no point declares ``model_name`` → warning
        - the declared name differs from the directory → error
        - they match → ok
        """
        declared = next(
            (c.model_name for _, c in self.valid_points if c.model_name),
            "",
        )

        if not declared:
            self._check_results.append(
                warn(
                    "model-name-consistency",
                    "warn",
                    self.model_dir,
                    model_dir_name=self.model_dir.name,
                )
            )
        else:
            dir_name = self.model_dir.name
            if declared != dir_name:
                self._check_results.append(
                    err(
                        "model-name-consistency",
                        "fail",
                        self.model_dir,
                        declared=declared,
                        dir_name=dir_name,
                    )
                )
            else:
                self._check_results.append(
                    ok("model-name-consistency", "pass", self.model_dir, dir_name=dir_name)
                )
        return self

    @model_validator(mode="after")
    def _check_config_consistency(self) -> ModelContext:
        """§9.1 "Configuration consistency": one curve, one dataset and one model.

        The model is checked over ``valid_points`` rather than ``loaded_points``: a
        point whose ``result_summary.json`` failed to load still made a §8.3 disclosure,
        and disagreeing about which model was measured is a defect either way.
        """
        names = {c.model_name for _, c in self.valid_points if c.model_name}
        if len(names) > 1:
            self._check_results.append(
                err(
                    "config-consistency-model",
                    "fail",
                    self.model_dir,
                    names=", ".join(repr(n) for n in sorted(names)),
                )
            )
        elif names:
            self._check_results.append(
                ok("config-consistency-model", "pass", self.model_dir, next=next(iter(names)))
            )
        if not self.loaded_points:
            return self
        datasets = {config.dataset for config, _ in self.loaded_points}
        if len(datasets) > 1:
            self._check_results.append(
                err("config-consistency-dataset", "fail", self.model_dir, datasets=datasets)
            )
        else:
            self._check_results.append(
                ok("config-consistency-dataset", "pass", self.model_dir, next=next(iter(datasets)))
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
        if self.regions is None or not self._present_concurrencies:
            return self

        covered: set[str] = set()
        for concurrency in self._present_concurrencies:
            if concurrency not in self.accuracy_by_point:
                continue
            if concurrency <= ULTRA_LOW_CONCURRENCY_MAX:
                covered.add("ultra_low_concurrency")
            band = covered_region(concurrency, self.regions)
            if band is not None and band != "low_latency":
                covered.add(band)

        missing = [band for band in _MANDATORY_BANDS if band not in covered]
        if missing:
            self._check_results.append(
                err(
                    "accuracy-coverage",
                    "missing-bands",
                    self.points_dir,
                    bands=", ".join(_BAND_LABELS[band] for band in missing),
                )
            )
        else:
            self._check_results.append(
                ok(
                    "accuracy-coverage",
                    "pass",
                    self.points_dir,
                    accuracy_by_point_count=len(self.accuracy_by_point),
                    value=4 if self.is_agentic else 5,
                )
            )

        for _path, config in self.offline_points:
            if config.concurrency in self.accuracy_by_point:
                self._check_results.append(
                    ok(
                        "accuracy-coverage",
                        "pass-2",
                        self.points_dir,
                        concurrency=config.concurrency,
                    )
                )
            else:
                self._check_results.append(
                    err(
                        "accuracy-coverage", "fail", self.points_dir, concurrency=config.concurrency
                    )
                )
        return self

    def _dataset_score(self, concurrency: int, dataset: str, rule: str) -> float | None:
        """One dataset's scalar score from a point's accuracy results, as a percentage.

        Both agentic scorers report a fraction — the reference ``SWEBenchScorer``
        returns ``resolved / denominator`` and ``AgenticInferenceInlineScorer`` a mean
        of per-turn scores in [0, 1] — while the README's thresholds are percentages,
        so the value is always rescaled. The unit is never guessed from the value: a
        true 0.9% reported as 0.009 would otherwise pass for 0.9 read as 90%. A value
        outside [0, 1] is reported under *rule* and yields ``None``.
        """
        scores = self.accuracy_by_point[concurrency].metric_scores().get(dataset)
        if not scores:
            return None
        value = scores.get("score")
        if value is None:
            value = next(iter(scores.values()), None)
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            self._check_results.append(
                err(
                    rule,
                    "score-not-fraction",
                    self.points_dir,
                    concurrency=concurrency,
                    dataset=dataset,
                    value=value,
                )
            )
            return None
        return value * 100.0

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
                warn("agentic-accuracy", "warn", self.model_dir, model_dir_name=self.model_dir.name)
            )
            return self
        if not targets.published:
            self._check_results.append(
                warn("agentic-accuracy", "warn-2", self.model_dir, targets_name=targets.name)
            )
            return self
        self._gate_agentic_inline(targets)
        self._gate_agentic_swebench(targets)
        self._gate_agentic_osl(targets)
        return self

    def _gate_agentic_inline(self, targets: AgenticTargets) -> None:
        """Inline accuracy, at each point that reports it.

        "Every … submitted Pareto point must satisfy" applies to the results submitted.
        Points without one are skipped; that one result per mandatory region exists is
        ``accuracy-coverage``'s check.

        Read from the entry scored on the performance run, which the client names
        ``performance`` rather than after the performance dataset.
        """
        assert targets.inline_min is not None
        seen = False
        for concurrency in sorted(self.accuracy_by_point):
            dataset = self.accuracy_by_point[concurrency].performance_dataset()
            if dataset is None:
                continue
            score = self._dataset_score(concurrency, dataset, "agentic-accuracy-inline")
            if score is None:
                continue
            seen = True
            if score < targets.inline_min:
                self._check_results.append(
                    err(
                        "agentic-accuracy-inline",
                        "fail",
                        self.points_dir,
                        concurrency=concurrency,
                        score=score,
                        inline_min=targets.inline_min,
                        targets_name=targets.name,
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "agentic-accuracy-inline",
                        "pass",
                        self.points_dir,
                        concurrency=concurrency,
                        score=score,
                        inline_min=targets.inline_min,
                    )
                )
        if not seen:
            self._check_results.append(
                warn(
                    "agentic-accuracy-inline",
                    "warn",
                    self.points_dir,
                    performance_dataset_type=PERFORMANCE_DATASET_TYPE,
                )
            )

    def _mandatory_band(self, concurrency: int) -> str | None:
        """The §5.3 mandatory band a point's accuracy result counts towards, or ``None``.

        One band per point, unlike ``accuracy-coverage``, which lets a low point count
        towards two bands at once: a mean over bands must not count one result twice.
        """
        if concurrency <= ULTRA_LOW_CONCURRENCY_MAX:
            return "ultra_low_concurrency"
        assert self.regions is not None
        band = covered_region(concurrency, self.regions)
        return band if band in _MANDATORY_BANDS else None

    def _gate_agentic_swebench(self, targets: AgenticTargets) -> None:
        """SWE-bench, mean-of-N over the four mandatory bands — §4.3's multi-turn branch.

        "The arithmetic mean of the N required accuracy results MUST meet the quality
        threshold; individual results need not." For an agentic benchmark §5.3 puts
        the N=4 required results at "the four mandatory concurrency points", and §4.3
        asks for "one accuracy validation run … for each" region. The mean is therefore
        one value per band, not one per point. A result at a point outside the four
        bands is not a required result and is left out.

        The rules name no tiebreak for a band with two results; submitter's-choice
        points share bands with the mandatory ones and look the same on disk. Picking
        one would let a submitter choose which result counts, so those results are
        averaged into one value for the band, and the curve is warned about it.

        A short set is still gated, on the mean of the bands present, and said to be
        short: the missing results are ``accuracy-coverage``'s report, and staying
        silent here would let a submission with three strong bands pass unremarked.
        """
        assert targets.swebench_min is not None
        if self.regions is None:
            return  # region-computation already reports why no band can be assigned
        by_band: dict[str, list[tuple[int, float]]] = {}
        for c in sorted(self.accuracy_by_point):
            score = self._dataset_score(c, SWEBENCH_DATASET, "agentic-accuracy-swebench")
            band = self._mandatory_band(c)
            if score is not None and band is not None:
                by_band.setdefault(band, []).append((c, score))
        if not by_band:
            self._check_results.append(
                warn(
                    "agentic-accuracy-swebench",
                    "warn",
                    self.points_dir,
                    swebench_dataset=SWEBENCH_DATASET,
                )
            )
            return
        for band, results in by_band.items():
            if len(results) > 1:
                listed = ", ".join(f"r{c}" for c, _ in results)
                self._check_results.append(
                    warn(
                        "agentic-accuracy-swebench",
                        "warn-2",
                        self.points_dir,
                        replace=band.replace("_", " "),
                        results_count=len(results),
                        listed=listed,
                    )
                )
        band_scores = [
            sum(score for _, score in results) / len(results) for results in by_band.values()
        ]
        mean = sum(band_scores) / len(band_scores)
        n = len(band_scores)
        basis = f"mean of {n}" + (
            f" band(s) (§4.3 requires {SWEBENCH_MEAN_OF_N})"
            if n != SWEBENCH_MEAN_OF_N
            else " bands"
        )
        if mean < targets.swebench_min:
            self._check_results.append(
                err(
                    "agentic-accuracy-swebench",
                    "fail",
                    self.points_dir,
                    basis=basis,
                    mean=mean,
                    swebench_min=targets.swebench_min,
                    targets_name=targets.name,
                )
            )
        elif n != SWEBENCH_MEAN_OF_N:
            self._check_results.append(
                warn(
                    "agentic-accuracy-swebench",
                    "warn-3",
                    self.points_dir,
                    basis=basis,
                    mean=mean,
                    swebench_min=targets.swebench_min,
                    swebench_mean_of_n=SWEBENCH_MEAN_OF_N,
                )
            )
        else:
            self._check_results.append(
                ok(
                    "agentic-accuracy-swebench",
                    "pass",
                    self.points_dir,
                    basis=basis,
                    mean=mean,
                    swebench_min=targets.swebench_min,
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
                        "pass",
                        self.points_dir,
                        concurrency=config.concurrency,
                        avg=avg,
                        low=low,
                        high=high,
                    )
                )
            else:
                self._check_results.append(
                    err(
                        "agentic-osl-range",
                        "fail",
                        self.points_dir,
                        concurrency=config.concurrency,
                        avg=avg,
                        low=low,
                        high=high,
                        targets_name=targets.name,
                    )
                )
        if missing:
            listed = ", ".join(f"r{c}" for c in sorted(missing))
            self._check_results.append(
                warn(
                    "agentic-osl-range",
                    "warn",
                    self.points_dir,
                    osl_full_run_field=OSL_FULL_RUN_FIELD,
                    listed=listed,
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
        if self.is_agentic and get_agentic_targets(self.model_dir.name) is not None:
            return self  # gated by _check_agentic_accuracy, which aggregates differently
        if get_thresholds(self.model_dir.name) is None:
            # A fact about the model, so said once for the curve rather than once for
            # every point that carries accuracy.
            self._check_results.append(
                warn("accuracy-gate", "no-thresholds", self.model_dir, model=self.model_dir.name)
            )
            return self
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
            return  # reported once for the curve by _check_accuracy
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
                        "fail",
                        json_path,
                        total=total,
                        suffix=suffix,
                        min_queries=min_queries,
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "accuracy-sample-count",
                        "pass",
                        json_path,
                        total=total,
                        suffix=suffix,
                        min_queries=min_queries,
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
                warn("accuracy-gate", "warn-2", json_path, only_metric=only_metric)
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
                        "fail",
                        json_path,
                        matched_key=matched_key,
                        score=score,
                        lower=lower,
                    )
                )
            elif upper is not None and score > upper:
                self._check_results.append(
                    err(
                        "accuracy-gate",
                        "fail-2",
                        json_path,
                        matched_key=matched_key,
                        score=score,
                        upper=upper,
                    )
                )
            else:
                bound = f"[{lower:.4f}, {upper:.4f}]" if upper is not None else f"≥ {lower:.4f}"
                self._check_results.append(
                    ok(
                        "accuracy-gate",
                        "pass",
                        json_path,
                        matched_key=matched_key,
                        score=score,
                        bound=bound,
                    )
                )
        return
