# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Offline point (§5.7) and §5.3's accuracy coverage.

The Offline point is the first point that is deliberately *not* a fixed-concurrency
run, so most of what is asserted here is which ordinary rules must stand down for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from submission_checker.models import (
    AccuracyResult,
    PointConfig,
    RegionPlacement,
    RuntimeSettings,
    Severity,
)

from .conftest import _REGIONS, _config, _model_ctx, _summary

#: One point's accuracy results, reused wherever coverage rather than score matters.
_ACCURACY = AccuracyResult({"cnn_dailymail": {"num_samples": 500, "score": {"rouge1": 45.0}}})


def _offline_config(concurrency: int = 24576, offline: str = "dedicated") -> PointConfig:
    """A point declaring itself the Offline run.

    The default concurrency is a realistic dataset cardinality (open_orca), which is
    what §5.7.1 fixes it to — and which is far outside C_max's 10 % margin.
    """
    return PointConfig(
        concurrency=concurrency,
        dataset="open_orca",
        offline=offline,
        runtime_settings=RuntimeSettings(
            load_pattern="offline",
            runtime=RuntimeSettings.Runtime(scheduler_rng_seed=1, sample_index_rng_seed=2),
        ),
    )


@pytest.mark.unit
class TestOfflineVocabulary:
    @pytest.mark.parametrize("value", ["dedicated", "elected", "none"])
    def test_valid_values_accepted(self, tmp_path: Path, value: str) -> None:
        config = PointConfig.model_validate(
            {
                "concurrency": 64,
                "offline": value,
                "runtime_settings": {"load_pattern": "concurrency", "runtime": {}},
            },
            context={"yaml_path": tmp_path / "point.yaml"},
        )
        assert not [
            r
            for r in config._check_results
            if r.rule == "offline-declared" and r.severity == Severity.ERROR
        ]

    def test_unknown_value_errors(self, tmp_path: Path) -> None:
        config = PointConfig.model_validate(
            {
                "concurrency": 64,
                "offline": "maybe",
                "runtime_settings": {"load_pattern": "concurrency", "runtime": {}},
            },
            context={"yaml_path": tmp_path / "point.yaml"},
        )
        assert [
            r
            for r in config._check_results
            if r.rule == "offline-declared" and r.severity == Severity.ERROR
        ]

    def test_absent_is_not_a_finding(self, tmp_path: Path) -> None:
        config = _config(concurrency=64)
        assert not [r for r in config._check_results if r.rule == "offline-declared"]
        assert not config.is_offline

    @pytest.mark.parametrize(
        "value, expected", [("dedicated", True), ("elected", True), ("none", False), (None, False)]
    )
    def test_is_offline(self, value: str | None, expected: bool) -> None:
        assert _offline_config(offline=value).is_offline is expected if value else True


@pytest.mark.unit
class TestRulesThatStandDownForOffline:
    """§5.7 makes the Offline point an exception to three fixed-concurrency rules."""

    def test_dedicated_run_is_exempt_from_the_load_pattern(self, tmp_path: Path) -> None:
        """§6.1: it uses the Offline load pattern, not fixed concurrency."""
        config = PointConfig.model_validate(
            {
                "concurrency": 24576,
                "offline": "dedicated",
                "runtime_settings": {"load_pattern": "offline", "runtime": {}},
            },
            context={"yaml_path": tmp_path / "point.yaml"},
        )
        assert not [
            r
            for r in config._check_results
            if r.rule == "load-pattern" and r.severity == Severity.ERROR
        ]

    def test_elected_point_is_not_exempt(self, tmp_path: Path) -> None:
        """An elected point is an ordinary run nominated afterwards, so §6.1 applies."""
        config = PointConfig.model_validate(
            {
                "concurrency": 1024,
                "offline": "elected",
                "runtime_settings": {"load_pattern": "offline", "runtime": {}},
            },
            context={"yaml_path": tmp_path / "point.yaml"},
        )
        assert [
            r
            for r in config._check_results
            if r.rule == "load-pattern" and r.severity == Severity.ERROR
        ]

    def test_dedicated_concurrency_is_exempt_from_the_region_range(self, tmp_path: Path) -> None:
        """§5.7.1 fixes it to the dataset cardinality, well past C_max's margin."""
        placement = RegionPlacement(
            config=_offline_config(), regions=_REGIONS, yaml_path=tmp_path / "point.yaml"
        )
        assert not [
            r
            for r in placement._check_results
            if r.rule == "concurrency-in-range" and r.severity == Severity.ERROR
        ]

    def test_an_ordinary_point_at_that_concurrency_is_not_exempt(self, tmp_path: Path) -> None:
        placement = RegionPlacement(
            config=_config(concurrency=24576), regions=_REGIONS, yaml_path=tmp_path / "p.yaml"
        )
        assert [
            r
            for r in placement._check_results
            if r.rule == "concurrency-in-range" and r.severity == Severity.ERROR
        ]

    def test_elected_point_is_not_exempt_from_the_region_range(self, tmp_path: Path) -> None:
        """§5.7.2 Option 2: the §5.7.1 exemptions do not reach an elected point.

        It "remains a fixed-concurrency pareto point" whose concurrency the submitter
        chose, so an out-of-range value is a real defect rather than a dataset property.
        """
        placement = RegionPlacement(
            config=_offline_config(concurrency=24576, offline="elected"),
            regions=_REGIONS,
            yaml_path=tmp_path / "point.yaml",
        )
        assert [
            r
            for r in placement._check_results
            if r.rule == "concurrency-in-range" and r.severity == Severity.ERROR
        ]

    def test_elected_point_in_range_passes(self, tmp_path: Path) -> None:
        """The ordinary path: an elected point sits at C_max, which is in range."""
        placement = RegionPlacement(
            config=_offline_config(concurrency=1024, offline="elected"),
            regions=_REGIONS,
            yaml_path=tmp_path / "point.yaml",
        )
        assert not [
            r
            for r in placement._check_results
            if r.rule == "concurrency-in-range" and r.severity == Severity.ERROR
        ]

    def test_dedicated_run_covers_no_region(self, tmp_path: Path) -> None:
        """§5.7.2: it "does not satisfy any region-coverage requirement of §5.3"."""
        placement = RegionPlacement(
            config=_offline_config(concurrency=512),
            regions=_REGIONS,
            yaml_path=tmp_path / "point.yaml",
        )
        assert placement.covered_region is None

    def test_elected_point_keeps_its_region(self, tmp_path: Path) -> None:
        placement = RegionPlacement(
            config=_offline_config(concurrency=512, offline="elected"),
            regions=_REGIONS,
            yaml_path=tmp_path / "point.yaml",
        )
        assert placement.covered_region == "high_concurrency"


@pytest.mark.unit
class TestPointCount:
    """§5.3: 1 + 3 + 3 + 1, where the last group is the Offline point."""

    def _points(self, tmp_path: Path, n: int, offline: str | None = None):
        pts = [(tmp_path / f"r{i}" / "point.yaml", _config(concurrency=i)) for i in range(1, n)]
        if offline:
            pts.append((tmp_path / "rOff" / "point.yaml", _offline_config(offline=offline)))
        return pts

    def test_dedicated_offline_raises_the_minimum_to_eight(self, tmp_path: Path) -> None:
        pts = self._points(tmp_path, 7, offline="dedicated")  # 6 + 1 = 7 points
        ctx = _model_ctx(tmp_path, all_point_count=7, valid_points=pts)
        assert [
            r
            for r in ctx._check_results
            if r.rule == "point-count" and r.severity == Severity.ERROR
        ]

    def test_eight_points_with_a_dedicated_offline_passes(self, tmp_path: Path) -> None:
        pts = self._points(tmp_path, 8, offline="dedicated")
        ctx = _model_ctx(tmp_path, all_point_count=8, valid_points=pts)
        assert not [
            r
            for r in ctx._check_results
            if r.rule == "point-count" and r.severity == Severity.ERROR
        ]

    def test_elected_offline_keeps_the_minimum_at_seven(self, tmp_path: Path) -> None:
        """§5.7.2: electing C_max adds no run, so no extra point is required."""
        pts = self._points(tmp_path, 7, offline="elected")
        ctx = _model_ctx(tmp_path, all_point_count=7, valid_points=pts)
        assert not [
            r
            for r in ctx._check_results
            if r.rule == "point-count" and r.severity == Severity.ERROR
        ]


@pytest.mark.unit
class TestOfflinePointPresent:
    def test_two_declarations_error(self, tmp_path: Path) -> None:
        pts = [
            (tmp_path / "a" / "point.yaml", _offline_config(concurrency=100)),
            (tmp_path / "b" / "point.yaml", _offline_config(concurrency=200)),
        ]
        ctx = _model_ctx(tmp_path, valid_points=pts)
        assert [
            r
            for r in ctx._check_results
            if r.rule == "offline-point-present" and r.severity == Severity.ERROR
        ]

    def test_absent_errors_on_a_single_turn_curve(self, tmp_path: Path) -> None:
        """§9.1 rejects a non-agentic submission with no Offline point.

        This used to warn because the checker could not tell the two benchmark types
        apart. §6.1's load pattern now says which, so the rule is the ERROR §9.1 asks
        for rather than a hedge.
        """
        ctx = _model_ctx(tmp_path, valid_points=[(tmp_path / "p.yaml", _config())])
        hits = [r for r in ctx._check_results if r.rule == "offline-point-present"]
        assert hits and hits[0].severity == Severity.ERROR

    def test_elected_must_be_the_c_max_point(self, tmp_path: Path) -> None:
        """§5.7.2 elects the C_max point specifically; _model_ctx's C_max is 1024."""
        pts = [(tmp_path / "p.yaml", _offline_config(concurrency=512, offline="elected"))]
        ctx = _model_ctx(tmp_path, valid_points=pts)
        assert [
            r
            for r in ctx._check_results
            if r.rule == "offline-point-present" and r.severity == Severity.ERROR
        ]

    def test_elected_at_c_max_passes(self, tmp_path: Path) -> None:
        pts = [(tmp_path / "p.yaml", _offline_config(concurrency=1024, offline="elected"))]
        ctx = _model_ctx(tmp_path, valid_points=pts)
        assert not [
            r
            for r in ctx._check_results
            if r.rule == "offline-point-present" and r.severity == Severity.ERROR
        ]


@pytest.mark.unit
class TestOfflineOrdering:
    """§5.7.2: Offline must beat the C_max point on throughput and concurrency."""

    def _ctx(self, tmp_path: Path, offline_tps: float, offline_concurrency: int = 24576):
        c_max_config = _config(concurrency=1024)
        offline_config = _offline_config(concurrency=offline_concurrency)
        # _summary()'s system_tps is total_tokens / elapsed; set both points explicitly.
        c_max_summary = _summary(total_tokens=1_200_000.0)  # 1000 tok/s over 1200 s
        offline_summary = _summary(total_tokens=offline_tps * 1200.0)
        return _model_ctx(
            tmp_path,
            valid_points=[
                (tmp_path / "a" / "point.yaml", c_max_config),
                (tmp_path / "b" / "point.yaml", offline_config),
            ],
            loaded_points=[
                (c_max_config, c_max_summary),
                (offline_config, offline_summary),
            ],
        )

    def test_offline_faster_than_c_max_passes(self, tmp_path: Path) -> None:
        ctx = self._ctx(tmp_path, offline_tps=1200.0)
        assert not [
            r
            for r in ctx._check_results
            if r.rule == "offline-ordering" and r.severity != Severity.INFO
        ]

    def test_within_the_two_percent_tolerance_passes(self, tmp_path: Path) -> None:
        """The margin absorbs run-to-run variation; 0.99 × is inside it."""
        ctx = self._ctx(tmp_path, offline_tps=990.0)
        assert not [
            r
            for r in ctx._check_results
            if r.rule == "offline-ordering" and r.severity != Severity.INFO
        ]

    def test_below_the_tolerance_is_flagged_not_rejected(self, tmp_path: Path) -> None:
        """§9.1's action for this row is "Flag non-compliant submission"."""
        ctx = self._ctx(tmp_path, offline_tps=800.0)
        hits = [r for r in ctx._check_results if r.rule == "offline-ordering"]
        assert any(r.severity == Severity.WARNING for r in hits)
        assert not any(r.severity == Severity.ERROR for r in hits)

    def test_concurrency_below_c_max_is_flagged(self, tmp_path: Path) -> None:
        ctx = self._ctx(tmp_path, offline_tps=1200.0, offline_concurrency=512)
        assert [
            r
            for r in ctx._check_results
            if r.rule == "offline-ordering" and r.severity == Severity.WARNING
        ]

    def test_not_applied_to_an_elected_point(self, tmp_path: Path) -> None:
        """An elected point *is* the C_max point; comparing it to itself is meaningless."""
        config = _offline_config(concurrency=1024, offline="elected")
        ctx = _model_ctx(
            tmp_path,
            valid_points=[(tmp_path / "p.yaml", config)],
            loaded_points=[(config, _summary())],
        )
        assert not [r for r in ctx._check_results if r.rule == "offline-ordering"]


@pytest.mark.unit
class TestAccuracyCoverage:
    """§5.3: accuracy at N points — the four mandatory bands plus Offline."""

    def _pts(self, tmp_path: Path, concurrencies):
        return [(tmp_path / f"r{c}" / "point.yaml", _config(concurrency=c)) for c in concurrencies]

    def test_all_four_bands_covered_passes(self, tmp_path: Path) -> None:
        """_REGIONS is C_max=1024 / C_min=32: low 33–42, med 43–131, high 132–1024."""
        concurrencies = [16, 40, 100, 512]
        ctx = _model_ctx(
            tmp_path,
            valid_points=self._pts(tmp_path, concurrencies),
            accuracy_by_point=dict.fromkeys(concurrencies, _ACCURACY),
        )
        assert not [
            r
            for r in ctx._check_results
            if r.rule == "accuracy-coverage" and r.severity == Severity.ERROR
        ]

    def test_missing_band_errors(self, tmp_path: Path) -> None:
        concurrencies = [16, 40, 100, 512]
        ctx = _model_ctx(
            tmp_path,
            valid_points=self._pts(tmp_path, concurrencies),
            accuracy_by_point={16: _ACCURACY, 40: _ACCURACY, 100: _ACCURACY},
        )
        hits = [
            r
            for r in ctx._check_results
            if r.rule == "accuracy-coverage" and r.severity == Severity.ERROR
        ]
        assert hits and "high concurrency" in hits[0].message

    def test_clustering_in_one_band_does_not_satisfy_the_count(self, tmp_path: Path) -> None:
        """Five accuracy runs all in High Concurrency is five runs, not coverage."""
        concurrencies = [200, 300, 400, 500, 600]
        ctx = _model_ctx(
            tmp_path,
            valid_points=self._pts(tmp_path, concurrencies),
            accuracy_by_point=dict.fromkeys(concurrencies, _ACCURACY),
        )
        assert [
            r
            for r in ctx._check_results
            if r.rule == "accuracy-coverage" and r.severity == Severity.ERROR
        ]

    def test_offline_point_must_carry_accuracy(self, tmp_path: Path) -> None:
        """§5.3 counts the Offline point among the N required."""
        covered = [16, 40, 100, 512]
        pts = self._pts(tmp_path, covered)
        pts.append((tmp_path / "rOff" / "point.yaml", _offline_config()))
        ctx = _model_ctx(
            tmp_path, valid_points=pts, accuracy_by_point=dict.fromkeys(covered, _ACCURACY)
        )
        hits = [
            r
            for r in ctx._check_results
            if r.rule == "accuracy-coverage" and r.severity == Severity.ERROR
        ]
        assert hits and "Offline point" in hits[0].message

    def test_offline_point_with_accuracy_passes(self, tmp_path: Path) -> None:
        covered = [16, 40, 100, 512]
        offline = _offline_config()
        pts = self._pts(tmp_path, covered)
        pts.append((tmp_path / "rOff" / "point.yaml", offline))
        by_point = dict.fromkeys(covered, _ACCURACY)
        by_point[offline.concurrency] = _ACCURACY
        ctx = _model_ctx(tmp_path, valid_points=pts, accuracy_by_point=by_point)
        assert not [
            r
            for r in ctx._check_results
            if r.rule == "accuracy-coverage" and r.severity == Severity.ERROR
        ]

    def test_skipped_without_a_region_basis(self, tmp_path: Path) -> None:
        ctx = _model_ctx(tmp_path, valid_points=[], regions=None)
        assert not [r for r in ctx._check_results if r.rule == "accuracy-coverage"]


@pytest.mark.unit
class TestTheTwoOptionsDifferEverywhereTheSpecSaysTheyDo:
    """§5.7.2 offers two ways to satisfy the Offline requirement, and they are not alike.

    Option 1 is a dedicated run under the Offline load pattern. Option 2 elects the
    C_max point, which "remains a fixed-concurrency pareto point" — so every §5.7.1
    exemption applies to the first and none to the second. Collected in one table
    because the difference is easy to lose in per-rule tests.
    """

    @pytest.mark.parametrize(
        "rule, dedicated_exempt, elected_exempt",
        [
            ("load-pattern", True, False),
            ("concurrency-in-range", True, False),
        ],
    )
    def test_exemptions_apply_to_option_1_only(
        self, tmp_path: Path, rule: str, dedicated_exempt: bool, elected_exempt: bool
    ) -> None:
        for offline, exempt in (("dedicated", dedicated_exempt), ("elected", elected_exempt)):
            config = PointConfig.model_validate(
                {
                    # Out of range for _REGIONS, and not the fixed-concurrency pattern.
                    "concurrency": 24576,
                    "offline": offline,
                    "runtime_settings": {"load_pattern": "offline", "runtime": {}},
                },
                context={"yaml_path": tmp_path / "point.yaml"},
            )
            if rule == "load-pattern":
                found = [
                    r
                    for r in config._check_results
                    if r.rule == rule and r.severity == Severity.ERROR
                ]
            else:
                placement = RegionPlacement(
                    config=config, regions=_REGIONS, yaml_path=tmp_path / "point.yaml"
                )
                found = [
                    r
                    for r in placement._check_results
                    if r.rule == rule and r.severity == Severity.ERROR
                ]
            assert bool(found) is (not exempt), (
                f"{rule}: offline={offline!r} should {'not ' if exempt else ''}report an error"
            )

    def test_only_a_dedicated_run_raises_the_point_minimum(self, tmp_path: Path) -> None:
        """§5.7.2: electing "contains one fewer distinct run", so the minimum stays 7."""
        base = [(tmp_path / f"r{i}" / "point.yaml", _config(concurrency=i)) for i in range(1, 7)]
        dedicated = _model_ctx(
            tmp_path,
            all_point_count=7,
            valid_points=[*base, (tmp_path / "o.yaml", _offline_config(offline="dedicated"))],
        )
        elected = _model_ctx(
            tmp_path,
            all_point_count=7,
            valid_points=[
                *base,
                (tmp_path / "o.yaml", _offline_config(concurrency=1024, offline="elected")),
            ],
        )
        assert dedicated.min_points == 8
        assert elected.min_points == 7

    def test_only_a_dedicated_run_is_ordering_checked(self, tmp_path: Path) -> None:
        """§5.7.2: for an election the constraints "are met with equality"."""
        for offline, expect_rule in (("dedicated", True), ("elected", False)):
            config = _offline_config(concurrency=1024, offline=offline)
            ctx = _model_ctx(
                tmp_path,
                valid_points=[(tmp_path / "p.yaml", config)],
                loaded_points=[(config, _summary())],
            )
            has = bool([r for r in ctx._check_results if r.rule == "offline-ordering"])
            assert has is expect_rule

    def test_only_a_dedicated_run_forfeits_its_region(self, tmp_path: Path) -> None:
        """§5.7.2: an elected point "keeps its role as the C_max point"."""
        for offline, expected in (("dedicated", None), ("elected", "high_concurrency")):
            placement = RegionPlacement(
                config=_offline_config(concurrency=512, offline=offline),
                regions=_REGIONS,
                yaml_path=tmp_path / "point.yaml",
            )
            assert placement.covered_region == expected


@pytest.mark.unit
class TestAgenticDetermination:
    """§6.1's load pattern is the only agentic signal any file the checker reads carries.

    Four §9.1 rows turn on it — §5.3's point minimum and accuracy count, §5.7's Offline
    point, and §9.1's "Offline point present" — so these assert the determination itself
    and each rule that swings on it.
    """

    def _agentic(self, concurrency: int = 64) -> PointConfig:
        return _config(concurrency=concurrency, lp_type="agentic_inference")

    def test_load_pattern_accepted(self, tmp_path: Path) -> None:
        """§6.1 admits it as a fixed-concurrency pattern; `poisson` and friends stay out."""
        config = PointConfig.model_validate(
            {
                "concurrency": 64,
                "runtime_settings": {"load_pattern": "agentic_inference", "runtime": {}},
            },
            context={"yaml_path": tmp_path / "point.yaml"},
        )
        assert not [r for r in config._check_results if r.rule == "load-pattern" and not r.passed]
        assert config.is_agentic

    @pytest.mark.parametrize("lp", ["concurrency", "offline", "poisson"])
    def test_other_patterns_are_not_agentic(self, tmp_path: Path, lp: str) -> None:
        assert not _config(lp_type=lp).is_agentic

    def test_curve_is_agentic_when_every_point_says_so(self, tmp_path: Path) -> None:
        pts = [(tmp_path / f"p{c}.yaml", self._agentic(c)) for c in (16, 64, 512)]
        assert _model_ctx(tmp_path, valid_points=pts).is_agentic

    def test_mixed_patterns_are_reported_and_read_as_single_turn(self, tmp_path: Path) -> None:
        """§8.5 makes one curve one benchmark, so disagreement is an error.

        The curve falls back to non-agentic: that keeps §5.7's Offline requirement in
        force rather than letting one mislabelled point switch it off.
        """
        pts = [
            (tmp_path / "a.yaml", self._agentic(16)),
            (tmp_path / "b.yaml", _config(concurrency=64)),
        ]
        ctx = _model_ctx(tmp_path, valid_points=pts)
        assert not ctx.is_agentic
        assert [
            r
            for r in ctx._check_results
            if r.rule == "benchmark-type-consistency" and r.severity == Severity.ERROR
        ]

    def test_consistent_curve_passes_the_consistency_rule(self, tmp_path: Path) -> None:
        pts = [(tmp_path / f"p{c}.yaml", self._agentic(c)) for c in (16, 64)]
        ctx = _model_ctx(tmp_path, valid_points=pts)
        assert not [
            r for r in ctx._check_results if r.rule == "benchmark-type-consistency" and not r.passed
        ]

    def test_absent_offline_is_correct_for_an_agentic_curve(self, tmp_path: Path) -> None:
        """§5.7: an agentic submission "neither requires nor may include" one."""
        pts = [(tmp_path / f"p{c}.yaml", self._agentic(c)) for c in (16, 64)]
        ctx = _model_ctx(tmp_path, valid_points=pts)
        hits = [r for r in ctx._check_results if r.rule == "offline-point-present"]
        assert hits and all(r.passed for r in hits)

    def test_declared_offline_on_an_agentic_curve_errors(self, tmp_path: Path) -> None:
        """The inversion: presence is the defect, not absence."""
        offline = _offline_config(concurrency=1024, offline="elected")
        offline.runtime_settings.load_pattern = "agentic_inference"
        pts = [(tmp_path / "a.yaml", self._agentic(16)), (tmp_path / "b.yaml", offline)]
        ctx = _model_ctx(tmp_path, valid_points=pts)
        assert [
            r
            for r in ctx._check_results
            if r.rule == "offline-point-present" and r.severity == Severity.ERROR
        ]

    def test_agentic_minimum_is_seven_even_with_a_dedicated_declaration(
        self, tmp_path: Path
    ) -> None:
        """§5.3: agentic is 1 + 3 + 3. A `dedicated` declaration must not lift it to 8 —
        the declaration is itself the error, reported by offline-point-present."""
        offline = _offline_config(concurrency=24576)
        offline.runtime_settings.load_pattern = "agentic_inference"
        pts = [(tmp_path / "a.yaml", self._agentic(16)), (tmp_path / "b.yaml", offline)]
        assert _model_ctx(tmp_path, valid_points=pts).min_points == 7

    def test_single_turn_minimum_is_still_eight_with_a_dedicated_run(self, tmp_path: Path) -> None:
        pts = [
            (tmp_path / "a.yaml", _config(concurrency=16)),
            (tmp_path / "b.yaml", _offline_config(concurrency=24576)),
        ]
        assert _model_ctx(tmp_path, valid_points=pts).min_points == 8
