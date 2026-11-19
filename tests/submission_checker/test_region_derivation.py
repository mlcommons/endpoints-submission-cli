# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for the §5.4 control-flow inversion: C_min derived from the submitted points.

v0.7 computed region boundaries from the declared ``C_max`` alone, before reading a
single point. v1.0 derives ``C_min`` from the points themselves, so the checker must
parse every ``point.yaml`` *first* and only then compute the boundaries. That inversion
is the highest-risk part of the migration — these tests pin the derivation, what happens
when it is only partly possible, and what happens when it is not possible at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from submission_checker import layout
from submission_checker.checker import SubmissionChecker
from submission_checker.models import CheckResult, Report, Severity
from submission_checker.models.regions import ULTRA_LOW_CONCURRENCY_MAX, compute_regions

from .conftest import FIXTURES, copy_submission


def _check(path: Path) -> Report:
    return SubmissionChecker(path).run()


def _results(report: Report, rule: str, severity: Severity | None = None) -> list[CheckResult]:
    return [
        r for r in report.results if r.rule == rule and (severity is None or r.severity == severity)
    ]


def _curve_facts(submission: Path) -> tuple[int, list[int]]:
    """The frontier's declared C_max and its point concurrencies, read off disk."""
    for _system, model_dir in layout.iter_curves(submission / layout.RESULTS_DIR):
        point_dirs = layout.iter_point_dirs(model_dir)
        desc = json.loads((point_dirs[0] / layout.SYSTEM_DESC_JSON).read_text())
        concurrencies = [layout.parse_point_dir(d.name) or 0 for d in point_dirs]
        return desc["max_supported_concurrency"], concurrencies
    raise AssertionError(f"{submission} has no frontier")


#: Every fixture tree, with the C_min the checker must derive for its first frontier.
#: C_min is the lowest submitted concurrency, clamped to 32 — §5.4 puts the Ultra Low
#: Concurrency point at or below 32, so a frontier starting higher still gets usable
#: boundaries and is told separately that it misses the band.
_DERIVATION_TABLE = [
    ("invalid_submission", 88, 16),
    ("sub_a", 2048, 4),
    ("sub_b", 2048, 4),
    ("sub_c", 512, 8),
    ("sub_d", 1024, 8),
    ("sub_e", 1024, 1),
    ("sub_f", 1024, 1),
    ("sub_g", 2048, 32),  # points start at 64 → clamped
    ("sub_h", 2048, 32),  # points start at 64 → clamped
    ("sub_i", 512, 1),
    ("sub_j", 16384, 32),
    ("valid_standardized", 1000, 16),
]


@pytest.mark.parametrize("name, expected_c_max, expected_c_min", _DERIVATION_TABLE)
def test_c_min_derivation_table(name: str, expected_c_max: int, expected_c_min: int) -> None:
    c_max, concurrencies = _curve_facts(FIXTURES[name])
    assert c_max == expected_c_max
    assert min(min(concurrencies), ULTRA_LOW_CONCURRENCY_MAX) == expected_c_min


@pytest.mark.parametrize("name, expected_c_max, expected_c_min", _DERIVATION_TABLE)
def test_reported_boundaries_match_the_derivation(
    name: str, expected_c_max: int, expected_c_min: int
) -> None:
    """The region-basis message states the C_min the boundaries were computed from."""
    report = _check(FIXTURES[name])
    basis = _results(report, "region-basis")
    assert basis, f"{name}: region-basis did not run"
    assert f"C_min = {expected_c_min}" in basis[0].message


def test_clamped_basis_says_so(sub_g: Path) -> None:
    """A frontier whose lowest point is above 32 must not silently get C_min = 64."""
    basis = _results(_check(sub_g), "region-basis")
    assert "C_min = 32" in basis[0].message
    assert "clamped from 64" in basis[0].message


def test_full_basis_is_informational(valid_standardized: Path) -> None:
    report = _check(valid_standardized)
    assert _results(report, "region-basis", Severity.INFO)
    assert not _results(report, "region-basis", Severity.WARNING)
    assert not _results(report, "region-basis", Severity.ERROR)


class TestPartialAndAbsentBasis:
    """What happens when some — or none — of the points can be parsed.

    A point whose ``point.yaml`` does not parse still exists, so it still contributes
    to the basis, with its concurrency read from the ``r<N>`` directory name (which
    ``point-dirname-concurrency`` holds to the declared value). Excluding it instead
    moves the floor, and with it every boundary, which reported regions as uncovered
    that the submission does cover. The report says which points the basis borrowed.
    """

    def _corrupt(self, submission: Path, concurrency: int) -> None:
        for _system, model_dir in layout.iter_curves(submission / layout.RESULTS_DIR):
            path = model_dir / layout.point_dir_name(concurrency) / layout.POINT_YAML
            if path.exists():
                path.write_text("{not: valid: yaml [")

    def test_lowest_point_corrupt_keeps_its_place_in_the_basis(
        self, valid_standardized: Path, tmp_path: Path
    ) -> None:
        root = copy_submission(valid_standardized, tmp_path)
        self._corrupt(root, 16)  # excluding it would move C_min to the next point, 20

        report = _check(root)
        basis = _results(report, "region-basis")
        assert basis and basis[0].severity == Severity.INFO
        assert "C_min = 16" in basis[0].message
        assert "all 7 points; 1 from a point.yaml that did not validate" in basis[0].message
        # The region r16 covers is still covered: no error caused by the corrupt file.
        assert not _results(report, "low-concurrency-coverage", Severity.ERROR)

    def test_partial_basis_still_reports_the_corrupt_file(
        self, valid_standardized: Path, tmp_path: Path
    ) -> None:
        root = copy_submission(valid_standardized, tmp_path)
        self._corrupt(root, 16)
        assert _results(_check(root), "point-config-valid", Severity.ERROR)

    def test_all_points_corrupt_errors_but_keeps_checking(
        self, valid_standardized: Path, tmp_path: Path
    ) -> None:
        """With no basis at all, the rest of the report must stay useful."""
        root = copy_submission(valid_standardized, tmp_path)
        for _system, model_dir in layout.iter_curves(root / layout.RESULTS_DIR):
            for point_dir in layout.iter_point_dirs(model_dir):
                (point_dir / layout.POINT_YAML).write_text("{not: valid: yaml [")

        report = _check(root)
        # Every concurrency comes from its r<N> name, so the basis is still complete.
        basis = _results(report, "region-basis")
        assert basis and "7 from a point.yaml that did not validate" in basis[0].message
        # point-count does not depend on the regions, so it must still run.
        assert _results(report, "point-count")
        # Per-point placement needs the parsed config, so it cannot run...
        assert not _results(report, "concurrency-in-range")
        # ...but coverage only needs concurrencies, and the frontier does cover each region.
        assert not _results(report, "low-concurrency-coverage", Severity.ERROR)
        # Each point says its own rules were skipped, so fixing it may report more.
        assert len(_results(report, "point-rules-skipped", Severity.WARNING)) == 7

    def test_no_basis_does_not_crash_the_coverage_rules(
        self, valid_standardized: Path, tmp_path: Path
    ) -> None:
        root = copy_submission(valid_standardized, tmp_path)
        for _system, model_dir in layout.iter_curves(root / layout.RESULTS_DIR):
            for point_dir in layout.iter_point_dirs(model_dir):
                (point_dir / layout.POINT_YAML).write_text("{not: valid: yaml [")
        report = _check(root)
        assert not report.passed  # it fails, but it produced a report


class TestBoundariesFollowTheDerivedCMin:
    """The whole point of the inversion: boundaries move with the submitter's points."""

    def test_same_c_max_different_c_min_gives_different_regions(self) -> None:
        low = compute_regions(1024, 4)
        high = compute_regions(1024, 32)
        assert low.low_concurrency != high.low_concurrency
        assert low.high_concurrency.end == high.high_concurrency.end == 1024

    def test_a_low_starting_curve_covers_low_concurrency(self, sub_e: Path) -> None:
        """sub_e's points (1–1024) missed v0.7's fixed 33–42 window; v1.0 moves it down."""
        c_max, concurrencies = _curve_facts(FIXTURES["sub_e"])
        regions = compute_regions(c_max, min(min(concurrencies), ULTRA_LOW_CONCURRENCY_MAX))
        assert regions.low_concurrency.start == 2
        assert any(regions.low_concurrency.contains(c) for c in concurrencies)
        assert not _results(_check(sub_e), "low-concurrency-coverage", Severity.ERROR)
