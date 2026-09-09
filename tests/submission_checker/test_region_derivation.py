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

from .conftest import TEST_SUBMISSIONS


def _check(path: Path) -> Report:
    return SubmissionChecker(path).run()


def _results(report: Report, rule: str, severity: Severity | None = None) -> list[CheckResult]:
    return [
        r for r in report.results if r.rule == rule and (severity is None or r.severity == severity)
    ]


def _curve_facts(submission: Path) -> tuple[int, list[int]]:
    """The curve's declared C_max and its point concurrencies, read off disk."""
    for _system, model_dir in layout.iter_curves(submission / layout.RESULTS_DIR):
        point_dirs = layout.iter_point_dirs(model_dir)
        desc = json.loads((point_dirs[0] / layout.SYSTEM_DESC_JSON).read_text())
        concurrencies = [layout.parse_point_dir(d.name) or 0 for d in point_dirs]
        return desc["max_supported_concurrency"], concurrencies
    raise AssertionError(f"{submission} has no curve")


#: Every fixture tree, with the C_min the checker must derive for its first curve.
#: C_min is the lowest submitted concurrency, clamped to 32 — §5.4 puts the Ultra Low
#: Concurrency point at or below 32, so a curve starting higher still gets usable
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
    c_max, concurrencies = _curve_facts(TEST_SUBMISSIONS / name)
    assert c_max == expected_c_max
    assert min(min(concurrencies), ULTRA_LOW_CONCURRENCY_MAX) == expected_c_min


@pytest.mark.parametrize("name, expected_c_max, expected_c_min", _DERIVATION_TABLE)
def test_reported_boundaries_match_the_derivation(
    name: str, expected_c_max: int, expected_c_min: int
) -> None:
    """The region-basis message states the C_min the boundaries were computed from."""
    report = _check(TEST_SUBMISSIONS / name)
    basis = _results(report, "region-basis")
    assert basis, f"{name}: region-basis did not run"
    assert f"C_min = {expected_c_min}" in basis[0].message


def test_clamped_basis_says_so(sub_g: Path) -> None:
    """A curve whose lowest point is above 32 must not silently get C_min = 64."""
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

    A corrupt ``point.yaml`` has no trustworthy concurrency. Letting it set the floor
    would move every boundary and cascade spurious ``concurrency-in-range`` failures
    onto the points that are fine, so unparseable points are excluded from the basis
    and the report says the basis was partial.
    """

    def _corrupt(self, submission: Path, concurrency: int) -> None:
        for _system, model_dir in layout.iter_curves(submission / layout.RESULTS_DIR):
            path = model_dir / layout.point_dir_name(concurrency) / layout.POINT_YAML
            if path.exists():
                path.write_text("{not: valid: yaml [")

    def test_lowest_point_corrupt_derives_from_the_rest(
        self, valid_standardized: Path, tmp_path: Path
    ) -> None:
        import shutil

        root = tmp_path / "sub"
        shutil.copytree(valid_standardized, root)
        self._corrupt(root, 16)  # next-lowest point is 20

        report = _check(root)
        basis = _results(report, "region-basis")
        assert basis and basis[0].severity == Severity.WARNING
        assert "C_min = 20" in basis[0].message
        assert "6 of 7 points" in basis[0].message

    def test_partial_basis_still_reports_the_corrupt_file(
        self, valid_standardized: Path, tmp_path: Path
    ) -> None:
        import shutil

        root = tmp_path / "sub"
        shutil.copytree(valid_standardized, root)
        self._corrupt(root, 16)
        assert _results(_check(root), "point-config-valid", Severity.ERROR)

    def test_all_points_corrupt_errors_but_keeps_checking(
        self, valid_standardized: Path, tmp_path: Path
    ) -> None:
        """With no basis at all, the rest of the report must stay useful."""
        import shutil

        root = tmp_path / "sub"
        shutil.copytree(valid_standardized, root)
        for _system, model_dir in layout.iter_curves(root / layout.RESULTS_DIR):
            for point_dir in layout.iter_point_dirs(model_dir):
                (point_dir / layout.POINT_YAML).write_text("{not: valid: yaml [")

        report = _check(root)
        assert _results(report, "region-basis", Severity.ERROR)
        # point-count does not depend on the regions, so it must still run.
        assert _results(report, "point-count")
        # Region-dependent rules are skipped rather than reported against no basis.
        assert not _results(report, "concurrency-in-range")
        assert not _results(report, "low-concurrency-coverage")

    def test_no_basis_does_not_crash_the_coverage_rules(
        self, valid_standardized: Path, tmp_path: Path
    ) -> None:
        import shutil

        root = tmp_path / "sub"
        shutil.copytree(valid_standardized, root)
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
        c_max, concurrencies = _curve_facts(TEST_SUBMISSIONS / "sub_e")
        regions = compute_regions(c_max, min(min(concurrencies), ULTRA_LOW_CONCURRENCY_MAX))
        assert regions.low_concurrency.start == 2
        assert any(regions.low_concurrency.contains(c) for c in concurrencies)
        assert not _results(_check(sub_e), "low-concurrency-coverage", Severity.ERROR)
