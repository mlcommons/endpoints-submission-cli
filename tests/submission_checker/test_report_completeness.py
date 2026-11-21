# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""One run should report every problem, not just the first.

A submitter fixes what the checker reports and runs it again. Every check that an
unrelated mistake hides costs another round, so these tests take the compliant
``valid_standardized`` submission, introduce one realistic mistake, and assert the
rest of the report still runs: the mistake is reported, nothing else is lost, and no
error is invented because of it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from submission_checker import layout
from submission_checker.checker import SubmissionChecker
from submission_checker.models import CheckResult, Report, Severity


def _check(path: Path) -> Report:
    return SubmissionChecker(path).run()


def _results(report: Report, rule: str, severity: Severity | None = None) -> list[CheckResult]:
    return [
        r for r in report.results if r.rule == rule and (severity is None or r.severity == severity)
    ]


def _rules(report: Report) -> set[str]:
    return {r.rule for r in report.results}


@pytest.fixture
def submission(valid_standardized: Path, tmp_path: Path) -> Path:
    root = tmp_path / "sub"
    shutil.copytree(valid_standardized, root)
    return root


@pytest.fixture
def baseline(valid_standardized: Path) -> Report:
    report = _check(valid_standardized)
    assert report.passed
    return report


def _curve(root: Path) -> Path:
    ((_system, model_dir),) = layout.iter_curves(root / layout.RESULTS_DIR)
    return model_dir


def _edit_system_descs(root: Path, **changes: object) -> None:
    for point_dir in layout.iter_point_dirs(_curve(root)):
        path = point_dir / layout.SYSTEM_DESC_JSON
        data = json.loads(path.read_text())
        for key, value in changes.items():
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
        path.write_text(json.dumps(data, indent=2))


@pytest.mark.unit
class TestMissingDirectories:
    def test_missing_docs_does_not_hide_the_results_checks(
        self, submission: Path, baseline: Report
    ) -> None:
        shutil.rmtree(submission / layout.DOCS_DIR)

        report = _check(submission)

        # The missing directory, and once (not once per point) the pointer naming it.
        assert sorted(e.rule for e in report.errors) == ["required-dir", "shared-path-resolution"]
        (pointer,) = _results(report, "shared-path-resolution", Severity.ERROR)
        assert "shared_docs 'docs'" in pointer.message and "7 point(s)" in pointer.message
        # Everything the clean run checked, it still checks.
        assert _rules(baseline) <= _rules(report)

    def test_missing_results_still_checks_src(self, submission: Path) -> None:
        shutil.rmtree(submission / layout.RESULTS_DIR)

        report = _check(submission)

        assert _results(report, "required-dir", Severity.ERROR)
        assert _results(report, "src-dir", Severity.INFO)

    def test_v0_7_layout_is_named_as_such(self, tmp_path: Path) -> None:
        root = tmp_path / "AMD"
        for name in ("pareto", "documentation", "systems", "src"):
            (root / name).mkdir(parents=True)

        report = _check(root)

        messages = [e.message for e in _results(report, "required-dir", Severity.ERROR)]
        assert len(messages) == 2
        assert all("v0.7 layout" in m and "pareto/" in m for m in messages)

    def test_v1_0_layout_gets_no_legacy_hint(self, submission: Path) -> None:
        shutil.rmtree(submission / layout.DOCS_DIR)
        assert all("v0.7" not in e.message for e in _check(submission).errors)


@pytest.mark.unit
class TestSrcCheckedOnce:
    def test_src_is_reported_once_however_many_curves(self, submission: Path) -> None:
        model_dir = _curve(submission)
        shutil.copytree(
            model_dir, submission / layout.RESULTS_DIR / "second_system" / model_dir.name
        )

        report = _check(submission)

        assert len(list(layout.iter_curves(submission / layout.RESULTS_DIR))) == 2
        assert len(_results(report, "src-dir")) == 1


@pytest.mark.unit
@pytest.mark.parametrize("model", ["kimi-k3", "qwen3_6-35b-a3b", "deepseek-v4_1-flash"])
def test_agentic_model_rejects_single_turn_runs(submission: Path, model: str) -> None:
    model_dir = _curve(submission)
    model_dir = model_dir.rename(model_dir.parent / model)
    for path in model_dir.rglob(layout.POINT_YAML):
        data = yaml.safe_load(path.read_text())
        data["model_name"] = model
        path.write_text(yaml.safe_dump(data))

    report = _check(submission)

    assert not report.passed
    assert _results(report, "benchmark-type-consistency", Severity.ERROR)


@pytest.mark.unit
class TestInvalidSystemDescription:
    def test_bad_division_does_not_hide_the_curve(self, submission: Path, baseline: Report) -> None:
        _edit_system_descs(submission, division="Standardised")

        report = _check(submission)

        assert {e.rule for e in report.errors} == {"system-description-valid"}
        # Regions still derive from the raw max_supported_concurrency...
        assert _results(report, "region-basis", Severity.INFO)
        # ...so the per-point and coverage rules all still run.
        lost = _rules(baseline) - _rules(report)
        assert lost <= {"system-description-valid"}, lost

    def test_unreadable_c_max_skips_only_the_region_rules(self, submission: Path) -> None:
        _edit_system_descs(submission, division="Standardised", max_supported_concurrency=None)

        report = _check(submission)

        assert _results(report, "region-basis", Severity.WARNING)
        assert not _results(report, "low-concurrency-coverage")  # needs regions
        # Rules that need neither the description nor the regions still run.
        assert _results(report, "point-count")
        assert _results(report, "ultra-low-concurrency-coverage")
        assert _results(report, "steady-state-valid")


@pytest.mark.unit
class TestInvalidPointConfig:
    """One bad field in r16's point.yaml — the case that used to invent errors."""

    @pytest.fixture
    def report(self, submission: Path) -> Report:
        path = _curve(submission) / "r16" / layout.POINT_YAML
        data = yaml.safe_load(path.read_text())
        data["runtime_settings"]["min_duration_ms"] = "20 minutes"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return _check(submission)

    def test_only_the_bad_field_is_an_error(self, report: Report) -> None:
        assert {e.rule for e in report.errors} == {"point-config-valid"}

    def test_the_point_keeps_its_place_in_the_curve(self, report: Report) -> None:
        basis = _results(report, "region-basis")
        assert basis and "C_min = 16" in basis[0].message
        # r16 still covers Low Concurrency and carries its accuracy results.
        assert not _results(report, "low-concurrency-coverage", Severity.ERROR)
        assert not _results(report, "accuracy-coverage", Severity.ERROR)

    def test_its_result_summary_is_still_checked(self, submission: Path) -> None:
        point_dir = _curve(submission) / "r16"
        (point_dir / layout.POINT_YAML).write_text("{not: valid: yaml [")
        (point_dir / layout.RESULT_SUMMARY_JSON).write_text("[]")

        report = _check(submission)

        assert _results(report, "result-file-valid", Severity.ERROR)

    def test_it_says_which_rules_it_could_not_run(self, report: Report) -> None:
        (skipped,) = _results(report, "point-rules-skipped", Severity.WARNING)
        assert "r16/" in skipped.message
