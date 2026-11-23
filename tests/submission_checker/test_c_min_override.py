"""Optional C_min declarations, legacy defaults, and accuracy assignment (§5.4)."""

from __future__ import annotations

import json
import shutil

import pytest
from pydantic import ValidationError

from submission_checker import layout
from submission_checker.checker import SubmissionChecker
from submission_checker.models import ModelContext, Severity, SystemDescription
from submission_checker.models.aggregate.context import accuracy_bands
from submission_checker.models.aggregate.point_placement import RegionPlacement
from submission_checker.models.regions import compute_regions
from submission_checker.models.run_requirements import min_duration_ms

from .conftest import _config, _system_desc
from .test_checks_agentic_accuracy import _accuracy, _agentic_config


@pytest.mark.parametrize("value", [None, True, False, 16.0, "16", 0, -1, 33, [], {}])
def test_override_rejects_non_json_integers_and_out_of_range(value):
    with pytest.raises(ValidationError, match="c_min_override"):
        _system_desc(c_min_override=value)


@pytest.mark.parametrize("value", [1, 16, 32])
def test_override_round_trips(value):
    desc = _system_desc(c_min_override=value)
    assert desc.c_min_override == value
    assert SystemDescription.model_validate_json(desc.model_dump_json()).c_min_override == value


def test_omitted_override_stays_omitted():
    desc = _system_desc()
    assert desc.c_min_override is None
    assert "c_min_override" not in desc.model_dump()
    assert "c_min_override" not in json.loads(desc.model_dump_json())
    assert SystemDescription.model_validate(desc.model_dump()).c_min_override is None


@pytest.fixture
def curve(tmp_path, valid_standardized):
    root = tmp_path / "submission"
    shutil.copytree(valid_standardized, root)
    ((_, model),) = layout.iter_curves(root / "results")
    return root, layout.iter_point_dirs(model)


def declare(points, value):
    for point in points:
        path = point / "system_desc.json"
        data = json.loads(path.read_text())
        data["c_min_override"] = value
        path.write_text(json.dumps(data))


def hits(report, rule, severity=None):
    return [
        r for r in report.results if r.rule == rule and (severity is None or r.severity == severity)
    ]


def test_override_can_select_a_measured_point_above_the_minimum(curve):
    root, points = curve
    declare(points, 20)
    report = SubmissionChecker(root).run()
    assert not hits(report, "c-min-override", Severity.ERROR)
    assert "C_min = 20 declared" in hits(report, "region-basis")[0].message
    assert "≤ 20" in hits(report, "ultra-low-concurrency-coverage")[0].message


@pytest.mark.parametrize("value", [16, 20])
def test_either_measured_boundary_is_valid(curve, value):
    root, points = curve
    declare(points, value)
    report = SubmissionChecker(root).run()
    assert not hits(report, "c-min-override", Severity.ERROR)
    assert f"C_min = {value} declared" in hits(report, "region-basis")[0].message


def test_override_requires_matching_point(curve):
    root, points = curve
    declare(points, 24)
    report = SubmissionChecker(root).run()
    assert "must match" in hits(report, "c-min-override", Severity.ERROR)[0].message
    assert not hits(report, "region-basis")


@pytest.mark.parametrize("other", [None, True, 20.0, "20", 16, 33])
def test_every_raw_description_is_validated_without_falling_back(curve, other):
    root, points = curve
    declare(points, 20)
    declare([points[-1]], other)
    report = SubmissionChecker(root).run()
    assert hits(report, "c-min-override", Severity.ERROR)
    assert not hits(report, "region-basis")


def test_mixed_omission_is_rejected(curve):
    root, points = curve
    declare(points[1:], 20)
    report = SubmissionChecker(root).run()
    assert hits(report, "c-min-override", Severity.ERROR)
    assert not hits(report, "region-basis")


def test_unrelated_invalid_description_does_not_erase_override(curve):
    root, points = curve
    declare(points, 20)
    for point in points:
        path = point / "system_desc.json"
        data = json.loads(path.read_text())
        data.pop("system_name")
        path.write_text(json.dumps(data))
    report = SubmissionChecker(root).run()
    assert hits(report, "system-description-valid", Severity.ERROR)
    assert "C_min = 20 declared" in hits(report, "region-basis")[0].message
    assert "≤ 20" in hits(report, "ultra-low-concurrency-coverage")[0].message


def test_matching_point_with_invalid_metadata_keeps_boundary(curve):
    root, points = curve
    declare(points, 20)
    point = next(p for p in points if p.name == "r20")
    (point / "point.yaml").write_text("{broken [")
    report = SubmissionChecker(root).run()
    assert hits(report, "point-config-valid", Severity.ERROR)
    assert "C_min = 20 declared" in hits(report, "region-basis")[0].message


def test_curves_choose_independent_overrides(curve):
    root, points = curve
    other = points[0].parent.with_name("other-model")
    shutil.copytree(points[0].parent, other)
    declare(points, 16)
    declare(layout.iter_point_dirs(other), 20)
    report = SubmissionChecker(root).run()
    assert not hits(report, "c-min-override", Severity.ERROR)
    assert {r.message for r in hits(report, "region-basis")} == {
        f"C_min = {v} declared by c_min_override; Ultra Low Concurrency spans 1–{v}"
        for v in (16, 20)
    }


@pytest.mark.parametrize(
    "overridden, expected",
    [(False, ["ultra_low_concurrency", "low_concurrency"]), (True, ["low_concurrency"])],
)
def test_accuracy_range_uses_override_only_when_declared(overridden, expected):
    regions = compute_regions(1024, 16, c_min_overridden=overridden)
    assert accuracy_bands(24, regions) == expected
    assert accuracy_bands(16, regions) == ["ultra_low_concurrency"]
    assert min_duration_ms(24, agentic=False) == 600_000
    assert min_duration_ms(24, agentic=True) == 600_000


@pytest.mark.parametrize("accuracy_point", [16, 32])
def test_override_accepts_choice_point_and_accuracy_anywhere_in_ultra_low(tmp_path, accuracy_point):
    regions = compute_regions(1024, 32, c_min_overridden=True)
    concurrencies = [16, 32, 40, 64, 256, 512, 1024]
    points = [(tmp_path / f"r{c}/point.yaml", _agentic_config(c)) for c in concurrencies]
    choice = _config(concurrency=16, region="submitters_choice")
    placement = RegionPlacement(config=choice, regions=regions, yaml_path=tmp_path / "point.yaml")
    assert not any(r.severity == Severity.ERROR for r in placement._check_results)
    ctx = ModelContext(
        system_id="sys",
        system_desc=_system_desc(c_min_override=32),
        model_dir=tmp_path / "kimi-k3",
        regions=regions,
        points_dir=tmp_path,
        all_point_count=7,
        valid_points=points,
        loaded_points=[],
        accuracy_by_point={c: _accuracy(swebench=96) for c in [accuracy_point, 40, 64, 256]},
    )
    assert hits(ctx_report(ctx), "accuracy-coverage", Severity.INFO)
    assert not hits(ctx_report(ctx), "accuracy-coverage", Severity.ERROR)
    assert hits(ctx_report(ctx), "agentic-accuracy-swebench", Severity.INFO)


def ctx_report(ctx):
    from types import SimpleNamespace

    return SimpleNamespace(results=ctx._check_results)


def test_legacy_overlap_uses_same_assignment_for_accuracy_and_mean(tmp_path):
    points = [(tmp_path / f"r{c}/point.yaml", _agentic_config(c)) for c in [16, 24, 64, 256]]
    ctx = ModelContext(
        system_id="sys",
        system_desc=_system_desc(),
        model_dir=tmp_path / "kimi-k3",
        regions=compute_regions(1024, 16),
        points_dir=tmp_path,
        all_point_count=4,
        valid_points=points,
        loaded_points=[],
        accuracy_by_point={
            c: _accuracy(swebench=s) for c, s in [(16, 90), (24, 100), (64, 94), (256, 94)]
        },
    )
    report = ctx_report(ctx)
    assert not hits(report, "accuracy-coverage", Severity.ERROR)
    (mean,) = hits(report, "agentic-accuracy-swebench")
    assert mean.severity == Severity.INFO
    assert "mean of 4 bands" in mean.message
    assert "94.50" in mean.message
