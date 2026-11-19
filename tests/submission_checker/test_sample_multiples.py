# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Issued samples must cover whole dataset passes (§6.4)."""

from pathlib import Path

import pytest

from submission_checker.models import PointConfig, PointResult, RuntimeSettings, Severity

from .conftest import _summary


def _check(tmp_path: Path, issued: int, concurrency: int = 64, dataset: str = "aime25", **runtime):
    runtime = {"n_samples_from_dataset": 30, **runtime}
    config = PointConfig(
        concurrency=concurrency,
        dataset=dataset,
        runtime_settings=RuntimeSettings(runtime=RuntimeSettings.Runtime(), **runtime),
    )
    result = PointResult.model_validate(
        {
            "config": config,
            "summary": _summary(n_issued=issued, n_completed=issued),
            "yaml_path": tmp_path / "point.yaml",
        },
        context={"summary_path": tmp_path / "result_summary.json"},
    )
    return [r for r in result._check_results if r.rule == "dataset-pass-multiple"]


@pytest.mark.unit
@pytest.mark.parametrize("issued", [-30, 0, 37, 51, 67])
def test_rejects_nonpositive_or_partial_passes(tmp_path, issued):
    results = _check(tmp_path, issued)
    assert any(r.severity == Severity.ERROR for r in results)


@pytest.mark.unit
@pytest.mark.parametrize("issued", [-30, 0])
def test_rejects_nonpositive_issued_count_without_dataset_size(tmp_path, issued):
    results = _check(tmp_path, issued, n_samples_from_dataset=None)
    assert any(r.severity == Severity.ERROR for r in results)
    assert not any(r.severity == Severity.INFO for r in results)


@pytest.mark.unit
@pytest.mark.parametrize("issued", [30, 60, 90])
def test_accepts_positive_whole_passes(tmp_path, issued):
    results = _check(tmp_path, issued)
    assert any(r.severity == Severity.INFO for r in results)
    assert not any(r.severity == Severity.ERROR for r in results)


@pytest.mark.unit
def test_low_concurrency_uses_recorded_subset_size(tmp_path):
    results = _check(tmp_path, 20, concurrency=32, n_samples_from_dataset=10)
    assert any(r.severity == Severity.INFO for r in results)
    assert not any(r.severity == Severity.ERROR for r in results)


@pytest.mark.unit
def test_partial_subset_pass_fails(tmp_path):
    results = _check(tmp_path, 25, concurrency=32, n_samples_from_dataset=10)
    assert any(r.severity == Severity.ERROR for r in results)


@pytest.mark.unit
def test_partial_pass_above_32_is_rejected(tmp_path):
    results = _check(tmp_path, 20, concurrency=33)
    assert any(r.severity == Severity.ERROR for r in results)


@pytest.mark.unit
def test_declared_size_supports_unmapped_dataset(tmp_path):
    results = _check(tmp_path, 40, dataset="custom", n_samples_from_dataset=20)
    assert any(r.severity == Severity.INFO for r in results)


@pytest.mark.unit
def test_unknown_size_reports_that_check_could_not_run(tmp_path):
    results = _check(tmp_path, 40, dataset="custom", n_samples_from_dataset=None)
    assert any(r.severity == Severity.WARNING for r in results)
    assert not any(r.severity == Severity.INFO for r in results)
    assert not any(r.severity == Severity.ERROR for r in results)


@pytest.mark.unit
def test_zero_declared_dataset_size_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RuntimeSettings(runtime=RuntimeSettings.Runtime(), n_samples_from_dataset=0)


@pytest.mark.unit
@pytest.mark.parametrize("size", [10, None])
def test_missing_issued_count_is_reported(tmp_path, size):
    config = PointConfig(
        concurrency=32,
        dataset="aime25",
        runtime_settings=RuntimeSettings(
            runtime=RuntimeSettings.Runtime(), n_samples_from_dataset=size
        ),
    )
    result = PointResult(
        config=config,
        summary=_summary(n_issued=20, n_completed=20),
        yaml_path=tmp_path / "point.yaml",
    )
    summary = result.summary.model_dump(exclude={"n_samples_issued"})
    result = PointResult(config=config, summary=summary, yaml_path=tmp_path / "point.yaml")
    checks = [r for r in result._check_results if r.rule == "dataset-pass-multiple"]
    assert any(r.severity == Severity.WARNING for r in checks)
    assert not any(r.severity == Severity.INFO for r in checks)
    assert not any(r.severity == Severity.ERROR for r in checks)
