# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for models.file validators — per-file rule checks."""

from __future__ import annotations

import pytest

from submission_checker.models import (
    AccuracyResult,
    PointConfig,
    Severity,
)

from .conftest import _REGIONS


@pytest.mark.unit
class TestLoadPatternValidator:
    def test_wrong_type(self, tmp_path):
        config = PointConfig.model_validate(
            {"concurrency": 64, "runtime_settings": {"load_pattern": "qps"}},
            context={"yaml_path": tmp_path / "point_64.yaml"},
        )
        errors = [
            r
            for r in config._check_results
            if r.severity == Severity.ERROR and r.rule == "load-pattern"
        ]
        assert errors

    def test_missing_target_concurrency(self, tmp_path):
        config = PointConfig.model_validate(
            {"concurrency": 0, "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_0.yaml"},
        )
        errors = [
            r
            for r in config._check_results
            if r.severity == Severity.ERROR and r.rule == "load-pattern"
        ]
        assert errors

    def test_valid(self, tmp_path):
        config = PointConfig.model_validate(
            {"concurrency": 64, "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_64.yaml"},
        )
        assert all(
            r.severity != Severity.ERROR for r in config._check_results if r.rule == "load-pattern"
        )


@pytest.mark.unit
class TestStreamingValidator:
    def test_stream_false_errors(self, tmp_path):
        config = PointConfig.model_validate(
            {"concurrency": 64, "runtime_settings": {"load_pattern": "concurrency", "stream_all_chunks": False}},
            context={"yaml_path": tmp_path / "point_64.yaml"},
        )
        errors = [
            r
            for r in config._check_results
            if r.rule == "streaming-config" and r.severity == Severity.ERROR
        ]
        assert errors

    def test_stream_true_passes(self, tmp_path):
        config = PointConfig.model_validate(
            {"concurrency": 64, "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_64.yaml"},
        )
        assert all(
            r.severity != Severity.ERROR
            for r in config._check_results
            if r.rule == "streaming-config"
        )


@pytest.mark.unit
class TestConcurrencyInRangeValidator:
    def test_out_of_range(self, tmp_path):
        config = PointConfig.model_validate(
            {"concurrency": 9999, "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_9999.yaml", "regions": _REGIONS},
        )
        errors = [
            r
            for r in config._check_results
            if r.rule == "concurrency-in-range" and r.severity == Severity.ERROR
        ]
        assert errors

    def test_in_range(self, tmp_path):
        config = PointConfig.model_validate(
            {"concurrency": 64, "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_64.yaml", "regions": _REGIONS},
        )
        assert all(
            r.severity != Severity.ERROR
            for r in config._check_results
            if r.rule == "concurrency-in-range"
        )

    def test_no_regions_skips_check(self, tmp_path):
        config = PointConfig.model_validate(
            {"concurrency": 9999, "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_9999.yaml"},
        )
        # No regions in context — concurrency-in-range should not be present
        rules = {r.rule for r in config._check_results}
        assert "concurrency-in-range" not in rules


@pytest.mark.unit
class TestRegionDeclaredValidator:
    def test_absent_region_no_check(self, tmp_path):
        """No region-declared result when region field is omitted."""
        config = PointConfig.model_validate(
            {"concurrency": 64, "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_64.yaml", "regions": _REGIONS},
        )
        assert not any(r.rule == "region-declared" for r in config._check_results)

    def test_invalid_region_value_errors(self, tmp_path):
        """An unrecognised region string must produce a region-declared error."""
        config = PointConfig.model_validate(
            {"concurrency": 64, "region": "not_a_region", "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_64.yaml"},
        )
        assert any(
            r.rule == "region-declared" and r.severity == Severity.ERROR
            for r in config._check_results
        )

    def test_valid_region_matches_computed(self, tmp_path):
        """Declared region matching the computed region produces an ok result."""
        # concurrency=64 → med_throughput for M=1024
        config = PointConfig.model_validate(
            {"concurrency": 64, "region": "med_throughput", "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_64.yaml", "regions": _REGIONS},
        )
        assert any(
            r.rule == "region-declared" and r.severity != Severity.ERROR
            for r in config._check_results
        )

    def test_region_mismatch_warns(self, tmp_path):
        """Declared region that doesn't match the computed region produces a warning."""
        # concurrency=64 → med_throughput, but we declare low_latency
        config = PointConfig.model_validate(
            {"concurrency": 64, "region": "low_latency", "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_64.yaml", "regions": _REGIONS},
        )
        assert any(
            r.rule == "region-declared" and r.severity == Severity.WARNING
            for r in config._check_results
        )

    def test_submitters_choice_no_cross_check(self, tmp_path):
        """submitters_choice is valid for any concurrency — no cross-check performed."""
        config = PointConfig.model_validate(
            {"concurrency": 64, "region": "submitters_choice", "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_64.yaml", "regions": _REGIONS},
        )
        assert any(
            r.rule == "region-declared" and r.severity != Severity.ERROR
            for r in config._check_results
        )
        assert not any(
            r.rule == "region-declared" and r.severity == Severity.WARNING
            for r in config._check_results
        )

    def test_valid_region_no_regions_context(self, tmp_path):
        """Valid region string without regions context emits ok without cross-check."""
        config = PointConfig.model_validate(
            {"concurrency": 64, "region": "high_throughput", "runtime_settings": {"load_pattern": "concurrency"}},
            context={"yaml_path": tmp_path / "point_64.yaml"},
        )
        assert any(
            r.rule == "region-declared" and r.severity != Severity.ERROR
            for r in config._check_results
        )


@pytest.mark.unit
class TestAccuracyResultModel:
    def test_scores_accessible(self):
        ar = AccuracyResult({
            "cnn_dailymail::llama3_8b": {"score": {"rouge1": "38.73", "rouge2": "16.10"}}
        })
        scores = ar.metric_scores()
        assert scores["cnn_dailymail::llama3_8b"]["rouge1"] == pytest.approx(38.73)

    def test_no_check_results_on_valid_input(self):
        ar = AccuracyResult({"ds": {"score": {"rouge1": "38.73"}}})
        assert not any(r.severity == Severity.ERROR for r in ar._check_results)

    def test_empty_dict_emits_accuracy_valid_error(self):
        ar = AccuracyResult({})
        assert any(
            r.rule == "accuracy-valid" and r.severity == Severity.ERROR
            for r in ar._check_results
        )
