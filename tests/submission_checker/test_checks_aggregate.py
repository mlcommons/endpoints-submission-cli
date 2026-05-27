# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for models.aggregate validators — cross-file rule checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from submission_checker.models import (
    AccuracyResult,
    MIN_QUERY_COUNT,
    ModelContext,
    PercentileStats,
    PointConfig,
    PointResult,
    PointSummary,
    RuntimeSettings,
    Severity,
)

from .conftest import (
    _REGIONS,
    _config,
    _model_ctx,
    _passed,
    _summary,
    _system_desc,
)

# ---------------------------------------------------------------------------
# Local helpers (specific to aggregate tests)
# ---------------------------------------------------------------------------

# Derived values for the default _summary():
#   total_output_tokens = 500_000, duration_ns = 1_200_000_000_000 (1200 s)
#   system_tps = 500_000 / 1200 ≈ 416.667 tok/s
#   tps_per_user (concurrency=64) ≈ 6.5104 tok/s/user


def _summary_with(**extras) -> PointSummary:
    """Return the default summary but with extra fields stored for consistency checks."""
    return PointSummary(
        n_samples_completed=1000,
        n_samples_issued=1000,
        n_samples_failed=0,
        duration_ns=1_200_000_000_000.0,
        ttft=PercentileStats(total=0.0, percentiles={"50": 150_000_000.0, "95": 300_000_000.0}),
        output_sequence_lengths=PercentileStats(total=500_000.0),
        **extras,
    )


def _config_with_dataset(dataset: str, concurrency: int = 64) -> PointConfig:
    return PointConfig(
        concurrency=concurrency,
        dataset=dataset,
        runtime_settings=RuntimeSettings(min_duration_ms=1_200_000),
    )


# ---------------------------------------------------------------------------
# PointResult validators
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunDurationValidator:
    def test_short_duration_warns(self, tmp_path):
        config = _config(concurrency=64)
        short = _summary(duration_ns=100_000_000_000.0)  # 100 s — below any minimum
        run_result = PointResult.model_validate(
            {"config": config, "summary": short, "yaml_path": tmp_path / "run_64.yaml"},
            context={"regions": _REGIONS, "summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "point-duration" and r.severity == Severity.WARNING
            for r in run_result._check_results
        )

    def test_sufficient_duration_passes(self, tmp_path):
        config = _config(concurrency=64)
        long_s = _summary(duration_ns=1_200_000_000_000.0)  # 1200 s
        run_result = PointResult.model_validate(
            {"config": config, "summary": long_s, "yaml_path": tmp_path / "run_64.yaml"},
            context={"regions": _REGIONS, "summary_path": tmp_path / "summary.json"},
        )
        assert all(
            r.severity != Severity.ERROR
            for r in run_result._check_results
            if r.rule == "point-duration"
        )

    def test_out_of_range_concurrency_skipped(self, tmp_path):
        config = _config(concurrency=9999)
        run_result = PointResult.model_validate(
            {"config": config, "summary": _summary(), "yaml_path": tmp_path / "run_9999.yaml"},
            context={"regions": _REGIONS, "summary_path": tmp_path / "summary.json"},
        )
        # run-duration should not appear because concurrency is out of range
        assert not any(r.rule == "point-duration" for r in run_result._check_results)

    def test_no_regions_skips(self, tmp_path):
        config = _config(concurrency=64)
        run_result = PointResult.model_validate(
            {"config": config, "summary": _summary(), "yaml_path": tmp_path / "run_64.yaml"},
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert not any(r.rule == "point-duration" for r in run_result._check_results)


@pytest.mark.unit
class TestMetricConsistencyValidator:
    def test_zero_duration_errors(self, tmp_path):
        config = _config(concurrency=64)
        run_result = PointResult.model_validate(
            {
                "config": config,
                "summary": _summary(duration_ns=0.0),
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-duration" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )

    def test_negative_duration_errors(self, tmp_path):
        config = _config(concurrency=64)
        run_result = PointResult.model_validate(
            {
                "config": config,
                "summary": _summary(duration_ns=-1.0),
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-duration" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )

    def test_accounting_mismatch_errors(self, tmp_path):
        config = _config(concurrency=64)
        run_result = PointResult.model_validate(
            {
                "config": config,
                "summary": _summary(n_completed=990, n_issued=1000, n_failed=5),
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        # completed(990) + failed(5) = 995 ≠ issued(1000)
        assert any(
            r.rule == "metric-consistency-accounting" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )

    def test_valid_summary_passes(self, tmp_path):
        config = _config(concurrency=64)
        run_result = PointResult.model_validate(
            {"config": config, "summary": _summary(), "yaml_path": tmp_path / "run_64.yaml"},
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert all(r.severity != Severity.ERROR for r in run_result._check_results)

    def test_negative_output_tokens_errors(self, tmp_path):
        """Negative output token total must produce a metric-consistency-output-tokens error."""
        config = _config(concurrency=64)
        run_result = PointResult.model_validate(
            {
                "config": config,
                "summary": _summary(total_tokens=-1.0),
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-output-tokens" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )


@pytest.mark.unit
class TestTpsConsistencyValidator:
    def test_system_tps_derivable_ok(self, tmp_path):
        run_result = PointResult.model_validate(
            {"config": _config(concurrency=64), "summary": _summary(), "yaml_path": tmp_path / "run_64.yaml"},
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-system-tps" and r.severity != Severity.ERROR
            for r in run_result._check_results
        )

    def test_system_tps_stored_match_ok(self, tmp_path):
        """Stored system_tps matching derived value within 1% passes."""
        run_result = PointResult.model_validate(
            {
                "config": _config(concurrency=64),
                "summary": _summary_with(system_tps=416.67),  # derived ≈ 416.667
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-system-tps" and r.severity != Severity.ERROR
            for r in run_result._check_results
        )

    def test_system_tps_stored_mismatch_errors(self, tmp_path):
        """Stored system_tps differing from derived by >1% is an error."""
        run_result = PointResult.model_validate(
            {
                "config": _config(concurrency=64),
                "summary": _summary_with(system_tps=999.0),  # derived ≈ 416.667
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-system-tps" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )

    def test_tps_per_user_derivable_ok(self, tmp_path):
        run_result = PointResult.model_validate(
            {"config": _config(concurrency=64), "summary": _summary(), "yaml_path": tmp_path / "run_64.yaml"},
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-tps-per-user" and r.severity != Severity.ERROR
            for r in run_result._check_results
        )

    def test_tps_per_user_stored_match_ok(self, tmp_path):
        """Stored tps_per_user matching system_tps/concurrency within 1% passes."""
        run_result = PointResult.model_validate(
            {
                "config": _config(concurrency=64),
                "summary": _summary_with(tps_per_user=6.51),  # derived ≈ 6.5104
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-tps-per-user" and r.severity != Severity.ERROR
            for r in run_result._check_results
        )

    def test_tps_per_user_stored_mismatch_errors(self, tmp_path):
        """Stored tps_per_user differing from system_tps/concurrency by >1% is an error."""
        run_result = PointResult.model_validate(
            {
                "config": _config(concurrency=64),
                "summary": _summary_with(tps_per_user=999.0),  # derived ≈ 6.5104
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-tps-per-user" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )

    def test_tps_per_user_zero_concurrency_errors(self, tmp_path):
        """concurrency=0 must error rather than divide by zero."""
        config = PointConfig(
            concurrency=0,
            dataset="mlperf-perf-dataset-v1",
            runtime_settings=RuntimeSettings(min_duration_ms=1_200_000),
        )
        run_result = PointResult.model_validate(
            {"config": config, "summary": _summary(), "yaml_path": tmp_path / "run_64.yaml"},
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert any(
            r.rule == "metric-consistency-tps-per-user" and r.severity == Severity.ERROR
            for r in run_result._check_results
        )


@pytest.mark.unit
class TestMinQueryCountValidator:
    def test_meets_minimum_ok(self, tmp_path):
        for dataset, min_q in MIN_QUERY_COUNT.items():
            run_result = PointResult.model_validate(
                {
                    "config": _config_with_dataset(dataset),
                    "summary": _summary(n_completed=min_q),
                    "yaml_path": tmp_path / "run_64.yaml",
                },
                context={"summary_path": tmp_path / "summary.json"},
            )
            assert any(
                r.rule == "min-query-count" and r.severity != Severity.ERROR
                for r in run_result._check_results
            ), f"Expected ok for dataset '{dataset}' with {min_q} completed"

    def test_below_minimum_errors(self, tmp_path):
        for dataset, min_q in MIN_QUERY_COUNT.items():
            if min_q == 0:
                continue
            run_result = PointResult.model_validate(
                {
                    "config": _config_with_dataset(dataset),
                    "summary": _summary(n_completed=min_q - 1),
                    "yaml_path": tmp_path / "run_64.yaml",
                },
                context={"summary_path": tmp_path / "summary.json"},
            )
            assert any(
                r.rule == "min-query-count" and r.severity == Severity.ERROR
                for r in run_result._check_results
            ), f"Expected error for dataset '{dataset}' with {min_q - 1} completed"

    def test_unknown_dataset_skipped(self, tmp_path):
        run_result = PointResult.model_validate(
            {
                "config": _config_with_dataset("unknown-dataset-xyz"),
                "summary": _summary(n_completed=0),
                "yaml_path": tmp_path / "run_64.yaml",
            },
            context={"summary_path": tmp_path / "summary.json"},
        )
        assert not any(r.rule == "min-query-count" for r in run_result._check_results)

    def test_dataset_a_boundary(self, tmp_path):
        """dataset-a requires exactly 1 query — 0 fails, 1 passes."""
        base = {"yaml_path": tmp_path / "run_64.yaml"}
        ctx = {"summary_path": tmp_path / "summary.json"}

        fail = PointResult.model_validate(
            {"config": _config_with_dataset("dataset-a"), "summary": _summary(n_completed=0), **base}, context=ctx
        )
        assert any(r.rule == "min-query-count" and r.severity == Severity.ERROR for r in fail._check_results)

        ok_result = PointResult.model_validate(
            {"config": _config_with_dataset("dataset-a"), "summary": _summary(n_completed=1), **base}, context=ctx
        )
        assert any(r.rule == "min-query-count" and r.severity != Severity.ERROR for r in ok_result._check_results)

    def test_dataset_c_boundary(self, tmp_path):
        """dataset-c requires 100 queries — 99 fails, 100 passes."""
        base = {"yaml_path": tmp_path / "run_64.yaml"}
        ctx = {"summary_path": tmp_path / "summary.json"}

        fail = PointResult.model_validate(
            {"config": _config_with_dataset("dataset-c"), "summary": _summary(n_completed=99), **base}, context=ctx
        )
        assert any(r.rule == "min-query-count" and r.severity == Severity.ERROR for r in fail._check_results)

        ok_result = PointResult.model_validate(
            {"config": _config_with_dataset("dataset-c"), "summary": _summary(n_completed=100), **base}, context=ctx
        )
        assert any(r.rule == "min-query-count" and r.severity != Severity.ERROR for r in ok_result._check_results)


# ---------------------------------------------------------------------------
# ModelContext validators
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunCountValidator:
    def test_too_few(self, tmp_path):
        ctx = _model_ctx(tmp_path, all_point_count=3)
        assert any(
            r.rule == "point-count" and r.severity == Severity.ERROR for r in ctx._check_results
        )

    def test_cap_exceeded(self, tmp_path):
        ctx = _model_ctx(tmp_path, all_point_count=33)
        assert any(r.rule == "point-cap" and r.severity == Severity.ERROR for r in ctx._check_results)

    def test_valid_count(self, tmp_path):
        ctx = _model_ctx(tmp_path, all_point_count=10)
        assert all(
            r.severity != Severity.ERROR
            for r in ctx._check_results
            if r.rule in ("point-count", "point-cap")
        )


@pytest.mark.unit
class TestRegionalCoverageValidator:
    def test_no_runs_all_regions_missing(self, tmp_path):
        ctx = _model_ctx(tmp_path, valid_points=[])
        coverage_rules = {
            "low-latency-coverage",
            "low-throughput-coverage",
            "med-throughput-coverage",
            "high-throughput-coverage",
        }
        errors = {
            r.rule
            for r in ctx._check_results
            if r.severity == Severity.ERROR and r.rule in coverage_rules
        }
        assert errors == coverage_rules

    def test_concurrency_in_low_latency(self, tmp_path):
        yaml_path = tmp_path / "llama3-70b" / "points" / "point_16.yaml"
        valid_points = [(yaml_path, _config(concurrency=16))]
        ctx = _model_ctx(tmp_path, valid_points=valid_points)
        assert all(
            r.severity != Severity.ERROR
            for r in ctx._check_results
            if r.rule == "low-latency-coverage"
        )


@pytest.mark.unit
class TestConfigConsistencyValidator:
    def test_inconsistent_datasets(self, tmp_path):
        c1 = PointConfig(
            concurrency=64,
            dataset="dataset-a",
            runtime_settings=RuntimeSettings(),
        )
        c2 = PointConfig(
            concurrency=128,
            dataset="dataset-b",
            runtime_settings=RuntimeSettings(),
        )
        s = _summary()
        ctx = _model_ctx(tmp_path, loaded_points=[(c1, s), (c2, s)])
        assert any(
            r.rule == "config-consistency-dataset" and r.severity == Severity.ERROR
            for r in ctx._check_results
        )

    def test_model_name_mismatch(self, tmp_path):
        model_dir = tmp_path / "actual-name"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "points").mkdir(exist_ok=True)
        (model_dir / "results").mkdir(exist_ok=True)
        (model_dir / "accuracy").mkdir(exist_ok=True)
        s = _summary()
        ctx = ModelContext(
            system_id="test-sys",
            system_desc=_system_desc(benchmark_model="expected-name"),
            model_dir=model_dir,
            regions=_REGIONS,
            points_dir=model_dir / "points",
            accuracy_dir=model_dir / "accuracy",
            all_point_count=7,
            valid_points=[],
            loaded_points=[(_config(), s)],
            accuracy_result=None,
        )
        assert any(
            r.rule == "config-consistency-model" and r.severity == Severity.WARNING
            for r in ctx._check_results
        )

    def test_empty_results_skips(self, tmp_path):
        ctx = _model_ctx(tmp_path, loaded_points=[])
        # config-consistency-dataset should not appear when loaded_points is empty
        assert not any(r.rule == "config-consistency-dataset" for r in ctx._check_results)


@pytest.mark.unit
class TestAccuracyGateValidator:
    def test_no_accuracy_result_skips(self, tmp_path):
        ctx = _model_ctx(tmp_path, accuracy_result=None)
        assert not any(r.rule == "accuracy-gate" for r in ctx._check_results)

    def test_passed_accuracy_gate(self, tmp_path):
        ar = AccuracyResult(metric="rouge1", score=0.45, quality_target=0.43, passed=True)
        ctx = _model_ctx(tmp_path, accuracy_result=ar)
        assert any(
            r.rule == "accuracy-gate" and r.severity == Severity.INFO for r in ctx._check_results
        )
        assert not any(
            r.rule == "accuracy-gate" and r.severity == Severity.ERROR for r in ctx._check_results
        )

    def test_failed_accuracy_gate(self, tmp_path):
        ar = AccuracyResult(metric="rouge1", score=0.30, quality_target=0.43, passed=False)
        ctx = _model_ctx(tmp_path, accuracy_result=ar)
        assert any(
            r.rule == "accuracy-gate" and r.severity == Severity.ERROR for r in ctx._check_results
        )
