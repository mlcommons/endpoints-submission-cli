# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for models.aggregate validators — cross-file rule checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from submission_checker.models import (
    MIN_QUERY_COUNT,
    AccuracyResult,
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
    _summary,
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
            {
                "config": _config(concurrency=64),
                "summary": _summary(),
                "yaml_path": tmp_path / "run_64.yaml",
            },
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
            {
                "config": _config(concurrency=64),
                "summary": _summary(),
                "yaml_path": tmp_path / "run_64.yaml",
            },
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
            {
                "config": _config_with_dataset("dataset-a"),
                "summary": _summary(n_completed=0),
                **base,
            },
            context=ctx,
        )
        assert any(
            r.rule == "min-query-count" and r.severity == Severity.ERROR
            for r in fail._check_results
        )

        ok_result = PointResult.model_validate(
            {
                "config": _config_with_dataset("dataset-a"),
                "summary": _summary(n_completed=1),
                **base,
            },
            context=ctx,
        )
        assert any(
            r.rule == "min-query-count" and r.severity != Severity.ERROR
            for r in ok_result._check_results
        )

    def test_dataset_c_boundary(self, tmp_path):
        """dataset-c requires 100 queries — 99 fails, 100 passes."""
        base = {"yaml_path": tmp_path / "run_64.yaml"}
        ctx = {"summary_path": tmp_path / "summary.json"}

        fail = PointResult.model_validate(
            {
                "config": _config_with_dataset("dataset-c"),
                "summary": _summary(n_completed=99),
                **base,
            },
            context=ctx,
        )
        assert any(
            r.rule == "min-query-count" and r.severity == Severity.ERROR
            for r in fail._check_results
        )

        ok_result = PointResult.model_validate(
            {
                "config": _config_with_dataset("dataset-c"),
                "summary": _summary(n_completed=100),
                **base,
            },
            context=ctx,
        )
        assert any(
            r.rule == "min-query-count" and r.severity != Severity.ERROR
            for r in ok_result._check_results
        )


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
        assert any(
            r.rule == "point-cap" and r.severity == Severity.ERROR for r in ctx._check_results
        )

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

    def test_empty_results_skips(self, tmp_path):
        ctx = _model_ctx(tmp_path, loaded_points=[])
        # config-consistency-dataset should not appear when loaded_points is empty
        assert not any(r.rule == "config-consistency-dataset" for r in ctx._check_results)


@pytest.mark.unit
class TestAccuracyGateValidator:
    def test_no_accuracy_result_skips(self, tmp_path):
        ctx = _model_ctx(tmp_path, accuracy_result=None)
        assert not any(r.rule == "accuracy-gate" for r in ctx._check_results)

    def test_unknown_model_emits_warning(self, tmp_path):
        # default model_name "llama3-70b" has no known thresholds
        ar = AccuracyResult({"ds": {"score": {"rouge1": "39.0"}}})
        ctx = _model_ctx(tmp_path, accuracy_result=ar)
        assert any(
            r.rule == "accuracy-gate" and r.severity == Severity.WARNING for r in ctx._check_results
        )

    def test_passed_accuracy_gate(self, tmp_path):
        # rouge1 = 39.0 > threshold 38.3914 for llama3.1-8b
        ar = AccuracyResult(
            {
                "cnn_dailymail::llama3_8b": {
                    "score": {"rouge1": "39.0", "rouge2": "16.0", "rougeL": "25.0"}
                }
            }
        )
        ctx = _model_ctx(tmp_path, accuracy_result=ar, model_name="Llama-3_1-8B-Instruct")
        assert any(
            r.rule == "accuracy-gate" and r.severity == Severity.INFO for r in ctx._check_results
        )
        assert not any(
            r.rule == "accuracy-gate" and r.severity == Severity.ERROR for r in ctx._check_results
        )

    def test_failed_accuracy_gate(self, tmp_path):
        # rouge1 = 30.0 < threshold 38.3914 for llama3.1-8b
        ar = AccuracyResult(
            {
                "cnn_dailymail::llama3_8b": {
                    "score": {"rouge1": "30.0", "rouge2": "12.0", "rougeL": "20.0"}
                }
            }
        )
        ctx = _model_ctx(tmp_path, accuracy_result=ar, model_name="Llama-3_1-8B-Instruct")
        assert any(
            r.rule == "accuracy-gate" and r.severity == Severity.ERROR for r in ctx._check_results
        )

    def test_sample_count_passes(self, tmp_path):
        # 13368 == min_queries for llama3.1-8b → ok
        ar = AccuracyResult(
            {
                "cnn_dailymail::llama3_8b": {
                    "num_samples": 13368,
                    "score": {"rouge1": "39.0"},
                }
            }
        )
        ctx = _model_ctx(tmp_path, accuracy_result=ar, model_name="Llama-3_1-8B-Instruct")
        assert any(
            r.rule == "accuracy-sample-count" and r.severity != Severity.ERROR
            for r in ctx._check_results
        )
        assert not any(
            r.rule == "accuracy-sample-count" and r.severity == Severity.ERROR
            for r in ctx._check_results
        )

    def test_sample_count_fails(self, tmp_path):
        # 1000 < 13368 min_queries for llama3.1-8b → error
        ar = AccuracyResult(
            {
                "cnn_dailymail::llama3_8b": {
                    "num_samples": 1000,
                    "score": {"rouge1": "39.0"},
                }
            }
        )
        ctx = _model_ctx(tmp_path, accuracy_result=ar, model_name="Llama-3_1-8B-Instruct")
        assert any(
            r.rule == "accuracy-sample-count" and r.severity == Severity.ERROR
            for r in ctx._check_results
        )

    def test_sample_count_missing_skips(self, tmp_path):
        # no num_samples field → no accuracy-sample-count check
        ar = AccuracyResult(
            {
                "cnn_dailymail::llama3_8b": {
                    "score": {"rouge1": "39.0"},
                }
            }
        )
        ctx = _model_ctx(tmp_path, accuracy_result=ar, model_name="Llama-3_1-8B-Instruct")
        assert not any(r.rule == "accuracy-sample-count" for r in ctx._check_results)


# ---------------------------------------------------------------------------
# Property-based tests — metric-consistency invariants
# ---------------------------------------------------------------------------


_FAKE_PATH = Path("/fake/point.yaml")


def _make_result(
    *,
    duration_ns=1_200_000_000_000.0,
    n_completed=1000,
    n_issued=1000,
    n_failed=0,
    total_tokens=500_000.0,
    concurrency=64,
    **extra_fields,
):
    """Build a PointResult with the given parameters, injecting any extra_fields
    into PointSummary's model_extra (e.g. system_tps=, tps_per_user=).

    Uses a static fake path so the helper can be called from Hypothesis tests
    without needing a function-scoped tmp_path fixture.
    """
    summary = PointSummary(
        n_samples_completed=n_completed,
        n_samples_issued=n_issued,
        n_samples_failed=n_failed,
        duration_ns=duration_ns,
        output_sequence_lengths=PercentileStats(total=float(total_tokens)),
        **extra_fields,
    )
    config = PointConfig(
        concurrency=concurrency,
        dataset="mlperf-perf-dataset-v1",
        runtime_settings=RuntimeSettings(min_duration_ms=1_200_000),
    )
    return PointResult.model_validate(
        {"config": config, "summary": summary, "yaml_path": _FAKE_PATH},
        context={"summary_path": _FAKE_PATH.parent / "s.json"},
    )


@pytest.mark.unit
class TestMetricConsistencyProperties:
    """Property-based tests covering the boundary mutations that unit tests miss.

    Each test encodes an invariant that the implementation must satisfy for ALL
    inputs in the given domain, not just the specific examples in the unit tests
    above. A mutation that tweaks a boundary or flips a sign will violate at
    least one of these invariants and be caught by Hypothesis's shrinker.
    """

    # ------------------------------------------------------------------ duration

    @given(
        duration_ns=st.floats(min_value=1.0, max_value=1e18, allow_nan=False, allow_infinity=False)
    )
    def test_positive_duration_always_emits_ok(self, duration_ns):
        result = _make_result(duration_ns=duration_ns)
        assert any(
            r.rule == "metric-consistency-duration" and r.severity != Severity.ERROR
            for r in result._check_results
        )

    @given(duration_ns=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
    def test_nonpositive_duration_always_errors(self, duration_ns):
        result = _make_result(duration_ns=duration_ns)
        assert any(
            r.rule == "metric-consistency-duration" and r.severity == Severity.ERROR
            for r in result._check_results
        )

    # ------------------------------------------------------------------ sample accounting

    @given(
        n_completed=st.integers(min_value=0, max_value=10_000),
        n_failed=st.integers(min_value=1, max_value=1_000),
    )
    def test_correct_accounting_with_failures_always_passes(self, n_completed, n_failed):
        # n_issued set to the arithmetically correct value; n_failed > 0 so that
        # the + → - mutation produces a wrong accounted total and errors.
        n_issued = n_completed + n_failed
        result = _make_result(n_completed=n_completed, n_issued=n_issued, n_failed=n_failed)
        assert any(
            r.rule == "metric-consistency-accounting" and r.severity != Severity.ERROR
            for r in result._check_results
        )

    @given(
        n_completed=st.integers(min_value=0, max_value=10_000),
        n_failed=st.integers(min_value=0, max_value=1_000),
        n_issued=st.integers(min_value=1, max_value=20_000),
    )
    def test_wrong_accounting_always_errors(self, n_completed, n_failed, n_issued):
        assume(n_completed + n_failed != n_issued)
        result = _make_result(n_completed=n_completed, n_issued=n_issued, n_failed=n_failed)
        assert any(
            r.rule == "metric-consistency-accounting" and r.severity == Severity.ERROR
            for r in result._check_results
        )

    def test_issued_zero_skips_accounting_check(self):
        # n_samples_issued=0 means the tool didn't track dispatch count — skip the check.
        # The > 0 → >= 0 mutation would run the check and emit a spurious ok result.
        result = _make_result(n_completed=0, n_issued=0, n_failed=0)
        assert not any(r.rule == "metric-consistency-accounting" for r in result._check_results)

    def test_issued_one_runs_accounting_check(self):
        # n_samples_issued=1 with correct accounting must emit an ok result.
        # The > 0 → > 1 mutation would skip the check entirely.
        result = _make_result(n_completed=1, n_issued=1, n_failed=0)
        assert any(
            r.rule == "metric-consistency-accounting" and r.severity != Severity.ERROR
            for r in result._check_results
        )

    # ------------------------------------------------------------------ output tokens

    @given(total_tokens=st.integers(min_value=0, max_value=10**8))
    def test_nonnegative_tokens_always_emits_ok(self, total_tokens):
        result = _make_result(total_tokens=float(total_tokens))
        assert any(
            r.rule == "metric-consistency-output-tokens" and r.severity != Severity.ERROR
            for r in result._check_results
        )

    @given(total_tokens=st.integers(max_value=-1))
    def test_negative_tokens_always_errors(self, total_tokens):
        result = _make_result(total_tokens=float(total_tokens))
        assert any(
            r.rule == "metric-consistency-output-tokens" and r.severity == Severity.ERROR
            for r in result._check_results
        )

    def test_zero_tokens_passes(self):
        # total_output_tokens=0 must be ok; the < 0 → <= 0 mutation would reject it.
        result = _make_result(total_tokens=0.0)
        assert any(
            r.rule == "metric-consistency-output-tokens" and r.severity != Severity.ERROR
            for r in result._check_results
        )

    def test_duration_one_ns_passes(self):
        # duration_ns=1 must be ok; the <= 0 → <= 1 mutation would reject it.
        result = _make_result(duration_ns=1.0)
        assert any(
            r.rule == "metric-consistency-duration" and r.severity != Severity.ERROR
            for r in result._check_results
        )


@pytest.mark.unit
class TestTpsConsistencyProperties:
    """Property-based and targeted tests for the TPS formula and concurrency guard."""

    # ------------------------------------------------------------------ concurrency guard

    @given(concurrency=st.integers(min_value=1, max_value=1000))
    def test_positive_concurrency_no_guard_error(self, concurrency):
        result = _make_result(concurrency=concurrency)
        assert not any(
            r.rule == "metric-consistency-tps-per-user"
            and r.severity == Severity.ERROR
            and "not positive" in r.message
            for r in result._check_results
        )

    @given(concurrency=st.integers(max_value=0))
    def test_nonpositive_concurrency_always_errors(self, concurrency):
        result = _make_result(concurrency=concurrency)
        assert any(
            r.rule == "metric-consistency-tps-per-user" and r.severity == Severity.ERROR
            for r in result._check_results
        )

    def test_concurrency_one_ok(self):
        # concurrency=1 must pass; the <= 0 → <= 1 mutation would reject it.
        result = _make_result(concurrency=1)
        assert not any(
            r.rule == "metric-consistency-tps-per-user"
            and r.severity == Severity.ERROR
            and "not positive" in r.message
            for r in result._check_results
        )

    # ------------------------------------------------------------------ TPS formula: / vs *

    def test_stored_tps_per_user_within_tolerance_of_large_derived_passes(self):
        # derived ≈ 500_000/1200/64 ≈ 6.51. stored=6.45 is ~0.9% off (within 1% tolerance).
        # With the / → * mutation: rel_err = 0.06 * 6.51 ≈ 0.39 > 0.01 → would error.
        result = _make_result(tps_per_user=6.45)
        assert any(
            r.rule == "metric-consistency-tps-per-user" and r.severity != Severity.ERROR
            for r in result._check_results
        )

    def test_stored_system_tps_within_tolerance_of_large_derived_passes(self):
        # derived ≈ 500_000/1200 ≈ 416.67. stored=418.75 is ~0.5% off (within 1% tolerance).
        # With the / → * mutation: rel_err = 2.08 * 416.67 ≈ 867 > 0.01 → would error.
        result = _make_result(system_tps=418.75)
        assert any(
            r.rule == "metric-consistency-system-tps" and r.severity != Severity.ERROR
            for r in result._check_results
        )

    # ------------------------------------------------------------------ TPS epsilon: 1e-9 vs 1.0

    def test_small_derived_tps_per_user_large_relative_error_errors(self):
        # total_tokens=1, duration=1200s, concurrency=2 → derived ≈ 4.2e-4 (< 1.0).
        # stored=0.01 is ~24× derived; rel_err ≈ 23 >> 0.01 with correct 1e-9 epsilon.
        # With the 1e-9 → 1.0 epsilon mutation: denominator becomes 1.0, rel_err ≈ 0.01 - 4.2e-4 ≈ 0.0096 < 0.01 → passes (wrong).
        result = _make_result(total_tokens=1.0, concurrency=2, tps_per_user=0.01)
        assert any(
            r.rule == "metric-consistency-tps-per-user" and r.severity == Severity.ERROR
            for r in result._check_results
        )
