# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for models.aggregate validators — cross-file rule checks."""

from __future__ import annotations

import pytest

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
        runtime_settings=RuntimeSettings(min_duration_ms=1_200_000, runtime=RuntimeSettings.Runtime(scheduler_random_seed=42, dataloader_random_seed=42)),
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
            runtime_settings=RuntimeSettings(min_duration_ms=1_200_000, runtime=RuntimeSettings.Runtime(scheduler_random_seed=42, dataloader_random_seed=42)),
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

    def test_aime25_boundary(self, tmp_path):
        """aime25 requires 30 queries — 29 fails, 30 passes."""
        base = {"yaml_path": tmp_path / "run_64.yaml"}
        ctx = {"summary_path": tmp_path / "summary.json"}

        fail = PointResult.model_validate(
            {"config": _config_with_dataset("aime25"), "summary": _summary(n_completed=29), **base},
            context=ctx,
        )
        assert any(
            r.rule == "min-query-count" and r.severity == Severity.ERROR
            for r in fail._check_results
        )

        ok_result = PointResult.model_validate(
            {"config": _config_with_dataset("aime25"), "summary": _summary(n_completed=30), **base},
            context=ctx,
        )
        assert not any(
            r.rule == "min-query-count" and r.severity == Severity.ERROR
            for r in ok_result._check_results
        )

    def test_gpqa_boundary(self, tmp_path):
        """gpqa requires 198 queries — 197 fails, 198 passes."""
        base = {"yaml_path": tmp_path / "run_64.yaml"}
        ctx = {"summary_path": tmp_path / "summary.json"}

        fail = PointResult.model_validate(
            {"config": _config_with_dataset("gpqa"), "summary": _summary(n_completed=197), **base},
            context=ctx,
        )
        assert any(
            r.rule == "min-query-count" and r.severity == Severity.ERROR
            for r in fail._check_results
        )

        ok_result = PointResult.model_validate(
            {"config": _config_with_dataset("gpqa"), "summary": _summary(n_completed=198), **base},
            context=ctx,
        )
        assert not any(
            r.rule == "min-query-count" and r.severity == Severity.ERROR
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
            dataset="open_orca",
            runtime_settings=RuntimeSettings(runtime=RuntimeSettings.Runtime(scheduler_random_seed=42, dataloader_random_seed=42)),
        )
        c2 = PointConfig(
            concurrency=128,
            dataset="cnn_dailymail",
            runtime_settings=RuntimeSettings(runtime=RuntimeSettings.Runtime(scheduler_random_seed=42, dataloader_random_seed=42)),
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

    def test_scalar_score_gated_as_single_metric_passes(self, tmp_path):
        # Endpoints results.json exposes a single *unnamed* scalar `score`
        # (== exact_match for deepseek). With a single-metric threshold it is gated
        # as that metric — with a warning — and 81.3355 ≥ 0.99*81.3582 (80.5446) passes.
        ar = AccuracyResult({"deepseek_r1_accuracy": {"num_samples": 4388, "score": 81.3355}})
        ctx = _model_ctx(tmp_path, accuracy_result=ar, model_name="deepseek_r1-torch-fp4")
        assert any(
            r.rule == "accuracy-gate"
            and r.severity == Severity.WARNING
            and "unnamed scalar" in r.message
            for r in ctx._check_results
        )
        assert any(
            r.rule == "accuracy-gate"
            and r.severity == Severity.INFO
            and "exact_match" in r.message
            for r in ctx._check_results
        )
        assert not any(
            r.rule == "accuracy-gate" and r.severity == Severity.ERROR
            for r in ctx._check_results
        )

    def test_scalar_score_gated_as_single_metric_fails(self, tmp_path):
        # 70.0 < 0.99*81.3582 (80.5446) → the mapped exact_match gate errors.
        ar = AccuracyResult({"deepseek_r1_accuracy": {"num_samples": 4388, "score": 70.0}})
        ctx = _model_ctx(tmp_path, accuracy_result=ar, model_name="deepseek_r1-torch-fp4")
        assert any(
            r.rule == "accuracy-gate"
            and r.severity == Severity.ERROR
            and "exact_match" in r.message
            for r in ctx._check_results
        )

    def test_scalar_score_not_mapped_for_multimetric_model(self, tmp_path):
        # llama3.1-8b declares 5 metrics, so a bare scalar `score` is ambiguous and
        # must NOT be mapped — no scalar warning, and no metric comparison fires.
        ar = AccuracyResult({"cnn_dailymail::llama3_8b": {"num_samples": 13368, "score": 39.0}})
        ctx = _model_ctx(tmp_path, accuracy_result=ar, model_name="Llama-3_1-8B-Instruct")
        assert not any(
            r.rule == "accuracy-gate" and "unnamed scalar" in r.message
            for r in ctx._check_results
        )
        assert not any(
            r.rule == "accuracy-gate" and r.severity in (Severity.INFO, Severity.ERROR)
            for r in ctx._check_results
        )

    # gpt-oss reports per-subset *fractional* scores; the gate aggregates them
    # (sample-weighted), rescales 0–1 → percentage, and gates one value (like inference).
    _GPTOSS = {
        "livecodebench::gptoss": {"num_samples": 1055, "score": 0.8458135860979463},
        "aime25::gptoss": {"num_samples": 30, "score": 0.7625},
        "gpqa::gptoss": {"num_samples": 198, "score": 0.7515151515151515},
    }

    def test_multidataset_aggregates_and_rescales_fractional_score(self, tmp_path):
        ctx = _model_ctx(
            tmp_path, accuracy_result=AccuracyResult(self._GPTOSS), model_name="gpt-oss-120b"
        )
        # sample-weighted mean ≈ 0.8293 → 82.93% ≥ 82.2987 → PASS (no gate error)
        assert any(
            r.rule == "accuracy-gate" and r.severity == Severity.INFO and "exact_match" in r.message
            for r in ctx._check_results
        )
        assert not any(
            r.rule == "accuracy-gate" and r.severity == Severity.ERROR for r in ctx._check_results
        )

    def test_multidataset_sample_count_is_summed(self, tmp_path):
        ctx = _model_ctx(
            tmp_path, accuracy_result=AccuracyResult(self._GPTOSS), model_name="gpt-oss-120b"
        )
        sc = [r for r in ctx._check_results if r.rule == "accuracy-sample-count"]
        assert len(sc) == 1  # one aggregate check, not one per subset
        # no n_repeats → issued == unique = 1283 < required 4395 → error
        assert sc[0].severity == Severity.ERROR and "1283" in sc[0].message

    def test_sample_count_uses_issued_with_repeats(self, tmp_path):
        # gpt-oss runs each dataset with repeats; MLPerf's required count (4395) is the
        # *issued* total = Σ(num_samples × n_repeats), not unique samples (1283).
        ar = AccuracyResult(
            {
                "aime25::gptoss": {"num_samples": 30, "n_repeats": 8, "score": 0.80},
                "gpqa::gptoss": {"num_samples": 198, "n_repeats": 5, "score": 0.80},
                "livecodebench::gptoss": {"num_samples": 1055, "n_repeats": 3, "score": 0.85},
            }
        )
        ctx = _model_ctx(tmp_path, accuracy_result=ar, model_name="gpt-oss-120b")
        sc = [r for r in ctx._check_results if r.rule == "accuracy-sample-count"]
        assert len(sc) == 1
        # 30*8 + 198*5 + 1055*3 = 4395 issued ≥ 4395 → not an error
        assert sc[0].severity != Severity.ERROR and "4395" in sc[0].message

    def test_multidataset_aggregate_score_below_threshold_fails(self, tmp_path):
        ar = AccuracyResult(
            {
                "a::gptoss": {"num_samples": 100, "score": 0.50},
                "b::gptoss": {"num_samples": 100, "score": 0.50},
            }
        )
        ctx = _model_ctx(tmp_path, accuracy_result=ar, model_name="gpt-oss-120b")
        # mean 0.50 → 50% < 82.2987 → error
        assert any(
            r.rule == "accuracy-gate" and r.severity == Severity.ERROR and "exact_match" in r.message
            for r in ctx._check_results
        )
