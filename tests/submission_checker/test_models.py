"""Tests for Pydantic submission models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from submission_checker.models import (
    AccuracyResult,
    CheckResult,
    Division,
    PercentileStats,
    PointConfig,
    PointSummary,
    Report,
    RuntimeSettings,
    Severity,
    SystemDescription,
)

# ---------------------------------------------------------------------------
# CheckResult / Report (infrastructure models)
# ---------------------------------------------------------------------------


def test_check_result_passed_for_non_error():
    result = CheckResult(rule="r", message="ok", severity=Severity.WARNING)
    assert result.passed


def test_check_result_not_passed_for_error():
    result = CheckResult(rule="r", message="bad", severity=Severity.ERROR)
    assert not result.passed


def test_check_result_is_immutable():
    result = CheckResult(rule="r", message="ok")
    with pytest.raises(ValidationError):
        result.rule = "changed"  # type: ignore[misc]


def test_check_result_serialises_to_dict():
    result = CheckResult(rule="r", message="ok", severity=Severity.INFO)
    data = result.model_dump()
    assert data["rule"] == "r"
    assert data["severity"] == Severity.INFO
    assert data["passed"] is True


def test_check_result_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        CheckResult(rule="r", message="x", severity="not-a-severity")  # type: ignore[arg-type]


def test_report_errors_and_warnings(tmp_path: Path):
    report = Report(submission_path=tmp_path)
    report.results = [
        CheckResult(rule="a", message="err", severity=Severity.ERROR),
        CheckResult(rule="b", message="warn", severity=Severity.WARNING),
        CheckResult(rule="c", message="info", severity=Severity.INFO),
    ]
    assert len(report.errors) == 1
    assert len(report.warnings) == 1
    assert not report.passed


def test_report_passed_with_no_errors(tmp_path: Path):
    report = Report(submission_path=tmp_path)
    report.results = [CheckResult(rule="b", message="warn", severity=Severity.WARNING)]
    assert report.passed


def test_report_model_dump_includes_computed_fields(tmp_path: Path):
    report = Report(
        submission_path=tmp_path,
        results=[CheckResult(rule="r", message="bad", severity=Severity.ERROR)],
    )
    data = report.model_dump()
    assert data["passed"] is False
    assert len(data["errors"]) == 1


# ---------------------------------------------------------------------------
# SystemDescription
# ---------------------------------------------------------------------------

_NODE_TYPE = {
    "system_node_ensemble_id": 0,
    "number_of_nodes": 1,
    "host_processor_model_name": "Intel Xeon Gold 6148",
    "host_processors_per_node": 2,
    "host_processor_core_count": 20,
    "host_memory_capacity": "384 GB",
    "accelerator_model_name": "NVIDIA A100-SXM4-80GB",
    "accelerators_per_node": 8,
    "accelerator_memory_capacity": "80 GB HBM2e",
    "host_networking": "InfiniBand EDR",
    "host_storage_type": "NVMe SSD",
    "host_storage_capacity": "10 TB",
    "operating_system": "Ubuntu 20.04",
}

_BASE_FLAT = {
    "submitter_org_names": "Test Org",
    "system_name": "test-node",
    "system_category": "datacenter",
    "system_availability_status": "Available",
    "max_supported_concurrency": 1024,
    "serving_framework": "vLLM 0.4.0",
    "node_types": [_NODE_TYPE],
    "division": "Standardized",
}


def test_system_description_valid():
    sd = SystemDescription(**_BASE_FLAT)
    assert sd.division == Division.STANDARDIZED
    assert sd.submitter_org_names == "Test Org"


def test_system_description_accepts_vcpu_without_core_count():
    node_vcpu = {
        **_NODE_TYPE,
        "host_processor_core_count": None,
        "host_processor_vcpu_count": 40,
    }
    sd = SystemDescription(**{**_BASE_FLAT, "node_types": [node_vcpu]})
    assert sd.node_types[0].host_processor_vcpu_count == 40
    assert sd.node_types[0].host_processor_core_count is None


def test_system_description_allows_extra_fields():
    sd = SystemDescription(**{**_BASE_FLAT, "extra_top_level": "some value"})
    assert sd.model_extra["extra_top_level"] == "some value"


def test_system_description_rejects_invalid_division():
    with pytest.raises(ValidationError):
        SystemDescription(**{**_BASE_FLAT, "division": "NotADivision"})


# ---------------------------------------------------------------------------
# PointConfig
# ---------------------------------------------------------------------------


def test_measurement_point_config_valid():
    cfg = PointConfig(
        concurrency=64,
        dataset="test-ds",
        runtime_settings=RuntimeSettings(min_duration_ms=1_200_000),
    )
    assert cfg.concurrency == 64
    assert cfg.dataset == "test-ds"
    assert cfg.runtime_settings.stream_all_chunks is True


def test_measurement_point_config_concurrency_stored():
    cfg = PointConfig(concurrency=128)
    assert cfg.concurrency == 128


def test_measurement_point_config_empty_dataset():
    cfg = PointConfig(concurrency=32)
    assert cfg.dataset == ""


def test_measurement_point_config_load_pattern_stored():
    cfg = PointConfig(concurrency=64)
    assert cfg.runtime_settings.load_pattern == "concurrency"


# ---------------------------------------------------------------------------
# PointSummary
# ---------------------------------------------------------------------------


def _make_summary(
    n_completed: int = 1000,
    n_issued: int = 1000,
    n_failed: int = 0,
    duration_ns: float = 600_000_000_000.0,
    total_tokens: float = 500_000.0,
    ttft_p50_ns: float = 150_000_000.0,
    ttft_p95_ns: float = 300_000_000.0,
) -> PointSummary:
    return PointSummary(
        n_samples_completed=n_completed,
        n_samples_issued=n_issued,
        n_samples_failed=n_failed,
        duration_ns=duration_ns,
        ttft=PercentileStats(
            total=ttft_p50_ns * n_completed, percentiles={"50": ttft_p50_ns, "95": ttft_p95_ns}
        ),
        output_sequence_lengths=PercentileStats(total=total_tokens),
    )


def test_point_result_summary_duration_ms():
    s = _make_summary(duration_ns=1_200_000_000_000.0)
    assert abs(s.duration_ms - 1_200_000.0) < 1.0


def test_point_result_summary_sample_count_alias():
    s = _make_summary(n_completed=512)
    assert s.sample_count == 512


def test_point_result_summary_system_tps():
    # 600 s run, 60000 tokens → 100 tok/s
    s = _make_summary(duration_ns=600_000_000_000.0, total_tokens=60_000.0)
    assert abs(s.system_tps - 100.0) < 0.01


def test_point_result_summary_ttft_ms_conversion():
    s = _make_summary(ttft_p50_ns=150_000_000.0, ttft_p95_ns=300_000_000.0)
    assert abs(s.ttft_p50_ms - 150.0) < 0.001
    assert abs(s.ttft_p95_ms - 300.0) < 0.001


def test_point_result_summary_total_output_tokens():
    s = _make_summary(total_tokens=22_407_098.0)
    assert s.total_output_tokens == 22_407_098


# ---------------------------------------------------------------------------
# AccuracyResult
# ---------------------------------------------------------------------------


def test_accuracy_result_stores_scores():
    ar = AccuracyResult({
        "cnn_dailymail::llama3_8b": {"score": {"rouge1": "38.73", "rouge2": "16.10"}}
    })
    scores = ar.metric_scores()
    assert scores["cnn_dailymail::llama3_8b"]["rouge1"] == pytest.approx(38.73)
    assert scores["cnn_dailymail::llama3_8b"]["rouge2"] == pytest.approx(16.10)


def test_accuracy_result_empty_emits_error():
    ar = AccuracyResult({})
    assert any(r.rule == "accuracy-valid" and r.severity == Severity.ERROR for r in ar._check_results)


# ---------------------------------------------------------------------------
# DatasetMetadata coercions
# ---------------------------------------------------------------------------


def test_dataset_metadata_numeric_string_coerced_to_float():
    sd = SystemDescription(**{
        **_BASE_FLAT,
        "input_token_average": "512.0",
        "output_token_average": "128.5",
    })
    assert sd.input_token_average == 512.0
    assert sd.output_token_average == 128.5


def test_dataset_metadata_invalid_string_raises():
    with pytest.raises((ValidationError, ValueError)):
        SystemDescription(**{
            **_BASE_FLAT,
            "input_token_average": "not-a-number",
        })


# ---------------------------------------------------------------------------
# measured_accuracy_score coercion
# ---------------------------------------------------------------------------


def test_accuracy_empty_string_measured_score_coerced_to_none():
    sd = SystemDescription(**{**_BASE_FLAT, "measured_accuracy_score": ""})
    assert sd.measured_accuracy_score is None
