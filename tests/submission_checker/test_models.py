"""Tests for Pydantic submission models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from submission_checker.models import (
    AccuracyResult,
    CheckResult,
    DatasetAccuracyScores,
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
    "host_network_card_count": "4x NIC",
    "host_storage_type": "NVMe SSD",
    "host_storage_capacity": "10 TB",
    "operating_system": "Ubuntu 20.04",
    "host_memory_configuration": "8x 64GB DDR5",
    "accelerator_memory_type": "HBM2e",
    "driver": "550.54",
    "filesystem": "ext4",
}

_BASE_FLAT = {
    "submitter_org_names": "Test Org",
    "submitter_contact": "contact@example.com",
    "system_name": "test-node",
    "system_category": "datacenter",
    "system_availability_status": "Available",
    "min_supported_concurrency": 32,
    "max_supported_concurrency": 1024,
    "system_size": "1 node",
    "system_node_ensemble_count": 1,
    "system_node_ensemble_total": 1,
    "serving_framework": "vLLM 0.4.0",
    "node_types": [_NODE_TYPE],
    "division": "Standardized",
    "model_id": "test-model",
    "model_precision": "FP16",
    "link_to_model": "https://example.com/model",
    "dataset_id": "cnn_dailymail",
    "dataset_name": "CNN/DailyMail",
    "input_token_average": 870.0,
    "output_token_average": 128.0,
    "dataset_type": "performance",
    "dataset_link": "https://example.com/dataset",
    "measured_accuracy_score": "38.7",
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


def test_node_requires_core_or_vcpu_count():
    # A node disclosing neither physical cores nor vCPUs is rejected.
    node = {**_NODE_TYPE, "host_processor_core_count": None, "host_processor_vcpu_count": None}
    with pytest.raises(ValidationError):
        SystemDescription(**{**_BASE_FLAT, "node_types": [node]})


@pytest.mark.parametrize(
    "field",
    [
        "submitter_contact",
        "system_size",
        "system_node_ensemble_count",
        "model_id",
        "model_precision",
        "link_to_model",
        "dataset_id",
        "input_token_average",
        "dataset_link",
        "measured_accuracy_score",
    ],
)
def test_required_system_fields_rejected_when_missing(field):
    payload = {k: v for k, v in _BASE_FLAT.items() if k != field}
    with pytest.raises(ValidationError):
        SystemDescription(**payload)


@pytest.mark.parametrize(
    "field",
    [
        "host_memory_configuration",
        "accelerator_memory_type",
        "host_network_card_count",
        "driver",
        "filesystem",
    ],
)
def test_required_node_fields_rejected_when_missing(field):
    node = {k: v for k, v in _NODE_TYPE.items() if k != field}
    with pytest.raises(ValidationError):
        SystemDescription(**{**_BASE_FLAT, "node_types": [node]})


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
        runtime_settings=RuntimeSettings(
            min_duration_ms=1_200_000,
            runtime=RuntimeSettings.Runtime(scheduler_random_seed=42, dataloader_random_seed=42),
        ),
    )
    assert cfg.concurrency == 64
    assert cfg.dataset == "test-ds"
    assert cfg.runtime_settings.stream_all_chunks is True


def test_measurement_point_config_concurrency_stored():
    cfg = PointConfig(
        concurrency=128,
        runtime_settings=RuntimeSettings(
            runtime=RuntimeSettings.Runtime(scheduler_random_seed=42, dataloader_random_seed=42)
        ),
    )
    assert cfg.concurrency == 128


def test_measurement_point_config_empty_dataset():
    cfg = PointConfig(
        concurrency=32,
        runtime_settings=RuntimeSettings(
            runtime=RuntimeSettings.Runtime(scheduler_random_seed=42, dataloader_random_seed=42)
        ),
    )
    assert cfg.dataset == ""


def test_measurement_point_config_load_pattern_stored():
    cfg = PointConfig(
        concurrency=64,
        runtime_settings=RuntimeSettings(
            runtime=RuntimeSettings.Runtime(scheduler_random_seed=42, dataloader_random_seed=42)
        ),
    )
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
    ar = AccuracyResult(
        {"cnn_dailymail::llama3_8b": {"score": {"rouge1": "38.73", "rouge2": "16.10"}}}
    )
    scores = ar.metric_scores()
    assert scores["cnn_dailymail::llama3_8b"]["rouge1"] == pytest.approx(38.73)
    assert scores["cnn_dailymail::llama3_8b"]["rouge2"] == pytest.approx(16.10)


def test_accuracy_result_empty_emits_error():
    ar = AccuracyResult({})
    assert any(
        r.rule == "accuracy-valid" and r.severity == Severity.ERROR for r in ar._check_results
    )


# ---------------------------------------------------------------------------
# DatasetMetadata coercions
# ---------------------------------------------------------------------------


def test_dataset_metadata_numeric_string_coerced_to_float():
    sd = SystemDescription(
        **{
            **_BASE_FLAT,
            "input_token_average": "512.0",
            "output_token_average": "128.5",
        }
    )
    assert sd.input_token_average == 512.0
    assert sd.output_token_average == 128.5


def test_dataset_metadata_invalid_string_raises():
    with pytest.raises((ValidationError, ValueError)):
        SystemDescription(
            **{
                **_BASE_FLAT,
                "input_token_average": "not-a-number",
            }
        )


# ---------------------------------------------------------------------------
# measured_accuracy_score
# ---------------------------------------------------------------------------


def test_accuracy_empty_string_measured_score_rejected():
    # measured_accuracy_score is required; an empty string coerces to None and is rejected.
    with pytest.raises(ValidationError):
        SystemDescription(**{**_BASE_FLAT, "measured_accuracy_score": ""})


def test_accuracy_scalar_string_measured_score_accepted():
    # The legacy scalar form (str/float) is still accepted for now.
    sd = SystemDescription(**{**_BASE_FLAT, "measured_accuracy_score": "38.7"})
    assert sd.measured_accuracy_score == "38.7"


def test_accuracy_scalar_float_measured_score_accepted():
    sd = SystemDescription(**{**_BASE_FLAT, "measured_accuracy_score": 91.2})
    assert sd.measured_accuracy_score == 91.2


def test_accuracy_structured_measured_score_accepted():
    # The structured per-dataset form: {dataset: {"scores": {score_name: score_value}}}.
    sd = SystemDescription(
        **{
            **_BASE_FLAT,
            "measured_accuracy_score": {
                "cnn_dailymail": {"scores": {"rouge1": 38.73, "rouge2": 16.1}}
            },
        }
    )
    entry = sd.measured_accuracy_score["cnn_dailymail"]
    assert isinstance(entry, DatasetAccuracyScores)
    assert entry.scores == {"rouge1": pytest.approx(38.73), "rouge2": pytest.approx(16.1)}


def test_accuracy_structured_missing_scores_key_rejected():
    # A dataset entry without the `scores` mapping is invalid.
    with pytest.raises(ValidationError):
        SystemDescription(**{**_BASE_FLAT, "measured_accuracy_score": {"ds": {"rouge1": 1.0}}})


def test_accuracy_structured_string_score_rejected():
    # score_value must be a float — strings are validated, not parsed.
    with pytest.raises(ValidationError):
        SystemDescription(
            **{**_BASE_FLAT, "measured_accuracy_score": {"ds": {"scores": {"x": "38.73"}}}}
        )
