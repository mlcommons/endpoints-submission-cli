"""Regression tests for native client reports and agentic point declarations."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from submission_checker.checker import _results_has_accuracy_scores
from submission_checker.models import PointConfig, Severity
from submission_checker.models.file.point_config import WarmupSpec
from submission_checker.models.file.point_summary import PointSummary
from submission_checker.models.loader import (
    load_accuracy_result,
    load_accuracy_scores,
    load_result_summary,
    load_system_power,
)

pytestmark = pytest.mark.unit

_SCORES = [
    {
        "dataset_name": "swe_bench",
        "score": 0.935,
        "unit_samples": 200,
        "num_repeats": 1,
        "total_samples": 200,
        "complete": True,
    },
    {
        "dataset_name": "performance",
        "score": 0.5887,
        "unit_samples": 100,
        "num_repeats": 2,
        "total_samples": 200,
        "complete": True,
    },
]
_NO_WARMUP = {
    "duration_s": 0,
    "requests_issued": 0,
    "requests_completed": 0,
    "data_source": "none; warmup disabled",
    "concurrency": 0,
}


@pytest.mark.parametrize("embedded", [False, True])
def test_native_accuracy_reads_each_dataset_and_preserves_file(tmp_path, embedded):
    path = tmp_path / ("results.json" if embedded else "accuracy_results.json")
    path.write_text(json.dumps({"average_accuracy": 0.935, "accuracy_scores": _SCORES}))
    before = path.read_bytes()
    if embedded:
        model, results, present = load_accuracy_scores(path)
        assert present
        assert _results_has_accuracy_scores(path)
    else:
        model, results = load_accuracy_result(path)
    assert model is not None
    assert not results
    assert path.read_bytes() == before
    assert model.metric_scores() == {
        "swe_bench": {"score": 0.935},
        "performance": {"score": 0.5887},
    }
    for entry in _SCORES:
        parsed = model.root[entry["dataset_name"]]
        assert all(parsed[key] == value for key, value in entry.items())
        assert parsed["num_samples"] == entry["unit_samples"]
        assert parsed["n_repeats"] == entry["num_repeats"]


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize(
    "scores",
    [
        [{"dataset_name": "duplicate", "score": 1}, {"dataset_name": "duplicate", "score": 0}],
        [{"score": 1}],
        [{"dataset_name": "  ", "score": 1}],
        [{"dataset_name": "missing-score", "total_samples": 200}],
        [{"dataset_name": "conflict", "score": 1, "unit_samples": 200, "num_samples": 100}],
        ["not-an-entry"],
        "not-a-list-or-mapping",
    ],
)
def test_bad_native_accuracy_is_an_error_not_silently_dropped(tmp_path, embedded, scores):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"accuracy_scores": scores}))
    if embedded:
        model, results, present = load_accuracy_scores(path)
        assert present
    else:
        model, results = load_accuracy_result(path)
    assert model is None
    assert any(r.rule == "accuracy-valid" and r.severity == Severity.ERROR for r in results)


def test_empty_native_accuracy_is_not_valid(tmp_path):
    path = tmp_path / "accuracy_results.json"
    path.write_text(json.dumps({"average_accuracy": 0, "accuracy_scores": []}))
    _, results = load_accuracy_result(path)
    assert any(r.rule == "accuracy-valid" and r.severity == Severity.ERROR for r in results)


def test_legacy_accuracy_mapping_still_loads(tmp_path):
    path = tmp_path / "accuracy_results.json"
    payload = {"aime": {"score": {"exact_match": 85}, "num_samples": 30}}
    path.write_text(json.dumps(payload))
    model, results = load_accuracy_result(path)
    assert model is not None and not results
    assert model.root == payload


@pytest.mark.parametrize("suffix", ["", ".0", ".00"])
def test_decimal_percentiles_produce_real_latencies(tmp_path, suffix):
    path = tmp_path / "result_summary.json"
    path.write_text(
        json.dumps(
            {
                "n_samples_completed": 10,
                "duration_ns": 10_000_000_000,
                "ttft": {
                    "percentiles": {
                        "50" + suffix: 372_810_574,
                        "90" + suffix: 679_663_943,
                        "95" + suffix: 857_008_282,
                        "99.9": 5_912_532_820,
                    }
                },
                "tpot": {"percentiles": {"90" + suffix: 8_124_112.430555556}},
            }
        )
    )
    before = path.read_bytes()
    summary, results = load_result_summary(path)
    assert summary is not None and not results
    assert summary.ttft_p50_ms == pytest.approx(372.810574)
    assert summary.ttft_p90_ms == pytest.approx(679.663943)
    assert summary.ttft_p95_ms == pytest.approx(857.008282)
    assert summary.tpot_p90_ms == pytest.approx(8.124112430555556)
    assert summary.ttft.percentiles["99.9"] == 5_912_532_820
    assert path.read_bytes() == before


@pytest.mark.parametrize("same_value", [True, False])
def test_duplicate_percentile_aliases_must_agree(same_value):
    payload = {
        "n_samples_completed": 1,
        "duration_ns": 1,
        "tpot": {"percentiles": {"90": 100, "90.0": 100 if same_value else 200}},
    }
    if same_value:
        assert PointSummary.model_validate(payload).tpot.percentiles == {"90": 100}
    else:
        with pytest.raises(ValidationError, match="Conflicting values"):
            PointSummary.model_validate(payload)


@pytest.mark.parametrize("concurrency", [0, 1, 2147483647])
def test_disabled_warmup_accepts_zero_and_needs_no_request_logs(concurrency):
    config = PointConfig.model_validate(
        {
            "concurrency": 12,
            "runtime_settings": {"runtime": {}},
            "warmup": {**_NO_WARMUP, "concurrency": concurrency},
        }
    )
    warmup_results = [r for r in config._check_results if r.rule.startswith("warmup-")]
    assert {r.rule for r in warmup_results} == {"warmup-present", "warmup-logs-retained"}
    assert all(r.severity == Severity.INFO for r in warmup_results)


@pytest.mark.parametrize(
    "changes",
    [
        {"concurrency": -1},
        {"duration_s": 1},
        {"requests_issued": 1},
        {"requests_issued": 1, "requests_completed": 1},
        {"requests_completed": 1},
    ],
)
def test_zero_concurrency_does_not_mask_active_or_invalid_warmup(changes):
    with pytest.raises(ValidationError):
        WarmupSpec.model_validate({**_NO_WARMUP, **changes})


def test_active_warmup_still_requires_log_disclosure():
    config = PointConfig.model_validate(
        {
            "concurrency": 12,
            "runtime_settings": {"runtime": {}},
            "warmup": {
                **_NO_WARMUP,
                "duration_s": 1,
                "requests_issued": 2,
                "requests_completed": 2,
                "concurrency": 2,
            },
        }
    )
    assert any(
        r.rule == "warmup-logs-retained" and r.severity == Severity.WARNING
        for r in config._check_results
    )


@pytest.mark.parametrize("load_pattern", ["agentic_inference", "concurrency", "qps", "poisson"])
@pytest.mark.parametrize("concurrency", [0, 12])
@pytest.mark.parametrize("stream_all_chunks", [False, True])
def test_load_pattern_and_client_forwarding_are_independent(
    load_pattern, concurrency, stream_all_chunks
):
    config = PointConfig.model_validate(
        {
            "concurrency": concurrency,
            "runtime_settings": {
                "runtime": {},
                "load_pattern": load_pattern,
                "stream_all_chunks": stream_all_chunks,
            },
        }
    )
    errors = {r.rule for r in config._check_results if r.severity == Severity.ERROR}
    assert ("load-pattern" not in errors) == (
        concurrency > 0 and load_pattern in ("concurrency", "agentic_inference")
    )
    assert "streaming-config" not in errors


def test_power_accepts_component_template_names_without_inventing_missing_values(tmp_path):
    path = tmp_path / "system_power.json"
    payload = {
        "cpu": {
            "num_cpu": 2,
            "tdp_per_cpu_watts": 200,
            "public_specification": "https://example.com/cpu",
        },
        "accelerator": {"num_accelerator": 4, "tdp_per_accelerator_watts": 500},
        "scale_up_network": {"num_switches": 1, "tdp_per_switch": 100},
        "overhead_fraction": 0.3,
    }
    path.write_text(json.dumps(payload))
    model, errors = load_system_power(path)
    assert model is not None and not errors
    assert model.cpu.count == 2
    assert model.cpu.link == "https://example.com/cpu"
    assert model.derived_power_w == 3250
    missing = deepcopy(payload)
    for component in ("cpu", "accelerator", "scale_up_network"):
        missing[component] = {
            key: value for key, value in missing[component].items() if not key.startswith("tdp_")
        }
    path.write_text(json.dumps(missing))
    model, errors = load_system_power(path)
    assert model is not None and not errors
    assert model.cpu.count == 2
    assert model.derived_power_w is None


def test_power_accepts_provisioned_watts_alias(tmp_path):
    path = tmp_path / "system_power.json"
    path.write_text(json.dumps({"provisioned_power_watts": 5000}))
    model, errors = load_system_power(path)
    assert model is not None and not errors
    assert model.provisioned_power_kw == 5
