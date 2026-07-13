# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the run_metadata.json model and loader."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from submission_checker.models import RunMetadata, Severity
from submission_checker.models.loader import load_run_metadata

# Fields whose key must be present but whose value may be null.
_NULLABLE_KEYS = ("config_summary_notes", "link_config", "link_logs")

# Latency families × statistics that must all be present and non-null.
_LATENCY_FIELDS = [
    f"measured_latency_{group}_{stat}"
    for group in ("ttft", "tpot", "request")
    for stat in ("min", "average", "p50", "p90", "p95", "p99", "p999", "max")
]


def _valid() -> dict:
    md = {
        "run_date": "2026-01-26",
        "node_config": "32x Xeon 6503P + 48x GB200",
        "config_summary": "EP 1, PP 2, TP 4, DP 1, batch=24",
        "config_summary_notes": None,
        "concurrency": 128,
        "system_tps": 306.94,
        "tps_per_user": 2.40,
        "ttft": 312.5,
        "qps": 5.07,
        "tps_utilization": 0.37,
        "measured_total_output_tokens": 60557,
        "measured_run_duration": 197.29,
        "measured_total_requests": 1000,
        "link_config": None,
        "link_logs": None,
    }
    for f in _LATENCY_FIELDS:
        md[f] = 1.0
    return md


@pytest.mark.unit
class TestRunMetadataModel:
    def test_valid_payload(self):
        md = RunMetadata.model_validate(_valid())
        assert md.concurrency == 128
        assert md.tps_utilization == 0.37

    @pytest.mark.parametrize("field", _LATENCY_FIELDS)
    def test_latency_must_not_be_null(self, field):
        payload = {**_valid(), field: None}
        with pytest.raises(ValidationError):
            RunMetadata.model_validate(payload)

    @pytest.mark.parametrize("field", _LATENCY_FIELDS)
    def test_latency_must_be_present(self, field):
        payload = {k: v for k, v in _valid().items() if k != field}
        with pytest.raises(ValidationError):
            RunMetadata.model_validate(payload)

    @pytest.mark.parametrize(
        "field",
        [
            "concurrency",
            "system_tps",
            "tps_per_user",
            "ttft",
            "qps",
            "tps_utilization",
            "measured_total_output_tokens",
            "measured_run_duration",
            "measured_total_requests",
            "run_date",
            "node_config",
            "config_summary",
        ],
    )
    def test_core_fields_must_not_be_null(self, field):
        payload = {**_valid(), field: None}
        with pytest.raises(ValidationError):
            RunMetadata.model_validate(payload)

    @pytest.mark.parametrize("field", _NULLABLE_KEYS)
    def test_nullable_keys_accept_null(self, field):
        md = RunMetadata.model_validate({**_valid(), field: None})
        assert getattr(md, field) is None

    @pytest.mark.parametrize("field", _NULLABLE_KEYS)
    def test_nullable_keys_must_be_present(self, field):
        # The key must exist even though the value may be null.
        payload = {k: v for k, v in _valid().items() if k != field}
        with pytest.raises(ValidationError):
            RunMetadata.model_validate(payload)

    def test_config_summary_string_min_length(self):
        with pytest.raises(ValidationError):
            RunMetadata.model_validate({**_valid(), "config_summary": "x"})
        md = RunMetadata.model_validate({**_valid(), "config_summary": "abcd"})
        assert md.config_summary == "abcd"

    def test_config_summary_as_object(self):
        cs = {
            "disaggregated": None,
            "expert_parallel": None,
            "tensor_parallel": 4,
            "pipeline_parallel": 2,
            "data_parallel": None,
            "batch": None,
        }
        md = RunMetadata.model_validate({**_valid(), "config_summary": cs})
        assert md.config_summary.tensor_parallel == 4
        assert md.config_summary.pipeline_parallel == 2

    def test_config_summary_object_all_null_ok(self):
        cs = dict.fromkeys(
            (
                "disaggregated",
                "expert_parallel",
                "tensor_parallel",
                "pipeline_parallel",
                "data_parallel",
                "batch",
            )
        )
        md = RunMetadata.model_validate({**_valid(), "config_summary": cs})
        assert md.config_summary.batch is None

    def test_config_summary_null_rejected(self):
        with pytest.raises(ValidationError):
            RunMetadata.model_validate({**_valid(), "config_summary": None})


@pytest.mark.unit
class TestLoadRunMetadata:
    def test_valid_file(self, tmp_path):
        p = tmp_path / "run_metadata.json"
        p.write_text(json.dumps(_valid()))
        model, results = load_run_metadata(p)
        assert model is not None
        assert results == []

    def test_invalid_file_returns_errors(self, tmp_path):
        p = tmp_path / "run_metadata.json"
        bad = {**_valid(), "measured_latency_tpot_p50": None}
        p.write_text(json.dumps(bad))
        model, results = load_run_metadata(p)
        assert model is None
        assert results
        assert all(r.rule == "run-metadata-valid" and r.severity == Severity.ERROR for r in results)

    def test_missing_file(self, tmp_path):
        model, results = load_run_metadata(tmp_path / "absent.json")
        assert model is None
        assert len(results) == 1
        assert results[0].severity == Severity.ERROR
