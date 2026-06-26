"""Tests for SubmissionChecker using pre-built fixtures from test_submissions/."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml

from submission_checker.checker import SubmissionChecker
from submission_checker.models import CheckResult, Report, Severity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _errors(report: Report, rule: str) -> list[CheckResult]:
    return [r for r in report.results if r.rule == rule and r.severity == Severity.ERROR]


def _warnings(report: Report, rule: str) -> list[CheckResult]:
    return [r for r in report.results if r.rule == rule and r.severity == Severity.WARNING]


def _check(path: Path) -> Report:
    return SubmissionChecker(path).run()


# ---------------------------------------------------------------------------
# valid_standardized — fully compliant synthetic fixture, must pass everything
# ---------------------------------------------------------------------------


class TestValidStandardized:
    def test_passes_overall(self, valid_standardized):
        assert _check(valid_standardized).passed

    def test_all_regions_covered(self, valid_standardized):
        report = _check(valid_standardized)
        for rule in [
            "low-latency-coverage",
            "low-throughput-coverage",
            "med-throughput-coverage",
            "high-throughput-coverage",
        ]:
            assert not _errors(report, rule), f"{rule} should pass"

    def test_metric_consistency(self, valid_standardized):
        report = _check(valid_standardized)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")

    def test_accuracy_gate(self, valid_standardized):
        assert not _errors(_check(valid_standardized), "accuracy-gate")

    def test_point_count(self, valid_standardized):
        assert not _errors(_check(valid_standardized), "point-count")


# ---------------------------------------------------------------------------
# invalid_submission — 3 points, failed accuracy; must trigger specific errors
# ---------------------------------------------------------------------------


class TestInvalidSubmission:
    def test_fails_overall(self, invalid_submission):
        assert not _check(invalid_submission).passed

    def test_point_count_error(self, invalid_submission):
        assert _errors(_check(invalid_submission), "point-count")

    def test_accuracy_gate_error(self, invalid_submission):
        assert _errors(_check(invalid_submission), "accuracy-gate")

    def test_missing_throughput_regions(self, invalid_submission):
        report = _check(invalid_submission)
        # Only 3 points (c16, c38, c88); max_supported_concurrency=88 → LT region is 33–36, not covered
        assert _errors(report, "low-throughput-coverage")


# ---------------------------------------------------------------------------
# sub_a / sub_b — MI355X 8/16-GPU, gpt-oss-120b, M=2048, 7 points
# Concurrencies: 4,16,64,128,512,1024,2048 — jumps 16→64, skipping LT (33–44)
# ---------------------------------------------------------------------------


class TestSubA:
    def test_point_count_passes(self, sub_a):
        assert not _errors(_check(sub_a), "point-count")

    def test_low_latency_covered(self, sub_a):
        assert not _errors(_check(sub_a), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_a):
        assert _errors(_check(sub_a), "low-throughput-coverage")

    def test_metric_consistency(self, sub_a):
        report = _check(sub_a)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")

    def test_accuracy_passes(self, sub_a):
        assert not _errors(_check(sub_a), "accuracy-gate")


class TestSubB:
    def test_low_throughput_missing(self, sub_b):
        assert _errors(_check(sub_b), "low-throughput-coverage")

    def test_metric_consistency(self, sub_b):
        report = _check(sub_b)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


# ---------------------------------------------------------------------------
# sub_c / sub_d — TPU 4/8-chip, qwen3-coder-480b, M=512/1024, 7/8 points
# ---------------------------------------------------------------------------


class TestSubC:
    def test_point_count_passes(self, sub_c):
        assert not _errors(_check(sub_c), "point-count")

    def test_low_latency_covered(self, sub_c):
        assert not _errors(_check(sub_c), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_c):
        assert _errors(_check(sub_c), "low-throughput-coverage")

    def test_metric_consistency(self, sub_c):
        report = _check(sub_c)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


class TestSubD:
    def test_low_throughput_missing(self, sub_d):
        assert _errors(_check(sub_d), "low-throughput-coverage")

    def test_metric_consistency(self, sub_d):
        report = _check(sub_d)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


# ---------------------------------------------------------------------------
# sub_e / sub_f — Gaudi, llama3-8b, M=1024, 11 points (1–1024)
# Concurrencies include 32 (LL) and 64 (MT) — LT (33–42) still skipped
# ---------------------------------------------------------------------------


class TestSubE:
    def test_point_count_passes(self, sub_e):
        assert not _errors(_check(sub_e), "point-count")

    def test_low_latency_covered(self, sub_e):
        assert not _errors(_check(sub_e), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_e):
        assert _errors(_check(sub_e), "low-throughput-coverage")

    def test_high_throughput_covered(self, sub_e):
        assert not _errors(_check(sub_e), "high-throughput-coverage")

    def test_metric_consistency(self, sub_e):
        report = _check(sub_e)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


class TestSubF:
    def test_metric_consistency(self, sub_f):
        report = _check(sub_f)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


# ---------------------------------------------------------------------------
# sub_g / sub_h — 8-GPU vLLM/SGLang, llama3-70b, M=2048, 10 points
# Minimum concurrency is 64 — both LL (1–32) and LT (33–44) missing
# ---------------------------------------------------------------------------


class TestSubG:
    def test_low_latency_missing(self, sub_g):
        assert _errors(_check(sub_g), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_g):
        assert _errors(_check(sub_g), "low-throughput-coverage")

    def test_point_count_passes(self, sub_g):
        assert not _errors(_check(sub_g), "point-count")

    def test_metric_consistency(self, sub_g):
        report = _check(sub_g)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


class TestSubH:
    def test_low_latency_missing(self, sub_h):
        assert _errors(_check(sub_h), "low-latency-coverage")

    def test_metric_consistency(self, sub_h):
        report = _check(sub_h)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


# ---------------------------------------------------------------------------
# sub_i — H200 8-GPU, deepseek-r1, M=512, 10 points (1–512)
# LT region is 33–40; 32 is LL, 64 is MT — LT skipped
# Short durations → run-duration WARNINGs (not errors) on some points
# ---------------------------------------------------------------------------


class TestSubI:
    def test_low_latency_covered(self, sub_i):
        assert not _errors(_check(sub_i), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_i):
        assert _errors(_check(sub_i), "low-throughput-coverage")

    def test_metric_consistency(self, sub_i):
        report = _check(sub_i)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")

    def test_point_duration_warnings_not_errors(self, sub_i):
        report = _check(sub_i)
        assert not _errors(report, "point-duration"), "point-duration fires as WARNING, not ERROR"


# ---------------------------------------------------------------------------
# sub_j — GB300 72-GPU, deepseek-r1, M=16384, 10 points (32–16384)
# LT region is 33–57; 32 is LL, 64 is MT — LT skipped
# ---------------------------------------------------------------------------


class TestSubJ:
    def test_low_latency_covered(self, sub_j):
        assert not _errors(_check(sub_j), "low-latency-coverage")

    def test_low_throughput_missing(self, sub_j):
        assert _errors(_check(sub_j), "low-throughput-coverage")

    def test_high_throughput_covered(self, sub_j):
        assert not _errors(_check(sub_j), "high-throughput-coverage")

    def test_metric_consistency(self, sub_j):
        report = _check(sub_j)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")


# ---------------------------------------------------------------------------
# Targeted edge-case tests that build synthetic dirs to cover checker.py paths
# ---------------------------------------------------------------------------

_SYSTEM_DESC = {
    "submitter_org_names": "Test Org",
    "submitter_contact": "contact@example.com",
    "system_name": "test-sys",
    "system_category": "datacenter",
    "system_availability_status": "Available",
    "max_supported_concurrency": 1024,
    "serving_framework": "vLLM",
    "node_types": [
        {
            "system_node_ensemble_id": 0,
            "number_of_nodes": 1,
            "host_processor_model_name": "AMD EPYC",
            "host_processors_per_node": 2,
            "host_processor_core_count": 64,
            "host_memory_capacity": "512 GB",
            "accelerator_model_name": "H100",
            "accelerators_per_node": 8,
            "accelerator_memory_capacity": "80 GB",
            "host_networking": "InfiniBand",
            "host_network_card_count": "4x NIC",
            "host_storage_type": "NVMe",
            "host_storage_capacity": "10 TB",
            "operating_system": "Ubuntu 22.04",
            "host_memory_configuration": "8x 64GB DDR5",
            "accelerator_memory_type": "HBM3",
            "driver": "550.54",
            "filesystem": "ext4",
        }
    ],
    "division": "Serviced",
    "system_size": "1 node",
    "system_node_ensemble_count": 1,
    "system_node_ensemble_total": 1,
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

_SUMMARY = {
    "n_samples_issued": 1000,
    "n_samples_completed": 1000,
    "n_samples_failed": 0,
    "duration_ns": 1_200_000_000_000.0,
    "ttft": {"total": 0.0, "percentiles": {"50": 150_000_000.0, "95": 300_000_000.0}},
    "output_sequence_lengths": {"total": 500_000.0, "percentiles": {}},
}

_ACCURACY = {
    "llm-perf-dataset-v1": {
        "dataset_name": "llm-perf-dataset-v1",
        "num_samples": 500,
        "score": {"rouge1": "45.12", "rouge2": "22.01", "rougeL": "30.45"},
        "n_repeats": 1,
    }
}

# Concurrencies that cover all four regions for M=1024
# LT: 33–42 → 38; MT: 43–175 → 88; HT: 176–1126 → 256, 512, 768, 1000
_CONCURRENCIES = [16, 38, 88, 256, 512, 768, 1000]


def _make_run_yaml(concurrency: int) -> dict:
    return {
        "concurrency": concurrency,
        "dataset": "llm-perf-dataset-v1",
        "runtime_settings": {
            "load_pattern": "concurrency",
            "min_duration_ms": 1_200_000,
            "stream_all_chunks": True,
            "runtime": {
                "scheduler_random_seed": 42,
                "dataloader_random_seed": 42,
            },
        },
    }


def _make_run_metadata(concurrency: int) -> dict:
    """A fully-populated, valid run_metadata.json payload."""
    md = {
        "run_date": "2026-01-26",
        "node_config": "8x H100 test node",
        "config_summary": "TP 1, PP 1, DP 1",
        "config_summary_notes": None,
        "concurrency": concurrency,
        "system_tps": 306.94,
        "tps_per_user": 2.40,
        "ttft": 312.5,
        "qps": 5.07,
        "tps_utilization": 1.0,
        "measured_total_output_tokens": 60557,
        "measured_run_duration": 197.29,
        "measured_total_requests": 1000,
        "link_config": None,
        "link_logs": None,
    }
    for group in ("ttft", "tpot", "request"):
        for stat, val in (
            ("min", 1.0), ("average", 2.0), ("p50", 2.0), ("p90", 3.0),
            ("p95", 3.5), ("p99", 4.0), ("p999", 4.5), ("max", 5.0),
        ):
            md[f"measured_latency_{group}_{stat}"] = val
    return md


def _build_submission(
    root: Path,
    system_id: str = "test-sys",
    system_desc: dict | None = None,
    concurrencies: list[int] | None = None,
    write_runs: bool = True,
    write_results: bool = True,
    write_accuracy_json: bool = True,
    write_run_metadata: bool = True,
    accuracy_data: dict | None = None,
    model: str = "llama3-70b",
) -> Path:
    """Build a minimal valid (or deliberately broken) submission directory."""
    desc = system_desc if system_desc is not None else _SYSTEM_DESC.copy()
    concs = concurrencies if concurrencies is not None else _CONCURRENCIES

    systems_dir = root / "systems"
    systems_dir.mkdir(parents=True)
    (systems_dir / f"{system_id}.json").write_text(json.dumps(desc))
    (root / "documentation").mkdir(parents=True, exist_ok=True)

    pareto_dir = root / "pareto"
    model_dir = pareto_dir / system_id / model
    points_dir = model_dir / "points"
    results_dir = model_dir / "results"
    for d in (points_dir, results_dir):
        d.mkdir(parents=True)

    if write_runs:
        for c in concs:
            (points_dir / f"point_{c}.yaml").write_text(yaml.dump(_make_run_yaml(c)))

    if write_results:
        for c in concs:
            result_dir = results_dir / f"point_{c}"
            result_dir.mkdir(parents=True)
            (result_dir / "results_summary.json").write_text(json.dumps(_SUMMARY))
            (result_dir / "config.yaml").write_text(yaml.dump({"concurrency": c}))
            if write_run_metadata:
                (result_dir / "run_metadata.json").write_text(json.dumps(_make_run_metadata(c)))
            # Write accuracy inside the first point only; checker scans all points
            if c == concs[0]:
                accuracy_dir = result_dir / "accuracy"
                accuracy_dir.mkdir()
                if write_accuracy_json:
                    data = accuracy_data if accuracy_data is not None else _ACCURACY
                    (accuracy_dir / "results.json").write_text(json.dumps(data))

    return root


class TestCheckerEdgeCases:
    """Targeted tests to cover checker.py error paths not exercised by fixture tests."""

    def test_nonexistent_path(self, tmp_path):
        """path-exists error when submission_path does not exist."""
        report = _check(tmp_path / "does_not_exist")
        assert _errors(report, "path-exists")

    def test_missing_required_dirs_early_exit(self, tmp_path):
        """SubmissionDir structure errors cause early return from run()."""
        # Only systems/ present — pareto/ missing → structure error → early exit
        (tmp_path / "systems").mkdir()
        report = _check(tmp_path)
        assert _errors(report, "required-dir")
        # Should not have processed any systems
        assert not any(r.rule == "system-description-present" for r in report.results)

    def test_no_system_json_files(self, tmp_path):
        """system-description-present error when systems/ has no *.json files."""
        (tmp_path / "systems").mkdir()
        (tmp_path / "pareto").mkdir()
        (tmp_path / "documentation").mkdir()
        report = _check(tmp_path)
        assert _errors(report, "system-description-present")

    def test_invalid_system_json(self, tmp_path):
        """system-description-valid error when system JSON is malformed."""
        (tmp_path / "systems").mkdir()
        (tmp_path / "pareto").mkdir()
        (tmp_path / "documentation").mkdir()
        (tmp_path / "systems" / "bad-sys.json").write_text("{bad json")
        report = _check(tmp_path)
        assert _errors(report, "system-description-valid")

    def test_missing_pareto_system_dir_early_exit(self, tmp_path):
        """pareto-dir-exists error when pareto/<system_id>/ is absent."""
        (tmp_path / "systems").mkdir()
        (tmp_path / "pareto").mkdir()
        (tmp_path / "documentation").mkdir()
        (tmp_path / "systems" / "test-sys.json").write_text(json.dumps(_SYSTEM_DESC))
        report = _check(tmp_path)
        assert _errors(report, "pareto-dir-exists")

    def test_empty_pareto_system_dir(self, tmp_path):
        """benchmark-model-dir error when pareto/<system_id>/ has no subdirectories."""
        (tmp_path / "systems").mkdir()
        (tmp_path / "documentation").mkdir()
        (tmp_path / "systems" / "test-sys.json").write_text(json.dumps(_SYSTEM_DESC))
        pareto_sys = tmp_path / "pareto" / "test-sys"
        pareto_sys.mkdir(parents=True)
        report = _check(tmp_path)
        assert _errors(report, "benchmark-model-dir")

    def test_missing_model_subdirs_early_exit(self, tmp_path):
        """pareto-subdir error when points/ or results/ is absent."""
        (tmp_path / "systems").mkdir()
        (tmp_path / "documentation").mkdir()
        (tmp_path / "systems" / "test-sys.json").write_text(json.dumps(_SYSTEM_DESC))
        model_dir = tmp_path / "pareto" / "test-sys" / "llama3-70b"
        # Only points/ present — results/ missing
        (model_dir / "points").mkdir(parents=True)
        report = _check(tmp_path)
        assert _errors(report, "pareto-subdir")
        # Should not attempt to list point_*.yaml (early exit after structure errors)
        assert not any(r.rule == "measurement-points-present" for r in report.results)

    def test_no_point_yamls(self, tmp_path):
        """measurement-points-present error when points/ has no point_*.yaml files."""
        root = _build_submission(tmp_path, write_runs=False, write_results=False)
        report = _check(root)
        assert _errors(report, "measurement-points-present")

    def test_missing_result_log(self, tmp_path):
        """result-file-present error when results/point_<N>/ log is absent."""
        root = _build_submission(tmp_path, write_results=False)
        report = _check(root)
        assert _errors(report, "result-file-present")

    def test_missing_run_metadata_errors(self, tmp_path):
        """run-metadata-present error when run_metadata.json is absent from a point."""
        root = _build_submission(tmp_path, write_run_metadata=False)
        report = _check(root)
        assert _errors(report, "run-metadata-present")

    def test_valid_run_metadata_ok(self, tmp_path):
        """A fully-populated run_metadata.json produces a run-metadata-valid ok and no errors."""
        root = _build_submission(tmp_path)
        report = _check(root)
        assert not _errors(report, "run-metadata-present")
        assert not _errors(report, "run-metadata-valid")
        assert any(r.rule == "run-metadata-valid" for r in report.results)

    def test_invalid_run_metadata_errors(self, tmp_path):
        """A null measurement in run_metadata.json produces a run-metadata-valid error."""
        root = _build_submission(tmp_path)
        # Null out a measurement that must be non-null in every point's metadata.
        for md in root.rglob("run_metadata.json"):
            data = json.loads(md.read_text())
            data["measured_latency_ttft_p99"] = None
            md.write_text(json.dumps(data))
        report = _check(root)
        assert _errors(report, "run-metadata-valid")

    def _set_tps(self, root, values: dict[int, tuple[float, float]]) -> None:
        """Set (system_tps, tps_utilization) per concurrency in each run_metadata.json."""
        for md in root.rglob("run_metadata.json"):
            data = json.loads(md.read_text())
            tps, util = values[data["concurrency"]]
            data["system_tps"], data["tps_utilization"] = tps, util
            md.write_text(json.dumps(data))

    def test_tps_utilization_consistent_passes(self, tmp_path):
        """Correctly normalised tps_utilization yields no tps-utilization error."""
        root = _build_submission(tmp_path, concurrencies=[16, 38])
        # max system_tps = 200 → expected utils 0.5 and 1.0
        self._set_tps(root, {16: (100.0, 0.5), 38: (200.0, 1.0)})
        report = _check(root)
        assert not _errors(report, "tps-utilization")
        assert any(r.rule == "tps-utilization" for r in report.results)

    def test_tps_utilization_within_tolerance_passes(self, tmp_path):
        """A value off by < 0.1 from expected is accepted."""
        root = _build_submission(tmp_path, concurrencies=[16, 38])
        # expected for 16 is 0.5; 0.55 is within abs tol 0.1
        self._set_tps(root, {16: (100.0, 0.55), 38: (200.0, 1.0)})
        report = _check(root)
        assert not _errors(report, "tps-utilization")

    def test_tps_utilization_out_of_tolerance_errors(self, tmp_path):
        """A value off by > 0.1 from expected produces a tps-utilization error."""
        root = _build_submission(tmp_path, concurrencies=[16, 38])
        # expected for 16 is 0.5; 0.8 is off by 0.3 > 0.1
        self._set_tps(root, {16: (100.0, 0.8), 38: (200.0, 1.0)})
        report = _check(root)
        assert _errors(report, "tps-utilization")

    def test_missing_config_yaml(self, tmp_path):
        """result-file-present error when config.yaml is absent from a result dir."""
        root = _build_submission(tmp_path)
        config_yaml = (
            root / "pareto" / "test-sys" / "llama3-70b" / "results" / "point_16" / "config.yaml"
        )
        config_yaml.unlink()
        report = _check(root)
        assert _errors(report, "result-file-present")

    def test_invalid_result_log(self, tmp_path):
        """result-file-valid error when the result log JSON is malformed."""
        root = _build_submission(tmp_path)
        # Overwrite one summary with invalid JSON
        bad_path = root / "pareto" / "test-sys" / "llama3-70b" / "results" / "point_16"
        bad_path.mkdir(parents=True, exist_ok=True)
        (bad_path / "results_summary.json").write_text("{bad")
        report = _check(root)
        assert _errors(report, "result-file-valid")

    def test_missing_accuracy_results_json(self, tmp_path):
        """accuracy-file error when accuracy/results.json is absent;
        accuracy-present error because no model has accuracy data."""
        root = _build_submission(tmp_path, write_accuracy_json=False)
        report = _check(root)
        assert _errors(report, "accuracy-file")
        assert _errors(report, "accuracy-present")

    def test_invalid_accuracy_json(self, tmp_path):
        """accuracy-valid error when accuracy/results.json is malformed."""
        root = _build_submission(
            tmp_path,
            accuracy_data={"cnn_dailymail": "not-a-dict"},  # value must be a dict
        )
        report = _check(root)
        assert _errors(report, "accuracy-valid")

    def test_accuracy_scores_in_results_json(self, tmp_path):
        """Accuracy is read from a point's results.json accuracy_scores (no accuracy/ dir)."""
        root = _build_submission(tmp_path, write_accuracy_json=False)
        # Drop accuracy_scores into the first point's results.json. The checker reads it
        # directly; the empty accuracy/ dir is not consulted (no accuracy-file error).
        first_point = (
            root / "pareto" / "test-sys" / "llama3-70b" / "results" / f"point_{_CONCURRENCIES[0]}"
        )
        (first_point / "results.json").write_text(
            json.dumps({"config": {}, "results": {}, "accuracy_scores": _ACCURACY, "responses": []})
        )
        report = _check(root)
        assert not _errors(report, "accuracy-present")
        assert not _errors(report, "accuracy-valid")
        assert not _errors(report, "accuracy-file")

    def test_invalid_accuracy_scores_in_results_json(self, tmp_path):
        """accuracy-valid error when results.json accuracy_scores is malformed."""
        root = _build_submission(tmp_path, write_accuracy_json=False)
        first_point = (
            root / "pareto" / "test-sys" / "llama3-70b" / "results" / f"point_{_CONCURRENCIES[0]}"
        )
        (first_point / "results.json").write_text(
            json.dumps({"accuracy_scores": {"cnn_dailymail": "not-a-dict"}})
        )
        report = _check(root)
        assert _errors(report, "accuracy-valid")

    def test_point_filename_concurrency_mismatch(self, tmp_path):
        """point-filename-concurrency warning when filename concurrency ≠ declared concurrency."""
        root = _build_submission(tmp_path)
        # Add a point file whose name says 999 but YAML declares 64
        mismatch_yaml = root / "pareto" / "test-sys" / "llama3-70b" / "points" / "point_999.yaml"
        mismatch_yaml.write_text(yaml.dump(_make_run_yaml(64)))
        # Also add the matching result dir so it doesn't error on result-file-present
        result_dir = root / "pareto" / "test-sys" / "llama3-70b" / "results" / "point_64"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "results_summary.json").write_text(json.dumps(_SUMMARY))
        report = _check(root)
        assert _warnings(report, "point-filename-concurrency")

    def test_invalid_point_yaml_is_skipped(self, tmp_path):
        """A point_*.yaml that fails validation does not crash the checker."""
        root = _build_submission(tmp_path)
        bad_yaml = root / "pareto" / "test-sys" / "llama3-70b" / "points" / "point_99.yaml"
        bad_yaml.write_text("{bad yaml [")
        report = _check(root)
        # Should produce a point-config-valid error for the bad file
        assert _errors(report, "point-config-valid")

    def test_region_computation_error(self, tmp_path):
        """region-computation error when compute_regions raises ValueError."""
        # compute_regions only raises if M <= 32, but SystemDescription enforces M > 32.
        # Patch compute_regions to simulate an unexpected ValueError.
        (tmp_path / "systems").mkdir()
        (tmp_path / "documentation").mkdir()
        (tmp_path / "systems" / "test-sys.json").write_text(json.dumps(_SYSTEM_DESC))
        pareto_sys = tmp_path / "pareto" / "test-sys"
        pareto_sys.mkdir(parents=True)
        (pareto_sys / "llama3-70b").mkdir()
        with patch(
            "submission_checker.checker.compute_regions",
            side_effect=ValueError("M must be > 32"),
        ):
            report = _check(tmp_path)
        assert _errors(report, "region-computation")

    def test_model_name_matches_dir(self, tmp_path):
        """ok when model_id in system_desc matches the model directory name."""
        desc = {**_SYSTEM_DESC, "model_id": "llama3-70b"}
        root = _build_submission(tmp_path, system_desc=desc, model="llama3-70b")
        report = _check(root)
        ok_results = [r for r in report.results if r.rule == "model-name-consistency" and r.passed]
        assert ok_results

    def test_model_name_mismatch_errors(self, tmp_path):
        """err when model_id in system_desc does not match the model directory name."""
        desc = {**_SYSTEM_DESC, "model_id": "mistral-7b"}
        root = _build_submission(tmp_path, system_desc=desc, model="llama3-70b")
        report = _check(root)
        assert _errors(report, "model-name-consistency")

    def test_model_name_huggingface_format_matches(self, tmp_path):
        """ok when model_id uses HuggingFace org/name format — last component compared."""
        desc = {**_SYSTEM_DESC, "model_id": "meta-llama/llama3-70b"}
        root = _build_submission(tmp_path, system_desc=desc, model="llama3-70b")
        report = _check(root)
        ok_results = [r for r in report.results if r.rule == "model-name-consistency" and r.passed]
        assert ok_results

    def test_model_name_allowed_passes(self, tmp_path):
        """ok when system_desc.model_name is one of the allowed benchmark models."""
        desc = {**_SYSTEM_DESC, "model_name": "gpt-oss-120b"}
        root = _build_submission(tmp_path, system_desc=desc, model="gpt-oss-120b")
        report = _check(root)
        assert not _errors(report, "model-name-valid")
        assert [r for r in report.results if r.rule == "model-name-valid" and r.passed]

    def test_model_name_not_allowed_errors(self, tmp_path):
        """err when system_desc.model_name is not an allowed benchmark model."""
        desc = {**_SYSTEM_DESC, "model_name": "mistral-7b"}
        root = _build_submission(tmp_path, system_desc=desc, model="mistral-7b")
        report = _check(root)
        assert _errors(report, "model-name-valid")

    def test_model_name_missing_errors(self, tmp_path):
        """err when system_desc has no model_name (must be one of the allowed set)."""
        root = _build_submission(tmp_path, model="llama3-70b")  # _SYSTEM_DESC has no model_name
        report = _check(root)
        assert _errors(report, "model-name-valid")

    def test_run_filename_non_numeric_suffix_ignored(self, tmp_path):
        """Filename parsing errors (non-numeric suffix) are silently ignored."""
        root = _build_submission(tmp_path)
        # point_abc.yaml — stem is "point_abc", int("abc") raises ValueError
        # This tests the except (IndexError, ValueError): pass branch
        bad_name = root / "pareto" / "test-sys" / "llama3-70b" / "points" / "point_abc.yaml"
        bad_name.write_text(yaml.dump(_make_run_yaml(64)))
        result_dir = root / "pareto" / "test-sys" / "llama3-70b" / "results" / "point_64"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "results_summary.json").write_text(json.dumps(_SUMMARY))
        report = _check(root)
        # No run-filename-concurrency warning — the ValueError was swallowed
        assert not _warnings(report, "point-filename-concurrency")
