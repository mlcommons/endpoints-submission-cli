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
        # Only 3 points (c16, c38, c88) — no high-throughput coverage
        assert _errors(report, "high-throughput-coverage")


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
    "division": "Serviced",
    "publication_status": "Available",
    "benchmark_model": "llama3-70b",
    "max_supported_concurrency": 1024,
    "endpoint_url": "http://localhost",
    "serving_framework": "vLLM",
    "submitter": "Test Org",
    "system_name": "test-sys",
    "system_type": "datacenter",
    "system_type_detail": "",
    "number_of_nodes": 1,
    "host_processors_per_node": 2,
    "host_processor_model_name": "AMD EPYC",
    "host_processor_core_count": 64,
    "host_memory_capacity": "512 GB",
    "host_storage_type": "NVMe",
    "host_storage_capacity": "10 TB",
    "host_networking": "InfiniBand",
    "host_networking_topology": "Single switch",
    "accelerators_per_node": 8,
    "accelerator_model_name": "H100",
    "accelerator_memory_capacity": "80 GB",
    "operating_system": "Ubuntu 22.04",
}

_SUMMARY = {
    "n_samples_issued": 1000,
    "n_samples_completed": 1000,
    "n_samples_failed": 0,
    "duration_ns": 1_200_000_000_000.0,
    "ttft": {"total": 0.0, "percentiles": {"50": 150_000_000.0, "95": 300_000_000.0}},
    "output_sequence_lengths": {"total": 500_000.0, "percentiles": {}},
}

_ACCURACY = {"metric": "rouge1", "score": 0.45, "quality_target": 0.43, "passed": True}

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
        },
    }


def _build_submission(
    root: Path,
    system_id: str = "test-sys",
    system_desc: dict | None = None,
    concurrencies: list[int] | None = None,
    write_runs: bool = True,
    write_results: bool = True,
    write_accuracy: bool = True,
    write_accuracy_json: bool = True,
    write_sweep_files: bool = True,
    write_run_metadata: bool = True,
    write_report: bool = True,
    write_src: bool = True,
    accuracy_data: dict | None = None,
    model: str = "llama3-70b",
) -> Path:
    """Build a minimal valid (or deliberately broken) submission directory."""
    desc = system_desc if system_desc is not None else _SYSTEM_DESC.copy()
    desc["benchmark_model"] = model
    concs = concurrencies if concurrencies is not None else _CONCURRENCIES

    system_dir = root / system_id
    system_dir.mkdir(parents=True)
    (system_dir / "system_desc.json").write_text(json.dumps(desc))

    model_dir = system_dir / model
    model_dir.mkdir(parents=True)

    if write_sweep_files:
        (model_dir / "sweep_summary.csv").write_text("concurrency,qps\n")
        (model_dir / "sweep_distributions.csv").write_text("concurrency,percentile\n")

    if write_runs:
        for c in concs:
            run_dir = model_dir / f"r{c}"
            run_dir.mkdir(parents=True)

            (run_dir / "point.yaml").write_text(yaml.dump(_make_run_yaml(c)))

            if write_results:
                (run_dir / "mlperf_endpoints_log_summary.json").write_text(json.dumps(_SUMMARY))
                (run_dir / "mlperf_endpoints_log_detail.json").write_text("{}")

            if write_run_metadata:
                (run_dir / "run_metadata.json").write_text("{}")

            if write_report:
                (run_dir / "report.txt").write_text(f"Run r{c}\n")

            if write_src:
                src_dir = run_dir / "src" / "vllm"
                src_dir.mkdir(parents=True)
                (src_dir / ".gitkeep").write_text("")

            acc_dir = run_dir / "accuracy"
            acc_dir.mkdir()
            if write_accuracy:
                (acc_dir / "accuracy.txt").write_text("ROUGE-1: 0.45")
            if write_accuracy_json:
                data = accuracy_data if accuracy_data is not None else _ACCURACY
                (acc_dir / "accuracy_result.json").write_text(json.dumps(data))

    return root


class TestCheckerEdgeCases:
    """Targeted tests to cover checker.py error paths not exercised by fixture tests."""

    def test_nonexistent_path(self, tmp_path):
        """path-exists error when submission_path does not exist."""
        report = _check(tmp_path / "does_not_exist")
        assert _errors(report, "path-exists")

    def test_no_system_dirs_early_exit(self, tmp_path):
        """system-dir-present error when no directories contain system_desc.json."""
        # Only docs/ present — no system_desc.json anywhere
        (tmp_path / "docs").mkdir()
        report = _check(tmp_path)
        assert _errors(report, "system-dir-present")
        # Should not have processed any systems
        assert not any(r.rule == "system-description-valid" for r in report.results)

    def test_invalid_system_json(self, tmp_path):
        """system-description-valid error when system_desc.json is malformed."""
        sys_dir = tmp_path / "test-sys"
        sys_dir.mkdir()
        (sys_dir / "system_desc.json").write_text("{bad json")
        report = _check(tmp_path)
        assert _errors(report, "system-description-valid")

    def test_no_model_dirs_early_exit(self, tmp_path):
        """benchmark-model-dir error when system dir has no model subdirectories."""
        sys_dir = tmp_path / "test-sys"
        sys_dir.mkdir()
        (sys_dir / "system_desc.json").write_text(json.dumps(_SYSTEM_DESC))
        # No model subdirs
        report = _check(tmp_path)
        assert _errors(report, "benchmark-model-dir")

    def test_docs_dir_not_treated_as_model(self, tmp_path):
        """docs/ inside a system dir is not treated as a model directory."""
        sys_dir = tmp_path / "test-sys"
        sys_dir.mkdir()
        (sys_dir / "system_desc.json").write_text(json.dumps(_SYSTEM_DESC))
        (sys_dir / "docs").mkdir()  # should be ignored as a model dir
        report = _check(tmp_path)
        assert _errors(report, "benchmark-model-dir")

    def test_no_run_dirs(self, tmp_path):
        """measurement-points-present error when model dir has no r<N>/ directories."""
        root = _build_submission(tmp_path, write_runs=False)
        report = _check(root)
        assert _errors(report, "measurement-points-present")

    def test_missing_result_log(self, tmp_path):
        """result-file-present error when mlperf_endpoints_log_summary.json is absent."""
        root = _build_submission(tmp_path, write_results=False)
        report = _check(root)
        assert _errors(report, "result-file-present")

    def test_missing_detail_log(self, tmp_path):
        """result-detail-present error when mlperf_endpoints_log_detail.json is absent."""
        root = _build_submission(tmp_path)
        # Remove the detail log for one run
        detail = root / "test-sys" / "llama3-70b" / "r16" / "mlperf_endpoints_log_detail.json"
        detail.unlink()
        report = _check(root)
        assert _errors(report, "result-detail-present")

    def test_invalid_result_log(self, tmp_path):
        """result-file-valid error when the result log JSON is malformed."""
        root = _build_submission(tmp_path)
        bad_path = root / "test-sys" / "llama3-70b" / "r16" / "mlperf_endpoints_log_summary.json"
        bad_path.write_text("{bad")
        report = _check(root)
        assert _errors(report, "result-file-valid")

    def test_missing_accuracy_txt(self, tmp_path):
        """accuracy-file error when accuracy/accuracy.txt is absent."""
        root = _build_submission(tmp_path, write_accuracy=False)
        report = _check(root)
        assert _errors(report, "accuracy-file")

    def test_missing_accuracy_json(self, tmp_path):
        """accuracy-file error when accuracy/accuracy_result.json is absent."""
        root = _build_submission(tmp_path, write_accuracy_json=False)
        report = _check(root)
        assert _errors(report, "accuracy-file")

    def test_invalid_accuracy_json(self, tmp_path):
        """accuracy-valid error when accuracy_result.json is malformed."""
        root = _build_submission(
            tmp_path,
            accuracy_data={"metric": "rouge1"},  # missing required fields
        )
        report = _check(root)
        assert _errors(report, "accuracy-valid")

    def test_dir_concurrency_mismatch_warning(self, tmp_path):
        """point-filename-concurrency warning when r<N>/ dir concurrency ≠ declared concurrency."""
        root = _build_submission(tmp_path)
        # Add a run dir r999 that declares concurrency 64
        mismatch_dir = root / "test-sys" / "llama3-70b" / "r999"
        mismatch_dir.mkdir()
        (mismatch_dir / "point.yaml").write_text(yaml.dump(_make_run_yaml(64)))
        (mismatch_dir / "mlperf_endpoints_log_summary.json").write_text(json.dumps(_SUMMARY))
        (mismatch_dir / "mlperf_endpoints_log_detail.json").write_text("{}")
        (mismatch_dir / "run_metadata.json").write_text("{}")
        (mismatch_dir / "report.txt").write_text("report\n")
        src = mismatch_dir / "src" / "vllm"
        src.mkdir(parents=True)
        (src / ".gitkeep").write_text("")
        acc = mismatch_dir / "accuracy"
        acc.mkdir()
        (acc / "accuracy.txt").write_text("ok")
        (acc / "accuracy_result.json").write_text(json.dumps(_ACCURACY))
        report = _check(root)
        assert _warnings(report, "point-filename-concurrency")

    def test_invalid_point_yaml_is_skipped(self, tmp_path):
        """A point.yaml that fails validation does not crash the checker."""
        root = _build_submission(tmp_path)
        bad_yaml = root / "test-sys" / "llama3-70b" / "r99"
        bad_yaml.mkdir()
        (bad_yaml / "point.yaml").write_text("{bad yaml [")
        (bad_yaml / "mlperf_endpoints_log_summary.json").write_text(json.dumps(_SUMMARY))
        (bad_yaml / "mlperf_endpoints_log_detail.json").write_text("{}")
        (bad_yaml / "run_metadata.json").write_text("{}")
        (bad_yaml / "report.txt").write_text("report\n")
        src = bad_yaml / "src" / "vllm"
        src.mkdir(parents=True)
        (src / ".gitkeep").write_text("")
        acc = bad_yaml / "accuracy"
        acc.mkdir()
        (acc / "accuracy.txt").write_text("ok")
        (acc / "accuracy_result.json").write_text(json.dumps(_ACCURACY))
        report = _check(root)
        assert _errors(report, "point-config-valid")

    def test_region_computation_error(self, tmp_path):
        """region-computation error when compute_regions raises ValueError."""
        sys_dir = tmp_path / "test-sys"
        sys_dir.mkdir()
        (sys_dir / "system_desc.json").write_text(json.dumps(_SYSTEM_DESC))
        (sys_dir / "llama3-70b").mkdir()
        with patch(
            "submission_checker.checker.compute_regions",
            side_effect=ValueError("M must be > 32"),
        ):
            report = _check(tmp_path)
        assert _errors(report, "region-computation")
