"""Tests for SubmissionChecker using pre-built fixtures from test_submissions/."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from submission_checker import layout
from submission_checker.checker import SubmissionChecker
from submission_checker.models import CheckResult, Report, Severity

#: The three §9.1 per-region coverage rules. Ultra Low Concurrency is checked
#: separately against the fixed 1–32 band, so it is not one of these.
_COVERAGE_RULES = (
    "low-concurrency-coverage",
    "med-concurrency-coverage",
    "high-concurrency-coverage",
)

_ALL_SUB_FIXTURES = (
    "sub_a",
    "sub_b",
    "sub_c",
    "sub_d",
    "sub_e",
    "sub_f",
    "sub_g",
    "sub_h",
    "sub_i",
    "sub_j",
)

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
    """The corpus's must-pass tree. C_max=1000 with a derived C_min=16."""

    def test_passes_overall(self, valid_standardized):
        report = _check(valid_standardized)
        assert report.passed, [f"{r.rule}: {r.message}" for r in report.errors]

    def test_all_regions_covered(self, valid_standardized):
        report = _check(valid_standardized)
        for rule in _COVERAGE_RULES:
            assert not _errors(report, rule), f"{rule} should pass"

    def test_metric_consistency(self, valid_standardized):
        report = _check(valid_standardized)
        assert not _errors(report, "metric-consistency-duration")
        assert not _errors(report, "metric-consistency-accounting")
        assert not _errors(report, "metric-consistency-tpot-p90")

    def test_accuracy_gate(self, valid_standardized):
        assert not _errors(_check(valid_standardized), "accuracy-gate")

    def test_point_count(self, valid_standardized):
        assert not _errors(_check(valid_standardized), "point-count")

    def test_seed_binding(self, valid_standardized):
        report = _check(valid_standardized)
        for rule in ("seed-set-consistency", "seed-set-membership", "seed-runtime-match"):
            assert not _errors(report, rule), f"{rule} should pass"

    def test_shared_paths_resolve(self, valid_standardized):
        assert not _errors(_check(valid_standardized), "shared-path-resolution")


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

    def test_missing_concurrency_regions(self, invalid_submission):
        """C_max=88, C_min=16 → low 17–20, med 21–33; points are 16/38/88."""
        report = _check(invalid_submission)
        assert _errors(report, "low-concurrency-coverage")
        assert _errors(report, "med-concurrency-coverage")


# ---------------------------------------------------------------------------
# Fixture coverage expectations, re-derived under v1.0's per-submission regions.
#
# v0.7 fixed Low Concurrency at 33–42 regardless of the submission, so most of the
# corpus skipped it. v1.0 derives the boundaries from each curve's own C_min, which
# moves the window down to wherever the submitter's points actually start — and eight
# of these ten trees now cover it. The three that still miss it are the ones whose
# points start high relative to C_max.
# ---------------------------------------------------------------------------

#: Fixtures whose points cover all three concurrency regions under v1.0.
_FULLY_COVERING = ("sub_a", "sub_b", "sub_c", "sub_d", "sub_e", "sub_f", "sub_i")

#: (fixture, rules that must error) for the trees with a genuine coverage gap.
_COVERAGE_GAPS = (
    # C_max=2048, points start at 64 → C_min clamps to 32, low is 33–45.
    ("sub_g", ("low-concurrency-coverage", "ultra-low-concurrency-coverage")),
    ("sub_h", ("low-concurrency-coverage", "ultra-low-concurrency-coverage")),
    # C_max=16384, points start at 32 → low is 33–57, and 64 is already Medium.
    ("sub_j", ("low-concurrency-coverage",)),
)


@pytest.mark.parametrize("fixture_name", _FULLY_COVERING)
def test_fixture_covers_every_region(request, fixture_name):
    report = _check(request.getfixturevalue(fixture_name))
    for rule in _COVERAGE_RULES:
        assert not _errors(report, rule), f"{fixture_name}: {rule} should pass"
    assert not _errors(report, "ultra-low-concurrency-coverage")


@pytest.mark.parametrize("fixture_name, failing_rules", _COVERAGE_GAPS)
def test_fixture_coverage_gaps(request, fixture_name, failing_rules):
    report = _check(request.getfixturevalue(fixture_name))
    for rule in failing_rules:
        assert _errors(report, rule), f"{fixture_name}: {rule} should fail"
    for rule in set(_COVERAGE_RULES) - set(failing_rules):
        assert not _errors(report, rule), f"{fixture_name}: {rule} should pass"


@pytest.mark.parametrize("fixture_name", _ALL_SUB_FIXTURES)
def test_fixture_point_count_passes(request, fixture_name):
    assert not _errors(_check(request.getfixturevalue(fixture_name)), "point-count")


@pytest.mark.parametrize("fixture_name", _ALL_SUB_FIXTURES)
def test_fixture_metric_consistency(request, fixture_name):
    report = _check(request.getfixturevalue(fixture_name))
    assert not _errors(report, "metric-consistency-duration")
    assert not _errors(report, "metric-consistency-accounting")


@pytest.mark.parametrize("fixture_name", ("sub_a", "sub_c", "sub_e", "sub_j"))
def test_fixture_accuracy_passes(request, fixture_name):
    assert not _errors(_check(request.getfixturevalue(fixture_name)), "accuracy-gate")


@pytest.mark.parametrize("fixture_name", _ALL_SUB_FIXTURES)
def test_run_duration_is_flagged_not_rejected(request, fixture_name):
    """§9.1's failure action for run duration is "Flag", never "Reject"."""
    report = _check(request.getfixturevalue(fixture_name))
    assert [r for r in report.results if r.rule == "point-duration"], "rule did not run"
    assert not _errors(report, "point-duration")


# ---------------------------------------------------------------------------
# Targeted edge-case tests that build synthetic dirs to cover checker.py paths
# ---------------------------------------------------------------------------

_SYSTEM_DESC = {
    "submitter_org_names": "Test Org",
    "submitter_contact": "contact@example.com",
    "system_name": "test-sys",
    "system_category": "datacenter",
    "publication_status": "Available",
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
            "host_networking": "InfiniBand",
            "host_network_card_count": "4x NIC",
            "host_storage_type": "NVMe",
            "host_storage_capacity": "10 TB",
            "operating_system": "Ubuntu 22.04",
            "host_memory_configuration": "8x 64GB DDR5",
            "driver": "550.54",
            "filesystem": "ext4",
            "accelerator_info": [
                {
                    "accelerator_model_name": "H100",
                    "accelerators_per_node": 8,
                    "accelerator_memory_capacity": "80 GB",
                    "accelerator_memory_type": "HBM3",
                }
            ],
        }
    ],
    "division": "Serviced",
    "system_size": "1 node",
    "system_node_ensemble_count": 1,
    "system_node_ensemble_total": 1,
    "model_id": "test-model",
    "node_config": "8x H100 test node",
    "config_summary": "TP 1, PP 1, DP 1",
    "tps_utilization": 1.0,
}

_SUMMARY = {
    "n_samples_issued": 1000,
    "n_samples_completed": 1000,
    "n_samples_failed": 0,
    "duration_ns": 1_200_000_000_000.0,
    "ttft": {
        "total": 0.0,
        "percentiles": {"50": 150_000_000.0, "90": 270_000_000.0, "95": 300_000_000.0},
    },
    "tpot": {"total": 0.0, "percentiles": {"50": 4_000_000.0, "90": 5_500_000.0}},
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

#: Concurrencies covering every region for C_max=1024 with a derived C_min=16:
#: low 17–26 → 20; med 27–117 → 88; high 118–1024 → 256, 512, 768, 1000.
_CONCURRENCIES = [16, 20, 88, 256, 512, 768, 1000]

#: Seed set A from data/seed_sets.yaml, mirrored from policies PR #117.
_SEEDS = {
    "scheduler_rng_seed": 10487924139932647040,
    "sample_index_rng_seed": 586478644936801402,
    "model_seed": 9315206023656308754,
}


def _make_run_yaml(concurrency: int) -> dict:
    """A §8.3-complete point.yaml, so a test sees only the defect it introduced."""
    return {
        "concurrency": concurrency,
        "dataset": "llm-perf-dataset-v1",
        "division": "Serviced",
        "max_supported_concurrency": 1024,
        "model_name": "llama3.1-8b",
        "model_precision": "FP16",
        "link_to_model": "https://example.com/model",
        "link_to_model_transformation": "https://example.com/quantization",
        "model_notes": "",
        "dataset_name": "CNN/DailyMail",
        "dataset_type": "Performance",
        "dataset_link": "https://example.com/dataset",
        "shared_src": "src",
        "shared_docs": "docs",
        "seed_set": "A",
        "target_cohort": "2026-09-C0",
        "warmup": {
            "duration_s": 60.0,
            "requests_issued": concurrency * 10,
            "requests_completed": concurrency * 10,
            "data_source": "llm-perf-dataset-v1 validation split",
            "concurrency": concurrency,
            "initialization_steps": ["model loaded"],
            "logs_retained": True,
        },
        "runtime_settings": {
            "load_pattern": "concurrency",
            "min_duration_ms": 1_200_000,
            "stream_all_chunks": True,
            "runtime": dict(_SEEDS),
        },
    }


def _build_submission(
    root: Path,
    system_id: str = "test-sys",
    system_desc: dict | None = None,
    concurrencies: list[int] | None = None,
    write_runs: bool = True,
    write_results: bool = True,
    write_accuracy_json: bool = True,
    write_system_desc: bool = True,
    write_config_yaml: bool = True,
    accuracy_data: dict | None = None,
    model: str = "llama3-70b",
) -> Path:
    """Build a minimal valid (or deliberately broken) submission directory.

    Since policies PR #119 there is no per-system file: the system description is
    written into every ``r<N>/`` alongside the point's own artifacts.
    """
    desc = system_desc if system_desc is not None else _SYSTEM_DESC.copy()
    concs = concurrencies if concurrencies is not None else _CONCURRENCIES

    results_dir = root / "results" / system_id
    results_dir.mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    impl_dir = root / "src" / "trtllm"
    impl_dir.mkdir(parents=True, exist_ok=True)
    (impl_dir / "README.md").write_text("# trtllm\n")

    model_dir = results_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    for c in concs:
        point_dir = model_dir / f"r{c}"
        point_dir.mkdir(parents=True, exist_ok=True)
        if write_system_desc:
            (point_dir / "system_desc.json").write_text(json.dumps(desc))
        if write_runs:
            (point_dir / "point.yaml").write_text(yaml.dump(_make_run_yaml(c)))
        if write_results:
            (point_dir / "result_summary.json").write_text(json.dumps(_SUMMARY))
            if write_config_yaml:
                (point_dir / "config.yaml").write_text(yaml.dump({"concurrency": c}))
            # Write accuracy into the first point only; checker scans all points
            if c == concs[0] and write_accuracy_json:
                data = accuracy_data if accuracy_data is not None else _ACCURACY
                (point_dir / "accuracy_results.json").write_text(json.dumps(data))

    return root


class TestCheckerEdgeCases:
    """Targeted tests to cover checker.py error paths not exercised by fixture tests."""

    def test_nonexistent_path(self, tmp_path):
        """path-exists error when submission_path does not exist."""
        report = _check(tmp_path / "does_not_exist")
        assert _errors(report, "path-exists")

    def test_missing_required_dirs_early_exit(self, tmp_path):
        """SubmissionDir structure errors cause early return from run()."""
        # Only results/ present — docs/ missing → structure error → early exit
        (tmp_path / "results").mkdir()
        report = _check(tmp_path)
        assert _errors(report, "required-dir")
        # Should not have processed any systems
        assert not any(r.rule == "system-description-present" for r in report.results)

    def test_no_system_directories(self, tmp_path):
        """system-results-dir error when results/ holds no system directories."""
        (tmp_path / "results").mkdir()
        (tmp_path / "docs").mkdir()
        report = _check(tmp_path)
        assert _errors(report, "system-results-dir")

    def test_missing_point_system_desc(self, tmp_path):
        """system-description-present error when a point has no system_desc.json."""
        root = _build_submission(tmp_path, write_system_desc=False)
        report = _check(root)
        assert _errors(report, "system-description-present")

    def test_invalid_system_json(self, tmp_path):
        """system-description-valid error when a point's system_desc.json is malformed."""
        root = _build_submission(tmp_path)
        for path in root.rglob("system_desc.json"):
            path.write_text("{bad json")
        report = _check(root)
        assert _errors(report, "system-description-valid")

    def test_system_desc_inconsistent_across_curve(self, tmp_path):
        """§8.5: every point of a curve must describe the same system."""
        root = _build_submission(tmp_path)
        changed = root / "results" / "test-sys" / "llama3-70b" / "r88" / "system_desc.json"
        changed.write_text(json.dumps({**_SYSTEM_DESC, "system_name": "a-different-system"}))
        report = _check(root)
        assert _errors(report, "system-description-consistency")

    def test_tps_utilization_may_differ_across_a_curve(self, tmp_path):
        """tps_utilization is per point, so it is exempt from the consistency check."""
        root = _build_submission(tmp_path)
        changed = root / "results" / "test-sys" / "llama3-70b" / "r88" / "system_desc.json"
        changed.write_text(json.dumps({**_SYSTEM_DESC, "tps_utilization": 0.42}))
        report = _check(root)
        assert not _errors(report, "system-description-consistency")

    def test_empty_system_results_dir(self, tmp_path):
        """benchmark-model-dir error when results/<system>/ has no subdirectories."""
        (tmp_path / "results" / "test-sys").mkdir(parents=True)
        (tmp_path / "docs").mkdir()
        report = _check(tmp_path)
        assert _errors(report, "benchmark-model-dir")

    def test_model_dir_without_point_dirs(self, tmp_path):
        """point-dirs error when a model directory holds no r<N>/ directories."""
        (tmp_path / "docs").mkdir()
        sys_dir = tmp_path / "results" / "test-sys"
        sys_dir.mkdir(parents=True)
        (sys_dir / "llama3-70b").mkdir(parents=True)
        report = _check(tmp_path)
        assert _errors(report, "point-dirs")
        # Should not attempt to load point.yaml (early exit after structure errors)
        assert not any(r.rule == "measurement-points-present" for r in report.results)

    def test_no_point_yamls(self, tmp_path):
        """measurement-points-present error when r<N>/ dirs have no point.yaml."""
        root = _build_submission(tmp_path, write_runs=False, write_results=False)
        report = _check(root)
        assert _errors(report, "measurement-points-present")

    def test_missing_result_log(self, tmp_path):
        """result-summary-present error when r<N>/result_summary.json is absent."""
        root = _build_submission(tmp_path, write_results=False)
        report = _check(root)
        assert _errors(report, "result-summary-present")

    def _set_tps(self, root, values: dict[int, tuple[float, float]]) -> None:
        """Set each point's measured throughput and its declared tps_utilization.

        ``system_tps`` is not stored any more — the checker derives it from the
        measurement — so the throughput is set by choosing the output-token total that
        produces it over the summary's fixed duration.
        """
        duration_s = _SUMMARY["duration_ns"] / 1e9
        for _system, model_dir in layout.iter_curves(root / "results"):
            for point_dir in layout.iter_point_dirs(model_dir):
                concurrency = layout.parse_point_dir(point_dir.name)
                if concurrency not in values:
                    continue
                tps, util = values[concurrency]
                summary = json.loads((point_dir / "result_summary.json").read_text())
                summary["output_sequence_lengths"] = {
                    "total": tps * duration_s,
                    "percentiles": {},
                }
                summary["system_tps"] = tps
                (point_dir / "result_summary.json").write_text(json.dumps(summary))
                desc_path = point_dir / "system_desc.json"
                desc = json.loads(desc_path.read_text())
                desc["tps_utilization"] = util
                desc_path.write_text(json.dumps(desc))

    def test_tps_utilization_consistent_passes(self, tmp_path):
        """Correctly normalised tps_utilization yields no tps-utilization error."""
        root = _build_submission(tmp_path, concurrencies=[16, 20])
        # max system_tps = 200 → expected utils 0.5 and 1.0
        self._set_tps(root, {16: (100.0, 0.5), 20: (200.0, 1.0)})
        report = _check(root)
        assert not _errors(report, "tps-utilization")
        assert any(r.rule == "tps-utilization" for r in report.results)

    def test_tps_utilization_within_tolerance_passes(self, tmp_path):
        """A value off by < 0.1 from expected is accepted."""
        root = _build_submission(tmp_path, concurrencies=[16, 20])
        # expected for 16 is 0.5; 0.55 is within abs tol 0.1
        self._set_tps(root, {16: (100.0, 0.55), 20: (200.0, 1.0)})
        report = _check(root)
        assert not _errors(report, "tps-utilization")

    def test_tps_utilization_out_of_tolerance_errors(self, tmp_path):
        """A value off by > 0.1 from expected produces a tps-utilization error."""
        root = _build_submission(tmp_path, concurrencies=[16, 20])
        # expected for 16 is 0.5; 0.8 is off by 0.3 > 0.1
        self._set_tps(root, {16: (100.0, 0.8), 20: (200.0, 1.0)})
        report = _check(root)
        assert _errors(report, "tps-utilization")

    def _add_curve(self, root, system_id, model, values: dict[int, tuple[float, float]]):
        """Add a second <system_id>/<model> pareto curve with per-point (tps, util)."""
        model_dir = root / "results" / system_id / model
        model_dir.mkdir(parents=True, exist_ok=True)
        duration_s = _SUMMARY["duration_ns"] / 1e9
        for c, (tps, util) in values.items():
            rd = model_dir / f"r{c}"
            rd.mkdir(exist_ok=True)
            (rd / "point.yaml").write_text(yaml.dump(_make_run_yaml(c)))
            summary = {
                **_SUMMARY,
                "output_sequence_lengths": {"total": tps * duration_s, "percentiles": {}},
                "system_tps": tps,
            }
            (rd / "result_summary.json").write_text(json.dumps(summary))
            (rd / "config.yaml").write_text(yaml.dump({"concurrency": c}))
            (rd / "system_desc.json").write_text(
                json.dumps({**_SYSTEM_DESC, "tps_utilization": util})
            )
            (rd / "accuracy_results.json").write_text(json.dumps(_ACCURACY))

    def test_tps_utilization_normalized_per_curve(self, tmp_path):
        """Each system/model curve normalizes to its OWN peak, not a submission-wide max.

        Regression test: a small system (peak 200) alongside a large one (peak 2000)
        must not be forced to divide by the large system's peak. With the old
        submission-wide max this errored on every small-system point.
        """
        root = _build_submission(tmp_path, system_id="sys-small", concurrencies=[16, 20])
        self._set_tps(root, {16: (100.0, 0.5), 20: (200.0, 1.0)})  # own peak 200
        self._add_curve(root, "sys-big", "llama3-70b", {16: (1000.0, 0.5), 20: (2000.0, 1.0)})
        report = _check(root)
        assert not _errors(report, "tps-utilization")
        assert any(r.rule == "tps-utilization" for r in report.results)

    def test_tps_utilization_per_curve_detects_error(self, tmp_path):
        """A wrongly-normalized value is still caught within its own curve."""
        root = _build_submission(tmp_path, system_id="sys-small", concurrencies=[16, 20])
        self._set_tps(root, {16: (100.0, 0.5), 20: (200.0, 1.0)})  # correct
        # sys-big point 16 expects 0.5 but stores 0.9 (off by 0.4 > tol)
        self._add_curve(root, "sys-big", "llama3-70b", {16: (1000.0, 0.9), 20: (2000.0, 1.0)})
        report = _check(root)
        assert _errors(report, "tps-utilization")

    def test_missing_config_yaml_is_not_an_error(self, tmp_path):
        """config.yaml is optional as of v1.0 — point.yaml carries the disclosure."""
        root = _build_submission(tmp_path, write_config_yaml=False)
        report = _check(root)
        assert not [r for r in report.errors if "config.yaml" in r.message]
        assert not _errors(report, "result-summary-present")

    def test_invalid_result_log(self, tmp_path):
        """result-file-valid error when the result log JSON is malformed."""
        root = _build_submission(tmp_path)
        # Overwrite one summary with invalid JSON
        bad_path = root / "results" / "test-sys" / "llama3-70b" / "r16"
        bad_path.mkdir(parents=True, exist_ok=True)
        (bad_path / "result_summary.json").write_text("{bad")
        report = _check(root)
        assert _errors(report, "result-file-valid")

    def test_missing_accuracy_results_json(self, tmp_path):
        """accuracy-present error because no model carries accuracy data at all."""
        root = _build_submission(tmp_path, write_accuracy_json=False)
        report = _check(root)
        assert _errors(report, "accuracy-present")

    def test_invalid_accuracy_json(self, tmp_path):
        """accuracy-valid error when accuracy_results.json is malformed."""
        root = _build_submission(
            tmp_path,
            accuracy_data={"cnn_dailymail": "not-a-dict"},  # value must be a dict
        )
        report = _check(root)
        assert _errors(report, "accuracy-valid")

    def test_accuracy_scores_in_results_json(self, tmp_path):
        """Accuracy falls back to a point's results.json accuracy_scores."""
        root = _build_submission(tmp_path, write_accuracy_json=False)
        # Drop accuracy_scores into the first point's results.json; with no
        # accuracy_results.json beside it, the checker falls back to this.
        first_point = root / "results" / "test-sys" / "llama3-70b" / f"r{_CONCURRENCIES[0]}"
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
        first_point = root / "results" / "test-sys" / "llama3-70b" / f"r{_CONCURRENCIES[0]}"
        (first_point / "results.json").write_text(
            json.dumps({"accuracy_scores": {"cnn_dailymail": "not-a-dict"}})
        )
        report = _check(root)
        assert _errors(report, "accuracy-valid")

    def test_point_dirname_concurrency_mismatch(self, tmp_path):
        """point-dirname-concurrency warning when r<N>/ disagrees with declared concurrency."""
        root = _build_submission(tmp_path)
        # A point directory named r999 whose point.yaml declares 64.
        mismatch_dir = root / "results" / "test-sys" / "llama3-70b" / "r999"
        mismatch_dir.mkdir(parents=True, exist_ok=True)
        (mismatch_dir / "point.yaml").write_text(yaml.dump(_make_run_yaml(64)))
        # Result log too, so it doesn't error on result-file-present instead.
        (mismatch_dir / "result_summary.json").write_text(json.dumps(_SUMMARY))
        report = _check(root)
        assert _warnings(report, "point-dirname-concurrency")

    def test_invalid_point_yaml_is_skipped(self, tmp_path):
        """A point.yaml that fails validation does not crash the checker."""
        root = _build_submission(tmp_path)
        bad_dir = root / "results" / "test-sys" / "llama3-70b" / "r99"
        bad_dir.mkdir(parents=True, exist_ok=True)
        (bad_dir / "point.yaml").write_text("{bad yaml [")
        report = _check(root)
        # Should produce a point-config-valid error for the bad file
        assert _errors(report, "point-config-valid")

    def test_region_computation_error(self, tmp_path):
        """region-computation error when compute_regions raises ValueError."""
        # compute_regions only raises if M <= 32, but SystemDescription enforces M > 32.
        # Patch compute_regions to simulate an unexpected ValueError.
        root = _build_submission(tmp_path)
        with patch(
            "submission_checker.checker.compute_regions",
            side_effect=ValueError("C_max must be > 32"),
        ):
            report = _check(root)
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

    def test_non_point_directory_is_ignored(self, tmp_path):
        """A directory that is not r<digits> is not treated as a Pareto point."""
        root = _build_submission(tmp_path)
        # "rabc" and "server_configs" are not point directories, so their contents
        # are never loaded and produce no concurrency warning.
        stray = root / "results" / "test-sys" / "llama3-70b" / "rabc"
        stray.mkdir(parents=True, exist_ok=True)
        (stray / "point.yaml").write_text(yaml.dump(_make_run_yaml(64)))
        report = _check(root)
        assert not _warnings(report, "point-dirname-concurrency")
