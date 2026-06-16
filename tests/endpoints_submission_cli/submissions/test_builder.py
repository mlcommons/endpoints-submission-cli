# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for submissions.builder module."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

from endpoints_submission_cli.exceptions import SubmissionBuildError
from endpoints_submission_cli.submissions.builder import (
    _compute_max_tps,
    _slugify,
    _truncate_responses,
    build_submission_folder,
    create_bundle_archive,
    extract_archive,
)


@pytest.mark.unit
class TestExtractArchive:
    def test_extracts_files(self, run_archive: Path, tmp_path: Path) -> None:
        dest = tmp_path / "extracted"
        extract_archive(run_archive, dest)
        # Should contain the system_desc.json somewhere
        files = list(dest.rglob("system_desc.json"))
        assert len(files) == 1

    def test_bad_archive_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.tar.gz"
        bad.write_bytes(b"not a tarball")
        with pytest.raises(SubmissionBuildError, match="Failed to extract"):
            extract_archive(bad, tmp_path / "out")

    def test_creates_dest_dir(self, run_archive: Path, tmp_path: Path) -> None:
        dest = tmp_path / "new" / "nested" / "dir"
        extract_archive(run_archive, dest)
        assert dest.is_dir()


@pytest.mark.unit
class TestBuildSubmissionFolder:
    def test_creates_systems_dir(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        assert (sub_dir / "systems").is_dir()

    def test_creates_pareto_dir(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        pareto = sub_dir / "pareto"
        assert pareto.is_dir()

    def test_system_json_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        jsons = list((sub_dir / "systems").glob("*.json"))
        assert len(jsons) == 1
        data = json.loads(jsons[0].read_text())
        assert data["division"] == "Standardized"
        assert "node_types" in data

    def test_point_yaml_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        yamls = list(sub_dir.rglob("point_*.yaml"))
        assert len(yamls) >= 1
        data = yaml.safe_load(yamls[0].read_text())
        assert data["concurrency"] == 4

    def test_log_summary_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        summaries = list(sub_dir.rglob("results_summary.json"))
        assert len(summaries) == 1
        data = json.loads(summaries[0].read_text())
        assert "n_samples_completed" in data
        assert "duration_ns" in data

    def test_accuracy_file_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        acc_jsons = list(sub_dir.rglob("accuracy/results.json"))
        assert len(acc_jsons) == 1

    def test_empty_run_list_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SubmissionBuildError, match="At least one"):
            build_submission_folder([], "standardized", "available", tmp_path)

    def test_missing_system_desc_in_archive_raises(self, tmp_path: Path) -> None:
        # Create archive without system_desc.json
        folder = tmp_path / "bad_run"
        folder.mkdir()
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "result_summary.json").write_text("{}")
        archive = tmp_path / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="bad_run")
        with pytest.raises(SubmissionBuildError, match="system_desc.json"):
            build_submission_folder([("bad", archive)], "standardized", "available", tmp_path / "out")

    def test_system_desc_missing_required_fields_raises(self, tmp_path: Path) -> None:
        # Missing required fields (submitter_org_names, system_name, node_types) must raise
        folder = tmp_path / "bad_run"
        folder.mkdir()
        bad_desc = {"system_category": "datacenter"}
        (folder / "system_desc.json").write_text(json.dumps(bad_desc))
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "result_summary.json").write_text("{}")
        archive = tmp_path / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="bad_run")
        with pytest.raises(SubmissionBuildError, match="schema validation"):
            build_submission_folder([("bad", archive)], "standardized", "available", tmp_path / "out")

    def test_multiple_runs_single_system(
        self, run_archive: Path, run_folder: Path, tmp_path: Path
    ) -> None:
        # Create a second archive with different concurrency
        import shutil

        second_folder = tmp_path / "run2"
        second_folder.mkdir()

        import json as _json
        cfg = yaml.safe_load((run_folder / "config.yaml").read_text())
        rs = _json.loads((run_folder / "result_summary.json").read_text())

        # Copy system info files from the first run
        for fname in ["system_desc.json", "mlperf-system-info-single-node-0.json", "serving_config.json"]:
            src = run_folder / fname
            if src.exists():
                shutil.copy(src, second_folder / fname)

        cfg["settings"]["load_pattern"]["target_concurrency"] = 16
        (second_folder / "config.yaml").write_text(yaml.dump(cfg))
        (second_folder / "result_summary.json").write_text(_json.dumps(rs))

        second_archive = tmp_path / "run2.tar.gz"
        with tarfile.open(second_archive, "w:gz") as tar:
            tar.add(second_folder, arcname="run2")

        sub_dir = build_submission_folder(
            [("run-001", run_archive), ("run-002", second_archive)],
            "standardized",
            "available",
            tmp_path / "sub",
        )
        yamls = list(sub_dir.rglob("point_*.yaml"))
        concurrencies = {yaml.safe_load(p.read_text())["concurrency"] for p in yamls}
        assert 4 in concurrencies
        assert 16 in concurrencies

    def test_system_json_division_from_cli(self, run_archive: Path, tmp_path: Path) -> None:
        # CLI division arg is authoritative — overwrites any placeholder or stale value in system_desc
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "serviced", "available", tmp_path
        )
        jsons = list((sub_dir / "systems").glob("*.json"))
        data = json.loads(jsons[0].read_text())
        assert data["division"] == "Serviced"

    def _make_archive(
        self, run_folder: Path, tmp_path: Path, name: str, concurrency: int, run_type: str
    ) -> Path:
        import shutil

        folder = tmp_path / name
        shutil.copytree(run_folder, folder)
        cfg = yaml.safe_load((folder / "config.yaml").read_text())
        cfg["settings"]["load_pattern"]["target_concurrency"] = concurrency
        cfg["datasets"][0]["type"] = run_type
        (folder / "config.yaml").write_text(yaml.dump(cfg))
        archive = tmp_path / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname=name)
        return archive

    def test_duplicate_performance_concurrency_raises(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        a1 = self._make_archive(run_folder, tmp_path, "run1", 4, "performance")
        a2 = self._make_archive(run_folder, tmp_path, "run2", 4, "performance")
        with pytest.raises(SubmissionBuildError, match="Duplicate"):
            build_submission_folder(
                [("run-001", a1), ("run-002", a2)], "standardized", "available", tmp_path / "sub"
            )

    def test_duplicate_accuracy_concurrency_raises(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        a1 = self._make_archive(run_folder, tmp_path, "run1", 4, "accuracy")
        a2 = self._make_archive(run_folder, tmp_path, "run2", 4, "accuracy")
        with pytest.raises(SubmissionBuildError, match="Duplicate"):
            build_submission_folder(
                [("run-001", a1), ("run-002", a2)], "standardized", "available", tmp_path / "sub"
            )

    def test_perf_and_accuracy_same_concurrency_ok(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        a_perf = self._make_archive(run_folder, tmp_path, "perf", 4, "performance")
        a_acc = self._make_archive(run_folder, tmp_path, "acc", 4, "accuracy")
        sub_dir = build_submission_folder(
            [("run-perf", a_perf), ("run-acc", a_acc)],
            "standardized",
            "available",
            tmp_path / "sub",
        )
        assert (sub_dir / "pareto").glob("*/*/points/point_4.yaml").__next__().exists()
        assert (sub_dir / "pareto").glob("*/*/results/point_4/accuracy/point_4.yaml").__next__().exists()

    def test_accuracy_run_routed_to_accuracy(self, run_folder: Path, tmp_path: Path) -> None:
        a_acc = self._make_archive(run_folder, tmp_path, "acc", 4, "accuracy")
        sub_dir = build_submission_folder(
            [("run-acc", a_acc)], "standardized", "available", tmp_path / "sub"
        )
        model_dirs = list((sub_dir / "pareto").glob("*/*"))
        assert len(model_dirs) == 1
        model_dir = model_dirs[0]
        acc_dir = model_dir / "results" / "point_4" / "accuracy"
        assert (acc_dir / "point_4.yaml").exists()
        assert (acc_dir / "results_summary.json").exists()
        assert not (model_dir / "points" / "point_4.yaml").exists()
        assert not (model_dir / "results" / "point_4" / "results_summary.json").exists()


@pytest.mark.unit
class TestCreateBundleArchive:
    def test_creates_archive(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path / "sub"
        )
        bundle = create_bundle_archive(sub_dir, tmp_path / "bundle.tar.gz")
        assert bundle.exists()
        with tarfile.open(bundle) as tar:
            names = tar.getnames()
        assert any("systems" in n for n in names)

    def test_default_dest(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path / "sub"
        )
        bundle = create_bundle_archive(sub_dir)
        expected = sub_dir.parent / f"{sub_dir.name}.tar.gz"
        try:
            assert bundle == expected
        finally:
            expected.unlink(missing_ok=True)


@pytest.mark.unit
class TestComputeMaxTps:
    def _make_run(self, system_tps: float | None) -> dict:
        meta: dict = {"system_tps": system_tps} if system_tps is not None else {}
        return {"_extra_files": {"run_metadata.json": json.dumps(meta).encode()}}

    def test_single_run(self) -> None:
        run_data = [self._make_run(1000.0)]
        assert _compute_max_tps(run_data) == 1000.0

    def test_multiple_runs_returns_max(self) -> None:
        run_data = [self._make_run(500.0), self._make_run(1500.0), self._make_run(1000.0)]
        assert _compute_max_tps(run_data) == 1500.0

    def test_missing_run_metadata_returns_none(self) -> None:
        run_data = [{"_extra_files": {}}]
        assert _compute_max_tps(run_data) is None

    def test_null_system_tps_skipped(self) -> None:
        run_data = [self._make_run(None), self._make_run(800.0)]
        assert _compute_max_tps(run_data) == 800.0


@pytest.mark.unit
class TestTpsUtilizationInjection:
    def _make_archive_with_metadata(
        self, run_folder: Path, system_tps: float, concurrency: int, tmp_path: Path, name: str
    ) -> Path:
        import shutil

        folder = tmp_path / name
        shutil.copytree(run_folder, folder)
        cfg = yaml.safe_load((folder / "config.yaml").read_text())
        cfg["settings"]["load_pattern"]["target_concurrency"] = concurrency
        (folder / "config.yaml").write_text(yaml.dump(cfg))
        (folder / "run_metadata.json").write_text(
            json.dumps({"system_tps": system_tps, "tps_utilization": None})
        )
        archive = tmp_path / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname=name)
        return archive

    def test_tps_utilization_written_for_single_run(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        archive = self._make_archive_with_metadata(run_folder, 1000.0, 4, tmp_path, "run1")
        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        meta_files = list(sub_dir.rglob("run_metadata.json"))
        assert len(meta_files) == 1
        data = json.loads(meta_files[0].read_text())
        assert data["tps_utilization"] == pytest.approx(1.0)

    def test_tps_utilization_normalized_across_runs(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        a1 = self._make_archive_with_metadata(run_folder, 1000.0, 4, tmp_path, "run1")
        a2 = self._make_archive_with_metadata(run_folder, 2000.0, 8, tmp_path, "run2")
        sub_dir = build_submission_folder(
            [("run-001", a1), ("run-002", a2)],
            "standardized",
            "available",
            tmp_path / "sub",
        )
        meta_files = sorted(sub_dir.rglob("run_metadata.json"))
        assert len(meta_files) == 2
        utilizations = sorted(json.loads(p.read_text())["tps_utilization"] for p in meta_files)
        assert utilizations == pytest.approx([0.5, 1.0])

    def test_no_run_metadata_no_crash(self, run_archive: Path, tmp_path: Path) -> None:
        # run_archive fixture has no run_metadata.json — should build without error
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        assert sub_dir.is_dir()


@pytest.mark.unit
class TestTruncateResponses:
    def _make_content(self, n_responses: int) -> bytes:
        data = {
            "config": {"mode": "perf"},
            "results": {"total": n_responses},
            "responses": [{"text": "hello world", "idx": i} for i in range(n_responses)],
        }
        return json.dumps(data).encode()

    def test_small_responses_unchanged(self) -> None:
        content = self._make_content(2)
        result = json.loads(_truncate_responses(content))
        assert len(result["responses"]) == 2

    def test_large_responses_truncated_under_10kb(self) -> None:
        content = self._make_content(10_000)
        result_bytes = _truncate_responses(content)
        result = json.loads(result_bytes)
        assert len(json.dumps(result["responses"]).encode()) <= 10 * 1024

    def test_other_keys_preserved(self) -> None:
        content = self._make_content(10_000)
        result = json.loads(_truncate_responses(content))
        assert result["config"] == {"mode": "perf"}
        assert result["results"]["total"] == 10_000

    def test_no_responses_key_unchanged(self) -> None:
        data = {"config": {}, "results": {}}
        content = json.dumps(data).encode()
        assert _truncate_responses(content) == content

    def test_invalid_json_returned_as_is(self) -> None:
        content = b"not json"
        assert _truncate_responses(content) == content


@pytest.mark.unit
class TestSlugify:
    def test_simple(self) -> None:
        assert _slugify("Test System") == "Test_System"

    def test_special_chars_replaced(self) -> None:
        assert _slugify("My System (v2)!") == "My_System_v2"

    def test_empty_string(self) -> None:
        assert _slugify("") == "unknown"

    def test_long_name_truncated(self) -> None:
        long = "A" * 100
        assert len(_slugify(long)) <= 64
