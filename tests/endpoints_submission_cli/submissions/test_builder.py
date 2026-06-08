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
    _slugify,
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
            [("run-001", run_archive)], "standardized", tmp_path, "available"
        )
        assert (sub_dir / "systems").is_dir()

    def test_creates_pareto_dir(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path, "available"
        )
        pareto = sub_dir / "pareto"
        assert pareto.is_dir()

    def test_system_json_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path, "available"
        )
        jsons = list((sub_dir / "systems").glob("*.json"))
        assert len(jsons) == 1
        data = json.loads(jsons[0].read_text())
        assert "model_metadata" in data
        assert data["model_metadata"]["division"] == "Standardized"

    def test_point_yaml_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path, "available"
        )
        yamls = list(sub_dir.rglob("point_*.yaml"))
        assert len(yamls) >= 1
        data = yaml.safe_load(yamls[0].read_text())
        assert data["concurrency"] == 4

    def test_log_summary_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path, "available"
        )
        summaries = list(sub_dir.rglob("mlperf_endpoints_log_summary.json"))
        assert len(summaries) == 1
        data = json.loads(summaries[0].read_text())
        assert "n_samples_completed" in data
        assert "duration_ns" in data

    def test_log_detail_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path, "available"
        )
        details = list(sub_dir.rglob("mlperf_endpoints_log_detail.json"))
        assert len(details) == 1

    def test_accuracy_files_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path, "available"
        )
        acc_txts = list(sub_dir.rglob("accuracy.txt"))
        acc_jsons = list(sub_dir.rglob("accuracy_result.json"))
        assert len(acc_txts) == 1
        assert len(acc_jsons) == 1

    def test_empty_run_list_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SubmissionBuildError, match="At least one"):
            build_submission_folder([], "standardized", tmp_path, "available")

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
            build_submission_folder([("bad", archive)], "standardized", tmp_path / "out", "available")

    def test_flat_system_desc_raises(self, tmp_path: Path) -> None:
        # Flat format (missing organization_metadata) must raise
        folder = tmp_path / "flat_run"
        folder.mkdir()
        flat_desc = {"system_name": "My System", "division": "Standardized"}
        (folder / "system_desc.json").write_text(json.dumps(flat_desc))
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "result_summary.json").write_text("{}")
        archive = tmp_path / "flat.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="flat_run")
        with pytest.raises(SubmissionBuildError, match="schema validation"):
            build_submission_folder([("flat", archive)], "standardized", tmp_path / "out", "available")

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
            tmp_path / "sub",
            "available",
        )
        yamls = list(sub_dir.rglob("point_*.yaml"))
        concurrencies = {yaml.safe_load(p.read_text())["concurrency"] for p in yamls}
        assert 4 in concurrencies
        assert 16 in concurrencies

    def test_system_json_division_from_cli(self, run_archive: Path, tmp_path: Path) -> None:
        # CLI division arg is authoritative — overwrites any placeholder or stale value in system_desc
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "serviced", tmp_path, "available"
        )
        jsons = list((sub_dir / "systems").glob("*.json"))
        data = json.loads(jsons[0].read_text())
        assert data["model_metadata"]["division"] == "Serviced"


@pytest.mark.unit
class TestCreateBundleArchive:
    def test_creates_archive(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path / "sub", "available"
        )
        bundle = create_bundle_archive(sub_dir, tmp_path / "bundle.tar.gz")
        assert bundle.exists()
        with tarfile.open(bundle) as tar:
            names = tar.getnames()
        assert any("systems" in n for n in names)

    def test_default_dest(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path / "sub", "available"
        )
        bundle = create_bundle_archive(sub_dir)
        expected = sub_dir.parent / f"{sub_dir.name}.tar.gz"
        try:
            assert bundle == expected
        finally:
            expected.unlink(missing_ok=True)


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
