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
        # Should contain the system_info.json somewhere
        files = list(dest.rglob("system_info.json"))
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
    def test_creates_system_desc_json(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        system_descs = list(sub_dir.rglob("system_desc.json"))
        assert len(system_descs) == 1

    def test_creates_model_dir(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        # <system>/<model>/ should exist
        system_dirs = [d for d in sub_dir.iterdir() if d.is_dir()]
        assert len(system_dirs) >= 1
        model_dirs = [d for d in system_dirs[0].iterdir() if d.is_dir()]
        assert len(model_dirs) >= 1

    def test_system_desc_json_content(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        descs = list(sub_dir.rglob("system_desc.json"))
        data = json.loads(descs[0].read_text())
        assert "division" in data
        assert data["division"] == "Standardized"

    def test_point_yaml_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        yamls = list(sub_dir.rglob("point.yaml"))
        assert len(yamls) >= 1
        data = yaml.safe_load(yamls[0].read_text())
        assert data["concurrency"] == 4

    def test_log_summary_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        summaries = list(sub_dir.rglob("mlperf_endpoints_log_summary.json"))
        assert len(summaries) == 1
        data = json.loads(summaries[0].read_text())
        assert "n_samples_completed" in data
        assert "duration_ns" in data

    def test_log_detail_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        details = list(sub_dir.rglob("mlperf_endpoints_log_detail.json"))
        assert len(details) == 1

    def test_run_metadata_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        assert len(list(sub_dir.rglob("run_metadata.json"))) == 1

    def test_report_txt_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        assert len(list(sub_dir.rglob("report.txt"))) == 1

    def test_sweep_csvs_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        assert len(list(sub_dir.rglob("sweep_summary.csv"))) == 1
        assert len(list(sub_dir.rglob("sweep_distributions.csv"))) == 1

    def test_src_impl_dir_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        src_dirs = [d for d in sub_dir.rglob("src") if d.is_dir()]
        assert len(src_dirs) >= 1
        # Each src/ must have at least one implementation subdir
        for src in src_dirs:
            impl_dirs = [d for d in src.iterdir() if d.is_dir()]
            assert len(impl_dirs) >= 1

    def test_accuracy_files_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        acc_txts = list(sub_dir.rglob("accuracy.txt"))
        acc_jsons = list(sub_dir.rglob("accuracy_result.json"))
        assert len(acc_txts) == 1
        assert len(acc_jsons) == 1

    def test_run_dir_pattern(self, run_archive: Path, tmp_path: Path) -> None:
        """Run directories follow the r<N> naming convention."""
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path
        )
        import re
        run_dirs = [d for d in sub_dir.rglob("*") if d.is_dir() and re.match(r"^r\d+$", d.name)]
        assert len(run_dirs) >= 1

    def test_empty_run_list_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SubmissionBuildError, match="At least one"):
            build_submission_folder([], "standardized", tmp_path)

    def test_missing_system_info_in_archive_raises(self, tmp_path: Path) -> None:
        # Create archive without system_info.json
        folder = tmp_path / "bad_run"
        folder.mkdir()
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "result_summary.json").write_text("{}")
        archive = tmp_path / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="bad_run")
        with pytest.raises(SubmissionBuildError, match="system_info.json"):
            build_submission_folder([("bad", archive)], "standardized", tmp_path / "out")

    def test_multiple_runs_single_system(
        self, run_archive: Path, run_folder: Path, tmp_path: Path
    ) -> None:
        # Create a second archive with different concurrency
        second_folder = tmp_path / "run2"
        second_folder.mkdir()

        import json as _json
        si = _json.loads((run_folder / "system_info.json").read_text())
        cfg = yaml.safe_load((run_folder / "config.yaml").read_text())
        rs = _json.loads((run_folder / "result_summary.json").read_text())

        cfg["settings"]["load_pattern"]["target_concurrency"] = 16
        (second_folder / "system_info.json").write_text(_json.dumps(si))
        (second_folder / "config.yaml").write_text(yaml.dump(cfg))
        (second_folder / "result_summary.json").write_text(_json.dumps(rs))

        second_archive = tmp_path / "run2.tar.gz"
        with tarfile.open(second_archive, "w:gz") as tar:
            tar.add(second_folder, arcname="run2")

        sub_dir = build_submission_folder(
            [("run-001", run_archive), ("run-002", second_archive)],
            "standardized",
            tmp_path / "sub",
        )
        import re
        run_dirs = {
            d.name
            for d in sub_dir.rglob("*")
            if d.is_dir() and re.match(r"^r\d+$", d.name)
        }
        assert "r4" in run_dirs
        assert "r16" in run_dirs

    def test_serviced_division_normalized(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "serviced", tmp_path
        )
        descs = list(sub_dir.rglob("system_desc.json"))
        data = json.loads(descs[0].read_text())
        assert data["division"] == "Serviced"


@pytest.mark.unit
class TestCreateBundleArchive:
    def test_creates_archive(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path / "sub"
        )
        bundle = create_bundle_archive(sub_dir, tmp_path / "bundle.tar.gz")
        assert bundle.exists()
        with tarfile.open(bundle) as tar:
            names = tar.getnames()
        assert any("system_desc.json" in n for n in names)

    def test_default_dest(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", tmp_path / "sub"
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
