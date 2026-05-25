# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for run_parser module."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
import pytest
import yaml

from endpoints_submission_cli.exceptions import RunFolderError
from endpoints_submission_cli.run_parser import (
    build_archive,
    parse_run_folder,
)


@pytest.mark.unit
class TestParseRunFolder:
    def test_valid_folder_returns_payload(self, run_folder: Path) -> None:
        payload = parse_run_folder(run_folder)

        assert "benchmark_version" in payload
        assert "started_at" in payload
        assert "finished_at" in payload
        assert "system_info" in payload
        assert "config" in payload
        assert "result_summary" in payload

    def test_system_info_passthrough(self, run_folder: Path) -> None:
        payload = parse_run_folder(run_folder)
        si = payload["system_info"]
        assert si["system_name"] == "Test System"
        assert si["accelerator_model_name"] == "NVIDIA H100 SXM5 80GB"

    def test_config_passthrough(self, run_folder: Path) -> None:
        payload = parse_run_folder(run_folder)
        cfg = payload["config"]
        assert cfg["model_params"]["name"] == "meta-llama/Llama-3.1-8B-Instruct"

    def test_result_summary_passthrough(self, run_folder: Path) -> None:
        payload = parse_run_folder(run_folder)
        rs = payload["result_summary"]
        assert rs["n_samples_completed"] == 2000

    def test_timestamps_are_iso_strings(self, run_folder: Path) -> None:
        payload = parse_run_folder(run_folder)
        assert "T" in payload["started_at"]
        assert "T" in payload["finished_at"]

    def test_finished_after_started(self, run_folder: Path) -> None:
        payload = parse_run_folder(run_folder)
        assert payload["started_at"] <= payload["finished_at"]

    def test_missing_system_info_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_run"
        folder.mkdir()
        (folder / "config.yaml").write_text("name: test")
        (folder / "result_summary.json").write_text("{}")
        with pytest.raises(RunFolderError, match="system_info.json"):
            parse_run_folder(folder)

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_run"
        folder.mkdir()
        (folder / "system_info.json").write_text("{}")
        (folder / "result_summary.json").write_text("{}")
        with pytest.raises(RunFolderError, match="config.yaml"):
            parse_run_folder(folder)

    def test_missing_result_summary_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_run"
        folder.mkdir()
        (folder / "system_info.json").write_text("{}")
        (folder / "config.yaml").write_text("name: test")
        with pytest.raises(RunFolderError, match="result_summary.json"):
            parse_run_folder(folder)

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RunFolderError, match="not a directory"):
            parse_run_folder(tmp_path / "nonexistent")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_json"
        folder.mkdir()
        (folder / "system_info.json").write_text("{bad json")
        (folder / "config.yaml").write_text("name: test")
        (folder / "result_summary.json").write_text("{}")
        with pytest.raises(RunFolderError, match="Invalid JSON"):
            parse_run_folder(folder)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_yaml"
        folder.mkdir()
        (folder / "system_info.json").write_text("{}")
        (folder / "config.yaml").write_text(": invalid: yaml: [")
        (folder / "result_summary.json").write_text("{}")
        with pytest.raises(RunFolderError, match="Invalid YAML"):
            parse_run_folder(folder)

    def test_config_yaml_not_mapping_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_yaml2"
        folder.mkdir()
        (folder / "system_info.json").write_text("{}")
        (folder / "config.yaml").write_text("- item1\n- item2\n")
        (folder / "result_summary.json").write_text("{}")
        with pytest.raises(RunFolderError, match="must be a YAML mapping"):
            parse_run_folder(folder)

    def test_zero_duration_timestamps_equal(self, tmp_path: Path) -> None:
        """When duration_ns is 0, started_at == finished_at."""
        folder = tmp_path / "zero_dur"
        folder.mkdir()
        (folder / "system_info.json").write_text("{}")
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "result_summary.json").write_text(json.dumps({"duration_ns": 0}))
        payload = parse_run_folder(folder)
        assert payload["started_at"] == payload["finished_at"]

    def test_benchmark_version_from_git_sha(self, tmp_path: Path) -> None:
        folder = tmp_path / "run_with_sha"
        folder.mkdir()
        (folder / "system_info.json").write_text("{}")
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "result_summary.json").write_text(json.dumps({"git_sha": "abc123"}))
        payload = parse_run_folder(folder)
        assert payload["benchmark_version"] == "abc123"

    def test_benchmark_version_unknown_when_no_git_sha(self, tmp_path: Path) -> None:
        folder = tmp_path / "run_no_sha"
        folder.mkdir()
        (folder / "system_info.json").write_text("{}")
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "result_summary.json").write_text("{}")
        payload = parse_run_folder(folder)
        assert payload["benchmark_version"] == "unknown"

    def test_google_sample_run(self, sample_google_run_dir: Path) -> None:
        """Parse a real Google sample run folder."""
        if not sample_google_run_dir.exists():
            pytest.skip("Sample Google run data not available")
        if not (sample_google_run_dir / "system_info.json").exists():
            pytest.skip("Sample run uses run_metadata.json format, not system_info.json")
        payload = parse_run_folder(sample_google_run_dir)
        assert payload["system_info"]["system_name"]
        assert payload["config"]
        assert payload["result_summary"]["n_samples_completed"] > 0


@pytest.mark.unit
class TestBuildArchive:
    def test_creates_tar_gz(self, run_folder: Path, tmp_path: Path) -> None:
        dest = tmp_path / "out.tar.gz"
        result = build_archive(run_folder, dest)
        assert result == dest
        assert dest.exists()

    def test_default_dest_beside_folder(self, run_folder: Path) -> None:
        expected = run_folder.parent / f"{run_folder.name}.tar.gz"
        try:
            result = build_archive(run_folder)
            assert result == expected
        finally:
            expected.unlink(missing_ok=True)

    def test_archive_contains_files(self, run_folder: Path, tmp_path: Path) -> None:
        dest = tmp_path / "out.tar.gz"
        build_archive(run_folder, dest)
        with tarfile.open(dest) as tar:
            names = tar.getnames()
        assert any("system_info.json" in n for n in names)
        assert any("config.yaml" in n for n in names)
        assert any("result_summary.json" in n for n in names)
