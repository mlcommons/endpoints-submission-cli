# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runs.parser module."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

from endpoints_submission_cli.exceptions import RunFolderError
from endpoints_submission_cli.runs.parser import (
    _started_at_to_run_date,
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
        assert si["node_types"][0]["accelerator_model_name"] == "NVIDIA H100 SXM5 80GB"

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
        with pytest.raises(RunFolderError, match="system_desc.json"):
            parse_run_folder(folder)

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_run"
        folder.mkdir()
        (folder / "system_desc.json").write_text("{}")
        (folder / "result_summary.json").write_text("{}")
        with pytest.raises(RunFolderError, match="config.yaml"):
            parse_run_folder(folder)

    def test_missing_result_summary_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_run"
        folder.mkdir()
        (folder / "system_desc.json").write_text("{}")
        (folder / "config.yaml").write_text("name: test")
        with pytest.raises(RunFolderError, match="result_summary.json"):
            parse_run_folder(folder)

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RunFolderError, match="not a directory"):
            parse_run_folder(tmp_path / "nonexistent")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_json"
        folder.mkdir()
        (folder / "system_desc.json").write_text("{bad json")
        (folder / "config.yaml").write_text("name: test")
        (folder / "result_summary.json").write_text("{}")
        with pytest.raises(RunFolderError, match="Invalid JSON"):
            parse_run_folder(folder)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_yaml"
        folder.mkdir()
        (folder / "system_desc.json").write_text("{}")
        (folder / "config.yaml").write_text(": invalid: yaml: [")
        (folder / "result_summary.json").write_text("{}")
        with pytest.raises(RunFolderError, match="Invalid YAML"):
            parse_run_folder(folder)

    def test_config_yaml_not_mapping_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "bad_yaml2"
        folder.mkdir()
        (folder / "system_desc.json").write_text("{}")
        (folder / "config.yaml").write_text("- item1\n- item2\n")
        (folder / "result_summary.json").write_text("{}")
        with pytest.raises(RunFolderError, match="must be a YAML mapping"):
            parse_run_folder(folder)

    def test_zero_duration_timestamps_equal(self, tmp_path: Path) -> None:
        """When duration_ns is 0, started_at == finished_at."""
        folder = tmp_path / "zero_dur"
        folder.mkdir()
        (folder / "system_desc.json").write_text("{}")
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "result_summary.json").write_text(json.dumps({"duration_ns": 0}))
        payload = parse_run_folder(folder)
        assert payload["started_at"] == payload["finished_at"]

    def test_benchmark_version_from_git_sha(self, tmp_path: Path) -> None:
        folder = tmp_path / "run_with_sha"
        folder.mkdir()
        (folder / "system_desc.json").write_text("{}")
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "result_summary.json").write_text(json.dumps({"git_sha": "abc123"}))
        payload = parse_run_folder(folder)
        assert payload["benchmark_version"] == "abc123"

    def test_benchmark_version_unknown_when_no_git_sha(self, tmp_path: Path) -> None:
        folder = tmp_path / "run_no_sha"
        folder.mkdir()
        (folder / "system_desc.json").write_text("{}")
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "result_summary.json").write_text("{}")
        payload = parse_run_folder(folder)
        assert payload["benchmark_version"] == "unknown"


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
        assert any("system_desc.json" in n for n in names)
        assert any("config.yaml" in n for n in names)
        assert any("result_summary.json" in n for n in names)


def _read_member(archive: Path, suffix: str) -> dict:
    with tarfile.open(archive) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(suffix))
        fh = tar.extractfile(member)
        assert fh is not None
        return json.loads(fh.read().decode("utf-8"))


@pytest.mark.unit
class TestStartedAtToRunDate:
    def test_utc_offset(self) -> None:
        assert _started_at_to_run_date("2026-03-15T10:30:00+00:00") == "2026-03-15"

    def test_z_suffix(self) -> None:
        # datetime.fromisoformat rejects a trailing 'Z' before Python 3.11.
        assert _started_at_to_run_date("2026-03-15T10:30:00Z") == "2026-03-15"

    def test_naive_datetime(self) -> None:
        assert _started_at_to_run_date("2026-03-15T10:30:00") == "2026-03-15"

    def test_microseconds(self) -> None:
        assert _started_at_to_run_date("2026-03-15T10:30:00.123456+00:00") == "2026-03-15"

    def test_offset_keeps_local_date(self) -> None:
        # The date is taken from the offset-aware timestamp as-is, not converted to UTC.
        assert _started_at_to_run_date("2026-01-01T01:00:00+05:30") == "2026-01-01"

    def test_invalid_returns_none(self) -> None:
        assert _started_at_to_run_date("not-a-date") is None


@pytest.mark.unit
class TestBuildArchiveRunDate:
    def test_run_date_injected_into_metadata(self, run_folder: Path, tmp_path: Path) -> None:
        (run_folder / "run_metadata.json").write_text(
            json.dumps({"system_tps": 100.0, "run_date": None})
        )
        dest = tmp_path / "out.tar.gz"
        build_archive(run_folder, dest, run_date="2026-03-15")
        meta = _read_member(dest, "run_metadata.json")
        assert meta["run_date"] == "2026-03-15"
        assert meta["system_tps"] == 100.0  # other fields preserved

    def test_source_folder_not_mutated(self, run_folder: Path, tmp_path: Path) -> None:
        meta_path = run_folder / "run_metadata.json"
        meta_path.write_text(json.dumps({"run_date": None}))
        build_archive(run_folder, tmp_path / "out.tar.gz", run_date="2026-03-15")
        # The on-disk source file must be untouched.
        assert json.loads(meta_path.read_text())["run_date"] is None

    def test_no_metadata_file_no_crash(self, run_folder: Path, tmp_path: Path) -> None:
        # run_folder fixture has no run_metadata.json — should archive without error.
        dest = tmp_path / "out.tar.gz"
        build_archive(run_folder, dest, run_date="2026-03-15")
        with tarfile.open(dest) as tar:
            assert not any(n.endswith("run_metadata.json") for n in tar.getnames())

    def test_no_run_date_leaves_metadata_untouched(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        (run_folder / "run_metadata.json").write_text(json.dumps({"run_date": "1999-01-01"}))
        dest = tmp_path / "out.tar.gz"
        build_archive(run_folder, dest)  # no run_date passed
        meta = _read_member(dest, "run_metadata.json")
        assert meta["run_date"] == "1999-01-01"

    def test_all_files_still_present_after_injection(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        (run_folder / "run_metadata.json").write_text(json.dumps({"run_date": None}))
        dest = tmp_path / "out.tar.gz"
        build_archive(run_folder, dest, run_date="2026-03-15")
        with tarfile.open(dest) as tar:
            names = tar.getnames()
        for expected in ("system_desc.json", "config.yaml", "result_summary.json"):
            assert any(n.endswith(expected) for n in names)
        # run_metadata.json appears exactly once (no duplicate from the substitution).
        assert sum(1 for n in names if n.endswith("run_metadata.json")) == 1
