# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runs.parser module."""

from __future__ import annotations

import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from endpoints_submission_cli.exceptions import RunFolderError
from endpoints_submission_cli.runs.parser import (
    _coerce_epoch,
    _extract_timestamps,
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

    def test_truncates_results_responses(self, run_folder: Path, tmp_path: Path) -> None:
        results_path = run_folder / "results.json"
        big = {
            "config": {"mode": "perf"},
            "results": {"total": 5000},
            "responses": [{"idx": i, "text": "lorem ipsum " * 10} for i in range(5000)],
        }
        results_path.write_text(json.dumps(big))
        original_bytes = results_path.read_bytes()

        dest = tmp_path / "out.tar.gz"
        build_archive(run_folder, dest)

        archived = _read_member(dest, "/results.json")
        # responses capped well below the original count...
        assert 0 < len(archived["responses"]) < 5000
        # ...while the other keys survive intact.
        assert archived["config"] == {"mode": "perf"}
        assert archived["results"] == {"total": 5000}
        # source folder is never mutated.
        assert results_path.read_bytes() == original_bytes

    def test_truncates_nested_accuracy_results(self, run_folder: Path, tmp_path: Path) -> None:
        # Accuracy run folders carry an accuracy/results.json; it must be truncated too.
        acc_dir = run_folder / "accuracy"
        acc_dir.mkdir()
        big = {
            "results": {"total": 5000},
            "responses": [{"idx": i, "text": "lorem ipsum " * 10} for i in range(5000)],
        }
        (acc_dir / "results.json").write_text(json.dumps(big))

        dest = tmp_path / "out.tar.gz"
        build_archive(run_folder, dest)

        archived = _read_member(dest, "/accuracy/results.json")
        assert 0 < len(archived["responses"]) < 5000
        assert archived["results"] == {"total": 5000}

    def test_results_without_responses_archived_unchanged(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        # The fixture's results.json has no "responses" key -> archived byte-for-byte.
        original = (run_folder / "results.json").read_bytes()
        dest = tmp_path / "out.tar.gz"
        build_archive(run_folder, dest)
        with tarfile.open(dest) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("/results.json"))
            fh = tar.extractfile(member)
            assert fh is not None
            assert fh.read() == original


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


_REF = datetime(2026, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
_REF_EPOCH_S = _REF.timestamp()  # seconds


@pytest.mark.unit
class TestCoerceEpoch:
    def test_seconds(self) -> None:
        assert _coerce_epoch(_REF_EPOCH_S) == _REF

    def test_milliseconds(self) -> None:
        assert _coerce_epoch(_REF_EPOCH_S * 1e3) == _REF

    def test_microseconds(self) -> None:
        assert _coerce_epoch(_REF_EPOCH_S * 1e6) == _REF

    def test_nanoseconds(self) -> None:
        assert _coerce_epoch(int(_REF_EPOCH_S * 1e9)) == _REF

    def test_zero_returns_none(self) -> None:
        # The real-world unpopulated value.
        assert _coerce_epoch(0) is None

    def test_negative_returns_none(self) -> None:
        assert _coerce_epoch(-5) is None

    def test_non_numeric_returns_none(self) -> None:
        assert _coerce_epoch("2026-03-15") is None
        assert _coerce_epoch(None) is None

    def test_bool_returns_none(self) -> None:
        # bool is an int subclass; must not be treated as an epoch.
        assert _coerce_epoch(True) is None

    def test_implausible_year_returns_none(self) -> None:
        # 100 seconds after the epoch → 1970, outside the sane window at every scale.
        assert _coerce_epoch(100) is None


@pytest.mark.unit
class TestExtractTimestamps:
    def test_uses_test_started_at_when_present(self) -> None:
        rs = {"test_started_at": int(_REF_EPOCH_S * 1e9), "duration_ns": 60 * 1e9}
        started, finished = _extract_timestamps(rs)
        assert started == _REF
        assert (finished - started).total_seconds() == pytest.approx(60.0)

    def test_falls_back_to_now_when_zero(self) -> None:
        # test_started_at = 0 (real-world case) → now-based fallback.
        before = datetime.now(tz=timezone.utc)
        started, finished = _extract_timestamps({"test_started_at": 0, "duration_ns": 60 * 1e9})
        after = datetime.now(tz=timezone.utc)
        assert before <= finished <= after
        assert (finished - started).total_seconds() == pytest.approx(60.0)

    def test_falls_back_when_field_absent(self) -> None:
        before = datetime.now(tz=timezone.utc)
        started, finished = _extract_timestamps({"duration_ns": 0})
        after = datetime.now(tz=timezone.utc)
        assert before <= started == finished <= after
