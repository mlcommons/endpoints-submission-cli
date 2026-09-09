# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared submission-layout helpers (§8.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from submission_checker import layout


class TestPointDirNames:
    @pytest.mark.parametrize("concurrency", [1, 4, 32, 256, 16384])
    def test_name_and_parse_round_trip(self, concurrency: int) -> None:
        assert layout.parse_point_dir(layout.point_dir_name(concurrency)) == concurrency

    @pytest.mark.parametrize("name", ["rabc", "r", "results", "r4x", "xr4", "server_configs", ""])
    def test_non_point_names_return_none(self, name: str) -> None:
        assert layout.parse_point_dir(name) is None

    def test_point_dirs_sort_numerically(self, tmp_path: Path) -> None:
        """r256 must follow r32, which a lexical sort gets backwards."""
        for concurrency in (256, 32, 4, 1024, 8):
            (tmp_path / layout.point_dir_name(concurrency)).mkdir()
        assert [d.name for d in layout.iter_point_dirs(tmp_path)] == [
            "r4",
            "r8",
            "r32",
            "r256",
            "r1024",
        ]

    def test_non_point_directories_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "r4").mkdir()
        (tmp_path / "server_configs").mkdir()
        (tmp_path / "rabc").mkdir()
        assert [d.name for d in layout.iter_point_dirs(tmp_path)] == ["r4"]

    def test_missing_directory_is_empty(self, tmp_path: Path) -> None:
        assert layout.iter_point_dirs(tmp_path / "absent") == []


class TestIterCurves:
    def test_yields_every_system_model_pair(self, tmp_path: Path) -> None:
        for system in ("sys_a", "sys_b"):
            for model in ("llama3.1-8b", "deepseek-r1"):
                (tmp_path / system / model).mkdir(parents=True)
        pairs = [(s.name, m.name) for s, m in layout.iter_curves(tmp_path)]
        assert pairs == [
            ("sys_a", "deepseek-r1"),
            ("sys_a", "llama3.1-8b"),
            ("sys_b", "deepseek-r1"),
            ("sys_b", "llama3.1-8b"),
        ]

    def test_files_are_not_curves(self, tmp_path: Path) -> None:
        (tmp_path / "sys_a").mkdir()
        (tmp_path / "stray.json").write_text("{}")
        (tmp_path / "sys_a" / "system_desc.json").write_text("{}")
        assert list(layout.iter_curves(tmp_path)) == []

    def test_missing_results_dir_yields_nothing(self, tmp_path: Path) -> None:
        assert list(layout.iter_curves(tmp_path / "absent")) == []


class TestResolveSharedPath:
    @pytest.fixture
    def submission(self, tmp_path: Path) -> Path:
        (tmp_path / "src" / "trtllm").mkdir(parents=True)
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "calibration.adoc").write_text("= Calibration\n")
        return tmp_path

    @pytest.mark.parametrize("value", ["src", "src/trtllm", "./docs", "docs/"])
    def test_resolvable_relative_paths(self, submission: Path, value: str) -> None:
        assert layout.resolve_shared_path(submission, value) is not None

    @pytest.mark.parametrize("value", ["", "   ", "src/absent", "docs/calibration.adoc"])
    def test_unresolvable_values(self, submission: Path, value: str) -> None:
        """A file is not a directory, and §9.1 asks for a directory."""
        assert layout.resolve_shared_path(submission, value) is None

    @pytest.mark.parametrize("value", ["../secrets", "src/../../etc", "/etc", "~/.ssh"])
    def test_escaping_paths_are_rejected_not_resolved(self, submission: Path, value: str) -> None:
        """A pointer that leaves the bundle cannot be reviewed from the bundle."""
        assert layout.resolve_shared_path(submission, value) is None

    def test_windows_separators_are_accepted(self, submission: Path) -> None:
        assert layout.resolve_shared_path(submission, "src\\trtllm") is not None

    def test_traversal_that_stays_inside_is_still_rejected(self, submission: Path) -> None:
        """'src/../docs' resolves inside the root, but '..' is refused on sight."""
        assert layout.resolve_shared_path(submission, "src/../docs") is None


class TestRequiredRunFiles:
    def test_config_yaml_is_not_required(self) -> None:
        """config.yaml became optional in v1.0; point.yaml carries the disclosure."""
        assert layout.CONFIG_YAML not in layout.REQUIRED_RUN_FILES
        assert layout.POINT_YAML in layout.REQUIRED_RUN_FILES

    def test_system_desc_is_required_per_run(self) -> None:
        assert layout.SYSTEM_DESC_JSON in layout.REQUIRED_RUN_FILES
