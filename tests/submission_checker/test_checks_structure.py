# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for models.structure — directory-layout gate checks (§8.1)."""

from __future__ import annotations

import pytest

from submission_checker.models import (
    ModelDir,
    Severity,
    SrcDir,
    SubmissionDir,
    SystemResults,
)

from .conftest import _passed


@pytest.mark.unit
class TestSubmissionDir:
    def test_missing_dir(self, tmp_path):
        (tmp_path / "results").mkdir()
        # docs/ intentionally absent
        sd = SubmissionDir(root=tmp_path)
        rules = {r.rule for r in sd._check_results if r.severity == Severity.ERROR}
        assert "required-dir" in rules

    def test_both_present(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / "docs").mkdir()
        sd = SubmissionDir(root=tmp_path)
        assert _passed(sd._check_results)

    def test_computed_paths(self, tmp_path):
        sd = SubmissionDir(root=tmp_path)
        assert sd.results_dir == tmp_path / "results"
        assert sd.docs_dir == tmp_path / "docs"


@pytest.mark.unit
class TestSystemResults:
    def test_missing_system_dir(self, tmp_path):
        sr = SystemResults(results_dir=tmp_path, system_id="sys-x")
        assert any(r.severity == Severity.ERROR for r in sr._check_results)

    def test_present(self, tmp_path):
        (tmp_path / "sys-x").mkdir()
        sr = SystemResults(results_dir=tmp_path, system_id="sys-x")
        assert _passed(sr._check_results)

    def test_system_dir_computed(self, tmp_path):
        sr = SystemResults(results_dir=tmp_path, system_id="sys-x")
        assert sr.system_dir == tmp_path / "sys-x"


@pytest.mark.unit
class TestModelDir:
    def test_no_point_dirs(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        md = ModelDir(root=model_dir, system_id="sys-x", benchmark_model="llama3-70b")
        assert any(r.severity == Severity.ERROR for r in md._check_results)

    def test_point_dirs_present(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        for d in ("r1", "r32"):
            (model_dir / d).mkdir()
        md = ModelDir(root=model_dir, system_id="sys-x", benchmark_model="llama3-70b")
        assert _passed(md._check_results)

    def test_point_dirs_sorted_numerically(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        for d in ("r256", "r1", "r32"):
            (model_dir / d).mkdir()
        md = ModelDir(root=model_dir, system_id="sys-x", benchmark_model="llama3-70b")
        # r256 must not sort before r32 the way a lexical sort would put it.
        assert [d.name for d in md.point_dirs] == ["r1", "r32", "r256"]

    def test_non_point_dirs_ignored(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "r1").mkdir()
        (model_dir / "scratch").mkdir()
        (model_dir / "results").mkdir()
        md = ModelDir(root=model_dir, system_id="sys-x", benchmark_model="llama3-70b")
        assert [d.name for d in md.point_dirs] == ["r1"]


@pytest.mark.unit
class TestSrcDir:
    def test_missing_src(self, tmp_path):
        sd = SrcDir(root=tmp_path)
        assert any(r.severity == Severity.ERROR for r in sd._check_results)

    def test_src_with_no_implementation_dir(self, tmp_path):
        (tmp_path / "src").mkdir()
        sd = SrcDir(root=tmp_path)
        assert any(r.severity == Severity.ERROR for r in sd._check_results)

    def test_implementation_without_readme(self, tmp_path):
        (tmp_path / "src" / "trtllm").mkdir(parents=True)
        sd = SrcDir(root=tmp_path)
        rules = {r.rule for r in sd._check_results if r.severity == Severity.ERROR}
        assert "src-readme" in rules

    def test_src_present(self, tmp_path):
        impl = tmp_path / "src" / "trtllm"
        impl.mkdir(parents=True)
        (impl / "README.md").write_text("# trtllm")
        sd = SrcDir(root=tmp_path)
        assert _passed(sd._check_results)

    def test_readme_match_is_case_insensitive(self, tmp_path):
        impl = tmp_path / "src" / "vllm"
        impl.mkdir(parents=True)
        (impl / "readme.md").write_text("# vllm")
        sd = SrcDir(root=tmp_path)
        assert _passed(sd._check_results)

    def test_enforced_regardless_of_division(self, tmp_path):
        """src/ is required for every division for now (see SrcDir docstring)."""
        impl = tmp_path / "src" / "sglang"
        impl.mkdir(parents=True)
        (impl / "README.md").write_text("# sglang")
        assert _passed(SrcDir(root=tmp_path)._check_results)
