# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for models.structure — directory-layout gate checks (§8.1)."""

from __future__ import annotations

import pytest

from submission_checker.models import (
    Division,
    ModelDir,
    Severity,
    SrcDir,
    SubmissionDir,
    SystemPareto,
)

from .conftest import _passed


@pytest.mark.unit
class TestSubmissionDir:
    def test_missing_dir(self, tmp_path):
        (tmp_path / "systems").mkdir()
        # pareto/ intentionally absent
        sd = SubmissionDir(root=tmp_path)
        rules = {r.rule for r in sd._check_results if r.severity == Severity.ERROR}
        assert "required-dir" in rules

    def test_both_present(self, tmp_path):
        (tmp_path / "systems").mkdir()
        (tmp_path / "pareto").mkdir()
        (tmp_path / "documentation").mkdir()
        (tmp_path / "documentation" / "README.md").write_text("docs")
        sd = SubmissionDir(root=tmp_path)
        assert _passed(sd._check_results)

    _REPRO_MSG = "Submission requires documentation and source code for reproducibility"

    def test_empty_documentation_fails_reproducibility(self, tmp_path):
        (tmp_path / "systems").mkdir()
        (tmp_path / "pareto").mkdir()
        (tmp_path / "documentation").mkdir()  # exists but empty
        sd = SubmissionDir(root=tmp_path)
        errors = [r for r in sd._check_results if r.severity == Severity.ERROR]
        assert any(r.rule == "reproducibility-content" and r.message == self._REPRO_MSG
                   for r in errors)

    def test_missing_documentation_uses_same_error(self, tmp_path):
        # A missing documentation/ reports the same single reproducibility error as
        # an empty one — no separate "required-dir" error for docs to fix first.
        (tmp_path / "systems").mkdir()
        (tmp_path / "pareto").mkdir()
        # documentation/ intentionally absent
        sd = SubmissionDir(root=tmp_path)
        errors = [r for r in sd._check_results if r.severity == Severity.ERROR]
        assert any(r.rule == "reproducibility-content" and r.message == self._REPRO_MSG
                   for r in errors)
        # documentation must NOT surface as a generic required-dir error
        assert not any(
            r.rule == "required-dir" and "documentation" in r.message for r in errors
        )

    def test_computed_paths(self, tmp_path):
        sd = SubmissionDir(root=tmp_path)
        assert sd.systems_dir == tmp_path / "systems"
        assert sd.pareto_dir == tmp_path / "pareto"


@pytest.mark.unit
class TestSystemPareto:
    def test_missing_system_pareto(self, tmp_path):
        sp = SystemPareto(pareto_dir=tmp_path, system_id="sys-x")
        assert any(r.severity == Severity.ERROR for r in sp._check_results)

    def test_present(self, tmp_path):
        (tmp_path / "sys-x").mkdir()
        sp = SystemPareto(pareto_dir=tmp_path, system_id="sys-x")
        assert _passed(sp._check_results)

    def test_system_dir_computed(self, tmp_path):
        sp = SystemPareto(pareto_dir=tmp_path, system_id="sys-x")
        assert sp.system_dir == tmp_path / "sys-x"


@pytest.mark.unit
class TestModelDir:
    def test_missing_subdir(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "points").mkdir()
        # results/ intentionally absent
        md = ModelDir(root=model_dir, system_id="sys-x", benchmark_model="llama3-70b")
        assert any(r.severity == Severity.ERROR for r in md._check_results)

    def test_all_present(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        for d in ("points", "results"):
            (model_dir / d).mkdir()
        md = ModelDir(root=model_dir, system_id="sys-x", benchmark_model="llama3-70b")
        assert _passed(md._check_results)

    def test_computed_paths(self, tmp_path):
        md = ModelDir(root=tmp_path, system_id="sys-x", benchmark_model="llama3-70b")
        assert md.points_dir == tmp_path / "points"
        assert md.results_dir == tmp_path / "results"


@pytest.mark.unit
class TestSrcDir:
    _REPRO_MSG = "Submission requires documentation and source code for reproducibility"

    def test_standardized_missing_src(self, tmp_path):
        # Missing src/<model>/ reports the same single reproducibility error as an
        # empty one — no separate "missing directory" error to fix first.
        sd = SrcDir(root=tmp_path, division=Division.STANDARDIZED, model="llama3-70b")
        errors = [r for r in sd._check_results if r.severity == Severity.ERROR]
        assert any(r.rule == "reproducibility-content" and r.message == self._REPRO_MSG
                   for r in errors)

    def test_standardized_src_present(self, tmp_path):
        (tmp_path / "src" / "llama3-70b").mkdir(parents=True)
        (tmp_path / "src" / "llama3-70b" / "run.sh").write_text("echo hi")
        sd = SrcDir(root=tmp_path, division=Division.STANDARDIZED, model="llama3-70b")
        assert _passed(sd._check_results)

    def test_standardized_empty_src_fails_reproducibility(self, tmp_path):
        (tmp_path / "src" / "llama3-70b").mkdir(parents=True)  # exists but empty
        sd = SrcDir(root=tmp_path, division=Division.STANDARDIZED, model="llama3-70b")
        errors = [r for r in sd._check_results if r.severity == Severity.ERROR]
        assert any(r.rule == "reproducibility-content" and r.message == self._REPRO_MSG
                   for r in errors)

    def test_non_standardized_skipped(self, tmp_path):
        sd = SrcDir(root=tmp_path, division=Division.SERVICED, model="llama3-70b")
        assert sd._check_results == []
