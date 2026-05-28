# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for models.structure — directory-layout gate checks (§8.1)."""

from __future__ import annotations

import pytest

from submission_checker.models import (
    ModelDir,
    RunDir,
    Severity,
)

from .conftest import _passed


@pytest.mark.unit
class TestModelDir:
    def test_missing_sweep_summary(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "sweep_distributions.csv").write_text("data\n")
        # sweep_summary.csv intentionally absent
        md = ModelDir(root=model_dir, system_name="sys-x", model_name="llama3-70b")
        rules = {r.rule for r in md._check_results if r.severity == Severity.ERROR}
        assert "model-sweep-file" in rules

    def test_missing_sweep_distributions(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "sweep_summary.csv").write_text("data\n")
        # sweep_distributions.csv intentionally absent
        md = ModelDir(root=model_dir, system_name="sys-x", model_name="llama3-70b")
        rules = {r.rule for r in md._check_results if r.severity == Severity.ERROR}
        assert "model-sweep-file" in rules

    def test_both_sweep_files_present(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "sweep_summary.csv").write_text("data\n")
        (model_dir / "sweep_distributions.csv").write_text("data\n")
        md = ModelDir(root=model_dir, system_name="sys-x", model_name="llama3-70b")
        assert _passed(md._check_results)

    def test_run_dirs_property_matches_r_pattern(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "r16").mkdir()
        (model_dir / "r256").mkdir()
        (model_dir / "docs").mkdir()         # should be excluded
        (model_dir / "sweep_summary.csv").touch()  # file, not dir
        md = ModelDir(root=model_dir, system_name="sys-x", model_name="llama3-70b")
        names = [d.name for d in md.run_dirs]
        assert "r16" in names
        assert "r256" in names
        assert "docs" not in names

    def test_run_dirs_empty_when_none(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        md = ModelDir(root=model_dir, system_name="sys-x", model_name="llama3-70b")
        assert md.run_dirs == []


@pytest.mark.unit
class TestRunDir:
    def _make_valid_run_dir(self, tmp_path) -> None:
        """Create a fully valid r16/ run directory."""
        run = tmp_path / "r16"
        run.mkdir()
        (run / "point.yaml").write_text("concurrency: 16\n")
        acc = run / "accuracy"
        acc.mkdir()
        (acc / "accuracy.txt").write_text("ROUGE-1: 0.45\n")
        (acc / "accuracy_result.json").write_text("{}")
        (run / "mlperf_endpoints_log_summary.json").write_text("{}")
        (run / "mlperf_endpoints_log_detail.json").write_text("{}")
        (run / "run_metadata.json").write_text("{}")
        (run / "report.txt").write_text("report\n")
        src = run / "src" / "vllm"
        src.mkdir(parents=True)
        (src / ".gitkeep").write_text("")

    def test_all_present_passes(self, tmp_path):
        self._make_valid_run_dir(tmp_path)
        rd = RunDir(root=tmp_path / "r16", system_name="sys-x", model_name="llama3-70b")
        assert _passed(rd._check_results)

    def test_missing_point_yaml(self, tmp_path):
        self._make_valid_run_dir(tmp_path)
        (tmp_path / "r16" / "point.yaml").unlink()
        rd = RunDir(root=tmp_path / "r16", system_name="sys-x", model_name="llama3-70b")
        rules = {r.rule for r in rd._check_results if r.severity == Severity.ERROR}
        assert "run-point-config" in rules

    def test_missing_accuracy_dir(self, tmp_path):
        self._make_valid_run_dir(tmp_path)
        import shutil
        shutil.rmtree(tmp_path / "r16" / "accuracy")
        rd = RunDir(root=tmp_path / "r16", system_name="sys-x", model_name="llama3-70b")
        rules = {r.rule for r in rd._check_results if r.severity == Severity.ERROR}
        assert "run-accuracy-dir" in rules

    def test_missing_summary_log(self, tmp_path):
        self._make_valid_run_dir(tmp_path)
        (tmp_path / "r16" / "mlperf_endpoints_log_summary.json").unlink()
        rd = RunDir(root=tmp_path / "r16", system_name="sys-x", model_name="llama3-70b")
        rules = {r.rule for r in rd._check_results if r.severity == Severity.ERROR}
        assert "run-result-files" in rules

    def test_missing_detail_log(self, tmp_path):
        self._make_valid_run_dir(tmp_path)
        (tmp_path / "r16" / "mlperf_endpoints_log_detail.json").unlink()
        rd = RunDir(root=tmp_path / "r16", system_name="sys-x", model_name="llama3-70b")
        rules = {r.rule for r in rd._check_results if r.severity == Severity.ERROR}
        assert "run-result-files" in rules

    def test_missing_run_metadata(self, tmp_path):
        self._make_valid_run_dir(tmp_path)
        (tmp_path / "r16" / "run_metadata.json").unlink()
        rd = RunDir(root=tmp_path / "r16", system_name="sys-x", model_name="llama3-70b")
        rules = {r.rule for r in rd._check_results if r.severity == Severity.ERROR}
        assert "run-metadata" in rules

    def test_missing_report_txt(self, tmp_path):
        self._make_valid_run_dir(tmp_path)
        (tmp_path / "r16" / "report.txt").unlink()
        rd = RunDir(root=tmp_path / "r16", system_name="sys-x", model_name="llama3-70b")
        rules = {r.rule for r in rd._check_results if r.severity == Severity.ERROR}
        assert "run-report" in rules

    def test_missing_src_dir(self, tmp_path):
        self._make_valid_run_dir(tmp_path)
        import shutil
        shutil.rmtree(tmp_path / "r16" / "src")
        rd = RunDir(root=tmp_path / "r16", system_name="sys-x", model_name="llama3-70b")
        rules = {r.rule for r in rd._check_results if r.severity == Severity.ERROR}
        assert "run-src-dir" in rules

    def test_src_dir_no_impl_subdirs(self, tmp_path):
        self._make_valid_run_dir(tmp_path)
        import shutil
        shutil.rmtree(tmp_path / "r16" / "src")
        (tmp_path / "r16" / "src").mkdir()
        # src/ exists but has no implementation subdirectories
        rd = RunDir(root=tmp_path / "r16", system_name="sys-x", model_name="llama3-70b")
        rules = {r.rule for r in rd._check_results if r.severity == Severity.ERROR}
        assert "run-src-dir" in rules
