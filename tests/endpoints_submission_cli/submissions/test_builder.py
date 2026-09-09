# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for submissions.builder module."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

from endpoints_submission_cli.exceptions import SubmissionBuildError, TruncationError
from endpoints_submission_cli.submissions.builder import (
    PENDING_SUBMISSION_ID,
    _compute_max_tps,
    _extract_concurrency,
    _extract_model,
    _extract_run_type,
    _slugify,
    build_submission_folder,
    create_bundle_archive,
    extract_archive,
    set_submission_id,
)
from endpoints_submission_cli.truncation import truncate_responses as _truncate_responses
from submission_checker.models import Severity


def _submission_root(org_dir: Path) -> Path:
    """The single <submission_id>/ level below the org directory."""
    subs = [d for d in org_dir.iterdir() if d.is_dir()]
    assert len(subs) == 1, f"expected one submission dir, got {[d.name for d in subs]}"
    return subs[0]


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
    def test_builds_under_pending_submission_id(self, run_archive: Path, tmp_path: Path) -> None:
        """Without an id the tree is built under the placeholder level."""
        org_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        assert (org_dir / PENDING_SUBMISSION_ID).is_dir()
        assert (org_dir / PENDING_SUBMISSION_ID / "results").is_dir()

    def test_builds_under_given_submission_id(self, run_archive: Path, tmp_path: Path) -> None:
        org_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path, "sub-123"
        )
        assert (org_dir / "sub-123" / "results").is_dir()
        assert not (org_dir / PENDING_SUBMISSION_ID).exists()

    def test_creates_results_dir(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = _submission_root(
            build_submission_folder(
                [("run-001", run_archive)], "standardized", "available", tmp_path
            )
        )
        assert (sub_dir / "results").is_dir()

    def test_creates_docs_dir(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = _submission_root(
            build_submission_folder(
                [("run-001", run_archive)], "standardized", "available", tmp_path
            )
        )
        assert (sub_dir / "docs").is_dir()

    def test_src_implementation_copied_through(self, run_archive: Path, tmp_path: Path) -> None:
        """The run archive's src/<implementation>/ is copied through verbatim."""
        sub_dir = _submission_root(
            build_submission_folder(
                [("run-001", run_archive)], "standardized", "available", tmp_path
            )
        )
        impl = sub_dir / "src" / "trtllm"
        assert (impl / "README.md").exists()
        assert (impl / "launch_sut.sh").exists()

    def test_system_desc_written_per_point(self, run_archive: Path, tmp_path: Path) -> None:
        """Policies PR #119 moved the §8.2 description into every r<N>/ directory."""
        sub_dir = _submission_root(
            build_submission_folder(
                [("run-001", run_archive)], "standardized", "available", tmp_path
            )
        )
        assert not list((sub_dir / "results").glob("*/system_desc_id.json"))
        jsons = list(sub_dir.rglob("system_desc.json"))
        assert len(jsons) == 1
        assert jsons[0].parent.name.startswith("r")
        data = json.loads(jsons[0].read_text())
        assert data["division"] == "Standardized"
        assert "node_types" in data

    def test_point_yaml_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        yamls = list(sub_dir.rglob("point.yaml"))
        assert len(yamls) >= 1
        data = yaml.safe_load(yamls[0].read_text())
        assert data["concurrency"] == 4

    def test_log_summary_created(self, run_archive: Path, tmp_path: Path) -> None:
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        summaries = list(sub_dir.rglob("result_summary.json"))
        assert len(summaries) == 1
        data = json.loads(summaries[0].read_text())
        assert "n_samples_completed" in data
        assert "duration_ns" in data

    def test_accuracy_file_created(self, run_folder: Path, tmp_path: Path) -> None:
        acc_archive = self._make_archive(run_folder, tmp_path, "acc", 4, "accuracy")
        sub_dir = build_submission_folder(
            [("run-001", acc_archive)], "standardized", "available", tmp_path / "sub"
        )
        acc_jsons = list(sub_dir.rglob("accuracy_results.json"))
        assert len(acc_jsons) == 1

    def test_empty_run_list_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SubmissionBuildError, match="At least one"):
            build_submission_folder([], "standardized", "available", tmp_path)

    def test_missing_system_desc_in_archive_raises(self, tmp_path: Path) -> None:
        # Create archive without system_desc.json
        folder = tmp_path / "bad_run"
        folder.mkdir()
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "point.yaml").write_text("concurrency: 4\n")
        (folder / "result_summary.json").write_text("{}")
        archive = tmp_path / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="bad_run")
        with pytest.raises(SubmissionBuildError, match="system_desc.json"):
            build_submission_folder(
                [("bad", archive)], "standardized", "available", tmp_path / "out"
            )

    def test_system_desc_missing_required_fields_raises(self, tmp_path: Path) -> None:
        # Missing required fields (submitter_org_names, system_name, node_types) must raise
        folder = tmp_path / "bad_run"
        folder.mkdir()
        bad_desc = {"system_category": "datacenter"}
        (folder / "system_desc.json").write_text(json.dumps(bad_desc))
        (folder / "config.yaml").write_text(yaml.dump({"name": "x"}))
        (folder / "point.yaml").write_text("concurrency: 4\n")
        (folder / "result_summary.json").write_text("{}")
        archive = tmp_path / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="bad_run")
        with pytest.raises(SubmissionBuildError, match="schema validation"):
            build_submission_folder(
                [("bad", archive)], "standardized", "available", tmp_path / "out"
            )

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
        for fname in [
            "system_desc.json",
            "mlperf-system-info-single-node-0.json",
            "serving_config.json",
        ]:
            src = run_folder / fname
            if src.exists():
                shutil.copy(src, second_folder / fname)

        cfg["settings"]["load_pattern"]["target_concurrency"] = 16
        (second_folder / "config.yaml").write_text(yaml.dump(cfg))
        # point.yaml is supplied per run now, so the second point declares its own.
        point = yaml.safe_load((run_folder / "point.yaml").read_text())
        point["concurrency"] = 16
        (second_folder / "point.yaml").write_text(yaml.dump(point))
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
        yamls = list(sub_dir.rglob("point.yaml"))
        concurrencies = {yaml.safe_load(p.read_text())["concurrency"] for p in yamls}
        assert 4 in concurrencies
        assert 16 in concurrencies

    def test_system_json_division_from_cli(self, run_archive: Path, tmp_path: Path) -> None:
        # CLI division arg is authoritative — overwrites any placeholder or stale value in system_desc
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "serviced", "available", tmp_path
        )
        jsons = list(_submission_root(sub_dir).rglob("system_desc.json"))
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

    def test_duplicate_accuracy_concurrency_raises(self, run_folder: Path, tmp_path: Path) -> None:
        a1 = self._make_archive(run_folder, tmp_path, "run1", 4, "accuracy")
        a2 = self._make_archive(run_folder, tmp_path, "run2", 4, "accuracy")
        with pytest.raises(SubmissionBuildError, match="Duplicate"):
            build_submission_folder(
                [("run-001", a1), ("run-002", a2)], "standardized", "available", tmp_path / "sub"
            )

    def test_perf_and_accuracy_same_concurrency_ok(self, run_folder: Path, tmp_path: Path) -> None:
        a_perf = self._make_archive(run_folder, tmp_path, "perf", 4, "performance")
        a_acc = self._make_archive(run_folder, tmp_path, "acc", 4, "accuracy")
        sub_dir = build_submission_folder(
            [("run-perf", a_perf), ("run-acc", a_acc)],
            "standardized",
            "available",
            tmp_path / "sub",
        )
        # Both runs describe the same Pareto point, so they share one r4/ directory.
        results = _submission_root(sub_dir) / "results"
        assert list(results.glob("*/*/r4/point.yaml"))
        assert list(results.glob("*/*/r4/result_summary.json"))
        assert list(results.glob("*/*/r4/accuracy_results.json"))

    def test_accuracy_run_routed_to_accuracy(self, run_folder: Path, tmp_path: Path) -> None:
        a_acc = self._make_archive(run_folder, tmp_path, "acc", 4, "accuracy")
        sub_dir = build_submission_folder(
            [("run-acc", a_acc)], "standardized", "available", tmp_path / "sub"
        )
        model_dirs = [d for d in (_submission_root(sub_dir) / "results").glob("*/*") if d.is_dir()]
        assert len(model_dirs) == 1
        point_dir = model_dirs[0] / "r4"
        # An accuracy-only concurrency still defines the point: it supplies both the
        # point.yaml/result_summary.json and the accuracy results.
        assert (point_dir / "point.yaml").exists()
        assert (point_dir / "result_summary.json").exists()
        assert (point_dir / "accuracy_results.json").exists()


@pytest.mark.unit
class TestSrcValidation:
    """src/ defects are the submitter's to fix — the builder must not paper over them."""

    def _archive_without(self, run_folder: Path, tmp_path: Path, *, drop: str) -> Path:
        import shutil

        folder = tmp_path / "variant"
        shutil.copytree(run_folder, folder)
        target = folder / drop
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        archive = tmp_path / "variant.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="variant")
        return archive

    def test_without_src_raises(self, run_folder: Path, tmp_path: Path) -> None:
        archive = self._archive_without(run_folder, tmp_path, drop="src")
        with pytest.raises(SubmissionBuildError, match="no implementation directory"):
            build_submission_folder(
                [("run-001", archive)], "standardized", "available", tmp_path / "out"
            )

    def test_implementation_without_readme_raises(self, run_folder: Path, tmp_path: Path) -> None:
        archive = self._archive_without(run_folder, tmp_path, drop="src/trtllm/README.md")
        with pytest.raises(SubmissionBuildError, match=r"Missing README\.md in src/trtllm/"):
            build_submission_folder(
                [("run-001", archive)], "standardized", "available", tmp_path / "out"
            )

    def test_loose_files_at_src_root_are_not_an_implementation(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        """Files dumped straight into src/ do not constitute an implementation folder."""
        import shutil

        folder = tmp_path / "loose"
        shutil.copytree(run_folder, folder)
        shutil.rmtree(folder / "src")
        (folder / "src").mkdir()
        (folder / "src" / "harness.py").write_text("print('hi')\n")
        archive = tmp_path / "loose.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="loose")
        with pytest.raises(SubmissionBuildError, match="loose file"):
            build_submission_folder(
                [("run-001", archive)], "standardized", "available", tmp_path / "out"
            )

    def test_enforced_for_every_division(self, run_folder: Path, tmp_path: Path) -> None:
        """Enforced for all divisions for now; expected to become division-specific."""
        for division in ("standardized", "serviced", "rdi"):
            archive = self._archive_without(run_folder, tmp_path / division, drop="src")
            with pytest.raises(SubmissionBuildError, match="no implementation directory"):
                build_submission_folder(
                    [("run-001", archive)], division, "available", tmp_path / f"out-{division}"
                )

    def test_readme_required_for_every_division(self, run_folder: Path, tmp_path: Path) -> None:
        for division in ("standardized", "serviced", "rdi"):
            archive = self._archive_without(
                run_folder, tmp_path / division, drop="src/trtllm/README.md"
            )
            with pytest.raises(SubmissionBuildError, match=r"Missing README\.md"):
                build_submission_folder(
                    [("run-001", archive)], division, "available", tmp_path / f"out-{division}"
                )


@pytest.mark.unit
class TestSetSubmissionId:
    def test_renames_placeholder(self, run_archive: Path, tmp_path: Path) -> None:
        org_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        dest = set_submission_id(org_dir, "sub-123")
        assert dest == org_dir / "sub-123"
        assert (dest / "results").is_dir()
        assert not (org_dir / PENDING_SUBMISSION_ID).exists()

    def test_missing_placeholder_raises(self, tmp_path: Path) -> None:
        org_dir = tmp_path / "org"
        org_dir.mkdir()
        with pytest.raises(SubmissionBuildError, match="nothing to rename"):
            set_submission_id(org_dir, "sub-123")

    def test_existing_destination_raises(self, run_archive: Path, tmp_path: Path) -> None:
        org_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        (org_dir / "sub-123").mkdir()
        with pytest.raises(SubmissionBuildError, match="already exists"):
            set_submission_id(org_dir, "sub-123")


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
        # The archive carries the <organisation>/<submission_id>/ levels.
        assert any(f"{PENDING_SUBMISSION_ID}/results" in n for n in names)

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
    """The denominator of tps_utilization, derived from the measurement itself.

    v0.7 read a submitter-declared ``system_tps`` out of ``run_metadata.json``; policies
    PR #119 deleted that file, so the builder now computes throughput the same way the
    checker does — output tokens over elapsed seconds from ``result_summary.json``.
    """

    _DURATION_NS = 1_200_000_000_000.0  # 1200 s

    def _make_run(self, system_tps: float | None) -> dict:
        if system_tps is None:
            return {"result_summary": {}}
        return {
            "result_summary": {
                "duration_ns": self._DURATION_NS,
                "output_sequence_lengths": {
                    "total": system_tps * (self._DURATION_NS / 1e9),
                },
            }
        }

    def test_single_run(self) -> None:
        assert _compute_max_tps([self._make_run(1000.0)]) == pytest.approx(1000.0)

    def test_multiple_runs_returns_max(self) -> None:
        run_data = [self._make_run(500.0), self._make_run(1500.0), self._make_run(1000.0)]
        assert _compute_max_tps(run_data) == pytest.approx(1500.0)

    def test_missing_result_summary_returns_none(self) -> None:
        assert _compute_max_tps([{"result_summary": {}}]) is None

    def test_underivable_run_skipped(self) -> None:
        run_data = [self._make_run(None), self._make_run(800.0)]
        assert _compute_max_tps(run_data) == pytest.approx(800.0)

    def test_zero_duration_is_not_a_division_error(self) -> None:
        run_data = [
            {
                "result_summary": {
                    "duration_ns": 0.0,
                    "output_sequence_lengths": {"total": 1000.0},
                }
            }
        ]
        assert _compute_max_tps(run_data) is None


@pytest.mark.unit
class TestTpsUtilizationInjection:
    _DURATION_NS = 1_200_000_000_000.0  # 1200 s

    def _make_archive(
        self, run_folder: Path, system_tps: float, concurrency: int, tmp_path: Path, name: str
    ) -> Path:
        """An archive whose measured throughput is *system_tps* at *concurrency*.

        The throughput is set through ``result_summary.json`` because that is where the
        builder now reads it from: §8.2 has no ``system_tps`` field, so a submitter
        cannot declare a throughput that disagrees with what they measured.
        """
        import shutil

        folder = tmp_path / name
        shutil.copytree(run_folder, folder)

        point = yaml.safe_load((folder / "point.yaml").read_text())
        point["concurrency"] = concurrency
        (folder / "point.yaml").write_text(yaml.dump(point))

        summary = json.loads((folder / "result_summary.json").read_text())
        summary["duration_ns"] = self._DURATION_NS
        summary["output_sequence_lengths"] = {
            "total": system_tps * (self._DURATION_NS / 1e9),
            "percentiles": {},
        }
        (folder / "result_summary.json").write_text(json.dumps(summary))

        archive = tmp_path / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname=name)
        return archive

    def test_tps_utilization_written_for_single_run(self, run_folder: Path, tmp_path: Path) -> None:
        archive = self._make_archive(run_folder, 1000.0, 4, tmp_path, "run1")
        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        descs = list(sub_dir.rglob("system_desc.json"))
        assert len(descs) == 1
        assert json.loads(descs[0].read_text())["tps_utilization"] == pytest.approx(1.0)

    def test_tps_utilization_normalized_across_runs(self, run_folder: Path, tmp_path: Path) -> None:
        a1 = self._make_archive(run_folder, 1000.0, 4, tmp_path, "run1")
        a2 = self._make_archive(run_folder, 2000.0, 8, tmp_path, "run2")
        sub_dir = build_submission_folder(
            [("run-001", a1), ("run-002", a2)],
            "standardized",
            "available",
            tmp_path / "sub",
        )
        descs = sorted(sub_dir.rglob("system_desc.json"))
        assert len(descs) == 2
        utilizations = sorted(json.loads(p.read_text())["tps_utilization"] for p in descs)
        assert utilizations == pytest.approx([0.5, 1.0])

    def test_underivable_throughput_no_crash(self, run_archive: Path, tmp_path: Path) -> None:
        """A run whose summary yields no throughput builds without tps_utilization."""
        sub_dir = build_submission_folder(
            [("run-001", run_archive)], "standardized", "available", tmp_path
        )
        assert sub_dir.is_dir()


@pytest.mark.unit
class TestPointYamlIsCopiedNotDerived:
    """point.yaml is supplied by the submitter and copied through untouched (issue #72).

    It used to be derived from config.yaml, which silently produced nulls for any §8.3
    field a harness did not put in its config — the checker then rejected the bundle.
    """

    def _archive(self, run_folder: Path, tmp_path: Path, name: str = "run") -> Path:
        import shutil

        folder = tmp_path / name
        shutil.copytree(run_folder, folder)
        archive = tmp_path / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname=name)
        return archive

    def test_copied_byte_for_byte(self, run_folder: Path, tmp_path: Path) -> None:
        original = (run_folder / "point.yaml").read_bytes()
        archive = self._archive(run_folder, tmp_path)
        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        assert next(sub_dir.rglob("point.yaml")).read_bytes() == original

    def test_unknown_keys_survive(self, run_folder: Path, tmp_path: Path) -> None:
        """The builder must not filter the disclosure down to fields it knows about."""
        import shutil

        folder = tmp_path / "extra"
        shutil.copytree(run_folder, folder)
        point = yaml.safe_load((folder / "point.yaml").read_text())
        point["submitter_specific_note"] = "keep me"
        point["runtime_settings"]["future_field"] = 123
        (folder / "point.yaml").write_text(yaml.dump(point))
        archive = tmp_path / "extra.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="extra")

        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        out = yaml.safe_load(next(sub_dir.rglob("point.yaml")).read_text())
        assert out["submitter_specific_note"] == "keep me"
        assert out["runtime_settings"]["future_field"] == 123

    def test_config_changes_do_not_alter_point_yaml(self, run_folder: Path, tmp_path: Path) -> None:
        """Proof the value is no longer sourced from config.yaml."""
        import shutil

        folder = tmp_path / "divergent"
        shutil.copytree(run_folder, folder)
        cfg = yaml.safe_load((folder / "config.yaml").read_text())
        cfg["settings"]["client"]["stream_all_chunks"] = False
        cfg["settings"]["runtime"]["min_duration_ms"] = 999
        (folder / "config.yaml").write_text(yaml.dump(cfg))
        archive = tmp_path / "divergent.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="divergent")

        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        out = yaml.safe_load(next(sub_dir.rglob("point.yaml")).read_text())
        # point.yaml still reports its own values, untouched by the config edits.
        assert out["runtime_settings"]["stream_all_chunks"] is True
        assert out["runtime_settings"]["min_duration_ms"] == 600000

    def test_missing_point_yaml_raises(self, run_folder: Path, tmp_path: Path) -> None:
        import shutil

        folder = tmp_path / "nopoint"
        shutil.copytree(run_folder, folder)
        (folder / "point.yaml").unlink()
        archive = tmp_path / "nopoint.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="nopoint")

        with pytest.raises(SubmissionBuildError, match="does not contain point.yaml"):
            build_submission_folder(
                [("run-001", archive)], "standardized", "available", tmp_path / "sub"
            )

    def test_accuracy_only_point_uses_its_own_point_yaml(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        """An accuracy-only concurrency still supplies the point, so its file is used."""
        import shutil

        folder = tmp_path / "acc"
        shutil.copytree(run_folder, folder)
        cfg = yaml.safe_load((folder / "config.yaml").read_text())
        cfg["datasets"][0]["type"] = "accuracy"
        (folder / "config.yaml").write_text(yaml.dump(cfg))
        point = yaml.safe_load((folder / "point.yaml").read_text())
        point["marker"] = "from-accuracy-run"
        (folder / "point.yaml").write_text(yaml.dump(point))
        archive = tmp_path / "acc.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="acc")

        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        out = yaml.safe_load(next(sub_dir.rglob("point.yaml")).read_text())
        assert out["marker"] == "from-accuracy-run"


@pytest.mark.unit
class TestRunArchiveLayouts:
    """The builder consumes archives `runs create` uploaded, in either phase layout.

    PR #78 moved the summary `runs create` reads to ``performance/result_summary.json``,
    the path mlcommons/endpoints actually writes. That left the builder — which reads
    the *uploaded archive* — still looking flat, so `submissions create` failed on every
    real run. The two changes touched different files, so git merged them cleanly and
    nothing caught it.

    The builder stays permissive where `runs create` is strict: it consumes archives the
    API already holds, including ones uploaded before the layout settled.
    """

    def _archive(self, run_folder: Path, tmp_path: Path, *, phase_dirs: bool) -> Path:
        import shutil

        folder = tmp_path / "run"
        shutil.copytree(run_folder, folder)
        if phase_dirs:
            summary = (folder / "result_summary.json").read_bytes()
            (folder / "result_summary.json").unlink()
            (folder / "performance").mkdir()
            (folder / "performance" / "result_summary.json").write_bytes(summary)
        archive = tmp_path / "run.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="run")
        return archive

    @pytest.mark.parametrize("phase_dirs", [True, False], ids=["endpoints", "flat"])
    def test_both_layouts_build(self, run_folder: Path, tmp_path: Path, phase_dirs: bool) -> None:
        archive = self._archive(run_folder, tmp_path, phase_dirs=phase_dirs)
        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        assert list(sub_dir.rglob("point.yaml"))

    def test_bundle_is_always_flat(self, run_folder: Path, tmp_path: Path) -> None:
        """§8.1 puts result_summary.json in r<N>/ — the phase directories are input-only."""
        archive = self._archive(run_folder, tmp_path, phase_dirs=True)
        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        summary = next(sub_dir.rglob("result_summary.json"))
        assert summary.parent.name.startswith("r")
        assert not list(sub_dir.rglob("performance/result_summary.json"))

    def test_missing_summary_names_the_expected_path(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        import shutil

        folder = tmp_path / "run"
        shutil.copytree(run_folder, folder)
        (folder / "result_summary.json").unlink()
        archive = tmp_path / "run.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="run")

        with pytest.raises(SubmissionBuildError, match="performance/result_summary.json"):
            build_submission_folder(
                [("run-001", archive)], "standardized", "available", tmp_path / "sub"
            )

    def test_endpoints_accuracy_artifact_is_used(self, run_folder: Path, tmp_path: Path) -> None:
        """endpoints writes accuracy/accuracy_results.json, not accuracy/results.json."""
        import shutil

        folder = tmp_path / "run"
        shutil.copytree(run_folder, folder)
        config = yaml.safe_load((folder / "config.yaml").read_text())
        config["datasets"] = [{"type": "accuracy"}]
        (folder / "config.yaml").write_text(yaml.dump(config))
        (folder / "results.json").unlink(missing_ok=True)
        (folder / "accuracy").mkdir(exist_ok=True)
        (folder / "accuracy" / "accuracy_results.json").write_text(
            json.dumps({"cnn_dailymail": {"num_samples": 10, "score": {"rouge1": 42.0}}})
        )
        archive = tmp_path / "run.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="run")

        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        written = next(sub_dir.rglob("accuracy_results.json"))
        assert json.loads(written.read_text())["cnn_dailymail"]["score"]["rouge1"] == 42.0


@pytest.mark.unit
class TestDirectoryNamingPrecedence:
    """§8.1 names the results directory ``<model_name>``, which lives in point.yaml.

    config.yaml became optional in v1.0, so deriving a path from it meant the tree's
    shape depended on a file that need not exist — and its ``model_params.name`` is a
    HuggingFace path, not the supported-model-list name §8.2 defines.
    """

    def test_point_yaml_model_name_wins(self) -> None:
        config = {"model_params": {"name": "meta-llama/Llama-3.1-8B-Instruct"}}
        assert _extract_model(config, {"model_name": "llama3.1-8b"}) == "llama3_1-8b"

    def test_config_is_only_a_fallback(self) -> None:
        config = {"model_params": {"name": "meta-llama/Llama-3.1-8B-Instruct"}}
        assert _extract_model(config, {}) == "Llama-3_1-8B-Instruct"

    def test_neither_source_yields_a_placeholder(self) -> None:
        assert _extract_model({}, {}) == "unknown_model"

    def test_concurrency_uses_the_same_precedence(self) -> None:
        """The two directory-naming helpers must not disagree about which file wins."""
        config = {"settings": {"load_pattern": {"target_concurrency": 999}}}
        assert _extract_concurrency(config, {"concurrency": 16}) == 16

    def test_built_tree_is_named_from_the_disclosure(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        import shutil

        folder = tmp_path / "run"
        shutil.copytree(run_folder, folder)
        archive = tmp_path / "run.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="run")

        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        # point.yaml declares model_name: llama3.1-8b; config.yaml says the HF path.
        model_dirs = [p.parent.parent.name for p in sub_dir.rglob("point.yaml")]
        assert model_dirs == ["llama3_1-8b"]


@pytest.mark.unit
class TestRunTypeResolution:
    """A run whose type cannot be determined is refused, not assumed.

    Defaulting to "performance" was silently destructive: an accuracy run shipped
    without a config.yaml would be filed as its concurrency's performance run, colliding
    with the real one and dropping the accuracy results from the bundle.
    """

    def test_config_dataset_type_wins(self) -> None:
        config = {"datasets": [{"type": "accuracy"}]}
        assert _extract_run_type(config, {"dataset_type": "Performance"}, "r1") == "accuracy"

    def test_point_dataset_type_is_the_fallback(self) -> None:
        assert _extract_run_type({}, {"dataset_type": "Accuracy"}, "r1") == "accuracy"
        assert _extract_run_type({}, {"dataset_type": "Performance"}, "r1") == "performance"

    def test_dataset_type_is_matched_case_insensitively(self) -> None:
        assert _extract_run_type({}, {"dataset_type": "accuracy"}, "r1") == "accuracy"

    def test_combined_dataset_type_is_ambiguous(self) -> None:
        """ "Accuracy + Performance" describes the dataset, not which one this run did."""
        with pytest.raises(SubmissionBuildError, match="covers"):
            _extract_run_type({}, {"dataset_type": "Accuracy + Performance"}, "run-001")

    def test_no_source_errors(self) -> None:
        with pytest.raises(SubmissionBuildError, match="accuracy or a performance run"):
            _extract_run_type({}, {}, "run-001")

    def test_error_names_the_run(self) -> None:
        with pytest.raises(SubmissionBuildError, match="run-042"):
            _extract_run_type({}, {}, "run-042")

    def test_build_succeeds_without_config_when_point_declares_the_type(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        import shutil

        folder = tmp_path / "run"
        shutil.copytree(run_folder, folder)
        (folder / "config.yaml").unlink()
        point = yaml.safe_load((folder / "point.yaml").read_text())
        point["dataset_type"] = "Performance"
        (folder / "point.yaml").write_text(yaml.dump(point))
        archive = tmp_path / "run.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="run")

        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        assert list(sub_dir.rglob("point.yaml"))

    def test_build_fails_when_neither_source_declares_the_type(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        import shutil

        folder = tmp_path / "run"
        shutil.copytree(run_folder, folder)
        (folder / "config.yaml").unlink()
        point = yaml.safe_load((folder / "point.yaml").read_text())
        point.pop("dataset_type", None)
        (folder / "point.yaml").write_text(yaml.dump(point))
        archive = tmp_path / "run.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="run")

        with pytest.raises(SubmissionBuildError, match="accuracy or a performance run"):
            build_submission_folder(
                [("run-001", archive)], "standardized", "available", tmp_path / "sub"
            )


@pytest.mark.unit
class TestSharedPathInjection:
    """shared_src / shared_docs are the one §8.3 pair the builder may fill in.

    Issue #72's rule is that the builder does not derive disclosure. These two keys are
    exempt because their referent — ``src/<impl>/``, the union of every run's ``src/``
    folder — does not exist until the builder assembles it, so no run archive can name
    it. Validate if present, inject if absent, never overwrite.
    """

    def _archive_with_point(self, run_folder: Path, tmp_path: Path, point_text: str) -> Path:
        import shutil

        folder = tmp_path / "run"
        shutil.copytree(run_folder, folder)
        (folder / "point.yaml").write_text(point_text)
        archive = tmp_path / "run.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="run")
        return archive

    def _point_without_pointers(self, run_folder: Path) -> dict:
        point = yaml.safe_load((run_folder / "point.yaml").read_text())
        point.pop("shared_src", None)
        point.pop("shared_docs", None)
        return point

    def test_injected_when_absent(self, run_folder: Path, tmp_path: Path) -> None:
        point = self._point_without_pointers(run_folder)
        archive = self._archive_with_point(run_folder, tmp_path, yaml.dump(point))
        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        built = yaml.safe_load(next(sub_dir.rglob("point.yaml")).read_text())
        assert built["shared_src"] == "src/trtllm"
        assert built["shared_docs"] == "docs"

    def test_injection_preserves_comments_and_key_order(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        """The original bytes survive as a prefix, so comments and order are intact."""
        point = self._point_without_pointers(run_folder)
        original = (
            "# submitter's own header comment\n"
            + yaml.dump(point, sort_keys=False)
            + "# trailing note\n"
        )
        archive = self._archive_with_point(run_folder, tmp_path, original)
        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        built_text = next(sub_dir.rglob("point.yaml")).read_text()

        assert built_text.startswith(original)
        assert "# submitter's own header comment" in built_text
        assert "# trailing note" in built_text
        assert yaml.safe_load(built_text) == {
            **point,
            "shared_src": "src/trtllm",
            "shared_docs": "docs",
        }

    def test_declared_pointer_is_not_overwritten(self, run_folder: Path, tmp_path: Path) -> None:
        """A submitter naming a resolvable directory is making a claim, not a mistake."""
        import shutil

        folder = tmp_path / "run"
        shutil.copytree(run_folder, folder)
        # A second implementation the submitter deliberately points at.
        other = folder / "src" / "vllm"
        other.mkdir(parents=True)
        (other / "README.md").write_text("# vllm\n")
        point = yaml.safe_load((folder / "point.yaml").read_text())
        point["shared_src"] = "src/vllm"
        (folder / "point.yaml").write_text(yaml.dump(point))
        archive = tmp_path / "run.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="run")

        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        built = yaml.safe_load(next(sub_dir.rglob("point.yaml")).read_text())
        assert built["shared_src"] == "src/vllm"

    def test_unresolvable_pointer_fails_the_build(self, run_folder: Path, tmp_path: Path) -> None:
        point = yaml.safe_load((run_folder / "point.yaml").read_text())
        point["shared_src"] = "src/does-not-exist"
        archive = self._archive_with_point(run_folder, tmp_path, yaml.dump(point))
        with pytest.raises(SubmissionBuildError, match="does not resolve"):
            build_submission_folder(
                [("run-001", archive)], "standardized", "available", tmp_path / "sub"
            )

    def test_traversal_pointer_fails_the_build(self, run_folder: Path, tmp_path: Path) -> None:
        """A pointer escaping the bundle cannot be reviewed from the bundle."""
        point = yaml.safe_load((run_folder / "point.yaml").read_text())
        point["shared_docs"] = "../../etc"
        archive = self._archive_with_point(run_folder, tmp_path, yaml.dump(point))
        with pytest.raises(SubmissionBuildError, match="does not resolve"):
            build_submission_folder(
                [("run-001", archive)], "standardized", "available", tmp_path / "sub"
            )

    def test_ambiguous_implementation_is_not_guessed(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        """Two implementations and no declared shared_src is an error, not a coin flip."""
        import shutil

        folder = tmp_path / "run"
        shutil.copytree(run_folder, folder)
        second = folder / "src" / "vllm"
        second.mkdir(parents=True)
        (second / "README.md").write_text("# vllm\n")
        point = self._point_without_pointers(run_folder)
        (folder / "point.yaml").write_text(yaml.dump(point))
        archive = tmp_path / "run.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname="run")

        with pytest.raises(SubmissionBuildError, match="more than one implementation"):
            build_submission_folder(
                [("run-001", archive)], "standardized", "available", tmp_path / "sub"
            )

    def test_injected_pointers_satisfy_the_checker(self, run_folder: Path, tmp_path: Path) -> None:
        """End to end: what the builder injects is what §9.1 accepts."""
        from submission_checker.checker import SubmissionChecker
        from submission_checker.models import Severity

        point = self._point_without_pointers(run_folder)
        archive = self._archive_with_point(run_folder, tmp_path, yaml.dump(point))
        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        report = SubmissionChecker(sub_dir).run()
        assert not [
            r
            for r in report.results
            if r.rule == "shared-path-resolution" and r.severity == Severity.ERROR
        ]


@pytest.mark.unit
class TestBuilderCheckerContract:
    """End-to-end: the builder's output must be consumable by the real checker.

    This is the coverage that was missing — builder unit tests only inspected
    individual YAML keys, and checker unit tests only validated hand-built dicts,
    so a structural mismatch (builder omitting a field the checker requires) slipped
    through. These tests run the builder's actual output through the real checker.
    """

    def _complete_run_archive(self, run_folder: Path, tmp_path: Path, name: str = "run") -> Path:
        """A run folder whose config has every field the checker needs to parse a point."""
        import shutil

        folder = tmp_path / name
        shutil.copytree(run_folder, folder)
        cfg = yaml.safe_load((folder / "config.yaml").read_text())
        cfg["settings"]["runtime"].update({"min_duration_ms": 600_000})
        cfg["settings"]["client"] = {"stream_all_chunks": True}
        cfg["settings"]["warmup"] = {
            "enabled": True,
            "salt": False,
            "duration_s": 60.0,
            "requests_issued": 100,
            "requests_completed": 100,
            "data_source": "warmup-ds",
            "concurrency": 4,
        }
        (folder / "config.yaml").write_text(yaml.dump(cfg))
        archive = tmp_path / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname=name)
        return archive

    def test_built_points_parse_through_checker_pointconfig(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        """Every built point_*.yaml must validate against the checker's PointConfig."""
        from submission_checker.models.file.point_config import PointConfig

        archive = self._complete_run_archive(run_folder, tmp_path)
        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        points = list(sub_dir.rglob("point.yaml"))
        assert points, "builder produced no point YAML"
        for py in points:
            data = yaml.safe_load(py.read_text())
            # Must not raise — this is exactly what failed before the runtime fix.
            cfg = PointConfig.model_validate(data, context={"yaml_path": py})
            # §4.6 binds the seeds to a published set; the fixture uses set A.
            assert cfg.seed_set == "A"
            assert cfg.runtime_settings.runtime.scheduler_rng_seed == 10487924139932647040
            assert cfg.runtime_settings.runtime.sample_index_rng_seed == 586478644936801402
            assert cfg.runtime_settings.runtime.model_seed == 9315206023656308754
            # No per-point structural/seed errors for a compliant input.
            errors = [r for r in cfg._check_results if r.severity == Severity.ERROR]
            assert not errors, (
                f"{py.name} unexpected errors: {[(r.rule, r.message) for r in errors]}"
            )

    #: A curve covering every §9.1 region for C_max=1024 with a derived C_min=16:
    #: low 17–26 → 20, med 27–117 → 88, high 118–1024 → 256, 512, 768, 1000.
    _COMPLIANT_CURVE = (16, 20, 88, 256, 512, 768, 1000)

    def _curve_archive(
        self, run_folder: Path, tmp_path: Path, concurrency: int, system_tps: float
    ) -> Path:
        """One point of a compliant curve, as a run archive."""
        import shutil

        name = f"run{concurrency}"
        folder = tmp_path / name
        shutil.copytree(run_folder, folder)

        duration_ns = 1_200_000_000_000.0  # §6.2's 1200 s concurrency-region minimum
        point = yaml.safe_load((folder / "point.yaml").read_text())
        point["concurrency"] = concurrency
        point["max_supported_concurrency"] = 1024
        point.pop("region", None)  # let the checker place it
        point["runtime_settings"]["min_duration_ms"] = 1_200_000
        point["warmup"] = {
            "duration_s": 60.0,
            "requests_issued": concurrency * 10,
            "requests_completed": concurrency * 10,
            "data_source": "cnn_dailymail validation split",
            "concurrency": concurrency,
            "initialization_steps": ["model loaded", "kv-cache warmed"],
            "logs_retained": True,
        }
        (folder / "point.yaml").write_text(yaml.dump(point))

        desc = json.loads((folder / "system_desc.json").read_text())
        desc["max_supported_concurrency"] = 1024
        # Both are the supported-model-list name (§8.2, §8.5), which is also what
        # names the results directory (§8.1) now that point.yaml is the source.
        desc["model_name"] = "llama3.1-8b"
        desc["model_id"] = "llama3.1-8b"
        (folder / "system_desc.json").write_text(json.dumps(desc))

        # §6.4 requires every cnn_dailymail sample to be scored for accuracy.
        results = json.loads((folder / "results.json").read_text())
        for entry in results["accuracy_scores"].values():
            entry["num_samples"] = 13368
        (folder / "results.json").write_text(json.dumps(results))

        summary = json.loads((folder / "result_summary.json").read_text())
        summary["duration_ns"] = duration_ns
        summary["n_samples_issued"] = 13368  # §6.4 minimum for cnn_dailymail
        summary["n_samples_completed"] = 13368
        summary["n_samples_failed"] = 0
        summary["output_sequence_lengths"] = {
            "total": system_tps * (duration_ns / 1e9),
            "percentiles": {},
        }
        summary.pop("system_tps", None)
        summary.pop("tps_per_user", None)
        (folder / "result_summary.json").write_text(json.dumps(summary))

        archive = tmp_path / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname=name)
        return archive

    def test_built_curve_passes_the_real_checker(self, run_folder: Path, tmp_path: Path) -> None:
        """The closing loop: a compliant curve built by the builder must pass §9.1.

        Every other test here checks one field or one rule. This one asserts the two
        halves of the pipeline actually agree — the failure mode that motivated issue
        #72 was precisely a builder writing a bundle its own checker rejected.
        """
        from submission_checker.checker import SubmissionChecker

        archives = [
            (f"run-{i:03d}", self._curve_archive(run_folder, tmp_path, c, 100.0 * (i + 1)))
            for i, c in enumerate(self._COMPLIANT_CURVE)
        ]
        sub_dir = build_submission_folder(archives, "standardized", "available", tmp_path / "sub")
        report = SubmissionChecker(sub_dir).run()
        assert report.passed, [f"{r.rule}: {r.message}" for r in report.errors]

    def test_built_curve_satisfies_the_new_v1_rules(self, run_folder: Path, tmp_path: Path) -> None:
        """Named explicitly, so a regression says which v1.0 rule broke."""
        from submission_checker.checker import SubmissionChecker

        archives = [
            (f"run-{i:03d}", self._curve_archive(run_folder, tmp_path, c, 100.0 * (i + 1)))
            for i, c in enumerate(self._COMPLIANT_CURVE)
        ]
        sub_dir = build_submission_folder(archives, "standardized", "available", tmp_path / "sub")
        report = SubmissionChecker(sub_dir).run()
        for rule in (
            "shared-path-resolution",
            "seed-set-consistency",
            "seed-set-membership",
            "seed-runtime-match",
            "target-cohort",
            "point-disclosure-complete",
            "metric-consistency-tpot-p90",
            "metric-consistency-tps-per-user",
            "tps-utilization",
            "system-description-consistency",
            "ultra-low-concurrency-coverage",
            "low-concurrency-coverage",
            "med-concurrency-coverage",
            "high-concurrency-coverage",
            "region-basis",
        ):
            ran = [r for r in report.results if r.rule == rule]
            assert ran, f"{rule} did not run"
            assert not [r for r in ran if r.severity == Severity.ERROR], (
                f"{rule}: {[r.message for r in ran if r.severity == Severity.ERROR]}"
            )

    def test_full_checker_runs_without_parse_failures(
        self, run_folder: Path, tmp_path: Path
    ) -> None:
        """Running the real SubmissionChecker on built output yields no parse/validation errors.

        Other rule failures (e.g. point-count) are acceptable here — we only assert that
        nothing failed to *parse*, which is the builder↔checker contract.
        """
        from submission_checker.checker import SubmissionChecker

        archive = self._complete_run_archive(run_folder, tmp_path)
        sub_dir = build_submission_folder(
            [("run-001", archive)], "standardized", "available", tmp_path / "sub"
        )
        report = SubmissionChecker(sub_dir).run()
        parse_failures = [
            r
            for r in report.results
            if r.severity == Severity.ERROR
            and ("validation error" in r.message.lower() or "field required" in r.message.lower())
        ]
        assert not parse_failures, (
            "checker hit parse/validation failures on builder output: "
            f"{[(r.rule, r.message) for r in parse_failures]}"
        )


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

    def _make_dict_content(self, n_responses: int) -> bytes:
        # Some run modes emit responses as a {sample_uuid: output_text} mapping.
        data = {
            "config": {"mode": "perf"},
            "accuracy_scores": {"ds": {"num_samples": n_responses, "score": 1.0}},
            "responses": {f"uuid-{i:08d}": "hello world " * 5 for i in range(n_responses)},
        }
        return json.dumps(data).encode()

    def test_small_dict_responses_unchanged(self) -> None:
        content = self._make_dict_content(2)
        result = json.loads(_truncate_responses(content))
        assert result["responses"] == json.loads(content)["responses"]
        assert result["accuracy_scores"] == {"ds": {"num_samples": 2, "score": 1.0}}

    def test_large_dict_responses_truncated_under_10kb(self) -> None:
        content = self._make_dict_content(10_000)
        result = json.loads(_truncate_responses(content))
        assert isinstance(result["responses"], dict)
        assert len(json.dumps(result["responses"]).encode()) <= 10 * 1024
        assert 0 < len(result["responses"]) < 10_000  # kept some, dropped the rest
        # non-responses fields untouched
        assert result["accuracy_scores"] == {"ds": {"num_samples": 10_000, "score": 1.0}}

    def test_unknown_responses_type_raises_hard_error(self) -> None:
        # A responses shape we cannot bound must fail loudly, never pass through silently.
        content = json.dumps({"config": {}, "responses": "x" * 50_000}).encode()
        with pytest.raises(TruncationError):
            _truncate_responses(content)

    def test_missing_responses_is_not_an_error(self) -> None:
        content = json.dumps({"config": {}, "accuracy_scores": {}}).encode()
        assert _truncate_responses(content) == content  # nothing to truncate, no raise

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


@pytest.mark.unit
class TestBuilderTruncatesResults:
    """build_submission_folder truncates the verbose responses in the perf results.json."""

    def _archive(self, run_folder: Path, tmp_path: Path, responses: object) -> Path:
        import shutil

        folder = tmp_path / "perf_run"
        shutil.copytree(run_folder, folder)
        results = json.loads((folder / "results.json").read_text())
        results["responses"] = responses
        (folder / "results.json").write_text(json.dumps(results))
        archive = tmp_path / "perf_run.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname=folder.name)
        return archive

    def _perf_results(self, sub_dir: Path) -> dict:
        perf = [p for p in sub_dir.rglob("results.json") if p.parent.name != "accuracy"]
        assert len(perf) == 1
        return json.loads(perf[0].read_text())

    def test_dict_responses_truncated_in_bundle(self, run_folder: Path, tmp_path: Path) -> None:
        # Regression: responses emitted as a {uuid: text} dict were not truncated.
        big = {f"uuid-{i:08d}": "hello world " * 5 for i in range(10_000)}
        sub_dir = build_submission_folder(
            [("run-001", self._archive(run_folder, tmp_path, big))],
            "standardized",
            "available",
            tmp_path / "sub",
        )
        written = self._perf_results(sub_dir)
        assert isinstance(written["responses"], dict)
        assert len(json.dumps(written["responses"]).encode()) <= 10 * 1024
        assert 0 < len(written["responses"]) < 10_000

    def test_list_responses_truncated_in_bundle(self, run_folder: Path, tmp_path: Path) -> None:
        big = [{"text": "hello world", "idx": i} for i in range(10_000)]
        sub_dir = build_submission_folder(
            [("run-001", self._archive(run_folder, tmp_path, big))],
            "standardized",
            "available",
            tmp_path / "sub",
        )
        written = self._perf_results(sub_dir)
        assert isinstance(written["responses"], list)
        assert len(json.dumps(written["responses"]).encode()) <= 10 * 1024
        assert 0 < len(written["responses"]) < 10_000
