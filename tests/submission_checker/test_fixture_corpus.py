# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests that the committed fixture corpus is in the v1.0 layout, and stays there.

``tests/tools/regenerate_fixtures.py`` migrated the corpus and is committed so the next
format change does not start from scratch. Its idempotence is the property that makes it
re-runnable, so it is asserted here rather than trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from submission_checker import layout

from .conftest import TEST_SUBMISSIONS

_SUBMISSIONS = sorted(p for p in TEST_SUBMISSIONS.iterdir() if p.is_dir())

#: Files no longer part of any layout the spec defines.
_RETIRED_FILES = ("run_metadata.json", "system_desc_id.json", "results_summary.json")


def test_regenerator_is_idempotent() -> None:
    """Running it against the committed corpus must report nothing to do.

    This doubles as a guard on the corpus: if someone hand-edits a fixture into a shape
    the regenerator disagrees with, this fails and names the file.
    """
    from tests.tools.regenerate_fixtures import main

    assert main(["--check", "--root", str(TEST_SUBMISSIONS)]) == 0


@pytest.mark.parametrize("submission", _SUBMISSIONS, ids=lambda p: p.name)
class TestCorpusLayout:
    def test_no_retired_files_remain(self, submission: Path) -> None:
        for name in _RETIRED_FILES:
            assert not list(submission.rglob(name)), f"{name} still present"

    def test_no_pre_r_naming_survives(self, submission: Path) -> None:
        """point_<N>.yaml predates the r<N>/point.yaml layout."""
        assert not list(submission.rglob("point_*.yaml"))

    def test_every_point_has_a_system_description(self, submission: Path) -> None:
        for _system, model_dir in layout.iter_curves(submission / layout.RESULTS_DIR):
            for point_dir in layout.iter_point_dirs(model_dir):
                assert (point_dir / layout.SYSTEM_DESC_JSON).is_file(), point_dir

    def test_shared_trees_exist(self, submission: Path) -> None:
        assert (submission / layout.SRC_DIR).is_dir()
        assert (submission / layout.DOCS_DIR).is_dir()

    def test_every_implementation_has_a_readme(self, submission: Path) -> None:
        impls = [d for d in (submission / layout.SRC_DIR).iterdir() if d.is_dir()]
        assert impls
        for impl in impls:
            assert (impl / layout.README_MD).is_file(), impl

    def test_points_declare_the_v1_seed_names(self, submission: Path) -> None:
        for path in (submission / layout.RESULTS_DIR).rglob(layout.POINT_YAML):
            runtime = yaml.safe_load(path.read_text())["runtime_settings"]["runtime"]
            assert "scheduler_rng_seed" in runtime, path
            assert "sample_index_rng_seed" in runtime, path
            assert "model_seed" in runtime, path
            assert "scheduler_random_seed" not in runtime, path
            assert "dataloader_random_seed" not in runtime, path

    def test_points_declare_the_shared_pointers(self, submission: Path) -> None:
        for path in (submission / layout.RESULTS_DIR).rglob(layout.POINT_YAML):
            point = yaml.safe_load(path.read_text())
            for key in ("shared_src", "shared_docs", "seed_set", "target_cohort"):
                assert point.get(key), f"{path}: {key}"
            assert layout.resolve_shared_path(submission, point["shared_src"]) is not None
            assert layout.resolve_shared_path(submission, point["shared_docs"]) is not None

    def test_system_descriptions_use_publication_status(self, submission: Path) -> None:
        """§8.2's name for availability; the older spellings are gone from the corpus."""
        for path in (submission / layout.RESULTS_DIR).rglob(layout.SYSTEM_DESC_JSON):
            desc = json.loads(path.read_text())
            assert "publication_status" in desc, path
            assert "system_availability_status" not in desc, path

    def test_node_types_nest_accelerator_info(self, submission: Path) -> None:
        """§8.2.1 moved the flat accelerator_* node fields into accelerator_info[]."""
        for path in (submission / layout.RESULTS_DIR).rglob(layout.SYSTEM_DESC_JSON):
            for node in json.loads(path.read_text()).get("node_types", []):
                assert "accelerator_info" in node, path
                assert "accelerator_model_name" not in node, path

    def test_summaries_report_a_tpot_p90(self, submission: Path) -> None:
        """§9.1 makes it the sole source of tps_per_user, so every point needs one."""
        for path in (submission / layout.RESULTS_DIR).rglob(layout.RESULT_SUMMARY_JSON):
            percentiles = json.loads(path.read_text()).get("tpot", {}).get("percentiles", {})
            assert float(percentiles.get("90", 0)) > 0, path

    def test_ttft_percentiles_are_monotonic(self, submission: Path) -> None:
        """The migrated P90 must sit between the P50 and P95 it was interpolated from."""
        for path in (submission / layout.RESULTS_DIR).rglob(layout.RESULT_SUMMARY_JSON):
            percentiles = json.loads(path.read_text()).get("ttft", {}).get("percentiles", {})
            ordered = [percentiles[k] for k in ("50", "90", "95") if k in percentiles]
            assert ordered == sorted(ordered), f"{path}: {percentiles}"
