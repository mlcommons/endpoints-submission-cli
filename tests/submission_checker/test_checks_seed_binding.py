# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for SeedBinding — §4.6 seed rotation, checked across a Pareto curve."""

from __future__ import annotations

from pathlib import Path

import pytest

from submission_checker.models import PointConfig, RuntimeSettings, SeedBinding, Severity
from submission_checker.seed_sets import SeedSet, load_seed_sets

_SET_A = load_seed_sets()["A"]
_SET_B = SeedSet(id="B", scheduler_rng_seed=11, sample_index_rng_seed=22, model_seed=33, cohorts=())
_REGISTRY = {"A": _SET_A}


def _point(
    tmp_path: Path,
    concurrency: int = 64,
    seed_set: str | None = "A",
    target_cohort: str | None = "2026-09-C0",
    seeds: dict[str, int] | None = None,
    legacy: bool = False,
) -> tuple[Path, PointConfig]:
    runtime_kwargs: dict[str, int] = {}
    if legacy:
        runtime_kwargs = {"scheduler_random_seed": 42, "dataloader_random_seed": 42}
    else:
        runtime_kwargs = dict(seeds if seeds is not None else _SET_A.seeds)
    config = PointConfig(
        concurrency=concurrency,
        dataset="cnn_dailymail",
        seed_set=seed_set,
        target_cohort=target_cohort,
        runtime_settings=RuntimeSettings(runtime=RuntimeSettings.Runtime(**runtime_kwargs)),
    )
    return tmp_path / f"r{concurrency}" / "point.yaml", config


def _binding(tmp_path: Path, points, registry=None) -> SeedBinding:
    return SeedBinding(
        points=points, registry=_REGISTRY if registry is None else registry, model_dir=tmp_path
    )


def _errors(binding: SeedBinding, rule: str) -> list:
    return [r for r in binding._check_results if r.rule == rule and r.severity == Severity.ERROR]


@pytest.mark.unit
class TestSeedSetConsistency:
    def test_identical_sets_pass(self, tmp_path: Path) -> None:
        points = [_point(tmp_path, c) for c in (16, 64, 256)]
        assert not _errors(_binding(tmp_path, points), "seed-set-consistency")

    def test_differing_sets_error(self, tmp_path: Path) -> None:
        """§9.1: "every point must record the same seed set"."""
        points = [_point(tmp_path, 16), _point(tmp_path, 64, seed_set="B")]
        assert _errors(_binding(tmp_path, points), "seed-set-consistency")

    def test_no_declared_set_is_left_to_the_disclosure_check(self, tmp_path: Path) -> None:
        points = [_point(tmp_path, 16, seed_set=None, legacy=True)]
        binding = _binding(tmp_path, points)
        assert not [r for r in binding._check_results if r.rule == "seed-set-consistency"]

    def test_no_points_is_not_a_finding(self, tmp_path: Path) -> None:
        assert not _binding(tmp_path, [])._check_results


@pytest.mark.unit
class TestSeedSetMembership:
    def test_published_set_passes(self, tmp_path: Path) -> None:
        assert not _errors(_binding(tmp_path, [_point(tmp_path)]), "seed-set-membership")

    def test_unpublished_set_errors(self, tmp_path: Path) -> None:
        points = [_point(tmp_path, seed_set="Z")]
        assert _errors(_binding(tmp_path, points), "seed-set-membership")

    def test_a_newer_registry_admits_a_newer_set(self, tmp_path: Path) -> None:
        """The --seed-sets override exists so a new set lands without a release."""
        points = [_point(tmp_path, seed_set="B", seeds=_SET_B.seeds)]
        binding = _binding(tmp_path, points, registry={"A": _SET_A, "B": _SET_B})
        assert not _errors(binding, "seed-set-membership")
        assert not _errors(binding, "seed-runtime-match")


@pytest.mark.unit
class TestSeedRuntimeMatch:
    def test_matching_seeds_pass(self, tmp_path: Path) -> None:
        assert not _errors(_binding(tmp_path, [_point(tmp_path)]), "seed-runtime-match")

    def test_mismatched_seed_errors(self, tmp_path: Path) -> None:
        """§2.1.1: the RNGs must actually be seeded from the bound set."""
        seeds = {**_SET_A.seeds, "model_seed": 12345}
        points = [_point(tmp_path, seeds=seeds)]
        assert _errors(_binding(tmp_path, points), "seed-runtime-match")

    def test_v07_seed_of_42_errors_against_a_declared_set(self, tmp_path: Path) -> None:
        """A bundle that declares set A but still seeds 42 is not compliant."""
        points = [_point(tmp_path, seeds={"scheduler_rng_seed": 42, "sample_index_rng_seed": 42})]
        assert _errors(_binding(tmp_path, points), "seed-runtime-match")

    def test_unknown_set_suppresses_the_match_check(self, tmp_path: Path) -> None:
        """One error per defect: an unpublished set is reported by membership alone."""
        points = [_point(tmp_path, seed_set="Z")]
        binding = _binding(tmp_path, points)
        assert not [r for r in binding._check_results if r.rule == "seed-runtime-match"]

    def test_legacy_seed_names_warn(self, tmp_path: Path) -> None:
        """v0.7's spellings are read, then flagged so the submitter learns what changed."""
        points = [_point(tmp_path, legacy=True, seed_set="A")]
        binding = _binding(tmp_path, points)
        assert [
            r
            for r in binding._check_results
            if r.rule == "seed-runtime-match" and r.severity == Severity.WARNING
        ]


@pytest.mark.unit
class TestSeedSetAdoption:
    def test_skipped_while_the_registry_has_no_cohorts(self, tmp_path: Path) -> None:
        """§4.6's four-cohort window cannot be evaluated against a cohort-less registry."""
        binding = _binding(tmp_path, [_point(tmp_path)])
        results = [r for r in binding._check_results if r.rule == "seed-set-adoption"]
        assert len(results) == 1
        assert results[0].severity == Severity.INFO
        assert "SKIPPED" in results[0].message

    def test_enforced_once_cohorts_exist(self, tmp_path: Path) -> None:
        published = SeedSet(id="A", **_SET_A.seeds, cohorts=("2026-09-C0",))
        binding = _binding(tmp_path, [_point(tmp_path)], registry={"A": published})
        assert not _errors(binding, "seed-set-adoption")

    def test_wrong_cohort_errors_once_cohorts_exist(self, tmp_path: Path) -> None:
        published = SeedSet(id="A", **_SET_A.seeds, cohorts=("2025-01-C0",))
        binding = _binding(tmp_path, [_point(tmp_path)], registry={"A": published})
        assert _errors(binding, "seed-set-adoption")
