# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for approved drafters (§2.9.4) and agentic metrics (§4.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from submission_checker.drafters import (
    ApprovedDrafter,
    DrafterListError,
    bundled_drafters_path,
    load_approved_drafters,
)
from submission_checker.models import (
    DrafterBinding,
    PointConfig,
    PointResult,
    PointSummary,
    RuntimeSettings,
    Severity,
)

from .conftest import _REGIONS

_WEIGHT_ENTRY = ApprovedDrafter(
    benchmark="llama3.1-8b",
    approved_cohort="2026-10-C1",
    model_id="acme/llama3-eagle",
    weight_checksum="sha256:abc",
)
_CONFIG_ENTRY = ApprovedDrafter(
    benchmark="llama3.1-8b",
    approved_cohort="2026-10-C1",
    target_checksum="sha256:target",
    configuration={"exit_layer": 24},
)


def _point(
    tmp_path: Path,
    declared: dict | None,
    target_cohort: str = "2026-12-C1",
    concurrency: int = 64,
) -> tuple[Path, PointConfig]:
    payload: dict = {
        "concurrency": concurrency,
        "dataset": "cnn_dailymail",
        "target_cohort": target_cohort,
        "runtime_settings": {"load_pattern": "concurrency", "runtime": {}},
    }
    if declared is not None:
        payload["speculative_decoding"] = declared
    return tmp_path / "point.yaml", PointConfig.model_validate(payload)


def _binding(tmp_path: Path, points, approved) -> DrafterBinding:
    return DrafterBinding(
        points=points, approved=approved, benchmark="llama3.1-8b", model_dir=tmp_path
    )


def _errors(binding: DrafterBinding, rule: str) -> list:
    return [r for r in binding._check_results if r.rule == rule and r.severity == Severity.ERROR]


@pytest.mark.unit
class TestDrafterList:
    def test_bundled_list_ships_empty(self) -> None:
        """§2.9.4's list is not published yet; empty means "none approved"."""
        assert bundled_drafters_path().is_file()
        assert load_approved_drafters() == {}

    def test_entries_group_by_benchmark(self, tmp_path: Path) -> None:
        path = tmp_path / "d.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "drafters": [
                        {"benchmark": "a", "weight_checksum": "x"},
                        {"benchmark": "a", "weight_checksum": "y"},
                        {"benchmark": "b", "target_checksum": "t", "configuration": {"n": 1}},
                    ]
                }
            )
        )
        loaded = load_approved_drafters(path)
        assert sorted(loaded) == ["a", "b"]
        assert len(loaded["a"]) == 2

    def test_entry_identifying_no_drafter_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "d.yaml"
        path.write_text(yaml.safe_dump({"drafters": [{"benchmark": "a"}]}))
        with pytest.raises(DrafterListError, match="identifies no drafter"):
            load_approved_drafters(path)

    def test_missing_benchmark_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "d.yaml"
        path.write_text(yaml.safe_dump({"drafters": [{"weight_checksum": "x"}]}))
        with pytest.raises(DrafterListError, match="no 'benchmark'"):
            load_approved_drafters(path)

    def test_unreadable_file(self, tmp_path: Path) -> None:
        with pytest.raises(DrafterListError, match="Cannot read"):
            load_approved_drafters(tmp_path / "absent.yaml")

    def test_environment_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "d.yaml"
        path.write_text(yaml.safe_dump({"drafters": [{"benchmark": "z", "weight_checksum": "x"}]}))
        monkeypatch.setenv("MLPERF_ENDPOINTS_APPROVED_DRAFTERS", str(path))
        assert list(load_approved_drafters()) == ["z"]


@pytest.mark.unit
class TestMatching:
    """§9.1 matches by checksum — a model ID alone must not be enough."""

    def test_weight_checksum_matches(self) -> None:
        assert _WEIGHT_ENTRY.matches({"weight_checksum": "sha256:abc"})

    def test_model_id_alone_does_not_match(self) -> None:
        """A renamed or re-uploaded drafter must not pass as an approved one."""
        assert not _WEIGHT_ENTRY.matches({"model_id": "acme/llama3-eagle"})

    def test_configuration_identified_needs_both_halves(self) -> None:
        assert _CONFIG_ENTRY.matches(
            {"target_checksum": "sha256:target", "configuration": {"exit_layer": 24}}
        )
        assert not _CONFIG_ENTRY.matches({"target_checksum": "sha256:target"})
        assert not _CONFIG_ENTRY.matches(
            {"target_checksum": "sha256:target", "configuration": {"exit_layer": 12}}
        )

    def test_is_configuration_identified(self) -> None:
        assert _CONFIG_ENTRY.is_configuration_identified
        assert not _WEIGHT_ENTRY.is_configuration_identified


@pytest.mark.unit
class TestApprovedDrafterRule:
    def test_point_without_speculation_is_not_checked(self, tmp_path: Path) -> None:
        binding = _binding(tmp_path, [_point(tmp_path, None)], [_WEIGHT_ENTRY])
        assert not binding._check_results

    def test_approved_drafter_passes(self, tmp_path: Path) -> None:
        points = [_point(tmp_path, {"weight_checksum": "sha256:abc"})]
        assert not _errors(_binding(tmp_path, points, [_WEIGHT_ENTRY]), "approved-drafter")

    def test_unapproved_drafter_errors(self, tmp_path: Path) -> None:
        points = [_point(tmp_path, {"weight_checksum": "sha256:other"})]
        assert _errors(_binding(tmp_path, points, [_WEIGHT_ENTRY]), "approved-drafter")

    def test_speculation_with_no_approved_list_errors(self, tmp_path: Path) -> None:
        """§2.9.4: a benchmark with no approved drafter disallows speculation entirely."""
        points = [_point(tmp_path, {"weight_checksum": "sha256:abc"})]
        hits = _errors(_binding(tmp_path, points, []), "approved-drafter")
        assert hits and "no drafter is approved" in hits[0].message.lower()


@pytest.mark.unit
class TestApprovalLeadTime:
    """§2.9.4: usable from two cohorts after the approval cohort."""

    def _bind(self, tmp_path: Path, target_cohort: str) -> DrafterBinding:
        points = [_point(tmp_path, {"weight_checksum": "sha256:abc"}, target_cohort)]
        return _binding(tmp_path, points, [_WEIGHT_ENTRY])

    def test_earliest_usable_cohort(self) -> None:
        """Approved 2026-10-C1 → 2026-11-C0 → 2026-11-C1."""
        assert str(_WEIGHT_ENTRY.earliest_target_cohort()) == "2026-11-C1"

    def test_at_the_earliest_cohort_passes(self, tmp_path: Path) -> None:
        assert not _errors(self._bind(tmp_path, "2026-11-C1"), "drafter-approval-lead-time")

    def test_one_cohort_too_early_errors(self, tmp_path: Path) -> None:
        assert _errors(self._bind(tmp_path, "2026-11-C0"), "drafter-approval-lead-time")

    def test_the_approval_cohort_itself_is_too_early(self, tmp_path: Path) -> None:
        assert _errors(self._bind(tmp_path, "2026-10-C1"), "drafter-approval-lead-time")

    def test_later_cohorts_pass(self, tmp_path: Path) -> None:
        assert not _errors(self._bind(tmp_path, "2027-05-C0"), "drafter-approval-lead-time")

    def test_unevaluable_without_an_approval_cohort(self, tmp_path: Path) -> None:
        entry = ApprovedDrafter(benchmark="llama3.1-8b", weight_checksum="sha256:abc")
        points = [_point(tmp_path, {"weight_checksum": "sha256:abc"})]
        binding = _binding(tmp_path, points, [entry])
        hits = [r for r in binding._check_results if r.rule == "drafter-approval-lead-time"]
        assert hits and hits[0].severity == Severity.WARNING

    def test_unapproved_drafter_skips_the_lead_time_check(self, tmp_path: Path) -> None:
        """One error per defect: an unapproved drafter is reported once."""
        points = [_point(tmp_path, {"weight_checksum": "sha256:nope"})]
        binding = _binding(tmp_path, points, [_WEIGHT_ENTRY])
        assert not [r for r in binding._check_results if r.rule == "drafter-approval-lead-time"]


@pytest.mark.unit
class TestAgenticMetrics:
    """§4.1: e2e_avg_interactivity = sum(output_tokens) / sum(e2e_turn_time)."""

    def _result(self, tmp_path: Path, **summary_extras) -> PointResult:
        summary = PointSummary(
            n_samples_completed=1000,
            n_samples_issued=1000,
            duration_ns=1_200_000_000_000.0,
            **summary_extras,
        )
        config = PointConfig(
            concurrency=64,
            dataset="cnn_dailymail",
            runtime_settings=RuntimeSettings(runtime=RuntimeSettings.Runtime()),
        )
        return PointResult.model_validate(
            {"config": config, "summary": summary, "yaml_path": tmp_path / "p.yaml"},
            context={"regions": _REGIONS, "summary_path": tmp_path / "s.json"},
        )

    def _hits(self, result: PointResult) -> list:
        return [r for r in result._check_results if r.rule == "agentic-metric-consistency"]

    def test_derived_from_the_two_sums(self, tmp_path: Path) -> None:
        summary = PointSummary(
            n_samples_completed=1,
            duration_ns=1.0,
            output_tokens_per_turn_total=6000.0,
            e2e_turn_time_seconds_total=120.0,
        )
        assert summary.e2e_avg_interactivity == pytest.approx(50.0)

    def test_single_turn_benchmark_reports_nothing(self, tmp_path: Path) -> None:
        """Neither input is present, so the rule stands down entirely."""
        assert not self._hits(self._result(tmp_path))

    def test_stored_value_matching_the_derivation_passes(self, tmp_path: Path) -> None:
        result = self._result(
            tmp_path,
            output_tokens_per_turn_total=6000.0,
            e2e_turn_time_seconds_total=120.0,
            e2e_avg_interactivity=50.0,
        )
        assert self._hits(result) and all(r.severity == Severity.INFO for r in self._hits(result))

    def test_stored_mismatch_errors(self, tmp_path: Path) -> None:
        result = self._result(
            tmp_path,
            output_tokens_per_turn_total=6000.0,
            e2e_turn_time_seconds_total=120.0,
            e2e_avg_interactivity=999.0,
        )
        assert [r for r in self._hits(result) if r.severity == Severity.ERROR]

    def test_reported_without_its_inputs_errors(self, tmp_path: Path) -> None:
        """§9.1: reported agentic metrics must be *derivable* from their definitions."""
        result = self._result(tmp_path, e2e_avg_interactivity=50.0)
        assert [r for r in self._hits(result) if r.severity == Severity.ERROR]

    def test_zero_turn_time_is_not_a_division_error(self, tmp_path: Path) -> None:
        summary = PointSummary(
            n_samples_completed=1,
            duration_ns=1.0,
            output_tokens_per_turn_total=10.0,
            e2e_turn_time_seconds_total=0.0,
        )
        assert summary.e2e_avg_interactivity is None
