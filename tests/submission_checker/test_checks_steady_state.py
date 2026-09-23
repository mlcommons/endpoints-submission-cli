# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for the §4.4 steady-state reporting basis.

The checker cannot run the detector — that is post-processing over `events.jsonl`,
which is not in the bundle — so what is asserted here is that the *reported* block is
internally coherent, and that §6.2 is measured against the window rather than the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from submission_checker.models import PointConfig, PointResult, Severity
from submission_checker.models.file.steady_state import (
    MIN_TREND_N,
    SteadyState,
    SteadyStateWindow,
)

from .conftest import _REGIONS, _summary


def _block(
    status: str = "windowable",
    verdict: str = "STEADY STATE",
    super_passes: int = 4,
    duration_s: float = 1200.0,
    state: dict[str, str] | None = None,
) -> dict:
    return {
        "status": status,
        "verdict": verdict,
        "window": {
            "super_pass_start": 1,
            "super_pass_end": super_passes,
            "super_pass_size": 1000,
            "n_samples": 1000 * super_passes,
            "duration_s": duration_s,
        },
        "state": state if state is not None else {"tpot_p50": "Plateau", "tpot_p90": "Plateau"},
    }


def _point(tmp_path: Path, steady_state: dict | None, concurrency: int = 64) -> PointConfig:
    payload = {
        "concurrency": concurrency,
        "dataset": "mlperf-perf-dataset-v1",
        "runtime_settings": {"load_pattern": "concurrency", "runtime": {}},
    }
    if steady_state is not None:
        payload["steady_state"] = steady_state
    return PointConfig.model_validate(payload, context={"yaml_path": tmp_path / "point.yaml"})


def _errors(config: PointConfig, rule: str) -> list:
    return [r for r in config._check_results if r.rule == rule and r.severity == Severity.ERROR]


@pytest.mark.unit
class TestWindowModel:
    def test_super_passes_is_inclusive(self) -> None:
        assert SteadyStateWindow(super_pass_start=1, super_pass_end=4).super_passes == 4

    def test_super_passes_unknown_without_bounds(self) -> None:
        assert SteadyStateWindow(super_pass_start=1).super_passes is None

    def test_only_windowable_is_official(self) -> None:
        """§4.4: every other status falls back to whole-run `total`."""
        assert SteadyState(status="windowable").is_official
        for status in ("insufficient_duration", "insufficient_passes", "partial_dataset"):
            assert not SteadyState(status=status).is_official

    def test_drifting_metrics_are_listed(self) -> None:
        block = SteadyState(
            state={"tpot_p50": "Plateau", "tpot_p90": "Drifting Up", "ttft_p90": "Drifting Down"}
        )
        assert block.drifting_metrics == ["tpot_p90", "ttft_p90"]


@pytest.mark.unit
class TestVocabulary:
    def test_spec_values_accepted(self, tmp_path: Path) -> None:
        assert not _errors(_point(tmp_path, _block()), "steady-state-valid")

    @pytest.mark.parametrize(
        "field, value",
        [("status", "great"), ("verdict", "looks_fine")],
    )
    def test_unknown_value_errors(self, tmp_path: Path, field: str, value: str) -> None:
        block = _block()
        block[field] = value
        assert _errors(_point(tmp_path, block), "steady-state-valid")

    def test_unknown_gating_state_errors(self, tmp_path: Path) -> None:
        block = _block(state={"tpot_p90": "wobbling"})
        assert _errors(_point(tmp_path, block), "steady-state-valid")

    @pytest.mark.parametrize(
        "verdict", ["STEADY STATE", "drifting_up", "drifting_down", "anomaly", "not found"]
    )
    def test_every_spec_verdict_is_accepted(self, tmp_path: Path, verdict: str) -> None:
        assert not _errors(_point(tmp_path, _block(verdict=verdict)), "steady-state-valid")

    def test_absent_block_is_not_a_finding(self, tmp_path: Path) -> None:
        config = _point(tmp_path, None)
        assert not [r for r in config._check_results if r.rule == "steady-state-valid"]


@pytest.mark.unit
class TestInternalConsistency:
    """The status table keys on facts the block also states outright."""

    def test_windowable_below_the_trend_floor_errors(self, tmp_path: Path) -> None:
        block = _block(super_passes=MIN_TREND_N - 1)
        hits = _errors(_point(tmp_path, block), "steady-state-consistency")
        assert hits and "MIN_TREND_N" in hits[0].message

    def test_windowable_at_the_floor_passes(self, tmp_path: Path) -> None:
        assert not _errors(
            _point(tmp_path, _block(super_passes=MIN_TREND_N)), "steady-state-consistency"
        )

    def test_windowable_with_a_drifting_metric_errors(self, tmp_path: Path) -> None:
        """§4.4 requires every gating metric to be a Plateau for the official result."""
        block = _block(state={"tpot_p50": "Plateau", "tpot_p90": "Drifting Up"})
        hits = _errors(_point(tmp_path, block), "steady-state-consistency")
        assert hits and "tpot_p90" in hits[0].message

    def test_insufficient_passes_above_the_floor_errors(self, tmp_path: Path) -> None:
        block = _block(status="insufficient_passes", super_passes=6)
        assert _errors(_point(tmp_path, block), "steady-state-consistency")

    def test_insufficient_duration_below_the_floor_is_fine(self, tmp_path: Path) -> None:
        """That status is about time, not passes, so a short window is consistent."""
        block = _block(status="insufficient_duration", super_passes=5, duration_s=10.0)
        assert not _errors(_point(tmp_path, block), "steady-state-consistency")


@pytest.mark.unit
class TestDurationBasis:
    """§6.2's minimum is measured over the window's issue-time span (§4.4)."""

    def _result(self, tmp_path: Path, steady_state: dict | None, run_duration_ns: float):
        config = _point(tmp_path, steady_state, concurrency=64)
        return PointResult.model_validate(
            {
                "config": config,
                "summary": _summary(duration_ns=run_duration_ns),
                "yaml_path": tmp_path / "point.yaml",
            },
            context={"regions": _REGIONS, "summary_path": tmp_path / "summary.json"},
        )

    def _duration_hits(self, result: PointResult) -> list:
        return [r for r in result._check_results if r.rule == "point-duration"]

    def test_window_span_is_used_when_present(self, tmp_path: Path) -> None:
        """A long wall-clock run whose window is short must still be flagged."""
        result = self._result(
            tmp_path, _block(duration_s=10.0), run_duration_ns=3_600_000_000_000.0
        )
        hits = self._duration_hits(result)
        assert any(r.severity == Severity.WARNING for r in hits)
        assert any("steady-state window" in r.message for r in hits)

    def test_long_window_passes(self, tmp_path: Path) -> None:
        result = self._result(
            tmp_path, _block(duration_s=1200.0), run_duration_ns=1_200_000_000_000.0
        )
        assert not [r for r in self._duration_hits(result) if r.severity == Severity.WARNING]

    def test_falls_back_to_whole_run_without_a_block(self, tmp_path: Path) -> None:
        result = self._result(tmp_path, None, run_duration_ns=1_200_000_000_000.0)
        hits = self._duration_hits(result)
        assert hits and any("whole-run" in r.message for r in hits)

    def test_wall_clock_no_longer_rescues_a_short_window(self, tmp_path: Path) -> None:
        """The drain is excluded by construction, so wall-clock overstates the basis."""
        short_window = self._result(
            tmp_path, _block(duration_s=60.0), run_duration_ns=9_999_000_000_000.0
        )
        assert [r for r in self._duration_hits(short_window) if r.severity == Severity.WARNING]


@pytest.mark.unit
class TestReportingBasis:
    def _result(self, tmp_path: Path, steady_state: dict | None):
        return PointResult.model_validate(
            {
                "config": _point(tmp_path, steady_state),
                "summary": _summary(),
                "yaml_path": tmp_path / "point.yaml",
            },
            context={"regions": _REGIONS, "summary_path": tmp_path / "summary.json"},
        )

    def _basis(self, result: PointResult) -> list:
        return [r for r in result._check_results if r.rule == "steady-state-basis"]

    def test_windowable_is_official(self, tmp_path: Path) -> None:
        hits = self._basis(self._result(tmp_path, _block()))
        assert hits and all(r.severity == Severity.INFO for r in hits)

    @pytest.mark.parametrize(
        "status", ["insufficient_duration", "insufficient_passes", "partial_dataset"]
    )
    def test_fallback_statuses_are_flagged(self, tmp_path: Path, status: str) -> None:
        block = _block(status=status, super_passes=2 if status == "insufficient_passes" else 4)
        hits = self._basis(self._result(tmp_path, block))
        assert any(r.severity == Severity.WARNING for r in hits)

    def test_missing_block_is_flagged(self, tmp_path: Path) -> None:
        hits = self._basis(self._result(tmp_path, None))
        assert any(r.severity == Severity.WARNING for r in hits)

    @pytest.mark.parametrize("verdict", ["drifting_up", "drifting_down"])
    def test_drift_is_flagged_separately(self, tmp_path: Path, verdict: str) -> None:
        """§4.4: a drifting metric is a range or slope, never a point estimate."""
        block = _block(status="insufficient_passes", verdict=verdict, super_passes=2)
        messages = [r.message for r in self._basis(self._result(tmp_path, block))]
        assert any("never as a point estimate" in m for m in messages)

    def test_anomaly_verdict_is_still_official(self, tmp_path: Path) -> None:
        """§4.4 accepts the first plateau of a staircase as the steady state."""
        hits = self._basis(self._result(tmp_path, _block(verdict="anomaly")))
        assert all(r.severity == Severity.INFO for r in hits)
