# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for the §4.4 steady-state reporting basis.

The checker cannot run the detector — that is post-processing over `events.jsonl`,
which is not in the bundle. The reference client runs it and writes its verdict to
`result_summary.json`; what is asserted here is how that verdict is read into §4.4's
terms, and that §6.2 is measured against the window rather than the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from submission_checker.models import PointConfig, PointResult, Severity
from submission_checker.models.file.steady_state import (
    MIN_TREND_N,
    SteadyState,
    SteadyStateWindow,
    from_client_verdict,
)

from .conftest import _REGIONS, _summary

_ABSENT = object()


def _verdict(
    *,
    found: bool = True,
    passes: int | None = 4,
    duration_s: float = 1200.0,
    n_super_passes: int = 5,
    warmup: int = 1,
    trend: dict[str, str] | None = None,
    anomaly: bool = False,
) -> dict[str, Any]:
    """A `steady_state` block in the shape the client writes (`sp_hi` is exclusive)."""
    start_ns = 1_000_000_000_000
    window = (
        None
        if passes is None
        else {
            "sp_lo": 0,
            "sp_hi": passes,
            "n_super_passes": passes,
            "n_samples": 1000 * passes,
            "start_ns": start_ns,
            "end_ns": start_ns + int(duration_s * 1e9),
        }
    )
    return {
        "found": found,
        "reason": None if found else "no admissible steady plateau",
        "superpass_size": 1000,
        "n_super_passes": n_super_passes,
        "warmup": warmup,
        "window": window,
        "anomaly": {"detected": anomaly, "change_point_sp": 3 if anomaly else None},
        "global_trend": trend
        if trend is not None
        else {
            "ttft_p50": "steady",
            "ttft_p90": "steady",
            "tpot_p50": "steady",
            "tpot_p90": "steady",
        },
        "drifting_up": [k for k, v in (trend or {}).items() if v == "up"],
    }


def _point(
    tmp_path: Path,
    concurrency: int = 64,
    load_pattern: str = "concurrency",
    **extra: Any,
) -> PointConfig:
    payload = {
        "concurrency": concurrency,
        "dataset": "mlperf-perf-dataset-v1",
        "runtime_settings": {"load_pattern": load_pattern, "runtime": {}},
        **extra,
    }
    return PointConfig.model_validate(payload, context={"yaml_path": tmp_path / "point.yaml"})


def _result(
    tmp_path: Path,
    steady_state: Any = _ABSENT,
    *,
    run_duration_ns: float = 1_200_000_000_000.0,
    config: PointConfig | None = None,
) -> PointResult:
    summary = _summary(duration_ns=run_duration_ns)
    if steady_state is not _ABSENT:
        summary = summary.model_copy(update={"steady_state": steady_state})
    return PointResult.model_validate(
        {
            "config": config if config is not None else _point(tmp_path),
            "summary": summary,
            "yaml_path": tmp_path / "point.yaml",
        },
        context={"regions": _REGIONS, "summary_path": tmp_path / "summary.json"},
    )


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
class TestFromClientVerdict:
    """The client's verdict, read into §4.4's status and shape tables."""

    def test_window_end_is_converted_to_inclusive(self) -> None:
        """The client's `sp_hi` is exclusive: [0, 4) is four super-passes, 0..3."""
        block = from_client_verdict(_verdict(passes=4), 1200.0)
        assert block.window.super_pass_start == 0
        assert block.window.super_pass_end == 3
        assert block.window.super_passes == 4

    def test_window_duration_is_the_issue_span(self) -> None:
        block = from_client_verdict(_verdict(duration_s=1500.0), 1200.0)
        assert block.window.duration_s == pytest.approx(1500.0)

    def test_clean_window_is_windowable(self) -> None:
        block = from_client_verdict(_verdict(), 1200.0)
        assert (block.status, block.verdict) == ("windowable", "STEADY STATE")
        assert set(block.state.values()) == {"Plateau"}

    def test_window_below_the_trend_floor_is_insufficient_passes(self) -> None:
        block = from_client_verdict(_verdict(passes=MIN_TREND_N - 1), 1200.0)
        assert block.status == "insufficient_passes"

    def test_no_plateau_is_insufficient_passes_and_not_found(self) -> None:
        block = from_client_verdict(_verdict(found=False, passes=None), 1200.0)
        assert (block.status, block.verdict) == ("insufficient_passes", "not found")

    def test_under_one_super_pass_is_partial_dataset(self) -> None:
        raw = _verdict(found=False, passes=None, n_super_passes=1, warmup=1)
        assert from_client_verdict(raw, 1200.0).status == "partial_dataset"

    def test_window_shorter_than_section_6_2_is_insufficient_duration(self) -> None:
        block = from_client_verdict(_verdict(duration_s=900.0), 1200.0)
        assert (block.status, block.verdict) == ("insufficient_duration", "STEADY STATE")

    def test_window_the_client_rejected_as_short_is_insufficient_duration(self) -> None:
        """`found: false` with a window is the longest plateau the client found too short."""
        block = from_client_verdict(_verdict(found=False, duration_s=1500.0), 1200.0)
        assert (block.status, block.verdict) == ("insufficient_duration", "not found")

    def test_unknown_minimum_skips_the_duration_test(self) -> None:
        assert from_client_verdict(_verdict(duration_s=10.0), None).status == "windowable"

    def test_drift_after_the_window_is_drifting_up(self) -> None:
        """A local plateau with a metric climbing later (§4.4 shape table)."""
        trend = {"ttft_p50": "up", "ttft_p90": "up", "tpot_p50": "steady", "tpot_p90": "steady"}
        block = from_client_verdict(_verdict(trend=trend), 1200.0)
        assert block.verdict == "drifting_up"
        assert block.drifting_metrics == ["ttft_p50", "ttft_p90"]

    def test_downward_trend_is_drifting_down(self) -> None:
        trend = {"tpot_p50": "down", "tpot_p90": "steady"}
        assert from_client_verdict(_verdict(trend=trend), 1200.0).verdict == "drifting_down"

    def test_anomaly_takes_precedence_over_drift(self) -> None:
        """§4.4 keeps the first plateau of a staircase as the result."""
        trend = {"tpot_p50": "up", "tpot_p90": "up"}
        block = from_client_verdict(_verdict(trend=trend, anomaly=True), 1200.0)
        assert block.verdict == "anomaly"
        assert block.anomaly is not None

    def test_untestable_trend_has_no_state(self) -> None:
        trend = {"tpot_p50": "steady", "tpot_p90": "insufficient"}
        assert from_client_verdict(_verdict(trend=trend), 1200.0).state == {"tpot_p50": "Plateau"}

    def test_a_block_not_in_the_client_shape_raises(self) -> None:
        with pytest.raises(ValidationError):
            from_client_verdict({"status": "windowable", "verdict": "STEADY STATE"}, 1200.0)


@pytest.mark.unit
class TestDurationBasis:
    """§6.2's minimum is measured over the window's issue-time span (§4.4)."""

    def _duration_hits(self, result: PointResult) -> list:
        return [r for r in result._check_results if r.rule == "point-duration"]

    def test_window_span_is_used_when_present(self, tmp_path: Path) -> None:
        """A long wall-clock run whose window is short must still be flagged."""
        result = _result(tmp_path, _verdict(duration_s=10.0), run_duration_ns=3_600_000_000_000.0)
        hits = self._duration_hits(result)
        assert any(r.severity == Severity.WARNING for r in hits)
        assert any("steady-state window" in r.message for r in hits)

    def test_long_window_passes(self, tmp_path: Path) -> None:
        result = _result(tmp_path, _verdict(duration_s=1200.0))
        assert not [r for r in self._duration_hits(result) if r.severity == Severity.WARNING]

    def test_falls_back_to_whole_run_without_a_verdict(self, tmp_path: Path) -> None:
        hits = self._duration_hits(_result(tmp_path, None))
        assert hits and any("whole-run" in r.message for r in hits)

    def test_a_rejected_window_is_not_the_basis(self, tmp_path: Path) -> None:
        """A window the client found too short is not a steady window."""
        result = _result(tmp_path, _verdict(found=False, duration_s=10.0))
        assert any("whole-run" in r.message for r in self._duration_hits(result))

    def test_wall_clock_no_longer_rescues_a_short_window(self, tmp_path: Path) -> None:
        """The drain is excluded by construction, so wall-clock overstates the basis."""
        short_window = _result(
            tmp_path, _verdict(duration_s=60.0), run_duration_ns=9_999_000_000_000.0
        )
        assert [r for r in self._duration_hits(short_window) if r.severity == Severity.WARNING]


@pytest.mark.unit
class TestReportingBasis:
    def _basis(self, result: PointResult) -> list:
        return [r for r in result._check_results if r.rule == "steady-state-basis"]

    def test_windowable_is_official(self, tmp_path: Path) -> None:
        hits = self._basis(_result(tmp_path, _verdict()))
        assert hits and all(r.severity == Severity.INFO for r in hits)

    @pytest.mark.parametrize(
        "verdict",
        [
            _verdict(duration_s=900.0),
            _verdict(passes=2),
            _verdict(found=False, passes=None, n_super_passes=1, warmup=1),
        ],
        ids=["insufficient_duration", "insufficient_passes", "partial_dataset"],
    )
    def test_fallback_statuses_are_flagged(self, tmp_path: Path, verdict: dict) -> None:
        hits = self._basis(_result(tmp_path, verdict))
        assert any(r.severity == Severity.WARNING for r in hits)

    def test_detection_off_is_flagged(self, tmp_path: Path) -> None:
        hits = self._basis(_result(tmp_path, None))
        assert [r.message for r in hits if r.severity == Severity.WARNING]
        assert "detection was off" in hits[0].message

    def test_client_without_the_field_is_flagged(self, tmp_path: Path) -> None:
        hits = self._basis(_result(tmp_path))
        assert hits and "predates steady-state reporting" in hits[0].message

    @pytest.mark.parametrize("raw", [{"status": "windowable"}, "windowable", 4])
    def test_unreadable_verdict_is_flagged(self, tmp_path: Path, raw: Any) -> None:
        hits = self._basis(_result(tmp_path, raw))
        assert hits and "not the block the client writes" in hits[0].message

    @pytest.mark.parametrize(
        "trend",
        [{"tpot_p50": "up", "tpot_p90": "steady"}, {"tpot_p50": "down", "tpot_p90": "steady"}],
        ids=["drifting_up", "drifting_down"],
    )
    def test_drift_is_flagged_separately(self, tmp_path: Path, trend: dict) -> None:
        """§4.4: a drifting metric is a range or slope, never a point estimate."""
        messages = [r.message for r in self._basis(_result(tmp_path, _verdict(trend=trend)))]
        assert any("never as a point estimate" in m for m in messages)

    def test_anomaly_verdict_is_still_official(self, tmp_path: Path) -> None:
        """§4.4 accepts the first plateau of a staircase as the steady state."""
        hits = self._basis(_result(tmp_path, _verdict(anomaly=True)))
        assert all(r.severity == Severity.INFO for r in hits)

    def test_a_point_yaml_block_is_not_read(self, tmp_path: Path) -> None:
        """The verdict comes from result_summary.json, not a copy in point.yaml."""
        copied = {"status": "windowable", "verdict": "STEADY STATE"}
        config = _point(tmp_path, steady_state=copied)
        hits = self._basis(_result(tmp_path, None, config=config))
        assert any(r.severity == Severity.WARNING for r in hits)

    def test_agentic_points_are_out_of_scope(self, tmp_path: Path) -> None:
        """§4.4 covers fixed-concurrency points; the client does not run it for agentic."""
        config = _point(tmp_path, load_pattern="agentic_inference")
        assert not self._basis(_result(tmp_path, None, config=config))

    def test_dedicated_offline_point_is_out_of_scope(self, tmp_path: Path) -> None:
        config = _point(tmp_path, offline="dedicated")
        assert not self._basis(_result(tmp_path, None, config=config))

    def test_elected_offline_point_is_still_checked(self, tmp_path: Path) -> None:
        """An elected C_max point remains a fixed-concurrency point (§5.7.2)."""
        config = _point(tmp_path, offline="elected")
        assert self._basis(_result(tmp_path, None, config=config))
