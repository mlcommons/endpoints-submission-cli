# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2 metric-distribution heuristics."""

from __future__ import annotations

from dataclasses import replace

from submission_reviewer import Level

from .conftest import PointSpec, clean_curve, summary_for, titled, with_point


def _distributions(report):
    return [f for f in report.findings if f.check == "metric_distributions"]


def test_clean_curve_has_no_findings(run_review) -> None:
    assert _distributions(run_review(clean_curve())) == []


def test_flat_ttft_across_concurrency(run_review) -> None:
    specs = [replace(s, ttft_p50_ms=100.0 + i * 0.5) for i, s in enumerate(clean_curve())]
    (found,) = titled(run_review(specs), "TTFT is flat")
    assert found.level is Level.ATTENTION
    assert found.point is None
    assert found.evidence["percentile"] == "p50"


def test_flat_ttft_over_a_narrow_range_is_not_flagged(run_review) -> None:
    # 64 → 256 is a 4x span, below the 8x the heuristic needs.
    specs = [
        PointSpec(concurrency=c, system_tps=40.0 * c, ttft_p50_ms=100.0, tpot_p50_ms=5.0 + c / 64)
        for c in (64, 128, 256)
    ]
    assert titled(run_review(specs), "TTFT is flat") == []


def test_identical_percentile_blocks(run_review) -> None:
    copied = summary_for(clean_curve()[2])["ttft"]
    specs = with_point(clean_curve(), 64, overrides={"ttft": copied})
    (found,) = titled(run_review(specs), "Identical ttft percentiles")
    assert found.level is Level.ATTENTION
    assert found.evidence["points"] == "r16,r64"


def test_out_of_order_percentiles(run_review) -> None:
    specs = with_point(
        clean_curve(),
        16,
        overrides={"ttft": {"total": 1.0, "percentiles": {"50": 9e7, "90": 8e7}}},
    )
    (found,) = titled(run_review(specs), "Impossible ttft")
    assert (found.level, found.point) == (Level.ATTENTION, "r16")
    assert "p50 (90.00) > p90 (80.00)" in found.detail


def test_percentile_outside_min_max(run_review) -> None:
    block = {"total": 1.0, "min": 5e7, "max": 1e8, "percentiles": {"50": 6e7, "99": 2e8}}
    specs = with_point(clean_curve(), 16, overrides={"ttft": block})
    (found,) = titled(run_review(specs), "Impossible ttft")
    assert "above max" in found.detail


def test_narrow_distribution(run_review) -> None:
    specs = with_point(clean_curve(), 64, tpot_spread=1.004)
    found = titled(run_review(specs), "narrow tpot")
    assert [(f.point, f.level) for f in found] == [("r64", Level.NOTE)]


def test_round_throughput(run_review) -> None:
    specs = clean_curve()
    for c, tps in ((16, 2000.0), (64, 6000.0), (256, 11500.0)):
        specs = with_point(specs, c, system_tps=tps)
    (found,) = titled(run_review(specs), "Round throughput")
    assert found.level is Level.NOTE
    assert found.evidence["count"] == 3


def test_two_round_values_are_not_enough(run_review) -> None:
    specs = with_point(with_point(clean_curve(), 16, system_tps=2000.0), 64, system_tps=6000.0)
    assert titled(run_review(specs), "Round throughput") == []
