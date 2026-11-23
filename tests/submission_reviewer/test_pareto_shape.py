# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2 Pareto-curve plausibility heuristics."""

from __future__ import annotations

from submission_reviewer import Level

from .conftest import CURVE, NEEDS_EXTRA_INPUT, PointSpec, clean_curve, titled, with_point


def _pareto(report):
    return [f for f in report.findings if f.check == "pareto_shape"]


def test_clean_curve_has_no_findings(run_review) -> None:
    report = run_review(clean_curve())
    assert report.findings == []
    assert {o.status for o in report.outcomes if o.check not in NEEDS_EXTRA_INPUT} == {"ran"}


def test_dip_that_recovers_asks_for_attention(run_review) -> None:
    specs = with_point(clean_curve(), 256, system_tps=5000.0)  # r64 is 6,244; r1024 14,577
    (dip,) = titled(run_review(specs), "dips")
    assert dip.level is Level.ATTENTION
    assert (dip.curve, dip.point) == (CURVE, "r256")
    assert dip.evidence["recovered_at"] == "r1024"


def test_saturation_without_recovery_is_not_a_dip(run_review) -> None:
    specs = with_point(clean_curve(), 1024, system_tps=0.85 * 11505.6)
    assert titled(run_review(specs), "dips") == []


def test_small_wobble_is_tolerated(run_review) -> None:
    # 3 % below r64, then recovering: inside the 5 % dip tolerance.
    specs = with_point(clean_curve(), 256, system_tps=0.97 * 6243.9)
    assert titled(run_review(specs), "dips") == []


def test_superlinear_scaling(run_review) -> None:
    specs = with_point(clean_curve(), 16, system_tps=615.4 * 6)  # 6x tps for 4x concurrency
    (found,) = titled(run_review(specs), "faster than concurrency")
    assert found.level is Level.ATTENTION
    assert found.point == "r16"
    assert found.evidence["concurrency_ratio"] == 4.0


def test_per_user_rate_above_tpot_bound(run_review) -> None:
    # 400 tokens/s per user, but a 5.2 ms median TPOT suggests about 192. Not a hard
    # bound (TPOT leaves out the first chunk), so it is a note.
    specs = with_point(clean_curve(), 4, system_tps=1600.0)
    (found,) = titled(run_review(specs), "high for the TPOT")
    assert found.level is Level.NOTE
    assert found.point == "r4"


def test_sharp_ttft_drop_asks_for_attention(run_review) -> None:
    # r256 has TTFT p50 250 ms; r1024 at 75 ms is a 70 % drop.
    specs = with_point(clean_curve(), 1024, ttft_p50_ms=75.0)
    (found,) = titled(run_review(specs), "TTFT falls")
    assert (found.level, found.point) == (Level.ATTENTION, "r1024")


def test_moderate_ttft_drop_is_a_note(run_review) -> None:
    specs = with_point(clean_curve(), 1024, ttft_p50_ms=175.0)  # 30 % below r256
    (found,) = titled(run_review(specs), "TTFT falls")
    assert found.level is Level.NOTE


def test_tps_per_user_rising_with_load(run_review) -> None:
    # r256's TPOT below r64's: per-user speed goes up as load grows.
    specs = with_point(clean_curve(), 256, tpot_p50_ms=6.0, system_tps=8000.0)
    found = titled(run_review(specs), "Per-user speed rises")
    assert [f.point for f in found] == ["r256"]
    assert found[0].level is Level.NOTE


def test_dominated_point(run_review) -> None:
    # r64 drops below r16 on both axes.
    specs = with_point(clean_curve(), 64, system_tps=2000.0, tpot_p50_ms=6.5)
    found = titled(run_review(specs), "dominated")
    assert [f.point for f in found] == ["r64"]
    assert found[0].evidence["dominated_by"] == "r16"


def test_littles_law(run_review) -> None:
    specs = clean_curve()
    specs = with_point(specs, 16, latency_ratio=1.0)
    specs = with_point(specs, 64, latency_ratio=1.5)
    specs = with_point(specs, 256, latency_ratio=0.4)
    found = {f.point: f for f in titled(run_review(specs), "In-flight")}
    assert set(found) == {"r64", "r256"}
    assert "more requests in flight" in found["r64"].detail
    assert "did not keep" in found["r256"].detail
    assert all(f.level is Level.NOTE for f in found.values())


def test_dedicated_offline_point_is_not_on_the_curve(run_review) -> None:
    offline = PointSpec(
        concurrency=13368, system_tps=9000.0, ttft_p50_ms=5.0, tpot_p50_ms=60.0, offline="dedicated"
    )
    report = run_review([*clean_curve(), offline])
    assert _pareto(report) == []


def test_single_point_curve_is_skipped(run_review) -> None:
    assert _pareto(run_review(clean_curve()[:1])) == []
