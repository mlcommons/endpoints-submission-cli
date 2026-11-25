# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: is the Pareto curve physically plausible?

"Throughput should generally increase with concurrency up to saturation, then plateau
or decrease." Each heuristic below looks for a curve that a serving system would not
produce under a fixed-concurrency closed loop:

- **Dip and recovery.** Throughput falls, then a later point climbs back above the
  level before the fall. A fall that never recovers is saturation, which is expected.
- **Superlinear scaling.** Throughput grows faster than concurrency between two points.
  Batching amortizes cost, but it cannot make each user faster as more users arrive.
- **Per-user rate high for the TPOT.** ``system_tps / concurrency`` is one user's mean
  output rate, which is normally below ``1000 / tpot_p50_ms``. It is not a hard bound:
  TPOT leaves out the first chunk, which can carry many tokens, so this is a hint.
- **TTFT falling with load.** More concurrent requests queue for prefill, so TTFT
  should not drop sharply as concurrency rises.
- **tps_per_user rising with load.** Per-user speed should fall as load grows.
- **Dominated points.** A point beaten on both axes by another adds nothing to the
  frontier and often means one of the two runs was not like the others.
- **Little's law.** ``latency.total / duration_ns`` is the mean number of requests in
  flight. A closed loop cannot hold more than its target, and should not hold far less.
"""

from __future__ import annotations

from itertools import pairwise

from ..context import INPUT_SUMMARIES, Curve, ReviewContext
from ..findings import Finding, Level
from ..thresholds import ParetoThresholds
from .base import Check, PointMetrics, fmt, metrics, pct

__all__ = ["CHECK"]

_NAME = "pareto_shape"
_AREA = "pareto_shape"


def _finding(
    level: Level,
    title: str,
    detail: str,
    curve: Curve,
    point: str | None,
    spec_ref: str,
    **evidence: float | int | str | None,
) -> Finding:
    return Finding(
        check=_NAME,
        focus_area=_AREA,
        level=level,
        title=title,
        detail=detail,
        curve=curve.label,
        point=point,
        spec_ref=spec_ref,
        evidence=evidence,
    )


def _dips(curve: Curve, pts: list[PointMetrics], t: ParetoThresholds) -> list[Finding]:
    out: list[Finding] = []
    for i in range(len(pts) - 1):
        before, after = pts[i], pts[i + 1]
        if before.system_tps <= 0 or after.system_tps >= before.system_tps * (1 - t.dip_tolerance):
            continue
        recovery = next((p for p in pts[i + 2 :] if p.system_tps > before.system_tps), None)
        if recovery is None:
            continue  # saturation: the curve plateaus or falls and stays there
        drop = 1 - after.system_tps / before.system_tps
        out.append(
            _finding(
                Level.ATTENTION,
                "Throughput dips, then recovers",
                f"system_tps falls {pct(drop)} from {before.name} ({fmt(before.system_tps)}) "
                f"to {after.name} ({fmt(after.system_tps)}), then rises again to "
                f"{fmt(recovery.system_tps)} at {recovery.name}. A saturated system "
                "plateaus or declines; check whether these points ran the same "
                "configuration.",
                curve,
                after.name,
                "§9.2, §5.1",
                drop=round(drop, 4),
                recovered_at=recovery.name,
            )
        )
    return out


def _superlinear(curve: Curve, pts: list[PointMetrics], t: ParetoThresholds) -> list[Finding]:
    out: list[Finding] = []
    for lo, hi in pairwise(pts):
        if lo.system_tps <= 0 or lo.concurrency <= 0 or hi.concurrency <= lo.concurrency:
            continue
        tps_ratio = hi.system_tps / lo.system_tps
        c_ratio = hi.concurrency / lo.concurrency
        if tps_ratio > c_ratio * (1 + t.superlinear_tolerance):
            out.append(
                _finding(
                    Level.ATTENTION,
                    "Throughput grows faster than concurrency",
                    f"system_tps grows {tps_ratio:.2f}× from {lo.name} to {hi.name} "
                    f"({fmt(lo.system_tps)} → {fmt(hi.system_tps)}) while concurrency "
                    f"grows only {c_ratio:.2f}×, so each user got faster under more load.",
                    curve,
                    hi.name,
                    "§9.2, §4.1",
                    tps_ratio=round(tps_ratio, 4),
                    concurrency_ratio=round(c_ratio, 4),
                )
            )
    return out


def _per_user_rate(curve: Curve, pts: list[PointMetrics], t: ParetoThresholds) -> list[Finding]:
    out: list[Finding] = []
    for p in pts:
        if p.tpot_p50_ms is None or p.concurrency <= 0:
            continue
        rate = p.system_tps / p.concurrency
        bound = 1000.0 / p.tpot_p50_ms
        if rate > bound * t.per_user_rate_factor:
            out.append(
                _finding(
                    Level.NOTE,
                    "Throughput is high for the TPOT",
                    f"system_tps / concurrency is {fmt(rate)} tokens/s per user, while the "
                    f"median TPOT of {fmt(p.tpot_p50_ms, 2)} ms suggests about {fmt(bound)}. "
                    "TPOT leaves out the first chunk, so this happens when responses arrive "
                    "in few large chunks; otherwise check that the throughput, TPOT and "
                    "concurrency come from the same run.",
                    curve,
                    p.name,
                    "§9.2, §4.1",
                    per_user_rate=round(rate, 3),
                    tpot_bound=round(bound, 3),
                )
            )
    return out


def _ttft_drops(curve: Curve, pts: list[PointMetrics], t: ParetoThresholds) -> list[Finding]:
    out: list[Finding] = []
    with_ttft = [p for p in pts if p.ttft_p50_ms is not None]
    for lo, hi in pairwise(with_ttft):
        assert lo.ttft_p50_ms is not None and hi.ttft_p50_ms is not None
        drop = 1 - hi.ttft_p50_ms / lo.ttft_p50_ms
        if drop <= t.ttft_drop_note:
            continue
        level = Level.ATTENTION if drop > t.ttft_drop_attention else Level.NOTE
        out.append(
            _finding(
                level,
                "TTFT falls as concurrency rises",
                f"TTFT p50 drops {pct(drop)} from {fmt(lo.ttft_p50_ms)} ms at {lo.name} to "
                f"{fmt(hi.ttft_p50_ms)} ms at {hi.name}. More concurrent requests should "
                "not reach their first token sooner; check whether both points ran the "
                "same configuration and dataset.",
                curve,
                hi.name,
                "§9.2, §4.1",
                drop=round(drop, 4),
            )
        )
    return out


def _tps_per_user_rises(
    curve: Curve, pts: list[PointMetrics], t: ParetoThresholds
) -> list[Finding]:
    out: list[Finding] = []
    with_tpu = [p for p in pts if p.tps_per_user is not None]
    for lo, hi in pairwise(with_tpu):
        assert lo.tps_per_user is not None and hi.tps_per_user is not None
        rise = hi.tps_per_user / lo.tps_per_user - 1
        if rise > t.tps_per_user_rise_note:
            out.append(
                _finding(
                    Level.NOTE,
                    "Per-user speed rises with load",
                    f"tps_per_user rises {pct(rise)} from {fmt(lo.tps_per_user)} at "
                    f"{lo.name} to {fmt(hi.tps_per_user)} at {hi.name}.",
                    curve,
                    hi.name,
                    "§9.2, §4.1",
                    rise=round(rise, 4),
                )
            )
    return out


def _dominated(curve: Curve, pts: list[PointMetrics], t: ParetoThresholds) -> list[Finding]:
    out: list[Finding] = []
    margin = 1 + t.dominance_margin
    with_tpu = [p for p in pts if p.tps_per_user is not None]
    for p in with_tpu:
        assert p.tps_per_user is not None
        p_tpu = p.tps_per_user
        better = next(
            (
                q
                for q in with_tpu
                if q.system_tps > p.system_tps * margin
                and q.tps_per_user is not None
                and q.tps_per_user > p_tpu * margin
            ),
            None,
        )
        if better is None:
            continue
        assert better.tps_per_user is not None
        out.append(
            _finding(
                Level.NOTE,
                "Point is dominated",
                f"{better.name} beats {p.name} on both system_tps "
                f"({fmt(better.system_tps)} vs {fmt(p.system_tps)}) and tps_per_user "
                f"({fmt(better.tps_per_user)} vs {fmt(p_tpu)}), so {p.name} is off the "
                "frontier.",
                curve,
                p.name,
                "§9.2, §5.2",
                dominated_by=better.name,
            )
        )
    return out


def _littles_law(curve: Curve, pts: list[PointMetrics], t: ParetoThresholds) -> list[Finding]:
    out: list[Finding] = []
    for p in pts:
        if p.implied_concurrency is None or p.concurrency <= 0:
            continue
        ratio = p.implied_concurrency / p.concurrency
        if t.littles_law_low <= ratio <= t.littles_law_high:
            continue
        why = (
            "more requests in flight than the target allows"
            if ratio > t.littles_law_high
            else "the client did not keep the target concurrency in flight"
        )
        out.append(
            _finding(
                Level.NOTE,
                "In-flight requests don't match the concurrency",
                f"latency.total / duration implies {fmt(p.implied_concurrency)} requests in "
                f"flight on average, against a target of {p.concurrency} ({ratio:.2f}×): "
                f"{why}.",
                curve,
                p.name,
                "§9.2, §6.1",
                implied_concurrency=round(p.implied_concurrency, 3),
                ratio=round(ratio, 4),
            )
        )
    return out


def run(ctx: ReviewContext) -> list[Finding]:
    """Return the curve-shape findings for every frontier in *ctx*."""
    t = ctx.thresholds.pareto
    out: list[Finding] = []
    for curve in ctx.curves:
        pts = [metrics(p) for p in curve.measured()]
        if len(pts) < 2:
            continue
        for heuristic in (
            _dips,
            _superlinear,
            _per_user_rate,
            _ttft_drops,
            _tps_per_user_rises,
            _dominated,
            _littles_law,
        ):
            out.extend(heuristic(curve, pts, t))
    return out


CHECK = Check(name=_NAME, focus_area=_AREA, requires=frozenset({INPUT_SUMMARIES}), run=run)
