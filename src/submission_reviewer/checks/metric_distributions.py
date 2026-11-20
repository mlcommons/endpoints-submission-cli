# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: do the metric distributions suggest artificial manipulation?

The rules' example is "suspiciously uniform TTFT values across very different
concurrency levels". Measured distributions come from HDR histograms of thousands of
requests, so they carry noise; the heuristics below look for values that are too
regular, too identical, or impossible:

- **Uniform TTFT.** Points spanning a wide concurrency range whose TTFT barely moves.
- **Identical percentile blocks.** Two points reporting exactly the same percentiles
  for a metric, which separate runs essentially never do.
- **Percentile order.** A higher percentile below a lower one, or outside min/max.
- **Narrow distributions.** A p90 hardly above the p50.
- **Round throughput.** Several points whose system_tps is a round number.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from itertools import pairwise

from submission_checker.models.file import PercentileStats

from ..context import INPUT_SUMMARIES, Curve, Point, ReviewContext
from ..findings import Finding, Level
from ..thresholds import DistributionThresholds
from .base import Check, fmt, metrics, pct, points_label

__all__ = ["CHECK"]

_NAME = "metric_distributions"
_AREA = "metric_distributions"

#: ``result_summary.json`` distribution blocks, with the unit their percentiles use
#: and the divisor that turns them into the unit shown.
_BLOCKS: dict[str, tuple[str, float]] = {
    "ttft": ("ms", 1e6),
    "tpot": ("ms", 1e6),
    "latency": ("ms", 1e6),
    "output_sequence_lengths": ("tokens", 1.0),
}


def _finding(
    level: Level,
    title: str,
    detail: str,
    curve: Curve,
    point: str | None,
    spec_ref: str = "§9.2, §4.1",
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


def _block(point: Point, name: str) -> PercentileStats | None:
    if point.summary is None:
        return None
    value = getattr(point.summary, name, None)
    if value is None:
        value = (point.summary.model_extra or {}).get(name)
        if isinstance(value, dict):
            try:
                value = PercentileStats.model_validate(value)
            except ValueError:
                return None
    return value if isinstance(value, PercentileStats) else None


def _ordered(block: PercentileStats) -> list[tuple[float, float]]:
    return sorted((float(k), v) for k, v in block.percentiles.items())


def _show(block: PercentileStats, unit: str, divisor: float, limit: int = 4) -> str:
    shown = [f"p{k:g} {fmt(v / divisor, 2)}" for k, v in _ordered(block)[:limit]]
    return ", ".join(shown) + f" {unit}"


def _uniform_ttft(curve: Curve, t: DistributionThresholds) -> list[Finding]:
    pts = [metrics(p) for p in curve.measured()]
    out: list[Finding] = []
    for label, values in (
        ("p50", [(p.name, p.concurrency, p.ttft_p50_ms) for p in pts]),
        ("p90", [(p.name, p.concurrency, p.ttft_p90_ms) for p in pts]),
    ):
        known = [(n, c, v) for n, c, v in values if v is not None and c > 0]
        if len(known) < 3:
            continue
        lo_c, hi_c = min(c for _, c, _ in known), max(c for _, c, _ in known)
        if hi_c / lo_c < t.uniform_ttft_min_span:
            continue
        ttfts = [v for _, _, v in known]
        mean = statistics.fmean(ttfts)
        cv = statistics.pstdev(ttfts) / mean if mean > 0 else 0.0
        if cv >= t.uniform_ttft_max_cv:
            continue
        out.append(
            _finding(
                Level.ATTENTION,
                "TTFT is flat across concurrency",
                f"TTFT {label} stays within {pct(cv)} (coefficient of variation) across "
                f"{len(known)} points from concurrency {lo_c} to {hi_c}: "
                f"{fmt(min(ttfts))}–{fmt(max(ttfts))} ms. Queueing for prefill normally "
                "raises TTFT as load grows.",
                curve,
                None,
                percentile=label,
                cv=round(cv, 4),
            )
        )
        break  # one finding per curve is enough; p90 adds nothing if p50 already fired
    return out


def _identical_blocks(curve: Curve, t: DistributionThresholds) -> list[Finding]:
    out: list[Finding] = []
    for name, (unit, divisor) in _BLOCKS.items():
        groups: dict[tuple[tuple[float, float], ...], list[Point]] = defaultdict(list)
        for p in curve.points:
            block = _block(p, name)
            if block is not None and len(block.percentiles) >= 2:
                groups[tuple(_ordered(block))].append(p)
        for same in groups.values():
            if len(same) < t.identical_block_min_points:
                continue
            block = _block(same[0], name)
            assert block is not None
            names = [p.name for p in same]
            out.append(
                _finding(
                    Level.ATTENTION,
                    f"Identical {name} percentiles on {len(same)} points",
                    f"{points_label(names)} report exactly the same {name} percentiles "
                    f"({_show(block, unit, divisor)}). Separate runs at different loads "
                    "do not produce identical distributions; check whether values were "
                    "copied between points.",
                    curve,
                    None,
                    metric=name,
                    points=",".join(names),
                )
            )
    return out


def _percentile_order(curve: Curve) -> list[Finding]:
    out: list[Finding] = []
    for p in curve.points:
        for name, (unit, divisor) in _BLOCKS.items():
            block = _block(p, name)
            if block is None or not block.percentiles:
                continue
            ordered = _ordered(block)
            problems = [
                f"p{k1:g} ({fmt(v1 / divisor, 2)}) > p{k2:g} ({fmt(v2 / divisor, 2)})"
                for (k1, v1), (k2, v2) in pairwise(ordered)
                if v1 > v2
            ]
            extra = block.model_extra or {}
            lo, hi = extra.get("min"), extra.get("max")
            if isinstance(lo, (int, float)) and ordered[0][1] < lo:
                problems.append(f"p{ordered[0][0]:g} is below min ({fmt(lo / divisor, 2)})")
            if isinstance(hi, (int, float)) and ordered[-1][1] > hi:
                problems.append(f"p{ordered[-1][0]:g} is above max ({fmt(hi / divisor, 2)})")
            if problems:
                out.append(
                    _finding(
                        Level.ATTENTION,
                        f"Impossible {name} percentiles",
                        f"{name} percentiles are out of order ({unit}): "
                        + "; ".join(problems)
                        + ". A measured distribution cannot produce this.",
                        curve,
                        p.name,
                        metric=name,
                    )
                )
    return out


def _narrow(curve: Curve, t: DistributionThresholds) -> list[Finding]:
    out: list[Finding] = []
    for p in curve.measured():
        for name in ("ttft", "tpot"):
            block = _block(p, name)
            if block is None:
                continue
            p50, p90 = block.percentiles.get("50"), block.percentiles.get("90")
            if not p50 or not p90 or p50 <= 0:
                continue
            ratio = p90 / p50
            if 1.0 <= ratio < t.narrow_ratio:
                out.append(
                    _finding(
                        Level.NOTE,
                        f"Very narrow {name} distribution",
                        f"{name} p90 is only {ratio:.3f}× its p50 "
                        f"({fmt(p50 / 1e6, 2)} → {fmt(p90 / 1e6, 2)} ms) across "
                        f"{p.summary.n_samples_completed if p.summary else '?'} requests.",
                        curve,
                        p.name,
                        metric=name,
                        ratio=round(ratio, 5),
                    )
                )
    return out


def _round_throughput(curve: Curve, t: DistributionThresholds) -> list[Finding]:
    step = t.round_step
    if step <= 0:
        return []
    round_points: list[tuple[str, float]] = []
    for p in curve.points:
        if p.summary is None:
            continue
        stored = (p.summary.model_extra or {}).get("system_tps")
        value = float(stored) if isinstance(stored, (int, float)) else p.summary.system_tps
        if value > 0 and math.isclose(value, round(value / step) * step, abs_tol=1e-6):
            round_points.append((p.name, value))
    if len(round_points) < t.round_min_points:
        return []
    shown = ", ".join(f"{n} {fmt(v, 0)}" for n, v in round_points[:6])
    return [
        _finding(
            Level.NOTE,
            "Round throughput values",
            f"{len(round_points)} points report a system_tps that is a whole multiple of "
            f"{fmt(step, 0)} ({shown}). Measured throughput is rarely round; check that "
            "the values were not entered by hand.",
            curve,
            None,
            count=len(round_points),
        )
    ]


def run(ctx: ReviewContext) -> list[Finding]:
    """Return the distribution findings for every frontier in *ctx*."""
    t = ctx.thresholds.distributions
    out: list[Finding] = []
    for curve in ctx.curves:
        out.extend(_uniform_ttft(curve, t))
        out.extend(_identical_blocks(curve, t))
        out.extend(_percentile_order(curve))
        out.extend(_narrow(curve, t))
        out.extend(_round_throughput(curve, t))
    return out


CHECK = Check(name=_NAME, focus_area=_AREA, requires=frozenset({INPUT_SUMMARIES}), run=run)
