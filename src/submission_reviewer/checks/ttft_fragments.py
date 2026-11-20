# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: are TTFT-triggering fragments genuine parts of the response?

TTFT stops at the first non-empty fragment (§4.1). Emitting "whitespace, control
characters, punctuation, or other meaningless leading content solely to stop the TTFT
clock" is not allowed. From each point's ``events.jsonl`` this check reads the first
streamed fragment of every performance response and reports:

- **Trivial first fragments**: the share of responses whose first fragment holds
  nothing but whitespace, punctuation or control characters, with the most common ones
  shown. Some models genuinely open with a newline, so the reviewer judges the share.
- **A pause after the first fragment**: when the log has per-chunk timestamps
  (``stream_all_chunks``), the gap between the first and second chunk compared with the
  later gaps. A token sent early to stop the clock leaves a long pause behind it;
  prefill/decode disaggregation can cause one too.

A dedicated Offline point is skipped: TTFT does not apply to it (§5.7.1).
"""

from __future__ import annotations

import statistics
import unicodedata
from collections import Counter

from ..context import INPUT_LOGS, Curve, Point, ReviewContext
from ..findings import Finding, Level
from ..logs import PointLog
from ..thresholds import LogThresholds
from .base import Check, finding_factory, pct, points_label

__all__ = ["CHECK", "is_trivial"]

_finding = finding_factory("ttft_fragments")
_SPEC = "§9.2, §4.1"


def is_trivial(fragment: str) -> bool:
    """Whether *fragment* holds only whitespace, punctuation or control characters."""
    return all(unicodedata.category(ch)[0] in "ZPC" for ch in fragment)


def _trivial_share(log: PointLog) -> tuple[int, int, Counter[str]]:
    fragments = [s.first_fragment for s in log.completions("performance") if s.first_fragment]
    trivial = Counter(f for f in fragments if is_trivial(f))
    return sum(trivial.values()), len(fragments), trivial


def _gap_ratio(log: PointLog) -> tuple[float, int] | None:
    """Median first-gap ratio over responses with at least three chunks."""
    ratios = []
    for s in log.completions("performance"):
        if s.t_first is None or s.t_second is None or s.t_last_chunk is None:
            continue
        if s.later_chunks < 2:
            continue
        later = (s.t_last_chunk - s.t_second) / (s.later_chunks - 1)
        if later > 0:
            ratios.append((s.t_second - s.t_first) / later)
    if len(ratios) < 10:
        return None
    return statistics.median(ratios), len(ratios)


def _curve(curve: Curve, ctx: ReviewContext, t: LogThresholds) -> list[Finding]:
    trivial_rows: list[tuple[Point, float, int, Counter[str]]] = []
    gap_rows: list[tuple[Point, float, int]] = []
    for point in curve.points:
        if point.is_dedicated_offline:
            continue
        log = ctx.logs.get(point.dir)
        if log is None:
            continue
        count, total, fragments = _trivial_share(log)
        if total and count / total > t.ttft_trivial_note:
            trivial_rows.append((point, count / total, total, fragments))
        gap = _gap_ratio(log)
        if gap is not None and gap[0] > t.first_gap_ratio_note:
            gap_rows.append((point, gap[0], gap[1]))

    out: list[Finding] = []
    if trivial_rows:
        worst = max(share for _, share, _, _ in trivial_rows)
        common: Counter[str] = Counter()
        for _, _, _, fragments in trivial_rows:
            common.update(fragments)
        shown = ", ".join(f"{frag!r} ×{n}" for frag, n in common.most_common(5))
        out.append(
            _finding(
                Level.ATTENTION if worst > t.ttft_trivial_attention else Level.NOTE,
                "Responses start with meaningless fragments",
                points_label([f"{p.name} {pct(share)} of {n}" for p, share, n, _ in trivial_rows])
                + f" of streamed responses begin with only whitespace, punctuation or control "
                f"characters (most common: {shown}). §4.1 bars leading content emitted only "
                "to stop the TTFT clock; check whether the model genuinely starts this way.",
                curve=curve.label,
                spec_ref=_SPEC,
                worst_share=round(worst, 4),
            )
        )
    if gap_rows:
        out.append(
            _finding(
                Level.NOTE,
                "Long pause after the first fragment",
                points_label([f"{p.name} {ratio:.1f}× ({n} responses)" for p, ratio, n in gap_rows])
                + ": the median gap between the first and second chunk, relative to the gaps "
                "after it. An early token sent to stop the clock leaves this pause; so can "
                "prefill/decode disaggregation.",
                curve=curve.label,
                spec_ref=_SPEC,
            )
        )
    return out


def run(ctx: ReviewContext) -> list[Finding]:
    """Return the first-fragment findings for every point with a log."""
    out: list[Finding] = []
    for curve in ctx.curves:
        out.extend(_curve(curve, ctx, ctx.thresholds.logs))
    return out


CHECK = Check(
    name="ttft_fragments",
    focus_area="ttft_fragments",
    requires=frozenset({INPUT_LOGS}),
    run=run,
)
