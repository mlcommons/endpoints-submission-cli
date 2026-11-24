# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: is content duplicated across response fields or padded to inflate output tokens?

§2.8 counts output tokens from the reconstructed assistant message, and prohibits
"deliberately duplicating substantially identical content across fields, or adding
padding content to any field, for the purpose of increasing reported token counts".
Two checks look for the signs:

``output_padding`` (always, from ``result_summary.json``):

- **Output length out of line** with the rest of the curve. Every point runs the same
  dataset and model, so mean output length should barely move with concurrency.
- **Token-limit stops**: a large share of responses ending with ``finish_reason``
  ``length``, which padding until the cap produces.

``output_padding_text`` (when ``events.jsonl`` is available), on the kept responses:

- **Duplicated fields**: the output repeating the reasoning trace, or the reverse.
- **Repetition**: responses mostly made of repeated passages. Models can loop on their
  own, usually ending at the token limit, so the share matters more than any one case.
- **Trailing filler**: responses ending in a long run of one character.
"""

from __future__ import annotations

import re
import statistics

from ..context import INPUT_LOGS, INPUT_SUMMARIES, Curve, Point, ReviewContext
from ..findings import Finding, Level
from ..logs import SampleRecord
from ..text import quote
from ..thresholds import LogThresholds
from .base import Check, finding_factory, fmt, pct, points_label

__all__ = ["AGGREGATE_CHECK", "TEXT_CHECK"]

_aggregate = finding_factory("output_padding")
_text = finding_factory("output_padding_text", "output_padding")
_SPEC = "§9.2, §2.8"
_SHINGLE = 8
_WORD = re.compile(r"\w+")


# ── Aggregate ─────────────────────────────────────────────────────────────────


def _mean_osl(point: Point, min_samples: int) -> float | None:
    s = point.summary
    if s is None or s.n_samples_completed < min_samples:
        return None
    return s.output_sequence_lengths.total / s.n_samples_completed


def _length_share(point: Point) -> float | None:
    counts = (
        (point.summary.model_extra or {}).get("finish_reason_counts") if point.summary else None
    )
    if not isinstance(counts, dict):
        return None
    numbers = {k: v for k, v in counts.items() if isinstance(v, (int, float))}
    total = sum(numbers.values())
    return numbers.get("length", 0) / total if total else None


def _aggregate_curve(curve: Curve, ctx: ReviewContext) -> list[Finding]:
    t = ctx.thresholds.outputs
    means = {p.name: m for p in curve.points if (m := _mean_osl(p, t.osl_min_samples)) is not None}
    out: list[Finding] = []
    if len(means) >= 3:
        median = statistics.median(means.values())
        off = [
            f"{name} {fmt(mean)} ({mean / median - 1:+.0%})"
            for name, mean in means.items()
            if median > 0 and abs(mean / median - 1) > t.osl_deviation_note
        ]
        if off:
            out.append(
                _aggregate(
                    Level.NOTE,
                    "Output length out of line with the curve",
                    f"Mean output tokens per response: {points_label(off)}, against a curve "
                    f"median of {fmt(median)}. The same dataset and model should give similar "
                    "lengths at every concurrency; check for padding or truncation.",
                    curve=curve.label,
                    spec_ref=_SPEC,
                    median=round(median, 2),
                )
            )
    capped = [
        f"{p.name} {pct(share)}"
        for p in curve.points
        if (share := _length_share(p)) is not None and share > t.length_finish_note
    ]
    if capped:
        out.append(
            _aggregate(
                Level.NOTE,
                "Most responses stop at the token limit",
                f"finish_reason “length” on {points_label(capped)} of responses. Padding until "
                "the cap ends this way; so does a max_tokens set below the dataset's needs.",
                curve=curve.label,
                spec_ref=_SPEC,
            )
        )
    return out


def run_aggregate(ctx: ReviewContext) -> list[Finding]:
    """Return the output-length findings for every frontier."""
    return [f for curve in ctx.curves for f in _aggregate_curve(curve, ctx)]


# ── Text ──────────────────────────────────────────────────────────────────────


def _shingles(words: list[str]) -> list[tuple[str, ...]]:
    return [tuple(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)]


def _duplicated(sample: SampleRecord, t: LogThresholds) -> bool:
    a = _WORD.findall((sample.output or "").lower())
    b = _WORD.findall((sample.reasoning or "").lower())
    if min(len(a), len(b)) < t.duplicate_min_words:
        return False
    sa, sb = set(_shingles(a)), set(_shingles(b))
    smaller = min(len(sa), len(sb))
    return smaller > 0 and len(sa & sb) / smaller >= t.duplicate_containment


def _repetitive(text: str, t: LogThresholds) -> bool:
    words = _WORD.findall(text.lower())
    if len(words) < t.repetition_min_words:
        return False
    shingles = _shingles(words)
    return bool(shingles) and 1 - len(set(shingles)) / len(shingles) >= t.repetition_ratio


def _filler(sample: SampleRecord, t: LogThresholds) -> bool:
    # The kept texts are truncated; the tail holds the response's real end.
    tail = (sample.tail or "")[-t.filler_run :]
    return len(tail) == t.filler_run and len(set(tail)) == 1


def _example(text: str) -> str:
    snippet = text.strip()
    return quote(repr(snippet[:80] + ("…" if len(snippet) > 80 else "")))


def _text_curve(curve: Curve, ctx: ReviewContext) -> list[Finding]:
    t = ctx.thresholds.logs
    rows: dict[str, list[tuple[str, float, int, str]]] = {"dup": [], "rep": [], "fill": []}
    for point in curve.points:
        log = ctx.logs.get(point.dir)
        if log is None:
            continue
        kept = [s for s in log.completions("performance") if s.output is not None]
        if not kept:
            continue
        texts = [((s.reasoning or "") + (s.output or ""), s) for s in kept]
        for key, hits in (
            ("dup", [s for _, s in texts if _duplicated(s, t)]),
            ("rep", [s for text, s in texts if _repetitive(text, t)]),
            ("fill", [s for _, s in texts if _filler(s, t)]),
        ):
            if hits:
                example = _example((hits[0].reasoning or "") + (hits[0].output or ""))
                if key == "dup":
                    example = _example(hits[0].output or "")
                elif key == "fill":
                    # The end is the point, and stripping would erase a whitespace run.
                    example = quote(repr("…" + (hits[0].tail or "")[-60:]))
                rows[key].append((point.name, len(hits) / len(kept), len(kept), example))

    specs = {
        "dup": (
            "Content duplicated across response fields",
            t.duplicate_share_note,
            t.duplicate_share_attention,
            "repeat their reasoning trace in the visible output (or the reverse)",
        ),
        "rep": (
            "Repetitive responses",
            t.repetition_share_note,
            t.repetition_share_attention,
            "are mostly repeated passages",
        ),
        "fill": (
            "Responses end in filler",
            t.filler_share_note,
            t.filler_share_attention,
            f"end in {t.filler_run}+ copies of one character",
        ),
    }
    out: list[Finding] = []
    for key, (title, note, attention, what) in specs.items():
        flagged = [r for r in rows[key] if r[1] > note]
        if not flagged:
            continue
        worst = max(r[1] for r in flagged)
        out.append(
            _text(
                Level.ATTENTION if worst > attention else Level.NOTE,
                title,
                points_label([f"{name} {pct(share)} of {n}" for name, share, n, _ in flagged])
                + f" of the sampled responses {what}; e.g. {flagged[0][3]}. §2.8 bars "
                "duplicating or padding content to raise the output-token count.",
                curve=curve.label,
                spec_ref=_SPEC,
                worst_share=round(worst, 4),
            )
        )
    return out


def run_text(ctx: ReviewContext) -> list[Finding]:
    """Return the response-text findings for every point with a log."""
    return [f for curve in ctx.curves for f in _text_curve(curve, ctx)]


AGGREGATE_CHECK = Check(
    name="output_padding",
    focus_area="output_padding",
    requires=frozenset({INPUT_SUMMARIES}),
    run=run_aggregate,
)
TEXT_CHECK = Check(
    name="output_padding_text",
    focus_area="output_padding",
    requires=frozenset({INPUT_LOGS}),
    run=run_text,
)
