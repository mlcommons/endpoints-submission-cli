# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: did warmup requests draw on any sample of the performance dataset?

The §9.2 bullet suggests reviewers "cross-check retained warmup logs against the
performance dataset". With ``events.jsonl`` the check can do that directly: the client
logs every warmup prompt (its responses are not logged) and every performance prompt,
so the reviewer can compare the two sets.

- **Overlap**: warmup prompts identical to a performance prompt once any §6.3.1 salt
  is stripped. Unsalted overlap asks for attention; salted overlap is context, since
  §6.3.1 allows it with KV-cache reuse disabled. Prompts are compared whole, so a
  truncated or paraphrased sample (also barred by §6.3.1) is not caught.
- **Log vs declaration**: a point that declares warmup requests but whose log has no
  warmup phase, or a logged warmup request count that differs from the declared one.
"""

from __future__ import annotations

from ..context import INPUT_LOGS, Curve, ReviewContext
from ..findings import Finding, Level
from ..text import quote
from .base import Check, finding_factory, points_label

__all__ = ["CHECK"]

_finding = finding_factory("warmup_prompts", "warmup")


def _curve(curve: Curve, ctx: ReviewContext) -> list[Finding]:
    unsalted: list[str] = []
    salted: list[str] = []
    examples: list[str] = []
    missing: list[str] = []
    count_mismatch: list[str] = []
    for point in curve.points:
        log = ctx.logs.get(point.dir)
        if log is None:
            continue
        warmup = log.issued("warmup")
        declared = point.config.warmup if point.config is not None else None
        if declared is not None and not declared.is_disabled:
            if not warmup:
                missing.append(point.name)
            elif declared.requests_issued != len(warmup):
                count_mismatch.append(
                    f"{point.name} ({declared.requests_issued} declared, {len(warmup)} logged)"
                )
        if not warmup:
            continue
        performance = {s.prompt_key for s in log.issued("performance") if s.prompt_key}
        overlap = [s for s in warmup if s.prompt_key in performance]
        if not overlap:
            continue
        distinct = {s.prompt_key for s in overlap}
        row = (
            f"{point.name} ({len(overlap)} of {len(warmup)} warmup requests, "
            f"{len(distinct)} prompts)"
        )
        (salted if all(s.salted for s in overlap) else unsalted).append(row)
        if not examples and overlap[0].prompt_key:
            examples.append(quote(repr(log.prompt_examples.get(overlap[0].prompt_key, ""))))

    out: list[Finding] = []
    if unsalted:
        out.append(
            _finding(
                Level.ATTENTION,
                "Warmup reused performance prompts",
                f"{points_label(unsalted)} sent prompts identical to performance-dataset "
                f"prompts, unsalted; e.g. {examples[0] if examples else '—'}. §6.3.1 bars "
                "performance samples from warmup.",
                curve=curve.label,
                spec_ref="§9.2, §6.3.1",
            )
        )
    if salted:
        out.append(
            _finding(
                Level.INFO,
                "Salted warmup reused performance prompts",
                f"{points_label(salted)} warmed up on performance prompts with a salt "
                "prefix, which §6.3.1 allows only with KV-cache reuse disabled.",
                curve=curve.label,
                spec_ref="§6.3.1",
            )
        )
    if missing:
        out.append(
            _finding(
                Level.NOTE,
                "Declared warmup is not in the log",
                f"{points_label(missing)} declare warmup requests, but their events.jsonl has "
                "no warmup phase. The retained log may not be the one from this run.",
                curve=curve.label,
                spec_ref="§9.2, §6.3.2",
            )
        )
    if count_mismatch:
        out.append(
            _finding(
                Level.NOTE,
                "Warmup count differs from the declaration",
                f"{points_label(count_mismatch)}.",
                curve=curve.label,
                spec_ref="§9.2, §6.3.3",
            )
        )
    return out


def run(ctx: ReviewContext) -> list[Finding]:
    """Return the warmup-log findings for every point with a log."""
    return [f for curve in ctx.curves for f in _curve(curve, ctx)]


CHECK = Check(
    name="warmup_prompts",
    focus_area="warmup",
    requires=frozenset({INPUT_LOGS}),
    run=run,
)
