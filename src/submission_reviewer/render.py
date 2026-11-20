# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Render review reports as the Markdown PR comment.

Every §9.2 focus area appears in the comment, under one of four headings:

- its own section, when a check found something;
- "Checked, nothing unusual", when every check for it ran clean;
- "Not evaluated", when a check for it was skipped or failed;
- "Manual review only", when no check covers it.

So the comment doubles as the reviewer's checklist. The first line is a hidden
:data:`MARKER`, which the workflow uses to find and update its own comment instead of
adding a new one on every push.
"""

from __future__ import annotations

from collections.abc import Sequence

from .findings import (
    FOCUS_AREAS,
    LEVEL_BADGE,
    LEVEL_ORDER,
    CheckOutcome,
    FocusArea,
    Level,
    ReviewReport,
)

__all__ = ["MARKER", "MAX_COMMENT_CHARS", "render_comment"]

#: Hidden first line identifying the reviewer's comment on a PR.
MARKER = "<!-- submission-reviewer -->"

#: GitHub rejects comments over 65,536 characters; stay well clear of it.
MAX_COMMENT_CHARS = 60_000

#: Findings shown per focus area before the rest are summarized as a count.
MAX_ROWS_PER_AREA = 30

#: Tables longer than this are folded into a <details> block.
_FOLD_ROWS = 8

_RULES_URL = (
    "https://github.com/mlcommons/endpoints_policies/blob/main/endpoints_rules.md"
    "#92-manual-review-focus-areas"
)

_TRUNCATED = (
    "\n\n> [!WARNING]\n> This comment was cut to fit GitHub's size limit. "
    "The job summary and the JSON output (`--format json`) have every finding.\n"
)


def _cell(text: str) -> str:
    """Make *text* safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _counts_line(report: ReviewReport) -> str:
    counts = report.counts
    parts = [
        f"{LEVEL_BADGE[Level.ATTENTION]} {counts[Level.ATTENTION.value]} attention",
        f"{LEVEL_BADGE[Level.NOTE]} {counts[Level.NOTE.value]} notes",
        f"{LEVEL_BADGE[Level.INFO]} {counts[Level.INFO.value]} info",
    ]
    return " · ".join(parts)


def _gap(outcome: CheckOutcome) -> str:
    """What *outcome* left unevaluated, or ``""`` when its check covered everything."""
    if outcome.status == "ran" and not outcome.reason:
        return ""
    verb = {"error": "failed", "skipped": "skipped", "ran": "covered only part"}
    return f"`{outcome.check}` {verb[outcome.status]}: {outcome.reason}"


def _area_section(area: FocusArea, report: ReviewReport) -> list[str]:
    found = sorted(
        (f for f in report.findings if f.focus_area == area.key),
        key=lambda f: (LEVEL_ORDER[f.level], f.curve or "", f.point or ""),
    )
    worst = found[0].level
    lines = [
        f"#### {LEVEL_BADGE[worst]} {area.title}",
        "",
        f"_{area.question}_ ({area.spec_ref})",
        "",
    ]
    if area.scope:
        lines += [f"Checked automatically: {area.scope}", ""]
    gaps = [_gap(o) for o in report.outcomes if o.focus_area == area.key and _gap(o)]
    if gaps:
        lines += ["Not evaluated: " + "; ".join(gaps), ""]
    table = ["| | Where | Finding | Details |", "| --- | --- | --- | --- |"]
    for f in found[:MAX_ROWS_PER_AREA]:
        where = " ".join(part for part in (f"`{f.curve}`" if f.curve else "", f.point) if part)
        details = f.detail + (f" ({f.spec_ref})" if f.spec_ref else "")
        table.append(
            f"| {LEVEL_BADGE[f.level]} | {_cell(where) or '—'} | {_cell(f.title)} | "
            f"{_cell(details)} |"
        )
    hidden = len(found) - MAX_ROWS_PER_AREA
    if len(found) > _FOLD_ROWS:
        lines += [
            "<details>",
            f"<summary>{len(found)} findings — click to expand</summary>",
            "",
            *table,
            "",
            "</details>",
        ]
    else:
        lines += table
    if hidden > 0:
        lines += ["", f"_…and {hidden} more; see the JSON output for the full list._"]
    return [*lines, ""]


def _bullets(title: str, items: list[str]) -> list[str]:
    if not items:
        return []
    return [f"#### {title}", "", *(f"- {item}" for item in items), ""]


def _report_section(report: ReviewReport) -> list[str]:
    lines = [f"### `{report.submission}`", "", _counts_line(report), ""]

    clean: list[str] = []
    not_evaluated: list[str] = []
    manual: list[str] = []
    for area in FOCUS_AREAS:
        outcomes = [o for o in report.outcomes if o.focus_area == area.key]
        if any(f.focus_area == area.key for f in report.findings):
            lines += _area_section(area, report)
        elif not outcomes:
            manual.append(f"**{area.title}** — {area.question} {area.manual_note}".rstrip())
        else:
            reasons = "; ".join(_gap(o) for o in outcomes if _gap(o))
            if not reasons:
                clean.append(f"**{area.title}**" + (f" — {area.scope}" if area.scope else ""))
            elif any(o.status == "ran" for o in outcomes):
                clean.append(f"**{area.title}** — partly; {reasons}")
            else:
                not_evaluated.append(f"**{area.title}** — {reasons}")

    lines += _bullets("✅ Checked, nothing unusual", clean)
    lines += _bullets("⏭️ Not evaluated", not_evaluated)
    lines += _bullets("👀 Manual review only", manual)
    if report.notes:
        lines += [
            "<details>",
            f"<summary>{len(report.notes)} file(s) could not be read</summary>",
            "",
            *(f"- {note}" for note in report.notes),
            "",
            "</details>",
            "",
        ]
    return lines


def render_comment(reports: Sequence[ReviewReport]) -> str:
    """Render *reports* as one Markdown PR comment, starting with :data:`MARKER`."""
    lines = [
        MARKER,
        "## 🔎 Review hints (§9.2)",
        "",
        f"Automated heuristics for the [§9.2 manual review focus areas]({_RULES_URL}). "
        "They point at what looks unusual so reviewers know where to start. They are not "
        "objections, and a clean result does not replace review. §9.1 compliance is "
        "reported separately by **Check submission**.",
        "",
        f"Levels: {LEVEL_BADGE[Level.ATTENTION]} unlikely in a genuine run, look first · "
        f"{LEVEL_BADGE[Level.NOTE]} unusual, often explainable · "
        f"{LEVEL_BADGE[Level.INFO]} context.",
        "",
    ]
    if not reports:
        lines.append("_No submission directory (one holding `results/`) was found._")
    for report in reports:
        lines += _report_section(report)
    text = "\n".join(lines).rstrip() + "\n"
    if len(text) > MAX_COMMENT_CHARS:
        text = text[: MAX_COMMENT_CHARS - len(_TRUNCATED)] + _TRUNCATED
    return text
