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

import re
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
from .text import quote

__all__ = ["MARKER", "MAX_COMMENT_CHARS", "escape", "render_comment"]

#: Hidden first line identifying the reviewer's comment on a PR.
MARKER = "<!-- submission-reviewer -->"

#: GitHub rejects comments over 65,536 characters; stay well clear of it.
MAX_COMMENT_CHARS = 60_000

#: Findings shown per focus area before the rest are summarized as a count.
MAX_ROWS_PER_AREA = 30

#: Row caps tried in turn until the comment fits :data:`MAX_COMMENT_CHARS`.
_ROW_CAPS = (MAX_ROWS_PER_AREA, 15, 8, 4, 2, 1, 0)

#: Tables longer than this are folded into a <details> block.
_FOLD_ROWS = 8

#: Longest text shown for each part of a finding, and for each note.
_TITLE_CHARS = 200
_WHERE_CHARS = 200
_DETAIL_CHARS = 1_500
_NOTE_CHARS = 500
_REASON_CHARS = 500
#: Notes listed before the rest are summarized as a count.
_MAX_NOTES = 50
#: Omitted submissions named in the truncation warning.
_MAX_OMITTED_NAMED = 10

_RULES_URL = (
    "https://github.com/mlcommons/endpoints_policies/blob/main/endpoints_rules.md"
    "#92-manual-review-focus-areas"
)
_JSON_HINT = "The JSON output (`--format json`) has every finding."

_ZWSP = "\u200b"
_BACKTICKS = re.compile(r"`+")
_MARKUP = re.compile(r"[\\\[\]<>&]")
_AUTOLINK = re.compile(r"://|(?i:www)\.")


def _escape_plain(text: str) -> str:
    """Neutralize the Markdown in *text*, which holds no backticks."""
    text = _MARKUP.sub(lambda m: "\\" + m.group(), text)
    # GitHub links bare URLs; escaping the ``:`` of ``://`` or the dot of ``www.`` stops it.
    text = _AUTOLINK.sub(lambda m: "\\://" if m.group() == "://" else m.group()[:-1] + "\\.", text)
    # A backslash does not stop an @-mention or an issue reference; a zero-width
    # space after the sign does.
    return text.replace("@", "@" + _ZWSP).replace("#", "#" + _ZWSP)


def escape(text: str) -> str:
    """Make *text* render as plain text, keeping its code spans.

    Finding text quotes submitter values in code spans (:func:`~.text.quote`), whose
    content GitHub shows literally. Everything outside them is escaped so that it
    cannot form a link, an image, raw HTML, an autolink or an @-mention, whatever a
    check forgot to quote. Code spans are found as GitHub finds them: a run of
    backticks closes at the next run of the same length, and a run that never closes
    is shown as literal backticks.
    """
    out: list[str] = []
    runs = list(_BACKTICKS.finditer(text))
    pos = i = 0
    while i < len(runs):
        run = runs[i]
        out.append(_escape_plain(text[pos : run.start()]))
        width = len(run.group())
        close = next((j for j in range(i + 1, len(runs)) if len(runs[j].group()) == width), None)
        if close is None:
            out.append("\\`" * width)
            pos, i = run.end(), i + 1
        else:
            out.append(text[run.start() : runs[close].end()])
            pos, i = runs[close].end(), close + 1
    out.append(_escape_plain(text[pos:]))
    return "".join(out)


def _clip(text: str, limit: int) -> str:
    """*text* cut to *limit* characters, before escaping, so no escape is cut in half."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(text: str, limit: int) -> str:
    """Make *text* safe inside a Markdown table cell."""
    return escape(_clip(text, limit)).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _line(text: str, limit: int) -> str:
    """Make *text* safe as one list item: escaped, and on one line."""
    return escape(_clip(" ".join(text.split()), limit))


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
    reason = _line(outcome.reason, _REASON_CHARS)
    return f"`{outcome.check}` {verb[outcome.status]}: {reason}"


def _area_section(area: FocusArea, report: ReviewReport, max_rows: int) -> list[str]:
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
    if max_rows == 0:
        return [*lines, f"_{len(found)} finding(s), left out to fit; {_JSON_HINT}_", ""]
    table = ["| | Where | Finding | Details |", "| --- | --- | --- | --- |"]
    for f in found[:max_rows]:
        where = " ".join(part for part in (quote(f.curve) if f.curve else "", f.point) if part)
        details = f.detail + (f" ({f.spec_ref})" if f.spec_ref else "")
        table.append(
            f"| {LEVEL_BADGE[f.level]} | {_cell(where, _WHERE_CHARS) or '—'} | "
            f"{_cell(f.title, _TITLE_CHARS)} | {_cell(details, _DETAIL_CHARS)} |"
        )
    shown = min(len(found), max_rows)
    if shown > _FOLD_ROWS:
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
    if len(found) > shown:
        lines += ["", f"_…and {len(found) - shown} more; see the JSON output for the full list._"]
    return [*lines, ""]


def _bullets(title: str, items: list[str]) -> list[str]:
    if not items:
        return []
    return [f"#### {title}", "", *(f"- {item}" for item in items), ""]


def _notes(report: ReviewReport) -> list[str]:
    if not report.notes:
        return []
    items = [f"- {_line(note, _NOTE_CHARS)}" for note in report.notes[:_MAX_NOTES]]
    if len(report.notes) > _MAX_NOTES:
        items.append(f"- …and {len(report.notes) - _MAX_NOTES} more; see the JSON output.")
    return [
        "<details>",
        f"<summary>{len(report.notes)} note(s) on the inputs</summary>",
        "",
        *items,
        "",
        "</details>",
        "",
    ]


def _report_section(report: ReviewReport, max_rows: int = MAX_ROWS_PER_AREA) -> list[str]:
    lines = [f"### {escape(quote(report.submission))}", "", _counts_line(report), ""]

    clean: list[str] = []
    not_evaluated: list[str] = []
    manual: list[str] = []
    for area in FOCUS_AREAS:
        outcomes = [o for o in report.outcomes if o.focus_area == area.key]
        if any(f.focus_area == area.key for f in report.findings):
            lines += _area_section(area, report, max_rows)
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
    lines += _notes(report)
    return lines


def _header(reports: Sequence[ReviewReport]) -> list[str]:
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
    return lines


def _warning(what: str) -> list[str]:
    return ["", "> [!WARNING]", f"> This comment was shortened to fit GitHub's size limit: {what}"]


def _join(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


def render_comment(reports: Sequence[ReviewReport]) -> str:
    """Render *reports* as one Markdown PR comment, starting with :data:`MARKER`.

    A comment over :data:`MAX_COMMENT_CHARS` is shortened by whole lines, never by
    cutting through one: first by showing fewer table rows per focus area, then by
    leaving out whole submissions from the end. The checklist of every submission
    shown is always kept, and a warning says what was left out.
    """
    header = _header(reports)
    for max_rows in _ROW_CAPS:
        sections = [_report_section(r, max_rows) for r in reports]
        lines = header + [line for section in sections for line in section]
        if max_rows != MAX_ROWS_PER_AREA:
            lines += _warning(f"fewer findings are listed per focus area. {_JSON_HINT}")
        text = _join(lines)
        if len(text) <= MAX_COMMENT_CHARS:
            return text
    # Even without tables it is too long: keep as many whole submissions as fit.
    for keep in range(len(sections) - 1, -1, -1):
        omitted = [quote(r.submission, limit=80) for r in reports[keep:]]
        named = ", ".join(omitted[:_MAX_OMITTED_NAMED])
        if len(omitted) > _MAX_OMITTED_NAMED:
            named += f" and {len(omitted) - _MAX_OMITTED_NAMED} more"
        lines = header + [line for section in sections[:keep] for line in section]
        lines += _warning(
            f"no findings are listed, and {len(omitted)} submission(s) are left out "
            f"({escape(named)}). {_JSON_HINT}"
        )
        text = _join(lines)
        if len(text) <= MAX_COMMENT_CHARS:
            return text
    return text
