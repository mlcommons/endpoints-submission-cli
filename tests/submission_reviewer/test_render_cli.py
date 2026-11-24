# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""The PR comment and the ``endpoints-submission-reviewer`` command."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from submission_reviewer import CheckOutcome, Finding, Level, ReviewReport, review
from submission_reviewer.cli import main
from submission_reviewer.render import MARKER, MAX_COMMENT_CHARS, escape, render_comment
from submission_reviewer.text import quote

from .conftest import clean_curve, with_point, write_submission

# ── Comment ───────────────────────────────────────────────────────────────────


def _finding(**kw: object) -> Finding:
    base: dict[str, object] = {
        "check": "pareto_shape",
        "focus_area": "pareto_shape",
        "level": Level.ATTENTION,
        "title": "Throughput dips, then recovers",
        "detail": "falls 20%",
        "curve": "sys/model",
        "point": "r256",
        "spec_ref": "§9.2",
    }
    return Finding.model_validate({**base, **kw})


def _report(findings: list[Finding], outcomes: list[CheckOutcome] | None = None) -> ReviewReport:
    return ReviewReport(
        submission="Org/abc",
        findings=findings,
        outcomes=outcomes
        if outcomes is not None
        else [
            CheckOutcome(check="pareto_shape", focus_area="pareto_shape", status="ran"),
            CheckOutcome(
                check="metric_distributions", focus_area="metric_distributions", status="ran"
            ),
        ],
    )


def test_comment_starts_with_the_marker_and_lists_every_area() -> None:
    text = render_comment([_report([_finding()])])
    assert text.startswith(MARKER + "\n")
    assert "#### 🔶 Pareto curve plausibility" in text
    assert "`sys/model` r256" in text
    clean = text.split("#### ✅ Checked, nothing unusual")[1].split("####")[0]
    assert "Metric distributions" in clean
    manual = text.split("#### 👀 Manual review only")[1]
    for title in ("Cross-submission consistency", "Warmup data", "Speculative decoding"):
        assert title in manual


def test_partial_areas_say_what_was_checked(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", clean_curve())
    clean = render_comment([review(root)]).split("#### ✅ Checked, nothing unusual")[1]
    assert "**Warmup data** — partly; `warmup_prompts` skipped: needs the per-request logs" in clean
    assert "**Division eligibility** — Declared division and status" in clean


def test_skipped_and_failed_checks_are_listed_as_not_evaluated() -> None:
    outcomes = [
        CheckOutcome(
            check="pareto_shape", focus_area="pareto_shape", status="skipped", reason="no data"
        ),
        CheckOutcome(
            check="metric_distributions",
            focus_area="metric_distributions",
            status="error",
            reason="Boom",
        ),
    ]
    section = render_comment([_report([], outcomes)]).split("#### ⏭️ Not evaluated")[1]
    assert "`pareto_shape` skipped: no data" in section
    assert "`metric_distributions` failed: Boom" in section


def test_table_cells_are_escaped() -> None:
    text = render_comment([_report([_finding(detail="a | b\nc")])])
    assert "a \\| b<br>c" in text


def test_long_tables_fold_and_cap() -> None:
    findings = [_finding(point=f"r{i}") for i in range(40)]
    text = render_comment([_report(findings)])
    assert "<summary>40 findings — click to expand</summary>" in text
    assert "…and 10 more" in text


def _well_formed(text: str) -> None:
    """Nothing was cut mid-line: every table row closes and every block is closed."""
    assert len(text) <= MAX_COMMENT_CHARS
    assert text.count("<details>") == text.count("</details>")
    for line in text.splitlines():
        if line.startswith("| "):
            assert line.endswith(" |"), line


def test_comment_fits_github_limit() -> None:
    findings = [
        _finding(check="x", focus_area=area, point=f"r{i}", detail="d" * 400)
        for area in ("pareto_shape", "metric_distributions", "warmup")
        for i in range(30)
    ]
    reports = [_report(findings) for _ in range(5)]
    text = render_comment(reports)
    _well_formed(text)
    assert "shortened to fit GitHub's size limit: fewer findings are listed" in text
    assert "--format json" in text.splitlines()[-1]
    # Every submission keeps its checklist.
    assert text.count("#### 👀 Manual review only") == 5
    assert all(f"### `{r.submission}`" in text for r in reports)


def test_whole_submissions_are_left_out_last() -> None:
    notes = [f"note {i} " + "n" * 400 for i in range(50)]
    reports = [
        ReviewReport(submission=f"Org/sub{i:02d}", findings=[_finding()], notes=notes)
        for i in range(8)
    ]
    text = render_comment(reports)
    _well_formed(text)
    shown = [r.submission for r in reports if f"### `{r.submission}`" in text]
    assert shown == [r.submission for r in reports[: len(shown)]]
    assert 0 < len(shown) < len(reports)
    warning = text.splitlines()[-1]
    assert f"{len(reports) - len(shown)} submission(s) are left out" in warning
    assert f"`Org/sub{len(shown):02d}`" in warning


def test_one_long_value_cannot_push_sections_out(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", with_point(clean_curve(), 256, system_tps=5000.0))
    for desc_path in root.rglob("system_desc.json"):
        desc = json.loads(desc_path.read_text())
        desc["availability_url"] = "https://example.com/" + "x" * 70_000
        desc["node_types"][0]["hw_notes"] = "TBD"
        desc_path.write_text(json.dumps(desc))
    text = render_comment([review(root, label="Org/sub")])
    _well_formed(text)
    assert "x" * 300 not in text
    for heading in ("System description", "Division eligibility", "Manual review only"):
        assert heading in text


# ── Submitter text ────────────────────────────────────────────────────────────

_PAYLOAD = (
    "TBD ![x](https://evil.test/p.png) [Approve](https://phish.example) "
    "</details><!-- submission-reviewer --> <img src=x> @octocat #1 www.phish.example"
)


def _html(text: str) -> str:
    """*text* as an independent CommonMark parser renders it, tables enabled."""
    markdown_it = pytest.importorskip("markdown_it")
    return markdown_it.MarkdownIt("commonmark").enable("table").render(text)


def _outside_code(html: str) -> str:
    return re.sub(r"<code>.*?</code>", "", html, flags=re.DOTALL)


def test_escape_keeps_code_spans_and_neutralizes_the_rest() -> None:
    assert escape("a `[x](y)` b") == "a `[x](y)` b"
    assert escape("``a ` b`` [c]") == r"``a ` b`` \[c\]"
    # A backtick run that never closes is shown literally, and cannot open a span.
    assert escape("`[x](y)") == r"\`\[x\](y)"
    # A backslash before a backtick is itself escaped, so the backtick still opens a span.
    assert escape(r"\`<b>`") == r"\\`<b>`"
    assert escape("@user #12") == "@\u200buser #\u200b12"
    assert escape("https://x.example www.y.example") == r"https\://x.example www\.y.example"


def test_quote_is_one_capped_code_span() -> None:
    assert quote("a `b`\n c") == "`a 'b' c`"
    assert quote("") == '`""`'
    assert quote("x" * 300) == "`" + "x" * 199 + "…`"


def test_submitter_text_cannot_add_markup() -> None:
    report = ReviewReport(
        submission="Org/abc",
        findings=[
            # A check that quoted the value, and one that (wrongly) did not.
            _finding(detail=f"hw_notes is {quote(_PAYLOAD)}", curve=_PAYLOAD + " `x"),
            _finding(title=_PAYLOAD, detail=_PAYLOAD, point="r1"),
        ],
        outcomes=[
            CheckOutcome(
                check="pareto_shape", focus_area="pareto_shape", status="ran", reason=_PAYLOAD
            )
        ],
        notes=[_PAYLOAD, "`" + _PAYLOAD],
    )
    text = render_comment([report])
    html = _outside_code(_html(text))
    # The only link is the rules link in the header; no image, no raw HTML from the payload.
    assert html.count("<a ") == 1
    assert "<img" not in html
    assert html.count("<details>") == html.count("</details>") == 1  # the notes block
    assert html.count("<!--") == 1  # the marker
    # GitHub links @-mentions and issue references in plain text. (Bare URLs too, which
    # this parser does not; test_escape_keeps_code_spans_and_neutralizes_the_rest
    # covers the escape that stops GitHub.)
    assert "@octocat" not in html
    assert "#1" not in html


def test_submitter_text_in_a_real_review(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", clean_curve())
    for desc_path in root.rglob("system_desc.json"):
        desc = json.loads(desc_path.read_text())
        desc["availability_url"] = _PAYLOAD
        desc["node_types"][0]["hw_notes"] = "N/A"
        desc["node_types"][0]["node_config"] = "8x B200 " + _PAYLOAD
        desc_path.write_text(json.dumps(desc))
    text = render_comment([review(root, label="Org/sub")])
    assert "Availability URL" in text  # the payload did reach a finding
    html = _outside_code(_html(text))
    assert html.count("<a ") == 1
    assert "<img" not in html
    assert html.count("<details>") == html.count("</details>")
    assert "@octocat" not in html


def test_real_review_renders(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", with_point(clean_curve(), 256, system_tps=5000.0))
    text = render_comment([review(root, label="Org/sub")])
    assert "### `Org/sub`" in text
    assert "Throughput dips, then recovers" in text


# ── Command ───────────────────────────────────────────────────────────────────


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A review-PR checkout: ``<org>/<submission_id>/`` below the repo root."""
    write_submission(
        tmp_path / "ACME_Corp" / "abc123", with_point(clean_curve(), 256, system_tps=5000.0)
    )
    return tmp_path


def test_cli_markdown_from_checkout_root(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    result = CliRunner().invoke(main, [str(checkout)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith(MARKER)
    assert "### `ACME_Corp/abc123`" in result.output


def test_cli_json_to_file(checkout: Path, tmp_path: Path) -> None:
    out = tmp_path / "review.json"
    result = CliRunner().invoke(main, [str(checkout), "--format", "json", "-o", str(out)])
    assert result.exit_code == 0, result.output
    (report,) = json.loads(out.read_text())["reports"]
    assert report["submission"] == "ACME_Corp/abc123"
    assert report["counts"]["attention"] >= 1
    assert any(f["title"] == "Throughput dips, then recovers" for f in report["findings"])


def test_cli_appends_step_summary(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("earlier step\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = CliRunner().invoke(main, [str(checkout), "--format", "json"])
    assert result.exit_code == 0, result.output
    text = summary.read_text()
    assert text.startswith("earlier step\n")
    assert MARKER in text


def test_cli_without_a_submission_fails(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, [str(tmp_path)])
    assert result.exit_code == 1
    assert "no submission" in result.output


def test_cli_rejects_a_bad_thresholds_file(checkout: Path, tmp_path: Path) -> None:
    bad = tmp_path / "t.yaml"
    bad.write_text("pareto:\n  nope: 1\n")
    result = CliRunner().invoke(main, [str(checkout), "--thresholds", str(bad)])
    assert result.exit_code == 1
    assert "nope" in result.output
