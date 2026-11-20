# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""``endpoints-submission-reviewer``: review hints for a submission, for a PR comment.

Reviews every submission found under PATH (the submission itself, its organization
directory, or a review-PR checkout holding ``<org>/<submission_id>/``) and writes the
result as Markdown, ready to post as a PR comment, or as JSON.

Findings are hints for reviewers, not failures, so the command exits 0 whatever it
finds. It exits non-zero only when it could not review at all: no submission under
PATH, or an unreadable thresholds file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from . import history
from .context import find_submissions
from .render import render_comment
from .runner import review
from .thresholds import ThresholdsError, load_thresholds

__all__ = ["main"]


def _label(submission: Path, start: Path) -> str:
    """Name *submission* relative to where the search started, e.g. ``ACME/abc123``."""
    if submission == start:
        return f"{start.parent.name}/{start.name}" if start.parent.name else start.name
    return str(submission.relative_to(start))


@click.command(name="endpoints-submission-reviewer")
@click.argument(
    "path", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path(".")
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    show_default=True,
    help="Markdown for a PR comment, or JSON for tooling.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Write to this file instead of standard output.",
)
@click.option(
    "--thresholds",
    "thresholds_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML overriding any of the packaged heuristic tolerances.",
)
@click.option(
    "--base-ref",
    metavar="REV",
    help="Git revision holding the original submission. Updates since then are compared with it.",
)
@click.option(
    "--pr-base",
    metavar="REV",
    help="Instead of --base-ref, in a review PR: the PR's base commit. The original "
    "submission is found among the PR's commits by the lifecycle manager's commit titles.",
)
@click.option(
    "--logs-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory mirroring results/ (<system>/<model>/r<N>/) holding each point's "
    "events.jsonl, for submissions that do not ship them.",
)
@click.option(
    "--step-summary/--no-step-summary",
    default=True,
    show_default=True,
    help="Also append the Markdown to $GITHUB_STEP_SUMMARY when it is set.",
)
def main(
    path: Path,
    fmt: str,
    output: Path | None,
    thresholds_path: Path | None,
    base_ref: str | None,
    pr_base: str | None,
    logs_dir: Path | None,
    step_summary: bool,
) -> None:
    """Review every submission under PATH and report §9.2 review hints."""
    try:
        thresholds = load_thresholds(thresholds_path)
    except ThresholdsError as exc:
        raise click.ClickException(str(exc)) from exc

    if base_ref and pr_base:
        raise click.UsageError("give --base-ref or --pr-base, not both")

    start = path.resolve()
    submissions = find_submissions(start)
    if not submissions:
        raise click.ClickException(f"no submission (a directory holding results/) under {path}")

    if pr_base:
        try:
            base_ref = history.find_original(start, pr_base)
        except history.GitError as exc:
            click.echo(f"warning: could not find the original submission: {exc}", err=True)
            base_ref = pr_base  # the review then explains what is missing in its notes

    reports = [
        review(s, thresholds, _label(s, start), base_ref=base_ref, logs_dir=logs_dir)
        for s in submissions
    ]
    markdown = render_comment(reports)
    if fmt == "json":
        text = json.dumps({"reports": [r.model_dump(mode="json") for r in reports]}, indent=2)
    else:
        text = markdown

    if output is None:
        click.echo(text, nl=not text.endswith("\n"))
    else:
        output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary and summary_file:
        with open(summary_file, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
