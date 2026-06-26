# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""``check-submission`` command — run the submission checker on a directory.

Designed to be dropped into a GitHub Actions (or any CI) pipeline: it prints a
human-readable table, optionally emits GitHub workflow annotations and a job
summary, and exits non-zero when the submission fails §9.1 compliance.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from submission_checker.checker import SubmissionChecker
from submission_checker.models import Report, Severity

__all__ = ["check_submission"]

# stdout: the report is this command's primary output, so it goes to stdout
# (Rich auto-disables colour when stdout is not a TTY, e.g. in CI logs).
_console = Console()

_SEVERITY_STYLE: dict[Severity, str] = {
    Severity.ERROR: "bold red",
    Severity.WARNING: "yellow",
    Severity.INFO: "dim",
}

# Map our severities onto GitHub Actions annotation levels.
_GITHUB_LEVEL: dict[Severity, str] = {
    Severity.ERROR: "error",
    Severity.WARNING: "warning",
    Severity.INFO: "notice",
}


def _gha_escape(value: str, *, prop: bool) -> str:
    """Escape a string for a GitHub Actions workflow command.

    See: https://docs.github.com/actions/using-workflows/workflow-commands-for-github-actions
    """
    out = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if prop:
        out = out.replace(":", "%3A").replace(",", "%2C")
    return out


def _rel_to_cwd(path: Path) -> str:
    """Path relative to cwd (the repo root under ``actions/checkout``) if possible."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _emit_github_annotations(report: Report) -> None:
    """Emit ``::error``/``::warning`` workflow commands so findings show up inline.

    Written to stderr: GitHub Actions parses workflow commands from the merged
    step log, so stdout stays clean for ``--json`` machine consumption.
    """
    for result in report.results:
        if result.severity == Severity.INFO:
            continue
        props = [f"title=submission-checker: {_gha_escape(result.rule, prop=True)}"]
        if result.path is not None:
            props.insert(0, f"file={_gha_escape(_rel_to_cwd(result.path), prop=True)}")
        ref = f" ({result.spec_ref})" if result.spec_ref else ""
        message = _gha_escape(f"[{result.rule}] {result.message}{ref}", prop=False)
        click.echo(f"::{_GITHUB_LEVEL[result.severity]} {','.join(props)}::{message}", err=True)


def _write_step_summary(report: Report, path: Path, summary_file: Path) -> None:
    """Append a Markdown summary to ``$GITHUB_STEP_SUMMARY`` for the job page."""
    errors, warnings = report.errors, report.warnings
    status = "✅ **PASSED**" if not errors else "❌ **FAILED**"
    lines = [
        "## Submission Checker",
        "",
        f"{status} — `{path}`",
        "",
        f"- Errors: **{len(errors)}**",
        f"- Warnings: **{len(warnings)}**",
        f"- Total checks: {len(report.results)}",
    ]
    findings = [r for r in report.results if r.severity != Severity.INFO]
    if findings:
        lines += [
            "",
            "| Severity | Rule | § Ref | Message | Path |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in findings:
            loc = _rel_to_cwd(r.path) if r.path else ""
            msg = r.message.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r.severity.value} | {r.rule} | {r.spec_ref} | {msg} | {loc} |")
    lines.append("")
    with open(summary_file, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _render_table(report: Report, path: Path, quiet: bool) -> Table:
    table = Table(title=f"Submission Check — {path}", show_lines=True)
    table.add_column("Rule", style="cyan", no_wrap=True)
    table.add_column("§ Ref", style="dim", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Message")
    table.add_column("Path", style="dim")

    for result in report.results:
        if quiet and result.severity == Severity.INFO:
            continue
        style = _SEVERITY_STYLE[result.severity]
        loc = (
            str(result.path.relative_to(path))
            if result.path and result.path.is_relative_to(path)
            else str(result.path or "")
        )
        table.add_row(
            result.rule,
            result.spec_ref,
            f"[{style}]{result.severity.value}[/{style}]",
            result.message,
            loc,
        )
    return table


@click.command(name="check-submission")
@click.argument("path", type=click.Path(exists=False, path_type=Path))
@click.option("--strict", is_flag=True, default=False, help="Treat warnings as errors.")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Hide INFO-level results.")
@click.option(
    "-j",
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Print the full report as JSON to stdout (suppresses the table).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the full report as JSON to FILE.",
)
@click.option(
    "--annotate/--no-annotate",
    default=None,
    help="Emit GitHub Actions annotations. Default: on when $GITHUB_ACTIONS is set.",
)
def check_submission(
    path: Path,
    strict: bool,
    quiet: bool,
    as_json: bool,
    output: Path | None,
    annotate: bool | None,
) -> None:
    r"""Run the submission checker on the submission directory at PATH.

    PATH is the submitting organisation's root directory. Intended for CI: it
    prints a results table to stdout, optionally emits GitHub Actions
    annotations/job summary, and sets the exit code from the outcome.

    Exit codes:

    \b
      0  Passed — no errors (and no warnings when --strict).
      1  Failed — one or more errors (or warnings when --strict).
      2  Usage error (bad arguments).
    """
    report = SubmissionChecker(path).run()

    in_github = os.environ.get("GITHUB_ACTIONS") == "true"
    do_annotate = in_github if annotate is None else annotate

    if output is not None:
        output.write_text(report.model_dump_json(indent=2))

    if do_annotate:
        _emit_github_annotations(report)
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            _write_step_summary(report, path, Path(summary_file))

    if as_json:
        click.echo(report.model_dump_json(indent=2))
    else:
        _console.print(_render_table(report, path, quiet))

    error_count = len(report.errors)
    warn_count = len(report.warnings)
    failed = error_count > 0 or (strict and warn_count > 0)

    if not as_json:
        verdict = "[bold red]FAILED[/]" if failed else "[bold green]PASSED[/]"
        _console.print(f"{verdict} — {error_count} error(s), {warn_count} warning(s)")

    raise SystemExit(1 if failed else 0)
