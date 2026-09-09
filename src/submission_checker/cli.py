"""Command-line interface for the submission checker."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .checker import SubmissionChecker
from .models import Severity, compute_regions
from .models.regions import ULTRA_LOW_CONCURRENCY_MAX
from .seed_sets import SEED_SETS_ENV_VAR

try:
    from endpoints_submission_cli._version_check import (
        register_upgrade_notice as _register_upgrade_notice,
    )
except ImportError:
    _register_upgrade_notice = None  # type: ignore[assignment]

console = Console()

_SEVERITY_STYLE: dict[Severity, str] = {
    Severity.ERROR: "bold red",
    Severity.WARNING: "yellow",
    Severity.INFO: "dim",
}


@click.group()
@click.version_option(package_name="endpoints-submission-cli")
def main() -> None:
    """MLPerf Endpoints submission checker — validate a submission directory."""
    if _register_upgrade_notice is not None:
        _register_upgrade_notice()


@main.command()
@click.argument("path", type=click.Path(exists=False, path_type=Path))
@click.option("--strict", is_flag=True, default=False, help="Treat warnings as errors.")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Hide INFO-level results.")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Write full results as JSON to FILE (in addition to the terminal table).",
)
@click.option(
    "--seed-sets",
    type=click.Path(path_type=Path),
    default=None,
    envvar=SEED_SETS_ENV_VAR,
    help=(
        "Published seed sets to check against (§4.6). Defaults to the set bundled with "
        f"this checker; also settable via ${SEED_SETS_ENV_VAR}."
    ),
)
def check(
    path: Path, strict: bool, quiet: bool, output: Path | None, seed_sets: Path | None
) -> None:
    r"""Check the submission at PATH for §9.1 compliance.

    PATH is the submitting organisation's directory or a submission directory below it.

    Exit codes:

    \b
      0  All checks passed (no errors; warnings ignored unless --strict).
      1  One or more errors found (or warnings when --strict is active).
    """
    checker = SubmissionChecker(path, seed_sets_path=seed_sets)
    report = checker.run()

    if output is not None:
        output.write_text(report.model_dump_json(indent=2))

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
        table.add_row(
            result.rule,
            result.spec_ref,
            f"[{style}]{result.severity.value}[/{style}]",
            result.message,
            str(result.path.relative_to(path))
            if result.path and result.path.is_relative_to(path)
            else str(result.path or ""),
        )

    console.print(table)

    error_count = len(report.errors)
    warn_count = len(report.warnings)
    total_failures = error_count + (warn_count if strict else 0)

    if total_failures:
        console.print(f"[bold red]FAILED[/] — {error_count} error(s), {warn_count} warning(s)")
    else:
        console.print(f"[bold green]PASSED[/] — {error_count} error(s), {warn_count} warning(s)")

    raise SystemExit(1 if total_failures else 0)


@main.command()
@click.option(
    "--max-concurrency",
    "-M",
    required=True,
    type=int,
    help="Declared Maximum Supported Concurrency (C_max).",
)
@click.option(
    "--min-concurrency",
    "-m",
    default=ULTRA_LOW_CONCURRENCY_MAX,
    show_default=True,
    type=int,
    help=(
        "Minimum concurrency (C_min). In a real submission this is derived from the "
        "lowest measurement point, not declared."
    ),
)
def regions(max_concurrency: int, min_concurrency: int) -> None:
    """Show computed region boundaries for a given (C_max, C_min) pair.

    Uses the §5.5 reference algorithm (banker's rounding).
    """
    try:
        r = compute_regions(max_concurrency, min_concurrency)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from None

    table = Table(
        title=f"Region Boundaries for C_max = {max_concurrency}, C_min = {min_concurrency}",
        show_lines=True,
    )
    table.add_column("Region", style="cyan")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")

    rows = [
        ("Low Latency", r.low_latency),
        ("Low Concurrency", r.low_concurrency),
        ("Medium Concurrency", r.med_concurrency),
        ("High Concurrency", r.high_concurrency),
        ("Margin (10% above C_max)", r.margin),
    ]
    for label, bounds in rows:
        table.add_row(label, str(bounds.start), str(bounds.end))

    console.print(table)
