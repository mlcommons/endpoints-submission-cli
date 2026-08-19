# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Rich-based output formatters for run records."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from ..formatting import fmt_bool, fmt_dt, fmt_int, fmt_str

__all__ = ["print_runs_table", "print_run_detail"]

_console = Console()


def print_runs_table(runs: list[dict[str, Any]]) -> None:
    """Print a summary table of run records (RunSummary schema)."""
    if not runs:
        _console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Runs", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    # no_wrap keeps the "*" test marker on the same line as the model instead of
    # doubling the row height; max_width stops the widened cell from starving the
    # other columns on a narrow terminal.
    table.add_column("Model", style="green", no_wrap=True, overflow="ellipsis", max_width=24)
    table.add_column("Concurrency", justify="right")
    table.add_column("Started At", style="dim")
    table.add_column("Finished At", style="dim")

    for run in runs:
        model = fmt_str(_run_model(run))
        if run.get("is_test"):
            # A one-char prefix rather than a column: a wider marker wraps the cell
            # onto a second line, and a trailing one is truncated away first.
            # Explained by the legend printed below the table.
            model = f"[yellow]*[/yellow] {model}"
        table.add_row(
            fmt_str(run.get("id")),
            model,
            fmt_int(_run_concurrency(run)),
            fmt_dt(run.get("started_at")),
            fmt_dt(run.get("finished_at")),
        )

    _console.print(table)
    if any(r.get("is_test") for r in runs):
        _console.print("[dim][yellow]*[/yellow] = test run[/dim]")


def _run_model(run: dict[str, Any]) -> Any:
    """Resolve a run's model, whether the record is a flat RunSummary or a full run.

    `runs list` returns a flat top-level ``model``; embedded runs from
    `submissions get` only carry the nested ``config``, so fall back to the
    same fields the API uses to derive the summary.
    """
    if run.get("model") is not None:
        return run["model"]
    config = run.get("config") or {}
    return (config.get("model_params") or {}).get("name") or config.get("model")


def _run_concurrency(run: dict[str, Any]) -> Any:
    """Resolve a run's concurrency for both flat and full run records (see _run_model)."""
    if run.get("concurrency") is not None:
        return run["concurrency"]
    config = run.get("config") or {}
    load_pattern = (config.get("settings") or {}).get("load_pattern") or {}
    target = load_pattern.get("target_concurrency")
    return target if target is not None else config.get("concurrency")


def print_run_detail(run: dict[str, Any]) -> None:
    """Print full details for a single run (RunOut schema)."""
    table = Table(title=f"Run {run.get('id', '')}", show_lines=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")

    rows = [
        ("ID", fmt_str(run.get("id"))),
        ("User ID", fmt_str(run.get("user_id"))),
        ("Model", fmt_str(_run_model(run))),
        ("Concurrency", fmt_int(_run_concurrency(run))),
        ("Benchmark Version", fmt_str(run.get("benchmark_version"))),
        ("API Version", fmt_str(run.get("api_version"))),
        ("CLI Version", fmt_str(run.get("cli_version"))),
        ("Started At", fmt_dt(run.get("started_at"))),
        ("Finished At", fmt_dt(run.get("finished_at"))),
        ("Expires At", fmt_dt(run.get("expires_at"))),
        ("Pinned", fmt_bool(run.get("pinned"))),
        ("Test Run", fmt_bool(run.get("is_test"))),
        ("Valid", fmt_bool(run.get("is_valid"))),
        ("Invalidation Reason", fmt_str(run.get("invalidation_reason"))),
        ("Archive URI", fmt_str(run.get("archive_uri"))),
    ]
    for field, value in rows:
        table.add_row(field, value)

    _console.print(table)

    # System info summary
    si = run.get("system_info", {})
    if si:
        si_table = Table(title="System Info", show_lines=True)
        si_table.add_column("Field", style="cyan")
        si_table.add_column("Value")
        for k, v in si.items():
            si_table.add_row(str(k), str(v))
        _console.print(si_table)

    # config and result_summary are nested blobs that would wreck the table.
    _console.print("[dim]Use -j/--json for the full config and result_summary.[/dim]")
