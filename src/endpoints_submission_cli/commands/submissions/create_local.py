# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""submissions create-local command."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import click
import yaml
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from ...exceptions import APIError, ArchiveError, RunFolderError, SubmissionCheckError
from ...runs import api as runs_api
from ...runs.parser import build_archive
from ...submissions import api as subs_api
from ...submissions.builder import create_bundle_archive
from ..common import _console, _get_token, _run_submission_checker

__all__ = ["submissions_create_local"]


@click.command("create-local")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
@click.option(
    "--path",
    "submission_path",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to the already-assembled submission directory (org-level).",
)
@click.option(
    "--division",
    required=True,
    help="Submission division (standardized|serviced|rdi).",
)
@click.option(
    "--scenario",
    required=True,
    type=click.Choice(["cop", "con"], case_sensitive=False),
    help="Scenario: cop (Client-on-Premises) or con (Client-over-Network).",
)
@click.option(
    "--availability",
    required=True,
    help="Availability status (available|preview|rdi).",
)
@click.option("--early-publish", is_flag=True, default=False, help="Request early publication.")
@click.option(
    "--publication-cycle",
    default=None,
    help="Target publication cycle (e.g. 2025-04-C1).",
)
@click.option(
    "--target-availability-date",
    default=None,
    help="Target availability date (YYYY-MM-DD). Required for preview availability.",
)
@click.option(
    "--embargo-date",
    default=None,
    help="Embargo datetime in ISO 8601 format (e.g. 2025-12-01T00:00:00).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run the Submission Checker and print layout — exit without calling the API.",
)
def submissions_create_local(
    token: str | None,
    submission_path: Path,
    division: str,
    scenario: str,
    availability: str,
    early_publish: bool,
    publication_cycle: str | None,
    target_availability_date: str | None,
    embargo_date: str | None,
    dry_run: bool,
) -> None:
    """Create a submission from a pre-assembled local folder.

    Scans pareto/<system>/<model>/results/point_*/ directories for run data,
    registers each as a run, then creates and uploads the submission bundle.

    Workflow:
      1. Discover run result directories under --path.
      2. Run the Submission Checker on --path — abort on errors.
      3. (dry-run exits here)
      4. Register each result directory: parse payload, POST /runs, upload archive.
      5. POST /submissions with all collected run_ids.
      6. Bundle --path and upload.
      7. PATCH submission status to REVIEW_PENDING.
    """
    if not submission_path.is_dir():
        _console.print(f"[bold red]Error:[/bold red] {submission_path} is not a directory.")
        sys.exit(1)

    result_dirs = _find_result_dirs(submission_path)
    if not result_dirs:
        _console.print(
            f"[bold red]Error:[/bold red] No result directories found under "
            f"{submission_path / 'pareto'}."
        )
        sys.exit(1)

    # 2. Run Submission Checker
    _console.print("[cyan]Running Submission Checker…[/cyan]")
    try:
        _run_submission_checker(submission_path)
    except SubmissionCheckError as exc:
        _console.print(f"[bold red]Submission checker failed:[/bold red]\n{exc}")
        sys.exit(1)

    if dry_run:
        _console.print("[bold green]Checker passed.[/bold green] Folder layout:\n")
        for p in sorted(submission_path.rglob("*")):
            indent = "  " * (len(p.relative_to(submission_path).parts) - 1)
            label = p.name + ("/" if p.is_dir() else "")
            _console.print(f"[dim]{indent}[/dim]{label}")
        _console.print("\n[dim](dry-run: no submission created)[/dim]")
        return

    resolved_token = _get_token(token)

    # 4. Register runs
    _console.print("[cyan]Registering runs…[/cyan]")
    run_ids: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=_console,
        ) as progress:
            task = progress.add_task("Registering runs", total=len(result_dirs))
            for result_dir in result_dirs:
                progress.update(task, description=f"Registering [cyan]{result_dir.name}[/cyan]")
                try:
                    payload = _parse_result_dir(result_dir)
                except RunFolderError as exc:
                    progress.stop()
                    _console.print(f"[bold red]Parse error ({result_dir.name}):[/bold red] {exc}")
                    _rollback_runs(resolved_token, run_ids)
                    sys.exit(1)

                try:
                    run_out = runs_api.create_run(resolved_token, payload)
                except APIError as exc:
                    progress.stop()
                    _console.print(
                        f"[bold red]API error creating run ({result_dir.name}):[/bold red] {exc}"
                    )
                    _rollback_runs(resolved_token, run_ids)
                    sys.exit(1)

                run_id: str = run_out["id"]
                archive_path = build_archive(result_dir, tmp_path / f"{result_dir.name}.tar.gz")
                try:
                    runs_api.upload_run_archive(resolved_token, run_id, archive_path)
                except (APIError, ArchiveError, OSError) as exc:
                    progress.stop()
                    _console.print(
                        f"[bold red]Archive upload failed ({result_dir.name}):[/bold red] {exc}\n"
                        f"[yellow]Rolling back run {run_id}…[/yellow]"
                    )
                    with contextlib.suppress(APIError):
                        runs_api.delete_run_archive(resolved_token, run_id)
                    with contextlib.suppress(APIError):
                        runs_api.delete_run(resolved_token, run_id)
                    _rollback_runs(resolved_token, run_ids)
                    sys.exit(1)

                run_ids.append(run_id)
                progress.advance(task)

        # 5. POST /submissions
        payload_sub: dict[str, Any] = {
            "division": division,
            "scenario": scenario,
            "availability": availability,
            "run_ids": run_ids,
            "early_publish": early_publish,
        }
        if publication_cycle:
            payload_sub["publication_cycle"] = publication_cycle
        if target_availability_date:
            payload_sub["target_availability_date"] = target_availability_date
        if embargo_date:
            payload_sub["embargo_date"] = embargo_date

        try:
            sub_out = subs_api.create_submission(resolved_token, payload_sub)
        except APIError as exc:
            _console.print(f"[bold red]API error creating submission:[/bold red] {exc}")
            _rollback_runs(resolved_token, run_ids)
            sys.exit(1)

        submission_id: str = sub_out["id"]

        # 6. Upload bundle
        _console.print("[cyan]Uploading submission bundle…[/cyan]")
        _write_cli_metadata(submission_path)
        archive_path = create_bundle_archive(submission_path, tmp_path / "bundle.tar.gz")
        try:
            subs_api.upload_submission_archive(resolved_token, submission_id, archive_path)
        except APIError as exc:
            _console.print(
                f"[bold red]Bundle upload failed:[/bold red] {exc}\n"
                f"[yellow]Withdrawing submission {submission_id}…[/yellow]"
            )
            with contextlib.suppress(APIError):
                subs_api.withdraw_submission(resolved_token, submission_id)
            sys.exit(1)

        # 7. Update status
        try:
            subs_api.update_submission(
                resolved_token,
                submission_id,
                {"status": "REVIEW_PENDING"},
            )
        except APIError as exc:
            _console.print(f"[yellow]Warning: status update failed (retryable):[/yellow] {exc}")

    _console.print(f"[bold green]Submission created:[/bold green] {submission_id}")


def _write_cli_metadata(submission_path: Path) -> None:
    """Write a cli_metadata.json marker so reviewers can identify create-local bundles."""
    try:
        version = importlib.metadata.version("endpoints-submission-cli")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    meta = {
        "command": "create-local",
        "cli_version": version,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    (submission_path / "cli_metadata.json").write_text(json.dumps(meta, indent=2))


def _find_result_dirs(submission_path: Path) -> list[Path]:
    """Return all point result directories under pareto/<system>/<model>/results/."""
    result_dirs: list[Path] = []
    pareto_dir = submission_path / "pareto"
    if not pareto_dir.is_dir():
        return result_dirs
    for system_dir in sorted(pareto_dir.iterdir()):
        if not system_dir.is_dir():
            continue
        for model_dir in sorted(system_dir.iterdir()):
            results_dir = model_dir / "results"
            if not results_dir.is_dir():
                continue
            for point_dir in sorted(results_dir.iterdir()):
                if point_dir.is_dir() and (point_dir / "system_desc.json").exists():
                    result_dirs.append(point_dir)
    return result_dirs


def _parse_result_dir(path: Path) -> dict[str, Any]:
    """Parse a submission result directory into a RunCreate payload.

    Like parse_run_folder but reads results_summary.json
    (the submission folder name) instead of result_summary.json.
    """
    for fname in ("system_desc.json", "config.yaml", "results_summary.json"):
        if not (path / fname).exists():
            raise RunFolderError(f"Result directory {path} is missing {fname}")

    try:
        system_info = cast(dict[str, Any], json.loads((path / "system_desc.json").read_text()))
    except json.JSONDecodeError as exc:
        raise RunFolderError(f"Invalid JSON in system_desc.json: {exc}") from exc

    try:
        raw_cfg = yaml.safe_load((path / "config.yaml").read_text())
        if not isinstance(raw_cfg, dict):
            raise RunFolderError("config.yaml must be a YAML mapping")
        config: dict[str, Any] = raw_cfg
    except yaml.YAMLError as exc:
        raise RunFolderError(f"Invalid YAML in config.yaml: {exc}") from exc

    try:
        result_summary = cast(
            dict[str, Any],
            json.loads((path / "results_summary.json").read_text()),
        )
    except json.JSONDecodeError as exc:
        raise RunFolderError(f"Invalid JSON in results_summary.json: {exc}") from exc

    now_utc = datetime.now(tz=timezone.utc)
    duration_s: float = result_summary.get("duration_ns", 0) / 1e9
    finished_at = now_utc
    started_at = finished_at - timedelta(seconds=duration_s)

    return {
        "benchmark_version": result_summary.get("git_sha") or "unknown",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "system_info": system_info,
        "config": config,
        "result_summary": result_summary,
    }


def _rollback_runs(token: str, run_ids: list[str]) -> None:
    """Delete all previously created run records on failure."""
    for rid in run_ids:
        _console.print(f"[yellow]Rolling back run {rid}…[/yellow]")
        with contextlib.suppress(APIError):
            runs_api.delete_run_archive(token, rid)
        with contextlib.suppress(APIError):
            runs_api.delete_run(token, rid)
