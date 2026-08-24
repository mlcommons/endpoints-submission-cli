# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""submissions create command."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import click
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from ...exceptions import APIError, SubmissionBuildError, SubmissionCheckError
from ...runs import api as runs_api
from ...submissions import api as subs_api

# from ...submissions import github as github_ops
from ...submissions.builder import (
    build_submission_folder,
    create_bundle_archive,
    set_submission_id,
)
from ..common import (
    _confirm_provisional,
    _console,
    _get_token,
    _run_submission_checker,
    _write_cli_metadata,
)

__all__ = ["submissions_create"]


@click.command("create")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
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
@click.option(
    "--run-ids",
    "run_ids",
    multiple=True,
    required=True,
    help="Run UUID(s) to include. Repeatable.",
)
@click.option(
    "--provisional",
    is_flag=True,
    default=False,
    help=(
        "Request provisional publication: results become publicly viewable on the "
        "visualizer during the next cohort with a 'peer review pending' disclaimer."
    ),
)
@click.option(
    "--yes",
    "-y",
    "assume_yes",
    is_flag=True,
    default=False,
    help="Skip the --provisional confirmation prompt (for non-interactive use).",
)
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
    help="Assemble folder, run checker, print layout — exit without submitting.",
)
@click.option(
    "--test",
    "is_test",
    is_flag=True,
    default=False,
    help="Mark the submission as a test submission (not a real results entry).",
)
def submissions_create(
    token: str | None,
    division: str,
    scenario: str,
    availability: str,
    run_ids: tuple[str, ...],
    provisional: bool,
    assume_yes: bool,
    publication_cycle: str | None,
    target_availability_date: str | None,
    embargo_date: str | None,
    dry_run: bool,
    is_test: bool,
) -> None:
    """Create a new submission from one or more registered runs.

    Workflow:
      1. Download run archives from the API.
      2. Assemble the submission folder structure.
      3. Run the Submission Checker — abort on errors.
      4. POST /submissions → get submission_id.
      5. Upload the submission bundle.
    """
    run_ids_list = list(run_ids)
    # Ask before any downloads — a declined prompt should cost nothing.
    if provisional and not dry_run:
        _confirm_provisional(assume_yes)
    resolved_token = _get_token(token)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. Download run archives
        _console.print("[cyan]Downloading run archives…[/cyan]")
        archives: list[tuple[str, Path]] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=_console,
        ) as progress:
            task = progress.add_task("Downloading run archives", total=len(run_ids_list))
            for run_id in run_ids_list:
                progress.update(task, description=f"Downloading [cyan]{run_id[:8]}…[/cyan]")
                try:
                    dest = runs_api.download_run_archive(
                        resolved_token, run_id, tmp_path / "archives"
                    )
                except APIError as exc:
                    progress.stop()
                    _console.print(f"[bold red]Failed to download run {run_id}:[/bold red] {exc}")
                    sys.exit(1)
                archives.append((run_id, dest))
                progress.advance(task)

        # 2. Assemble submission folder
        _console.print("[cyan]Assembling submission folder…[/cyan]")
        try:
            submission_dir = build_submission_folder(
                archives, division, availability, tmp_path / "bundle"
            )
        except SubmissionBuildError as exc:
            _console.print(f"[bold red]Build error:[/bold red] {exc}")
            sys.exit(1)

        # 3. Run Submission Checker
        _console.print("[cyan]Running Submission Checker…[/cyan]")
        try:
            _run_submission_checker(submission_dir)
        except SubmissionCheckError as exc:
            _console.print(f"[bold red]Submission checker failed:[/bold red]\n{exc}")
            sys.exit(1)

        if dry_run:
            _console.print("[bold green]Checker passed.[/bold green] Folder layout:\n")
            for p in sorted(submission_dir.rglob("*")):
                indent = "  " * (len(p.relative_to(submission_dir).parts) - 1)
                label = p.name + ("/" if p.is_dir() else "")
                _console.print(f"[dim]{indent}[/dim]{label}")
            _console.print("\n[dim](dry-run: no submission created)[/dim]")
            return

        # 4. POST /submissions
        payload: dict[str, Any] = {
            "division": division,
            "scenario": scenario,
            "availability": availability,
            "run_ids": run_ids_list,
            "is_test": is_test,
            # Wire field is still early_publish — the API schema has not been renamed.
            "early_publish": provisional,
        }
        if publication_cycle:
            payload["publication_cycle"] = publication_cycle
        if target_availability_date:
            payload["target_availability_date"] = target_availability_date
        if embargo_date:
            payload["embargo_date"] = embargo_date

        try:
            sub_out = subs_api.create_submission(resolved_token, payload)
        except APIError as exc:
            _console.print(f"[bold red]API error creating submission:[/bold red] {exc}")
            sys.exit(1)

        submission_id: str = sub_out["id"]

        # 5. Upload submission bundle
        # The tree was built under a placeholder because the id only exists once the
        # POST above succeeds; rename it before bundling so the archive carries the
        # required <organisation>/<submission_id>/ layout.
        _console.print("[cyan]Uploading submission bundle…[/cyan]")
        try:
            submission_root = set_submission_id(submission_dir, submission_id)
        except SubmissionBuildError as exc:
            _console.print(f"[bold red]Build error:[/bold red] {exc}")
            sys.exit(1)
        # The marker belongs to the submission, so it goes inside <submission_id>/ —
        # not beside it at the organisation level, which is shared across submissions.
        _write_cli_metadata(submission_root, "create", sub_out)
        # Bundle the organisation dir so the archive keeps the <org>/<submission_id>/ layout.
        archive_path = create_bundle_archive(submission_dir, tmp_path / "bundle.tar.gz")
        try:
            subs_api.upload_submission_archive(resolved_token, submission_id, archive_path)
        except APIError as exc:
            _console.print(
                f"[bold red]Bundle upload failed:[/bold red] {exc}\n"
                f"[yellow]Deleting submission record {submission_id}…[/yellow]"
            )
            try:
                subs_api.withdraw_submission(resolved_token, submission_id)
            except APIError as rb_exc:
                _console.print(f"[bold red]Rollback failed:[/bold red] {rb_exc}")
            sys.exit(1)

        # 7. PATCH with pr_url, pr_number, status
        try:
            subs_api.update_submission(
                resolved_token,
                submission_id,
                {"status": "REVIEW_PENDING"},
            )
        except APIError as exc:
            _console.print(f"[yellow]Warning: PATCH pr_url failed (retryable):[/yellow] {exc}")

    _console.print(
        f"[bold green]Submission created:[/bold green] {submission_id}"
        # f"[bold green]PR:[/bold green] {pr_url}"
    )
