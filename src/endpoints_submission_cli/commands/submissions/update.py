# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""submissions update command."""

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
from ...submissions.builder import build_submission_folder, create_bundle_archive
from ...submissions.formatters import print_submission_detail
from ..common import (
    _console,
    _get_token,
    _run_submission_checker,
    _write_cli_metadata,
)

__all__ = ["submissions_update"]


def _rollback_update(token: str, submission_id: str, original_run_ids: list[str]) -> None:
    _console.print(f"[yellow]Rolling back: restoring {len(original_run_ids)} run(s)…[/yellow]")
    try:
        subs_api.update_submission(token, submission_id, {"run_ids": original_run_ids})
    except APIError as exc:
        _console.print(f"[bold red]Rollback failed:[/bold red] {exc}")


@click.command("update")
@click.option("--submission-id", required=True, help="Submission UUID.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
@click.option("--run-ids", "run_ids", multiple=True, help="Replace run UUID list. Repeatable.")
@click.option(
    "--target-availability-date", default=None, help="Target availability date (YYYY-MM-DD)."
)
@click.option("--publication-cycle", default=None, help="Publication cycle (e.g. 2025-04-C1).")
@click.option(
    "--embargo-date",
    default=None,
    help="Embargo datetime in ISO 8601 format (e.g. 2025-12-01T00:00:00).",
)
def submissions_update(
    submission_id: str,
    token: str | None,
    run_ids: tuple[str, ...],
    target_availability_date: str | None,
    publication_cycle: str | None,
    embargo_date: str | None,
) -> None:
    """Update fields on an existing submission.

    Providing --run-ids triggers a full rebuild (download → build → checker → upload).
    All other flags are DB-only PATCHes with no rebuild.

    Runs may only be *removed* this way. Submission Rules §8 no longer provide a
    post-submission window for adding measurement points, so a --run-ids list that
    would add one is rejected.
    """
    resolved_token = _get_token(token)

    if (
        not run_ids
        and target_availability_date is None
        and publication_cycle is None
        and embargo_date is None
    ):
        _console.print("[yellow]Nothing to update — provide at least one field.[/yellow]")
        return

    # Date/metadata-only path: no rebuild needed
    if not run_ids:
        patch: dict[str, Any] = {}
        if target_availability_date is not None:
            patch["target_availability_date"] = target_availability_date
        if publication_cycle is not None:
            patch["publication_cycle"] = publication_cycle
        if embargo_date is not None:
            patch["embargo_date"] = embargo_date
        try:
            sub_out = subs_api.update_submission(resolved_token, submission_id, patch)
        except APIError as exc:
            _console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)
        print_submission_detail(sub_out)
        return

    # Run-IDs path: full rebuild
    desired_run_ids = list(run_ids)
    try:
        current_sub = subs_api.get_submission(resolved_token, submission_id)
    except APIError as exc:
        _console.print(f"[bold red]Error fetching submission:[/bold red] {exc}")
        sys.exit(1)

    original_run_ids: list[str] = current_sub.get("run_ids", [])
    division: str = current_sub.get("division", "standardized")
    availability: str = current_sub.get("availability", "available")
    added = [r for r in desired_run_ids if r not in original_run_ids]
    removed = [r for r in original_run_ids if r not in desired_run_ids]
    if added:
        # Submission Rules §8 removed the post-submission update window that allowed
        # this (endpoints_policies 7fd3e89). Withdrawing a faulty point is still
        # permitted, so a --run-ids list that only drops runs still goes through.
        _console.print(
            "[bold red]Cannot add runs to an existing submission.[/bold red]\n"
            f"  Would add: {', '.join(r[:8] for r in added)}\n"
            "  MLPerf Endpoints Submission Rules §8 no longer provide a post-submission\n"
            "  window for adding measurement points. A submission's points are fixed at\n"
            "  creation; faulty points may be withdrawn, but none may be added.\n"
            "  To submit a different set of runs, create a new submission."
        )
        sys.exit(1)
    if removed:
        _console.print(
            f"[cyan]Removing {len(removed)} run(s): {', '.join(r[:8] for r in removed)}…[/cyan]"
        )

    if not added and not removed:
        if (
            target_availability_date is not None
            or publication_cycle is not None
            or embargo_date is not None
        ):
            metadata_patch: dict[str, Any] = {}
            if target_availability_date is not None:
                metadata_patch["target_availability_date"] = target_availability_date
            if publication_cycle is not None:
                metadata_patch["publication_cycle"] = publication_cycle
            if embargo_date is not None:
                metadata_patch["embargo_date"] = embargo_date
            try:
                sub_out = subs_api.update_submission(resolved_token, submission_id, metadata_patch)
            except APIError as exc:
                _console.print(f"[bold red]Error:[/bold red] {exc}")
                sys.exit(1)
            print_submission_detail(sub_out)
        else:
            _console.print("[yellow]Run list unchanged. Nothing to do.[/yellow]")
        return

    # PATCH DB with new run list (and any metadata fields) in one call
    run_patch: dict[str, Any] = {"run_ids": desired_run_ids}
    if target_availability_date is not None:
        run_patch["target_availability_date"] = target_availability_date
    if publication_cycle is not None:
        run_patch["publication_cycle"] = publication_cycle
    if embargo_date is not None:
        run_patch["embargo_date"] = embargo_date
    try:
        subs_api.update_submission(resolved_token, submission_id, run_patch)
    except APIError as exc:
        _console.print(f"[bold red]Error updating submission:[/bold red] {exc}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Download all desired archives
        _console.print(f"[cyan]Downloading {len(desired_run_ids)} run archive(s)…[/cyan]")
        archives: list[tuple[str, Path]] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=_console,
        ) as progress:
            task = progress.add_task("Downloading run archives", total=len(desired_run_ids))
            for rid in desired_run_ids:
                progress.update(task, description=f"Downloading [cyan]{rid[:8]}…[/cyan]")
                try:
                    dest = runs_api.download_run_archive(resolved_token, rid, tmp_path / "archives")
                except APIError as exc:
                    progress.stop()
                    _console.print(f"[bold red]Failed to download run {rid}:[/bold red] {exc}")
                    _rollback_update(resolved_token, submission_id, original_run_ids)
                    sys.exit(1)
                archives.append((rid, dest))
                progress.advance(task)

        # Assemble submission folder
        _console.print("[cyan]Assembling submission folder…[/cyan]")
        try:
            submission_dir = build_submission_folder(
                archives, division, availability, tmp_path / "bundle", submission_id
            )
        except SubmissionBuildError as exc:
            _console.print(f"[bold red]Build error:[/bold red] {exc}")
            _rollback_update(resolved_token, submission_id, original_run_ids)
            sys.exit(1)

        # Run Submission Checker
        _console.print("[cyan]Running Submission Checker…[/cyan]")
        try:
            _run_submission_checker(submission_dir)
        except SubmissionCheckError as exc:
            _console.print(f"[bold red]Submission checker failed:[/bold red]\n{exc}")
            _rollback_update(resolved_token, submission_id, original_run_ids)
            sys.exit(1)

        upload_source = submission_dir
        # Upload merged bundle to blob storage
        _console.print("[cyan]Uploading submission bundle…[/cyan]")
        _write_cli_metadata(submission_dir / submission_id, "update", current_sub)
        archive_path = create_bundle_archive(upload_source, tmp_path / "bundle.tar.gz")
        try:
            subs_api.upload_submission_archive(resolved_token, submission_id, archive_path)
        except APIError as exc:
            _console.print(f"[bold red]Bundle upload failed:[/bold red] {exc}")
            _rollback_update(resolved_token, submission_id, original_run_ids)
            sys.exit(1)

        # Push merged branch to GitHub (non-fatal)
    _console.print(f"[bold green]Submission {submission_id} updated.[/bold green]")
