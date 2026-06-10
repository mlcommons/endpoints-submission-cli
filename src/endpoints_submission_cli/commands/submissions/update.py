# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""submissions update command."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import click
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from ...exceptions import APIError, GitHubError, SubmissionBuildError, SubmissionCheckError
from ...runs import api as runs_api
from ...submissions import api as subs_api
from ...submissions import github as github_ops
from ...submissions.builder import build_submission_folder, create_bundle_archive
from ...submissions.formatters import print_submission_detail
from ..common import _console, _get_token, _run_submission_checker

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
@click.option("--target-availability-date", default=None, help="Target availability date (YYYY-MM-DD).")
@click.option("--publication-cycle", default=None, help="Publication cycle (e.g. 2025-04-C1).")
@click.option("--embargo-date", default=None, help="Embargo datetime in ISO 8601 format (e.g. 2025-12-01T00:00:00).")
def submissions_update(
    submission_id: str,
    token: str | None,
    run_ids: tuple[str, ...],
    target_availability_date: str | None,
    publication_cycle: str | None,
    embargo_date: str | None,
) -> None:
    """Update fields on an existing submission.

    Providing --run-ids triggers a full rebuild (download → build → checker → upload → PR update).
    All other flags are DB-only PATCHes with no rebuild.
    """
    resolved_token = _get_token(token)

    if not run_ids and target_availability_date is None and publication_cycle is None and embargo_date is None:
        _console.print("[yellow]Nothing to update — provide at least one field.[/yellow]")
        return

    # Date/metadata-only path: no rebuild needed
    if not run_ids:
        patch: dict = {}
        if target_availability_date is not None:
            patch["target_availability_date"] = target_availability_date
        if publication_cycle is not None:
            patch["publication_cycle"] = publication_cycle
        if embargo_date is not None:
            patch["embargo_date"] = embargo_date
        try:
            sub_out = subs_api.update_submission(
                resolved_token, submission_id, patch
            )
        except APIError as exc:
            _console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)
        print_submission_detail(sub_out)
        return

    # Run-IDs path: full rebuild
    desired_run_ids = list(run_ids)
    """
    target_repo = github_ops.get_target_repo()
    _console.print("[cyan]Checking GitHub prerequisites…[/cyan]")
    try:
        repo_ok, repo_warning = github_ops.check_prerequisites(target_repo)
    except GitHubError as exc:
        _console.print(f"[bold red]GitHub prerequisite check failed:[/bold red] {exc}")
        sys.exit(1)
    if not repo_ok:
        _console.print(f"[yellow]Warning:[/yellow] {repo_warning}")
    """
    try:
        current_sub = subs_api.get_submission(resolved_token, submission_id)
    except APIError as exc:
        _console.print(f"[bold red]Error fetching submission:[/bold red] {exc}")
        sys.exit(1)

    original_run_ids: list[str] = current_sub.get("run_ids", [])
    division: str = current_sub.get("division", "standardized")
    availability: str = current_sub.get("availability", "available")
    pr_number: int | None = current_sub.get("pr_number")

    added = [r for r in desired_run_ids if r not in original_run_ids]
    removed = [r for r in original_run_ids if r not in desired_run_ids]
    if added:
        _console.print(f"[cyan]Adding {len(added)} run(s): {', '.join(r[:8] for r in added)}…[/cyan]")
    if removed:
        _console.print(f"[cyan]Removing {len(removed)} run(s): {', '.join(r[:8] for r in removed)}…[/cyan]")

    if not added and not removed:
        if target_availability_date is not None or publication_cycle is not None or embargo_date is not None:
            metadata_patch: dict = {}
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
    patch: dict = {"run_ids": desired_run_ids}
    if target_availability_date is not None:
        patch["target_availability_date"] = target_availability_date
    if publication_cycle is not None:
        patch["publication_cycle"] = publication_cycle
    if embargo_date is not None:
        patch["embargo_date"] = embargo_date
    try:
        subs_api.update_submission(resolved_token, submission_id, patch)
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
                    dest = runs_api.download_run_archive(
                        resolved_token, rid, tmp_path / "archives"
                    )
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
            submission_dir = build_submission_folder(archives, division, availability, tmp_path / "bundle")
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
        repo_dir = None
        """
        # Build commit message before merge (needed by commit_and_push)
        _parts = []
        if added:
            _parts.append(f"add {', '.join(r[:8] for r in added)}")
        if removed:
            _parts.append(f"remove {', '.join(r[:8] for r in removed)}")
        _commit_msg = f"update: {'; '.join(_parts)} ({len(desired_run_ids)} runs total)"

        # Merge fresh build with existing PR branch content (fatal — rollback on failure)
        if pr_number:
            _console.print("[cyan]Preparing PR branch merge…[/cyan]")
            try:
                repo_dir, merged_org_dir = github_ops.prepare_pr_branch_merge(
                    submission_dir,
                    target_repo,
                    tmp_path / "gh",
                    branch=f"submission-{submission_id}",
                )
                upload_source = merged_org_dir
            except GitHubError as exc:
                _console.print(f"[bold red]PR branch merge failed:[/bold red] {exc}")
                _rollback_update(resolved_token, submission_id, original_run_ids)
                sys.exit(1)
        """
        # Upload merged bundle to blob storage
        _console.print("[cyan]Uploading submission bundle…[/cyan]")
        archive_path = create_bundle_archive(upload_source, tmp_path / "bundle.tar.gz")
        try:
            subs_api.upload_submission_archive(resolved_token, submission_id, archive_path)
        except APIError as exc:
            _console.print(f"[bold red]Bundle upload failed:[/bold red] {exc}")
            _rollback_update(resolved_token, submission_id, original_run_ids)
            sys.exit(1)

        # Push merged branch to GitHub (non-fatal)
        """
        if pr_number and repo_dir:
            _console.print("[cyan]Updating GitHub PR…[/cyan]")
            try:
                github_ops.commit_and_push(repo_dir, _commit_msg)
            except GitHubError as exc:
                _console.print(
                    f"[yellow]GitHub push failed (blob updated, DB updated):[/yellow] {exc}\n"
                    f"Re-run [bold]submissions update[/bold] to retry."
                )
        """
    _console.print(f"[bold green]Submission {submission_id} updated.[/bold green]")
