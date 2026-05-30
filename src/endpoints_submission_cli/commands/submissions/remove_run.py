# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""submissions remove-run command."""

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
from ..common import _console, _get_token, _run_submission_checker

__all__ = ["submissions_remove_run"]


def _rollback_remove_run(token: str, submission_id: str, run_id: str) -> None:
    _console.print(f"[yellow]Rolling back: re-adding run {run_id} to record…[/yellow]")
    try:
        subs_api.add_run_to_submission(token, submission_id, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Rollback failed:[/bold red] {exc}")


@click.command("remove-run")
@click.option("--submission-id", required=True, help="Submission UUID.")
@click.option("--run-id", required=True, help="Run UUID to remove.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
def submissions_remove_run(submission_id: str, run_id: str, token: str | None) -> None:
    """Remove a run from an existing submission and update the GitHub PR.

    Workflow:
      1. Check GitHub prerequisites (gh installed and authenticated)
      2. DELETE /submissions/{id}/runs/{run_id}
      3. Download remaining run archives (with progress) — skipped if no runs remain
      4. Rebuild submission folder — skipped if no runs remain
      5. Run Submission Checker — rollback and abort on errors; skipped if no runs remain
      6. Upload updated bundle to blob storage — skipped if no runs remain
      7. Clone repo, check out existing PR branch, surgically update files, push — skipped if no runs remain
    """
    resolved_token = _get_token(token)
    target_repo = github_ops.get_target_repo()
    """
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
        sub_out = subs_api.remove_run_from_submission(resolved_token, submission_id, run_id)
    except APIError as exc:
        _console.print(f"[bold red]API error removing run:[/bold red] {exc}")
        sys.exit(1)

    pr_number = sub_out.get("pr_number")
    all_run_ids: list[str] = sub_out.get("run_ids", [])

    if not all_run_ids:
        _console.print("[yellow]No runs remain in submission after removal.[/yellow]")
        _console.print(
            f"[bold green]Run {run_id} removed from submission {submission_id}.[/bold green]"
        )
        return

    _console.print(
        f"[cyan]Rebuilding submission with {len(all_run_ids)} remaining run(s) "
        f"(removed [bold]{run_id[:8]}…[/bold])…[/cyan]"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. Download remaining run archives
        _console.print("[cyan]Downloading run archives…[/cyan]")
        archives: list[tuple[str, Path]] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=_console,
        ) as progress:
            task = progress.add_task("Downloading run archives", total=len(all_run_ids))
            for rid in all_run_ids:
                progress.update(task, description=f"Downloading [cyan]{rid[:8]}…[/cyan]")
                try:
                    dest = runs_api.download_run_archive(
                        resolved_token, rid, tmp_path / "archives"
                    )
                except APIError as exc:
                    progress.stop()
                    _console.print(
                        f"[bold red]Failed to download run {rid}:[/bold red] {exc}"
                    )
                    _rollback_remove_run(resolved_token, submission_id, run_id)
                    sys.exit(1)
                archives.append((rid, dest))
                progress.advance(task)

        # 2. Assemble submission folder
        _console.print("[cyan]Assembling submission folder…[/cyan]")
        division = sub_out.get("division", "standardized")
        try:
            submission_dir = build_submission_folder(archives, division, tmp_path / "bundle")
        except SubmissionBuildError as exc:
            _console.print(f"[bold red]Build error:[/bold red] {exc}")
            _rollback_remove_run(resolved_token, submission_id, run_id)
            sys.exit(1)

        # 3. Run Submission Checker
        _console.print("[cyan]Running Submission Checker…[/cyan]")
        try:
            _run_submission_checker(submission_dir)
        except SubmissionCheckError as exc:
            _console.print(f"[bold red]Submission checker failed:[/bold red]\n{exc}")
            _rollback_remove_run(resolved_token, submission_id, run_id)
            sys.exit(1)

        upload_source = submission_dir
        repo_dir = None
        """
        # 4. Merge fresh build with existing PR branch content (fatal — rollback on failure)
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
                _rollback_remove_run(resolved_token, submission_id, run_id)
                sys.exit(1)
        """

        # 5. Upload merged bundle to blob storage
        _console.print("[cyan]Uploading submission bundle…[/cyan]")
        archive_path = create_bundle_archive(upload_source, tmp_path / "bundle.tar.gz")
        try:
            subs_api.upload_submission_archive(resolved_token, submission_id, archive_path)
        except APIError as exc:
            _console.print(f"[bold red]Bundle upload failed:[/bold red] {exc}")
            _rollback_remove_run(resolved_token, submission_id, run_id)
            sys.exit(1)

        """
        # 6. Push merged branch to GitHub (non-fatal)
        if pr_number and repo_dir:
            _console.print("[cyan]Updating GitHub PR…[/cyan]")
            try:
                github_ops.commit_and_push(repo_dir, f"update: remove run {run_id[:8]}")
            except GitHubError as exc:
                _console.print(
                    f"[yellow]GitHub push failed (blob updated, DB updated):[/yellow] {exc}\n"
                    f"Re-run [bold]submissions remove-run[/bold] to retry."
                )
        """
        
    _console.print(
        f"[bold green]Run {run_id} removed from submission {submission_id}.[/bold green]"
    )
