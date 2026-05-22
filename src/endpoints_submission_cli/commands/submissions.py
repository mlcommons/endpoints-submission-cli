# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for managing MLPerf submissions."""

from __future__ import annotations

import datetime
import sys
import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .. import api_client, github_ops
from ..exceptions import (
    APIError,
    AuthError,
    GitHubError,
    SubmissionBuildError,
    SubmissionCheckError,
)
from ..formatters import output_json, print_submission_detail, print_submissions_table
from ..submission_builder import build_submission_folder, create_bundle_archive

__all__ = ["submissions"]

_console = Console(stderr=True)


def _get_token(token: str | None) -> str:
    try:
        return api_client.get_token(token)
    except AuthError as exc:
        _console.print(f"[bold red]Auth error:[/bold red] {exc}")
        sys.exit(1)


_SEVERITY_STYLE = {
    "error": "bold red",
    "warning": "yellow",
    "info": "dim",
}


def _run_submission_checker(submission_dir: Path) -> None:
    """Run the SubmissionChecker on *submission_dir*; print results table; raise on errors."""
    from submission_checker.checker import SubmissionChecker

    report = SubmissionChecker(submission_dir).run()

    # --- write Rich table to log file ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path.cwd() / f"submission_checker_{timestamp}.log"

    table = Table(show_lines=True, expand=False)
    table.add_column("Rule", style="cyan", no_wrap=True)
    table.add_column("§ Ref", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Message")
    table.add_column("Path")

    for r in report.results:
        sev = r.severity.value
        style = _SEVERITY_STYLE.get(sev, "")
        table.add_row(
            r.rule,
            r.spec_ref,
            f"[{style}]{sev}[/{style}]" if style else sev,
            r.message,
            str(r.path) if r.path else "",
        )

    with open(log_path, "w", encoding="utf-8") as fh:
        file_console = Console(file=fh, no_color=True, width=220)
        file_console.print(
            f"Submission Checker Report — {timestamp}\n"
            f"Directory : {submission_dir}\n"
            f"Results   : {len(report.results)} checks"
            f" ({len(report.errors)} error(s), {len(report.warnings)} warning(s))\n"
        )
        file_console.print(table)

    _console.print(f"[dim]Checker report written to {log_path}[/dim]")

    errors = report.errors
    if errors:
        msgs = "\n".join(f"  [{e.rule}] {e.message}" for e in errors)
        raise SubmissionCheckError(f"Submission checker found {len(errors)} error(s):\n{msgs}")


# ---------------------------------------------------------------------------
# submissions list
# ---------------------------------------------------------------------------


@click.group(name="submissions")
def submissions() -> None:
    """Manage MLPerf submissions."""


@submissions.command("list")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
@click.option("-j", "--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
def submissions_list(token: str | None, as_json: bool) -> None:
    """List all submissions for the authenticated user."""
    resolved_token = _get_token(token)
    try:
        sub_list = api_client.list_submissions(resolved_token)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    if as_json:
        output_json(sub_list)
    else:
        print_submissions_table(sub_list)


# ---------------------------------------------------------------------------
# submissions create
# ---------------------------------------------------------------------------


@submissions.command("create")
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
    "--dry-run",
    is_flag=True,
    default=False,
    help="Assemble folder, run checker, print layout — exit without submitting.",
)
def submissions_create(
    token: str | None,
    division: str,
    availability: str,
    run_ids: tuple[str, ...],
    early_publish: bool,
    publication_cycle: str | None,
    target_availability_date: str | None,
    dry_run: bool,
) -> None:
    """Create a new submission from one or more registered runs.

    Workflow:
      1. Download run archives from the API.
      2. Assemble the submission folder structure.
      3. Run the Submission Checker — abort on errors.
      4. POST /submissions → get submission_id.
      5. Upload the submission bundle.
      6. Create a GitHub PR.
      7. PATCH submission with pr_url and pr_number.
    """
    run_ids_list = list(run_ids)
    resolved_token = _get_token(token)
    if not dry_run:
        target_repo = github_ops.get_target_repo()
        _console.print("[cyan]Checking GitHub prerequisites…[/cyan]")
        try:
            repo_ok, repo_warning = github_ops.check_prerequisites(target_repo)
        except GitHubError as exc:
            _console.print(f"[bold red]GitHub prerequisite check failed:[/bold red] {exc}")
            sys.exit(1)
        if not repo_ok:
            _console.print(f"[yellow]Warning:[/yellow] {repo_warning}")

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
                    dest = api_client.download_run_archive(
                        resolved_token, run_id, tmp_path / "archives"
                    )
                except APIError as exc:
                    progress.stop()
                    _console.print(
                        f"[bold red]Failed to download run {run_id}:[/bold red] {exc}"
                    )
                    sys.exit(1)
                archives.append((run_id, dest))
                progress.advance(task)

        # 2. Assemble submission folder
        _console.print("[cyan]Assembling submission folder…[/cyan]")
        try:
            submission_dir = build_submission_folder(archives, division, tmp_path / "bundle")
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
        payload: dict = {
            "division": division,
            "availability": availability,
            "run_ids": run_ids_list,
            "early_publish": early_publish,
        }
        if publication_cycle:
            payload["publication_cycle"] = publication_cycle
        if target_availability_date:
            payload["target_availability_date"] = target_availability_date

        try:
            sub_out = api_client.create_submission(resolved_token, payload)
        except APIError as exc:
            _console.print(f"[bold red]API error creating submission:[/bold red] {exc}")
            sys.exit(1)

        submission_id: str = sub_out["id"]

        # 5. Upload submission bundle
        _console.print("[cyan]Uploading submission bundle…[/cyan]")
        archive_path = create_bundle_archive(submission_dir, tmp_path / "bundle.tar.gz")
        try:
            api_client.upload_submission_archive(resolved_token, submission_id, archive_path)
        except APIError as exc:
            _console.print(
                f"[bold red]Bundle upload failed:[/bold red] {exc}\n"
                f"[yellow]Deleting submission record {submission_id}…[/yellow]"
            )
            try:
                api_client.withdraw_submission(resolved_token, submission_id)
            except APIError as rb_exc:
                _console.print(f"[bold red]Rollback failed:[/bold red] {rb_exc}")
            sys.exit(1)

        # 6. Push submission branch and create GitHub PR
        _console.print("[cyan]Creating GitHub PR…[/cyan]")
        branch = f"submission-{submission_id}"
        try:
            github_ops.prepare_submission_branch(
                submission_dir, branch, target_repo, tmp_path / "gh"  # type: ignore[possibly-undefined]
            )
            pr_url, pr_number = github_ops.create_pr(submission_id, branch, target_repo)
        except GitHubError as exc:
            _console.print(
                f"[bold red]PR creation failed:[/bold red] {exc}\n"
                f"[yellow]Rolling back submission {submission_id}…[/yellow]"
            )
            try:
                api_client.withdraw_submission(resolved_token, submission_id)
                _console.print("[green]Rollback successful — submission withdrawn.[/green]")
            except APIError as rb_exc:
                _console.print(
                    f"[bold red]Rollback also failed:[/bold red] {rb_exc}\n"
                    f"Orphaned submission ID: {submission_id}"
                )
            sys.exit(1)

        # 7. PATCH with pr_url, pr_number, status
        try:
            api_client.update_submission(
                resolved_token,
                submission_id,
                {"pr_url": pr_url, "pr_number": pr_number, "status": "REVIEW_PENDING"},
            )
        except APIError as exc:
            _console.print(f"[yellow]Warning: PATCH pr_url failed (retryable):[/yellow] {exc}")

    _console.print(
        f"[bold green]Submission created:[/bold green] {submission_id}\n"
        f"[bold green]PR:[/bold green] {pr_url}"
    )


# ---------------------------------------------------------------------------
# submissions get
# ---------------------------------------------------------------------------


@submissions.command("get")
@click.option("--submission-id", required=True, help="Submission UUID.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
@click.option("-j", "--json", "as_json", is_flag=True, default=False, help="Output raw JSON.")
def submissions_get(submission_id: str, token: str | None, as_json: bool) -> None:
    """Get full submission details including embedded runs."""
    resolved_token = _get_token(token)
    try:
        sub = api_client.get_submission(resolved_token, submission_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    if as_json:
        output_json(sub)
    else:
        print_submission_detail(sub)


# ---------------------------------------------------------------------------
# submissions update
# ---------------------------------------------------------------------------


@submissions.command("update")
@click.option("--submission-id", required=True, help="Submission UUID.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
@click.option("--run-ids", "run_ids", multiple=True, help="Replace run UUID list. Repeatable.")
@click.option("--target-availability-date", default=None, help="Target availability date (YYYY-MM-DD).")
def submissions_update(
    submission_id: str,
    token: str | None,
    run_ids: tuple[str, ...],
    target_availability_date: str | None,
) -> None:
    """Update run IDs or target availability date on an existing submission.

    Only the fields you provide are changed.
    """
    resolved_token = _get_token(token)

    patch: dict = {}
    if run_ids:
        patch["run_ids"] = list(run_ids)
    if target_availability_date is not None:
        patch["target_availability_date"] = target_availability_date

    if not patch:
        _console.print("[yellow]Nothing to update — provide at least one field.[/yellow]")
        return

    try:
        sub_out = api_client.update_submission(resolved_token, submission_id, patch)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    print_submission_detail(sub_out)


# ---------------------------------------------------------------------------
# submissions withdraw
# ---------------------------------------------------------------------------


@submissions.command("withdraw")
@click.option("--submission-id", required=True, help="Submission UUID.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
def submissions_withdraw(submission_id: str, token: str | None) -> None:
    """Withdraw a submission: mark WITHDRAWN, close its PR, delete its archive.

    Order of operations: DB update → close PR → delete archive.
    If PR close fails the submission is already WITHDRAWN; retry PR close manually.
    If archive deletion fails the orphaned URI is logged for garbage collection.
    """
    resolved_token = _get_token(token)
    target_repo = github_ops.get_target_repo()

    try:
        sub_out = api_client.withdraw_submission(resolved_token, submission_id)
    except APIError as exc:
        _console.print(f"[bold red]Error withdrawing submission:[/bold red] {exc}")
        sys.exit(1)

    pr_number = sub_out.get("pr_number")

    if pr_number:
        try:
            github_ops.close_pr(pr_number, target_repo)
        except GitHubError as exc:
            _console.print(
                f"[yellow]PR close failed (submission already WITHDRAWN):[/yellow] {exc}\n"
                f"Close manually: gh pr close {pr_number} --repo {target_repo}"
            )

    try:
        api_client.delete_submission_archive(resolved_token, submission_id)
    except APIError as exc:
        _console.print(f"[yellow]Archive deletion failed (orphaned):[/yellow] {exc}")

    _console.print(f"[bold green]Submission withdrawn:[/bold green] {submission_id}")


# ---------------------------------------------------------------------------
# submissions add-run
# ---------------------------------------------------------------------------


@submissions.command("add-run")
@click.option("--submission-id", required=True, help="Submission UUID.")
@click.option("--run-id", required=True, help="Run UUID to add.")
@click.option(
    "--token",
    envvar="PRISM_USER_API_TOKEN",
    default=None,
    help="PRISM API key (mlc_...).",
)
def submissions_add_run(submission_id: str, run_id: str, token: str | None) -> None:
    """Add a run to an existing submission and update the GitHub PR.

    Workflow:
      1. POST /submissions/{id}/runs/{run_id}
      2. Download all run archives (with progress)
      3. Rebuild submission folder
      4. Run Submission Checker — rollback and abort on errors
      5. Upload updated bundle
      6. Clone repo, check out PR branch, copy files, push
    """
    resolved_token = _get_token(token)
    target_repo = github_ops.get_target_repo()

    try:
        sub_out = api_client.add_run_to_submission(resolved_token, submission_id, run_id)
    except APIError as exc:
        _console.print(f"[bold red]API error adding run:[/bold red] {exc}")
        sys.exit(1)

    pr_number = sub_out.get("pr_number")
    all_run_ids: list[str] = sub_out.get("run_ids", [])

    _console.print(
        f"[cyan]Rebuilding submission with {len(all_run_ids)} run(s) "
        f"(added [bold]{run_id[:8]}…[/bold])…[/cyan]"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. Download all run archives
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
                    dest = api_client.download_run_archive(
                        resolved_token, rid, tmp_path / "archives"
                    )
                except APIError as exc:
                    progress.stop()
                    _console.print(
                        f"[bold red]Failed to download run {rid}:[/bold red] {exc}"
                    )
                    _rollback_add_run(resolved_token, submission_id, run_id)
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
            _rollback_add_run(resolved_token, submission_id, run_id)
            sys.exit(1)

        # 3. Run Submission Checker
        _console.print("[cyan]Running Submission Checker…[/cyan]")
        try:
            _run_submission_checker(submission_dir)
        except SubmissionCheckError as exc:
            _console.print(f"[bold red]Submission checker failed:[/bold red]\n{exc}")
            _rollback_add_run(resolved_token, submission_id, run_id)
            sys.exit(1)

        # 4. Upload updated bundle
        _console.print("[cyan]Uploading submission bundle…[/cyan]")
        archive_path = create_bundle_archive(submission_dir, tmp_path / "bundle.tar.gz")
        try:
            api_client.upload_submission_archive(resolved_token, submission_id, archive_path)
        except APIError as exc:
            _console.print(f"[bold red]Bundle upload failed:[/bold red] {exc}")
            _rollback_add_run(resolved_token, submission_id, run_id)
            sys.exit(1)

        # 5. Update PR branch (clone → checkout → copy files → push)
        if pr_number:
            _console.print("[cyan]Updating GitHub PR…[/cyan]")
            try:
                github_ops.update_pr_branch(
                    pr_number,
                    submission_dir,
                    target_repo,
                    tmp_path / "gh",
                    f"update: add run {run_id[:8]}",
                    branch=f"submission-{submission_id}",
                )
            except GitHubError as exc:
                _console.print(
                    f"[yellow]GitHub push failed (blob updated, DB updated):[/yellow] {exc}\n"
                    f"Re-run [bold]submissions add-run[/bold] to retry."
                )

    _console.print(f"[bold green]Run {run_id} added to submission {submission_id}.[/bold green]")


# ---------------------------------------------------------------------------
# submissions remove-run
# ---------------------------------------------------------------------------


@submissions.command("remove-run")
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
      1. DELETE /submissions/{id}/runs/{run_id}
      2. Download remaining run archives (with progress)
      3. Rebuild submission folder
      4. Run Submission Checker — rollback and abort on errors
      5. Upload updated bundle
      6. Clone repo, check out PR branch, copy files, push
    """
    resolved_token = _get_token(token)
    target_repo = github_ops.get_target_repo()

    try:
        sub_out = api_client.remove_run_from_submission(resolved_token, submission_id, run_id)
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
                    dest = api_client.download_run_archive(
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

        # 4. Upload updated bundle
        _console.print("[cyan]Uploading submission bundle…[/cyan]")
        archive_path = create_bundle_archive(submission_dir, tmp_path / "bundle.tar.gz")
        try:
            api_client.upload_submission_archive(resolved_token, submission_id, archive_path)
        except APIError as exc:
            _console.print(f"[bold red]Bundle upload failed:[/bold red] {exc}")
            _rollback_remove_run(resolved_token, submission_id, run_id)
            sys.exit(1)

        # 5. Update PR branch (clone → checkout → copy files → push)
        if pr_number:
            _console.print("[cyan]Updating GitHub PR…[/cyan]")
            try:
                github_ops.update_pr_branch(
                    pr_number,
                    submission_dir,
                    target_repo,
                    tmp_path / "gh",
                    f"update: remove run {run_id[:8]}",
                    branch=f"submission-{submission_id}",
                )
            except GitHubError as exc:
                _console.print(
                    f"[yellow]GitHub push failed (blob updated, DB updated):[/yellow] {exc}\n"
                    f"Re-run [bold]submissions remove-run[/bold] to retry."
                )

    _console.print(
        f"[bold green]Run {run_id} removed from submission {submission_id}.[/bold green]"
    )


# ---------------------------------------------------------------------------
# Rollback helpers
# ---------------------------------------------------------------------------


def _rollback_add_run(token: str, submission_id: str, run_id: str) -> None:
    _console.print(f"[yellow]Rolling back: removing run {run_id} from record…[/yellow]")
    try:
        api_client.remove_run_from_submission(token, submission_id, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Rollback failed:[/bold red] {exc}")


def _rollback_remove_run(token: str, submission_id: str, run_id: str) -> None:
    _console.print(f"[yellow]Rolling back: re-adding run {run_id} to record…[/yellow]")
    try:
        api_client.add_run_to_submission(token, submission_id, run_id)
    except APIError as exc:
        _console.print(f"[bold red]Rollback failed:[/bold red] {exc}")
