# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for managing MLPerf submissions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from rich.console import Console

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

submissions = App(name="submissions", help="Manage MLPerf submissions.")
_console = Console(stderr=True)


def _get_token(token: str | None) -> str:
    try:
        return api_client.get_token(token)
    except AuthError as exc:
        _console.print(f"[bold red]Auth error:[/bold red] {exc}")
        raise SystemExit(1) from None


def _run_submission_checker(submission_dir: Path) -> None:
    """Run the SubmissionChecker on *submission_dir*; raise on errors."""
    from submission_checker.checker import SubmissionChecker

    report = SubmissionChecker(submission_dir).run()
    errors = report.errors
    if errors:
        msgs = "\n".join(f"  [{e.rule}] {e.message}" for e in errors)
        raise SubmissionCheckError(f"Submission checker found {len(errors)} error(s):\n{msgs}")


# ---------------------------------------------------------------------------
# submissions list
# ---------------------------------------------------------------------------


@submissions.command(name="list")
def submissions_list(
    *,
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
    json: Annotated[
        bool,
        Parameter(name=["-j", "--json"], help="Output raw JSON."),
    ] = False,
) -> None:
    """List all submissions for the authenticated user."""
    resolved_token = _get_token(token)
    try:
        sub_list = api_client.list_submissions(resolved_token)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    if json:
        output_json(sub_list)
    else:
        print_submissions_table(sub_list)


# ---------------------------------------------------------------------------
# submissions create
# ---------------------------------------------------------------------------


@submissions.command(name="create")
def submissions_create(
    *,
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
    division: Annotated[str, Parameter(help="Submission division (standardized|serviced|rdi).")],
    availability: Annotated[str, Parameter(help="Availability status (available|preview|rdi).")],
    run_ids: Annotated[
        list[str],
        Parameter(name="--run-ids", help="Run UUID(s) to include. Repeatable."),
    ],
    early_publish: Annotated[
        bool,
        Parameter(name="--early-publish", help="Request early publication."),
    ] = False,
    publication_cycle: Annotated[
        str | None,
        Parameter(name="--publication-cycle", help="Target publication cycle (e.g. 2025-04-C1)."),
    ] = None,
    target_availability_date: Annotated[
        str | None,
        Parameter(
            name="--target-availability-date",
            help="Target availability date (YYYY-MM-DD). Required for preview availability.",
        ),
    ] = None,
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
    resolved_token = _get_token(token)
    target_repo = github_ops.get_target_repo()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. Download run archives
        _console.print("[cyan]Downloading run archives…[/cyan]")
        archives: list[tuple[str, Path]] = []
        for run_id in run_ids:
            try:
                dest = api_client.download_run_archive(
                    resolved_token, run_id, tmp_path / "archives"
                )
            except APIError as exc:
                _console.print(f"[bold red]Failed to download run {run_id}:[/bold red] {exc}")
                raise SystemExit(1) from None
            archives.append((run_id, dest))

        # 2. Assemble submission folder
        _console.print("[cyan]Assembling submission folder…[/cyan]")
        try:
            submission_dir = build_submission_folder(archives, division, tmp_path / "bundle")
        except SubmissionBuildError as exc:
            _console.print(f"[bold red]Build error:[/bold red] {exc}")
            raise SystemExit(1) from None

        # 3. Run Submission Checker
        _console.print("[cyan]Running Submission Checker…[/cyan]")
        try:
            _run_submission_checker(submission_dir)
        except SubmissionCheckError as exc:
            _console.print(f"[bold red]Submission checker failed:[/bold red]\n{exc}")
            raise SystemExit(1) from None

        # 4. POST /submissions
        payload: dict = {
            "division": division,
            "availability": availability,
            "run_ids": run_ids,
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
            raise SystemExit(1) from None

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
            raise SystemExit(1) from None

        # 6. Create GitHub PR
        _console.print("[cyan]Creating GitHub PR…[/cyan]")
        branch = f"submission-{submission_id}"
        try:
            pr_url, pr_number = github_ops.create_pr(submission_id, branch, target_repo)
        except GitHubError as exc:
            _console.print(
                f"[yellow]PR creation failed (submission {submission_id} exists in DB):"
                f"[/yellow] {exc}\n"
                "Retry with: endpoints-submission-cli submissions update "
                f"--submission-id {submission_id} --pr-url <url> --pr-number <n>"
            )
            _console.print(
                f"[bold green]Submission created (no PR yet):[/bold green] {submission_id}"
            )
            return

        # 7. PATCH with pr_url, pr_number
        try:
            api_client.update_submission(
                resolved_token,
                submission_id,
                {"pr_url": pr_url, "pr_number": pr_number},
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


@submissions.command(name="get")
def submissions_get(
    *,
    submission_id: Annotated[str, Parameter(name="--submission-id", help="Submission UUID.")],
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
    json: Annotated[
        bool,
        Parameter(name=["-j", "--json"], help="Output raw JSON."),
    ] = False,
) -> None:
    """Get full submission details including embedded runs."""
    resolved_token = _get_token(token)
    try:
        sub = api_client.get_submission(resolved_token, submission_id)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    if json:
        output_json(sub)
    else:
        print_submission_detail(sub)


# ---------------------------------------------------------------------------
# submissions update
# ---------------------------------------------------------------------------


@submissions.command(name="update")
def submissions_update(  # noqa: PLR0913
    *,
    submission_id: Annotated[str, Parameter(name="--submission-id", help="Submission UUID.")],
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
    status: Annotated[str | None, Parameter(name="--status")] = None,
    run_ids: Annotated[list[str] | None, Parameter(name="--run-ids")] = None,
    availability_qualified_at: Annotated[
        str | None, Parameter(name="--availability-qualified-at")
    ] = None,
    compliance_passed_at: Annotated[
        str | None, Parameter(name="--compliance-passed-at")
    ] = None,
    first_published_at: Annotated[
        str | None, Parameter(name="--first-published-at")
    ] = None,
    peer_review_started_at: Annotated[
        str | None, Parameter(name="--peer-review-started-at")
    ] = None,
    objection_resolution_started_at: Annotated[
        str | None, Parameter(name="--objection-resolution-started-at")
    ] = None,
    finalized_at: Annotated[str | None, Parameter(name="--finalized-at")] = None,
    pr_url: Annotated[str | None, Parameter(name="--pr-url")] = None,
    pr_number: Annotated[int | None, Parameter(name="--pr-number")] = None,
    archive_uri: Annotated[str | None, Parameter(name="--archive-uri")] = None,
    publication_cycle: Annotated[str | None, Parameter(name="--publication-cycle")] = None,
    target_availability_date: Annotated[
        str | None, Parameter(name="--target-availability-date")
    ] = None,
) -> None:
    """Update one or more fields on an existing submission.

    Only the fields you provide are changed.
    """
    resolved_token = _get_token(token)

    patch: dict = {}
    if status is not None:
        patch["status"] = status
    if run_ids is not None:
        patch["run_ids"] = run_ids
    if availability_qualified_at is not None:
        patch["availability_qualified_at"] = availability_qualified_at
    if compliance_passed_at is not None:
        patch["compliance_passed_at"] = compliance_passed_at
    if first_published_at is not None:
        patch["first_published_at"] = first_published_at
    if peer_review_started_at is not None:
        patch["peer_review_started_at"] = peer_review_started_at
    if objection_resolution_started_at is not None:
        patch["objection_resolution_started_at"] = objection_resolution_started_at
    if finalized_at is not None:
        patch["finalized_at"] = finalized_at
    if pr_url is not None:
        patch["pr_url"] = pr_url
    if pr_number is not None:
        patch["pr_number"] = pr_number
    if archive_uri is not None:
        patch["archive_uri"] = archive_uri
    if publication_cycle is not None:
        patch["publication_cycle"] = publication_cycle
    if target_availability_date is not None:
        patch["target_availability_date"] = target_availability_date

    if not patch:
        _console.print("[yellow]Nothing to update — provide at least one field.[/yellow]")
        return

    try:
        sub_out = api_client.update_submission(resolved_token, submission_id, patch)
    except APIError as exc:
        _console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    print_submission_detail(sub_out)


# ---------------------------------------------------------------------------
# submissions withdraw
# ---------------------------------------------------------------------------


@submissions.command(name="withdraw")
def submissions_withdraw(
    *,
    submission_id: Annotated[str, Parameter(name="--submission-id", help="Submission UUID.")],
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
) -> None:
    """Withdraw a submission: mark WITHDRAWN, close its PR, delete its archive.

    Order of operations: DB update → close PR → delete archive.
    If PR close fails the submission is already WITHDRAWN; retry PR close manually.
    If archive deletion fails the orphaned URI is logged for garbage collection.
    """
    resolved_token = _get_token(token)
    target_repo = github_ops.get_target_repo()

    # 1. DELETE /submissions → status WITHDRAWN
    try:
        sub_out = api_client.withdraw_submission(resolved_token, submission_id)
    except APIError as exc:
        _console.print(f"[bold red]Error withdrawing submission:[/bold red] {exc}")
        raise SystemExit(1) from None

    pr_number = sub_out.get("pr_number")

    # 2. Close PR (best-effort)
    if pr_number:
        try:
            github_ops.close_pr(pr_number, target_repo)
        except GitHubError as exc:
            _console.print(
                f"[yellow]PR close failed (submission already WITHDRAWN):[/yellow] {exc}\n"
                f"Close manually: gh pr close {pr_number} --repo {target_repo}"
            )

    # 3. Delete archive (best-effort)
    try:
        api_client.delete_submission_archive(resolved_token, submission_id)
    except APIError as exc:
        _console.print(f"[yellow]Archive deletion failed (orphaned):[/yellow] {exc}")

    _console.print(f"[bold green]Submission withdrawn:[/bold green] {submission_id}")


# ---------------------------------------------------------------------------
# submissions add-run
# ---------------------------------------------------------------------------


@submissions.command(name="add-run")
def submissions_add_run(
    *,
    submission_id: Annotated[str, Parameter(name="--submission-id", help="Submission UUID.")],
    run_id: Annotated[str, Parameter(name="--run-id", help="Run UUID to add.")],
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
) -> None:
    """Add a run to an existing submission and update the GitHub PR.

    Workflow:
      1. POST /submissions/{id}/runs/{run_id}
      2. gh pr checkout → pull latest PR branch
      3. Download new run archive
      4. Rebuild submission folder, run Submission Checker
      5. Upload updated bundle
      6. Commit and push to PR branch
      7. PATCH submission with new archive_uri
    """
    resolved_token = _get_token(token)

    # 1. Register run addition in API
    try:
        sub_out = api_client.add_run_to_submission(resolved_token, submission_id, run_id)
    except APIError as exc:
        _console.print(f"[bold red]API error adding run:[/bold red] {exc}")
        raise SystemExit(1) from None

    pr_number = sub_out.get("pr_number")
    all_run_ids: list[str] = sub_out.get("run_ids", [])

    _console.print(f"[cyan]Rebuilding submission bundle with {len(all_run_ids)} run(s)…[/cyan]")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Download all run archives
        archives: list[tuple[str, Path]] = []
        for rid in all_run_ids:
            try:
                dest = api_client.download_run_archive(
                    resolved_token, rid, tmp_path / "archives"
                )
            except APIError as exc:
                _console.print(f"[bold red]Failed to download run {rid}:[/bold red] {exc}")
                _rollback_add_run(resolved_token, submission_id, run_id)
                raise SystemExit(1) from None
            archives.append((rid, dest))

        division = sub_out.get("division", "standardized")
        try:
            submission_dir = build_submission_folder(archives, division, tmp_path / "bundle")
        except SubmissionBuildError as exc:
            _console.print(f"[bold red]Build error:[/bold red] {exc}")
            _rollback_add_run(resolved_token, submission_id, run_id)
            raise SystemExit(1) from None

        try:
            _run_submission_checker(submission_dir)
        except SubmissionCheckError as exc:
            _console.print(f"[bold red]Submission checker failed:[/bold red]\n{exc}")
            _rollback_add_run(resolved_token, submission_id, run_id)
            raise SystemExit(1) from None

        archive_path = create_bundle_archive(submission_dir, tmp_path / "bundle.tar.gz")
        try:
            api_client.upload_submission_archive(resolved_token, submission_id, archive_path)
        except APIError as exc:
            _console.print(f"[bold red]Bundle upload failed:[/bold red] {exc}")
            _rollback_add_run(resolved_token, submission_id, run_id)
            raise SystemExit(1) from None

        # Update PR if one exists
        if pr_number:
            try:
                github_ops.checkout_pr(pr_number, tmp_path)
                # Copy updated bundle content into the checked-out workspace
                # (the PR branch tracks the submission folder, not the tar.gz)
                github_ops.commit_and_push(
                    tmp_path, f"update: add run {run_id}"
                )
            except GitHubError as exc:
                _console.print(
                    f"[yellow]GitHub push failed (blob updated, DB updated):[/yellow] {exc}\n"
                    "Retry: git push on the PR branch."
                )

    _console.print(f"[bold green]Run {run_id} added to submission {submission_id}.[/bold green]")


# ---------------------------------------------------------------------------
# submissions remove-run
# ---------------------------------------------------------------------------


@submissions.command(name="remove-run")
def submissions_remove_run(
    *,
    submission_id: Annotated[str, Parameter(name="--submission-id", help="Submission UUID.")],
    run_id: Annotated[str, Parameter(name="--run-id", help="Run UUID to remove.")],
    token: Annotated[
        str | None,
        Parameter(env_var="PRISM_USER_API_TOKEN", help="PRISM API key (mlc_...)."),
    ] = None,
) -> None:
    """Remove a run from an existing submission and update the GitHub PR.

    Workflow mirrors add-run: removes the run from the API record, rebuilds
    the submission folder, re-runs the checker, uploads the updated bundle,
    and pushes a new commit to the PR branch.
    """
    resolved_token = _get_token(token)

    # 1. Remove run from API record
    try:
        sub_out = api_client.remove_run_from_submission(resolved_token, submission_id, run_id)
    except APIError as exc:
        _console.print(f"[bold red]API error removing run:[/bold red] {exc}")
        raise SystemExit(1) from None

    pr_number = sub_out.get("pr_number")
    all_run_ids: list[str] = sub_out.get("run_ids", [])

    if not all_run_ids:
        _console.print("[yellow]No runs remain in submission after removal.[/yellow]")
        _console.print(
            f"[bold green]Run {run_id} removed from submission {submission_id}.[/bold green]"
        )
        return

    _console.print(f"[cyan]Rebuilding submission bundle with {len(all_run_ids)} run(s)…[/cyan]")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        archives: list[tuple[str, Path]] = []
        for rid in all_run_ids:
            try:
                dest = api_client.download_run_archive(
                    resolved_token, rid, tmp_path / "archives"
                )
            except APIError as exc:
                _console.print(f"[bold red]Failed to download run {rid}:[/bold red] {exc}")
                _rollback_remove_run(resolved_token, submission_id, run_id)
                raise SystemExit(1) from None
            archives.append((rid, dest))

        division = sub_out.get("division", "standardized")
        try:
            submission_dir = build_submission_folder(archives, division, tmp_path / "bundle")
        except SubmissionBuildError as exc:
            _console.print(f"[bold red]Build error:[/bold red] {exc}")
            _rollback_remove_run(resolved_token, submission_id, run_id)
            raise SystemExit(1) from None

        try:
            _run_submission_checker(submission_dir)
        except SubmissionCheckError as exc:
            _console.print(f"[bold red]Submission checker failed:[/bold red]\n{exc}")
            _rollback_remove_run(resolved_token, submission_id, run_id)
            raise SystemExit(1) from None

        archive_path = create_bundle_archive(submission_dir, tmp_path / "bundle.tar.gz")
        try:
            api_client.upload_submission_archive(resolved_token, submission_id, archive_path)
        except APIError as exc:
            _console.print(f"[bold red]Bundle upload failed:[/bold red] {exc}")
            _rollback_remove_run(resolved_token, submission_id, run_id)
            raise SystemExit(1) from None

        if pr_number:
            try:
                github_ops.checkout_pr(pr_number, tmp_path)
                github_ops.commit_and_push(
                    tmp_path, f"update: remove run {run_id}"
                )
            except GitHubError as exc:
                _console.print(
                    f"[yellow]GitHub push failed (blob updated, DB updated):[/yellow] {exc}\n"
                    "Retry: git push on the PR branch."
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
