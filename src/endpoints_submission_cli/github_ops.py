# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""GitHub CLI (gh) wrapper for submission PR operations.

All functions shell out to the ``gh`` CLI via subprocess. Unit tests should
mock ``subprocess.run`` / ``subprocess.check_output`` to avoid requiring a
live gh installation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .exceptions import GitHubError

__all__ = [
    "check_prerequisites",
    "prepare_submission_branch",
    "create_pr",
    "checkout_pr",
    "commit_and_push",
    "close_pr",
    "get_target_repo",
]

_DEFAULT_TARGET_REPO = "MLCommons-Systems/test-endpoints-submission-repo"


def get_target_repo() -> str:
    """Return the target GitHub repo from MLPERF_SUBMISSION_REPO env var or default."""
    return os.environ.get("MLPERF_SUBMISSION_REPO", _DEFAULT_TARGET_REPO)


def _run(
    cmd: list[str], cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a shell command, raising GitHubError on failure."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise GitHubError(
            f"Command {cmd[0]!r} failed (exit {exc.returncode}): {stderr}"
        ) from exc
    except FileNotFoundError as exc:
        raise GitHubError(
            f"Command {cmd[0]!r} not found — is it installed and on PATH?"
        ) from exc


def check_prerequisites(target_repo: str) -> tuple[bool, str]:
    """Verify gh is installed and authenticated; warn if target_repo is unreachable.

    Supports both token and SSH-based gh authentication.

    Returns:
        ``(repo_ok, warning)`` — ``repo_ok`` is False when the repo view failed,
        ``warning`` carries the raw gh error so the caller can display it.

    Raises:
        GitHubError: If gh is not installed or the user is not authenticated.
    """
    _run(["gh", "--version"])
    _run(["gh", "auth", "status"])
    result = _run(["gh", "repo", "view", target_repo], check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        return False, (
            f"Could not verify access to {target_repo!r}: {detail}\n"
            "Continuing — PR creation will fail if access is genuinely missing."
        )
    return True, ""


def prepare_submission_branch(
    submission_dir: Path,
    branch: str,
    target_repo: str,
    work_dir: Path,
) -> Path:
    """Clone target_repo, create branch, copy submission_dir in, commit and push.

    Works with both HTTPS-token and SSH authentication because it delegates
    the clone to ``gh repo clone``, which honours whichever auth gh is using.

    Args:
        submission_dir: Assembled org-level submission directory to copy in.
        branch: Branch name to create (e.g. ``"submission-<uuid>"``).
        target_repo: ``owner/repo`` slug.
        work_dir: Directory in which to create the local clone.

    Returns:
        Path to the cloned repository root.

    Raises:
        GitHubError: If any git/gh command fails.
    """
    repo_dir = work_dir / "repo"
    _run(["gh", "repo", "clone", target_repo, str(repo_dir), "--", "--depth", "1"])
    _run(["git", "checkout", "-b", branch], cwd=repo_dir)
    dest = repo_dir / submission_dir.name
    shutil.copytree(submission_dir, dest, dirs_exist_ok=True)
    commit_and_push(repo_dir, f"submission: {branch}")
    return repo_dir


def create_pr(
    submission_id: str,
    branch: str,
    target_repo: str,
    body: str = "",
) -> tuple[str, int]:
    """Create a GitHub PR for the given submission branch.

    Returns:
        Tuple of (pr_url, pr_number).

    Raises:
        GitHubError: If ``gh pr create`` fails.
    """
    pr_body = body or f"Submission {submission_id}"
    result = _run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            submission_id,
            "--body",
            pr_body,
            "--head",
            branch,
            "--repo",
            target_repo,
        ]
    )
    # gh pr create prints the PR URL to stdout
    pr_url = result.stdout.strip()
    if not pr_url.startswith("http"):
        raise GitHubError(f"Unexpected gh pr create output: {pr_url!r}")
    pr_number = _parse_pr_number(pr_url)
    return pr_url, pr_number


def checkout_pr(pr_number: int, cwd: Path) -> None:
    """Check out the PR branch in the given working directory.

    Raises:
        GitHubError: If ``gh pr checkout`` fails.
    """
    _run(["gh", "pr", "checkout", str(pr_number)], cwd=cwd)


def commit_and_push(cwd: Path, message: str) -> None:
    """Stage all changes, commit, and push to the current branch.

    Uses ``--set-upstream origin HEAD`` so new branches without a remote
    tracking ref are pushed correctly.

    Raises:
        GitHubError: If any git command fails.
    """
    _run(["git", "add", "."], cwd=cwd)
    _run(["git", "commit", "-m", message], cwd=cwd)
    _run(["git", "push", "--set-upstream", "origin", "HEAD"], cwd=cwd)


def close_pr(pr_number: int, target_repo: str) -> None:
    """Close a PR without merging (used for submission withdrawal).

    Raises:
        GitHubError: If ``gh pr close`` fails.
    """
    _run(
        [
            "gh",
            "pr",
            "close",
            str(pr_number),
            "--repo",
            target_repo,
        ]
    )


def _parse_pr_number(pr_url: str) -> int:
    """Extract the PR number from a GitHub PR URL."""
    # URL format: https://github.com/<owner>/<repo>/pull/<number>
    try:
        return int(pr_url.rstrip("/").split("/")[-1])
    except (ValueError, IndexError) as exc:
        raise GitHubError(f"Cannot parse PR number from URL: {pr_url!r}") from exc
