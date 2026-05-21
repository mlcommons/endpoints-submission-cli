# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""GitHub CLI (gh) wrapper for submission PR operations.

All functions shell out to the ``gh`` CLI via subprocess. Unit tests should
mock ``subprocess.run`` / ``subprocess.check_output`` to avoid requiring a
live gh installation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .exceptions import GitHubError

__all__ = [
    "create_pr",
    "checkout_pr",
    "commit_and_push",
    "close_pr",
    "get_target_repo",
]

_DEFAULT_TARGET_REPO = "mlcommons/inference_results_rolling"


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

    Raises:
        GitHubError: If any git command fails.
    """
    _run(["git", "add", "."], cwd=cwd)
    _run(["git", "commit", "-m", message], cwd=cwd)
    _run(["git", "push"], cwd=cwd)


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
