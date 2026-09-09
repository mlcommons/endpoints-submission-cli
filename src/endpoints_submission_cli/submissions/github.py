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

from submission_checker import layout

from ..exceptions import GitHubError

__all__ = [
    "check_prerequisites",
    "prepare_submission_branch",
    "prepare_pr_branch_merge",
    "update_pr_branch",
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
        raise GitHubError(f"Command {cmd[0]!r} failed (exit {exc.returncode}): {stderr}") from exc
    except FileNotFoundError as exc:
        raise GitHubError(f"Command {cmd[0]!r} not found — is it installed and on PATH?") from exc


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


def prepare_pr_branch_merge(
    submission_dir: Path,
    target_repo: str,
    work_dir: Path,
    branch: str,
) -> tuple[Path, Path]:
    """Clone target_repo, check out branch, and apply the surgical merge of submission_dir.

    Does NOT commit or push — call ``commit_and_push(repo_dir, message)`` afterward.

    Uses ``git fetch + git checkout`` instead of ``gh pr checkout`` to avoid a
    known incompatibility between shallow clones and ``gh pr checkout``'s tracking
    branch setup (``fatal: cannot set up tracking information``).

    Args:
        submission_dir: Assembled org-level submission directory (fresh build).
        target_repo: ``owner/repo`` slug.
        work_dir: Directory in which to create the local clone.
        branch: Name of the existing remote branch to check out.

    Returns:
        ``(repo_dir, merged_org_dir)`` — the clone root and the org-level directory
        inside it after the merge has been applied. ``merged_org_dir`` is the source
        to use for the blob storage archive so that blob storage and the GitHub PR
        branch always contain identical content.

    Raises:
        GitHubError: If any git/gh command fails.
    """
    repo_dir = work_dir / "repo"
    _run(["gh", "repo", "clone", target_repo, str(repo_dir), "--", "--depth", "1"])
    # Fetch only the PR branch (shallow) then create a local tracking branch.
    # This avoids the 'cannot set up tracking information' error that gh pr checkout
    # triggers on shallow clones.
    _run(["git", "fetch", "--depth", "1", "origin", branch], cwd=repo_dir)
    _run(["git", "checkout", "-b", branch, "FETCH_HEAD"], cwd=repo_dir)

    repo_org_dir = repo_dir / submission_dir.name  # e.g. repo_dir / "NVIDIA"

    if repo_org_dir.exists():
        _merge_submission_tree(submission_dir, repo_org_dir)
    else:
        # Org dir not yet on the PR branch — full copy (first push edge case).
        shutil.copytree(submission_dir, repo_org_dir)

    return repo_dir, repo_org_dir


def _merge_submission_tree(fresh_dir: Path, repo_dir: Path) -> None:
    """Apply a fresh build onto the copy already on the PR branch.

    Measurement artifacts are replaced from the fresh build; anything a reviewer edits
    on the branch is preserved. Reviewers annotate ``system_desc.json`` during review, so
    a rebuild must not overwrite it — but a *new* point has no reviewed version yet, so
    that one is seeded.

    Walks §8.1's tree (``results/<system>/<model>/r<N>/``). It previously walked v0.7's
    ``pareto/<system>/<model>/{points,results,accuracy}/`` and copied a top-level
    ``systems/`` directory; against current builder output the walk silently matched
    nothing and the ``systems/`` copy raised ``FileNotFoundError``, so every amendment
    (``submissions update``, ``add-run``, ``remove-run``) failed once the org directory
    existed on the branch.
    """
    fresh_results = fresh_dir / layout.RESULTS_DIR
    repo_results = repo_dir / layout.RESULTS_DIR

    # Shared trees are wholly builder-generated, so they are replaced outright.
    for shared in (layout.SRC_DIR, layout.DOCS_DIR):
        fresh_shared, repo_shared = fresh_dir / shared, repo_dir / shared
        if repo_shared.exists():
            shutil.rmtree(repo_shared)
        if fresh_shared.exists():
            shutil.copytree(fresh_shared, repo_shared)

    if not fresh_results.is_dir():
        if repo_results.exists():
            shutil.rmtree(repo_results)
        return

    # Drop curves the fresh build no longer contains (e.g. after remove-run).
    fresh_curves = {(s.name, m.name) for s, m in layout.iter_curves(fresh_results)}
    for repo_system, repo_model in list(layout.iter_curves(repo_results)):
        if (repo_system.name, repo_model.name) not in fresh_curves:
            shutil.rmtree(repo_model)
            if not any(repo_system.iterdir()):
                shutil.rmtree(repo_system)

    for _fresh_system, fresh_model in layout.iter_curves(fresh_results):
        rel = fresh_model.relative_to(fresh_results)
        repo_model = repo_results / rel
        repo_model.mkdir(parents=True, exist_ok=True)
        _merge_curve(fresh_model, repo_model)


def _merge_curve(fresh_model: Path, repo_model: Path) -> None:
    """Update one Pareto curve's point directories in place."""
    fresh_point_names = {p.name for p in layout.iter_point_dirs(fresh_model)}
    for repo_point in layout.iter_point_dirs(repo_model):
        if repo_point.name not in fresh_point_names:
            shutil.rmtree(repo_point)

    for fresh_point in layout.iter_point_dirs(fresh_model):
        repo_point = repo_model / fresh_point.name
        is_new_point = not repo_point.exists()
        repo_point.mkdir(parents=True, exist_ok=True)

        for src_path in sorted(fresh_point.iterdir()):
            dest = repo_point / src_path.name
            if src_path.is_dir():
                # e.g. server_configs/ — submitter-owned, replaced wholesale.
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src_path, dest)
            elif src_path.name != layout.SYSTEM_DESC_JSON:
                shutil.copy2(src_path, dest)

        # system_desc.json: preserve the reviewed version; seed only a new point.
        repo_sysdesc = repo_point / layout.SYSTEM_DESC_JSON
        if is_new_point or not repo_sysdesc.exists():
            fresh_sysdesc = fresh_point / layout.SYSTEM_DESC_JSON
            if fresh_sysdesc.exists():
                shutil.copy2(fresh_sysdesc, repo_sysdesc)


def update_pr_branch(
    pr_number: int,
    submission_dir: Path,
    target_repo: str,
    work_dir: Path,
    message: str,
    branch: str,
) -> None:
    """Clone target_repo, check out an existing PR branch, merge, commit, and push.

    Delegates merge logic to ``prepare_pr_branch_merge``; call that directly when
    you need the merged content before uploading to blob storage.

    Args:
        pr_number: GitHub PR number (kept for call-site clarity).
        submission_dir: Assembled org-level submission directory to copy in.
        target_repo: ``owner/repo`` slug.
        work_dir: Directory in which to create the local clone.
        message: Git commit message.
        branch: Name of the existing remote branch to check out.

    Raises:
        GitHubError: If any git/gh command fails.
    """
    repo_dir, _ = prepare_pr_branch_merge(submission_dir, target_repo, work_dir, branch)
    commit_and_push(repo_dir, message)


def checkout_pr(pr_number: int, cwd: Path) -> None:
    """Check out the PR branch in the given working directory.

    Raises:
        GitHubError: If ``gh pr checkout`` fails.
    """
    _run(["gh", "pr", "checkout", str(pr_number)], cwd=cwd)


def commit_and_push(cwd: Path, message: str) -> None:
    """Stage all changes, commit (if any), and push to the current branch.

    Uses ``--set-upstream origin HEAD`` so new branches without a remote
    tracking ref are pushed correctly. Skips the commit when the working tree
    is clean after staging (e.g. re-building an identical submission).

    Raises:
        GitHubError: If any git command fails.
    """
    _run(["git", "add", "."], cwd=cwd)
    status = _run(["git", "status", "--porcelain"], cwd=cwd)
    if status.stdout.strip():
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
