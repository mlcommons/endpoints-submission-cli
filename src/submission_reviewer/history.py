# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Read the original version of a submission from git history.

A review PR starts with one commit holding the submission as submitted; every update
the submitter makes during review lands as a later commit on the same branch. Given the
first commit (``--base-ref``), this module recovers the submission as it was then, so
:mod:`submission_reviewer.checks.updates` can compare it with the current files.

Only reads are performed, and only through plumbing commands: ``ls-tree`` lists the
tree and ``cat-file --batch`` reads blobs. ``git archive`` is deliberately avoided: it
applies the repository's own ``.gitattributes`` (``export-ignore``, ``export-subst``),
which a submitter controls and could use to hide or rewrite files. Only the small
metadata and result files the checks read are extracted, so a large log in the
submission costs nothing.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from submission_checker import layout

__all__ = ["BaseSnapshot", "GitError", "changed_files", "extract_base", "find_original"]

#: The files :func:`submission_reviewer.context.build_context` reads.
_EXTRACTED = frozenset(
    {
        layout.POINT_YAML,
        layout.SYSTEM_DESC_JSON,
        layout.RESULT_SUMMARY_JSON,
        layout.SYSTEM_POWER_JSON,
        layout.CONFIG_YAML,
    }
)
#: Larger blobs are not extracted; none of the files above comes near it.
_MAX_BLOB_BYTES = 20 * 1024 * 1024
_TIMEOUT_S = 120
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})

#: Commit subjects the lifecycle manager writes on a review branch
#: (``lifecycle_manager.lifecycle``): the initial archive, possibly in several batches
#: with the same subject; a placeholder when the archive was not ready yet; and syncs.
_ADD_SUBJECT = re.compile(r"^Add submission \S+$")
_PLACEHOLDER_SUBJECT = re.compile(r"^Placeholder for \S+")
_SYNC_SUBJECT = re.compile(r"^Update submission \S+$")
#: Batches of one archive are committed back to back; a later sync of the same
#: submission comes on another lifecycle run, hours apart. Same-subject commits
#: further apart than this are separate pushes.
_BATCH_WINDOW_S = 600


class GitError(RuntimeError):
    """A git command failed, or the history does not hold what was asked for."""


@dataclass(frozen=True)
class BaseSnapshot:
    """Where the original submission was extracted to.

    Attributes:
        commit: The full commit hash *ref* resolved to.
        root: The extracted ``<submission_id>/`` directory.
    """

    commit: str
    root: Path


def _git(cwd: Path, *args: str, stdin: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            input=stdin,
            capture_output=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitError("git is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {args[0]} timed out") from exc
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip().splitlines()
        raise GitError(f"git {args[0]} failed: {message[-1] if message else result.returncode}")
    return result.stdout


def _locate(root: Path) -> tuple[Path, str]:
    """The repository top level, and *root*'s path inside it in POSIX form."""
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip())
    try:
        rel = root.resolve().relative_to(top.resolve())
    except ValueError as exc:
        raise GitError(f"{root} is not inside the repository at {top}") from exc
    return top, rel.as_posix() if rel.parts else "."


def _resolve(top: Path, ref: str) -> str:
    if not ref or ref.startswith("-"):
        raise GitError(f"invalid revision {ref!r}")
    try:
        return _git(top, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").decode().strip()
    except GitError as exc:
        raise GitError(f"{ref!r} is not a commit in this repository") from exc


def _safe_relative(path: str) -> PurePosixPath | None:
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("..", "") for part in pure.parts):
        return None
    return pure


def _read_blobs(top: Path, shas: list[str]) -> dict[str, bytes]:
    """Read blobs by hash with one ``cat-file --batch`` call."""
    if not shas:
        return {}
    out = _git(top, "cat-file", "--batch", stdin=("\n".join(shas) + "\n").encode())
    blobs: dict[str, bytes] = {}
    pos = 0
    for _ in shas:
        header_end = out.index(b"\n", pos)
        sha, kind, size = out[pos:header_end].decode().split(" ")
        start = header_end + 1
        if kind != "blob":
            raise GitError(f"object {sha} is a {kind}, not a blob")
        blobs[sha] = out[start : start + int(size)]
        pos = start + int(size) + 1  # the content is followed by a newline
    return blobs


def extract_base(root: Path, ref: str, dest: Path) -> BaseSnapshot:
    """Write *root*'s files as of *ref* under *dest* and return where they landed.

    Args:
        root: The current ``<submission_id>/`` directory, inside a git checkout.
        ref: A revision holding the original submission, e.g. the PR's first commit.
        dest: An empty directory to extract into.

    Raises:
        GitError: If git fails, *ref* is not a commit, or the submission is not in it.
    """
    top, rel = _locate(root)
    commit = _resolve(top, ref)
    listing = _git(top, "ls-tree", "-r", "-z", "--full-tree", commit, "--", rel)
    wanted: list[tuple[str, PurePosixPath]] = []
    for entry in filter(None, listing.split(b"\0")):
        meta, _, raw_path = entry.partition(b"\t")
        mode, kind, sha = meta.decode().split(" ")
        path = _safe_relative(raw_path.decode(errors="replace"))
        if kind != "blob" or mode not in _REGULAR_FILE_MODES or path is None:
            continue
        if path.name in _EXTRACTED:
            wanted.append((sha, path))
    if not wanted:
        raise GitError(f"the submission directory {rel} holds no results in {ref}")

    sizes = _git(
        top, "cat-file", "--batch-check", stdin=("\n".join(s for s, _ in wanted) + "\n").encode()
    ).decode()
    small = {
        line.split(" ")[0]
        for line in sizes.splitlines()
        if line.count(" ") == 2 and int(line.split(" ")[2]) <= _MAX_BLOB_BYTES
    }
    blobs = _read_blobs(top, sorted(small))
    for sha, path in wanted:
        if sha in blobs:
            target = dest.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blobs[sha])
    base_root = dest if rel == "." else dest.joinpath(*PurePosixPath(rel).parts)
    return BaseSnapshot(commit=commit, root=base_root)


def changed_files(root: Path, ref: str) -> list[tuple[str, str]]:
    """Files under *root* that differ between *ref* and the working tree.

    Returns:
        ``(status, path)`` pairs, *path* relative to *root* and *status* one of git's
        ``A`` (added), ``D`` (deleted), ``M`` (modified) or ``T`` (type changed).

    Raises:
        GitError: If git fails or *ref* is not a commit.
    """
    top, rel = _locate(root)
    commit = _resolve(top, ref)
    out = _git(
        top,
        "diff",
        "--name-status",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        "-z",
        commit,
        "--",
        rel,
    )
    fields = out.decode(errors="replace").split("\0")
    prefix = "" if rel == "." else rel + "/"
    changes: list[tuple[str, str]] = []
    for status, path in zip(fields[0::2], fields[1::2], strict=False):
        if status and path.startswith(prefix):
            changes.append((status[0], path[len(prefix) :]))
    return changes


def find_original(path: Path, pr_base: str) -> str:
    """The commit holding a review PR's original submission.

    The lifecycle manager opens a review PR by committing the submission archive on a
    branch made from the base, as one or more commits titled ``Add submission <id>``
    (a large archive is split into batches). Later syncs add ``Update submission <id>``
    commits, and people may add their own. When the archive was not ready, the first
    commit is a ``Placeholder for <id>`` and the archive arrives with the first sync.

    So the original is the last commit of the leading run of commits sharing the first
    real commit's subject, when that subject is the manager's ``Add`` or ``Update``,
    each committed within :data:`_BATCH_WINDOW_S` of the one before; otherwise it is
    the first commit after any placeholders. The time limit keeps a later sync, which
    has the same subject as the first one, out of the original.

    Args:
        path: Any directory inside the checkout.
        pr_base: The PR's base commit; the PR's commits are ``pr_base..HEAD``.

    Raises:
        GitError: If git fails, *pr_base* is not a commit, or the PR has no commits.
    """
    top, _ = _locate(path)
    base = _resolve(top, pr_base)
    out = _git(top, "log", "--reverse", "--format=%H%x1f%ct%x1f%s", f"{base}..HEAD").decode()
    commits: list[tuple[str, int, str]] = []
    for line in out.splitlines():
        parts = line.split("\x1f", 2)
        if len(parts) == 3 and parts[1].isdigit():
            commits.append((parts[0], int(parts[1]), parts[2]))
    while commits and _PLACEHOLDER_SUBJECT.match(commits[0][2]):
        commits = commits[1:]
    if not commits:
        raise GitError(f"the PR has no commits after {pr_base} besides placeholders")
    sha, when, subject = commits[0]
    if _ADD_SUBJECT.match(subject) or _SYNC_SUBJECT.match(subject):
        for next_sha, next_when, next_subject in commits[1:]:
            if next_subject != subject or next_when - when > _BATCH_WINDOW_S:
                break
            sha, when = next_sha, next_when
    return sha
