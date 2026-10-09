# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
r"""MCP server exposing every CLI command as a tool.

Installed by the ``mcp`` extra and started by the ``endpoints-submission-mcp`` script::

    claude mcp add mlperf -e PRISM_USER_API_TOKEN=mlc_... \
        -- uvx --from 'endpoints-submission-cli[mcp]' endpoints-submission-mcp

Each tool runs the CLI itself under the interpreter this server runs in, so a tool does
exactly what the installed CLI would, and the server needs no knowledge of the
checker's or the API client's internals. Commands with a ``--json`` mode return the
parsed JSON. The rest print human-readable status, which is returned as text.

Every tool carries MCP annotations, so a client can tell the read-only tools from the
ones that change state on PRISM, and those from the ones that cannot be undone
(``withdraw_submission``, ``delete_run``). Clients use these to decide what to confirm
with the user before calling.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

__all__ = [
    "CliError",
    "build_server",
    "check_submission",
    "create_run",
    "create_submission",
    "delete_run",
    "download_run",
    "download_submission",
    "get_run",
    "get_submission",
    "install_skill",
    "list_runs",
    "list_submissions",
    "main",
    "pin_run",
    "remove_run_from_submission",
    "unpin_run",
    "update_submission",
    "withdraw_submission",
]

#: Seconds before a CLI call is abandoned. ``submissions create`` downloads every run,
#: assembles the bundle, runs the checker and uploads it, so this is generous.
_TIMEOUT_S = 900

#: The upgrade notice the CLI prints to stderr on exit. It is not part of any answer.
_UPGRADE_NOTICE = "A new version of endpoints-submission-cli is available"

_INSTRUCTIONS = """\
Tools for MLPerf Endpoints submissions, one per endpoints-submission-cli command.

check_submission validates a local submission folder against the rules' §9.1
automated checks. The other tools act on the PRISM API, authenticated by the
PRISM_USER_API_TOKEN environment variable the server was started with.

Tools that change state on PRISM are annotated as such. Before calling one, tell the
user exactly what it will do and get their agreement. withdraw_submission and
delete_run cannot be undone. create_run and create_submission accept dry_run=True,
which validates and shows the result without changing anything: use it first.
Pass test=True when the user is only trying things out."""


class CliError(RuntimeError):
    """The CLI failed, with its own error message."""


def _clean(text: str) -> str:
    """*text* without the upgrade notice, which is about the CLI, not the command."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _UPGRADE_NOTICE in line:
            # The notice is two lines: the announcement and the pip command.
            del lines[index : index + 2]
            break
    return "\n".join(lines).strip()


def _run(args: list[str], ok_codes: tuple[int, ...]) -> subprocess.CompletedProcess[str]:
    """Run ``endpoints-submission-cli <args>``, raising :class:`CliError` on failure.

    stdin is closed, so a command that would prompt fails at once instead of waiting
    on a terminal that is not there. The terminal is made wide and colourless so Rich
    does not wrap or style the status text returned to the client.
    """
    env = {**os.environ, "COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}
    try:
        done = subprocess.run(
            [sys.executable, "-m", "endpoints_submission_cli", *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise CliError(f"endpoints-submission-cli timed out after {_TIMEOUT_S}s") from exc
    if done.returncode not in ok_codes:
        message = _clean(f"{done.stdout}\n{done.stderr}")
        raise CliError(message or f"endpoints-submission-cli exited {done.returncode}")
    return done


def _run_cli(*args: str, ok_codes: tuple[int, ...] = (0,)) -> Any:
    """Run a command in its ``--json`` mode and return the parsed output."""
    done = _run([*args, "--json"], ok_codes)
    try:
        return json.loads(done.stdout)
    except ValueError as exc:
        raise CliError(f"endpoints-submission-cli printed no JSON: {done.stdout[:200]!r}") from exc


def _run_text(*args: str) -> dict[str, str]:
    """Run a command with no ``--json`` mode and return what it reported.

    These commands report on both streams — the result on stdout, status on stderr —
    so both are returned, in that order.
    """
    done = _run(list(args), (0,))
    return {"output": _clean(f"{done.stdout}\n{done.stderr}")}


def _repeat(flag: str, values: list[str] | None) -> list[str]:
    """``--flag a --flag b`` for a repeatable click option."""
    return [arg for value in values or [] for arg in (flag, value)]


def _optional(flag: str, value: str | None) -> list[str]:
    return [] if value is None else [flag, value]


def _switch(flag: str, on: bool) -> list[str]:
    return [flag] if on else []


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------


def check_submission(path: str, strict: bool = False) -> dict[str, Any]:
    """Validate a local submission folder against the §9.1 automated checks.

    Args:
        path: The ``<submission_id>/`` directory, or the organisation directory above
            a single submission.
        strict: Treat warnings as errors.

    Returns:
        The checker's report: ``results`` is a list of ``{rule, message, severity,
        path, spec_ref}``. A failing submission is a normal result, not an error.
    """
    args = ["check-submission", path, "--no-annotate", *_switch("--strict", strict)]
    # The command exits 1 when the submission fails; the report is still the answer.
    report: dict[str, Any] = _run_cli(*args, ok_codes=(0, 1))
    return report


def install_skill(
    project: bool = False, dest: str | None = None, force: bool = False
) -> dict[str, str]:
    """Install the mlperf-submissions Claude Code skill.

    Args:
        project: Install into ``./.claude/skills/`` under the server's working
            directory instead of ``~/.claude/skills/``.
        dest: Install into this skills directory instead. Not with ``project``.
        force: Replace an installed copy that differs from this version's, losing any
            local edits to it.
    """
    return _run_text(
        "install-skill",
        *_switch("--project", project),
        *_optional("--dest", dest),
        *_switch("--force", force),
    )


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def list_runs() -> Any:
    """List the authenticated user's runs on PRISM."""
    return _run_cli("runs", "list")


def get_run(run_id: str) -> Any:
    """Get the full details of one run.

    Args:
        run_id: The run's UUID.
    """
    return _run_cli("runs", "get", "--run-id", run_id)


def download_run(run_id: str, directory: str) -> dict[str, str]:
    """Download a run's archive (.tar.gz) into a local directory.

    Args:
        run_id: The run's UUID.
        directory: An existing local directory to save the archive in.
    """
    return _run_text("runs", "get", "--run-id", run_id, "--download-to", directory)


def create_run(
    path: str,
    expires_at: str | None = None,
    pinned: bool = False,
    test: bool = False,
    dry_run: bool = False,
) -> Any:
    """Register a run on PRISM from a local benchmark result folder.

    Args:
        path: The local run folder.
        expires_at: Expiry in ISO 8601 (e.g. ``2026-01-01T00:00:00``). Defaults to the
            server's policy.
        pinned: Pin the run at once, so it never expires.
        test: Mark it a test run, not a real results entry. Fixed at creation.
        dry_run: Parse the folder and return the payload without calling the API.

    Returns:
        With ``dry_run``, the parsed payload as JSON; otherwise what the CLI reported,
        including the new run's ID.
    """
    args = [
        "runs",
        "create",
        "--path",
        path,
        *_optional("--expires-at", expires_at),
        *_switch("--pinned", pinned),
        *_switch("--test", test),
    ]
    if dry_run:
        # --dry-run prints the payload as JSON, though the command has no --json flag.
        done = _run([*args, "--dry-run"], (0,))
        try:
            return json.loads(done.stdout)
        except ValueError as exc:
            raise CliError(f"runs create --dry-run printed no JSON: {done.stdout[:200]!r}") from exc
    return _run_text(*args)


def delete_run(run_id: str) -> dict[str, str]:
    """Delete a run and its stored archive. Cannot be undone.

    Args:
        run_id: The run's UUID.
    """
    return _run_text("runs", "delete", "--run-id", run_id)


def pin_run(run_id: str) -> dict[str, str]:
    """Pin a run so it never expires.

    Args:
        run_id: The run's UUID.
    """
    return _run_text("runs", "pin", "--run-id", run_id)


def unpin_run(run_id: str) -> dict[str, str]:
    """Unpin a run, restoring normal expiry.

    Args:
        run_id: The run's UUID.
    """
    return _run_text("runs", "unpin", "--run-id", run_id)


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------


def list_submissions() -> Any:
    """List the authenticated user's submissions on PRISM."""
    return _run_cli("submissions", "list")


def get_submission(submission_id: str) -> Any:
    """Get a submission's full details, including its runs.

    Args:
        submission_id: The submission's UUID.
    """
    return _run_cli("submissions", "get", "--submission-id", submission_id)


def download_submission(submission_id: str, directory: str) -> dict[str, str]:
    """Download a submission's archive (.tar.gz) into a local directory.

    Args:
        submission_id: The submission's UUID.
        directory: An existing local directory to save the archive in.
    """
    return _run_text(
        "submissions", "get", "--submission-id", submission_id, "--download-to", directory
    )


def create_submission(
    division: Literal["standardized", "serviced", "rdi"],
    scenario: Literal["cop", "con"],
    availability: Literal["available", "preview", "rdi"],
    run_ids: list[str],
    shared_src: list[str] | None = None,
    shared_docs: list[str] | None = None,
    provisional: bool = False,
    confirm_public_provisional: bool = False,
    target_availability_date: str | None = None,
    embargo_date: str | None = None,
    dry_run: bool = False,
    test: bool = False,
) -> dict[str, str]:
    """Create a submission from registered runs: download, assemble, check, upload.

    Args:
        division: The submission's division.
        scenario: ``cop`` (Client on Premises) or ``con`` (Client over Network).
        availability: The system's availability status.
        run_ids: The runs to include, by UUID.
        shared_src: Local directories merged into the bundle's shared ``src/``, each
            holding one or more ``<implementation>/`` folders.
        shared_docs: Local directories merged into the bundle's shared ``docs/``.
        provisional: Request provisional publication: the results become publicly
            viewable on the visualizer during the next cohort, with a "peer review
            pending" disclaimer.
        confirm_public_provisional: Must be true with ``provisional``. Set it only after
            the user has agreed to their results being made public before review.
        target_availability_date: ``YYYY-MM-DD``. Required for ``preview``.
        embargo_date: Embargo in ISO 8601 (e.g. ``2025-12-01T00:00:00``).
        dry_run: Assemble the bundle and run the checker, without submitting.
        test: Mark it a test submission, not a real results entry.
    """
    if provisional and not confirm_public_provisional:
        # The CLI asks this on a terminal; here the caller has to have asked it.
        raise CliError(
            "provisional publication makes these results publicly viewable on the"
            " visualizer during the next cohort, before peer review. Ask the user, and"
            " call again with confirm_public_provisional=true only if they agree."
        )
    return _run_text(
        "submissions",
        "create",
        "--division",
        division,
        "--scenario",
        scenario,
        "--availability",
        availability,
        *_repeat("--run-ids", run_ids),
        *_repeat("--shared-src", shared_src),
        *_repeat("--shared-docs", shared_docs),
        # --yes answers the CLI's own prompt, which confirm_public_provisional stands for.
        *(["--provisional", "--yes"] if provisional else []),
        *_optional("--target-availability-date", target_availability_date),
        *_optional("--embargo-date", embargo_date),
        *_switch("--dry-run", dry_run),
        *_switch("--test", test),
    )


def update_submission(
    submission_id: str,
    run_ids: list[str] | None = None,
    target_availability_date: str | None = None,
    embargo_date: str | None = None,
) -> dict[str, str]:
    """Update fields on an existing submission.

    Args:
        submission_id: The submission's UUID.
        run_ids: Replaces the submission's whole run list.
        target_availability_date: ``YYYY-MM-DD``.
        embargo_date: Embargo in ISO 8601 (e.g. ``2025-12-01T00:00:00``).
    """
    return _run_text(
        "submissions",
        "update",
        "--submission-id",
        submission_id,
        *_repeat("--run-ids", run_ids),
        *_optional("--target-availability-date", target_availability_date),
        *_optional("--embargo-date", embargo_date),
    )


def remove_run_from_submission(submission_id: str, run_id: str) -> dict[str, str]:
    """Remove one run from a submission.

    Args:
        submission_id: The submission's UUID.
        run_id: The UUID of the run to remove.
    """
    return _run_text(
        "submissions", "remove-run", "--submission-id", submission_id, "--run-id", run_id
    )


def withdraw_submission(submission_id: str) -> dict[str, str]:
    """Withdraw a submission. Cannot be undone.

    Args:
        submission_id: The submission's UUID.
    """
    return _run_text("submissions", "withdraw", "--submission-id", submission_id)


def build_server() -> MCPServer:
    """The server, with every tool registered. Imports ``mcp`` here, not at the top.

    The base install does not depend on ``mcp``, so importing this module must work
    without it; only building the server needs the extra.
    """
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    def reported(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Re-raise a CLI failure as ``ToolError``, whose message reaches the client.

        MCPServer treats any other exception as a crash and sends only "Error executing
        tool", which would hide "No API token provided" and the like from the model.
        """

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except CliError as exc:
                raise ToolError(str(exc)) from exc

        return wrapper

    def annotations(
        *, read_only: bool, remote: bool, destructive: bool = False, idempotent: bool = False
    ) -> ToolAnnotations:
        if read_only:
            return ToolAnnotations(read_only_hint=True, open_world_hint=remote)
        return ToolAnnotations(
            read_only_hint=False,
            destructive_hint=destructive,
            idempotent_hint=idempotent,
            open_world_hint=remote,
        )

    tools: list[tuple[Callable[..., Any], ToolAnnotations]] = [
        # Local
        (check_submission, annotations(read_only=True, remote=False)),
        (install_skill, annotations(read_only=False, remote=False, idempotent=True)),
        # PRISM, read-only
        (list_runs, annotations(read_only=True, remote=True)),
        (get_run, annotations(read_only=True, remote=True)),
        (list_submissions, annotations(read_only=True, remote=True)),
        (get_submission, annotations(read_only=True, remote=True)),
        # PRISM reads that write a local file
        (download_run, annotations(read_only=False, remote=True, idempotent=True)),
        (download_submission, annotations(read_only=False, remote=True, idempotent=True)),
        # PRISM writes. Additive ones are not destructive; replacing or removing is.
        (create_run, annotations(read_only=False, remote=True)),
        (pin_run, annotations(read_only=False, remote=True, idempotent=True)),
        (unpin_run, annotations(read_only=False, remote=True, idempotent=True)),
        (create_submission, annotations(read_only=False, remote=True)),
        (update_submission, annotations(read_only=False, remote=True, destructive=True)),
        (
            remove_run_from_submission,
            annotations(read_only=False, remote=True, destructive=True),
        ),
        (withdraw_submission, annotations(read_only=False, remote=True, destructive=True)),
        (delete_run, annotations(read_only=False, remote=True, destructive=True)),
    ]

    server = MCPServer("endpoints-submission-cli", instructions=_INSTRUCTIONS)
    for fn, hints in tools:
        server.tool(annotations=hints)(reported(fn))
    return server


def main() -> None:
    """Entry point called by the ``endpoints-submission-mcp`` script (stdio transport)."""
    try:
        server = build_server()
    except ModuleNotFoundError as exc:
        if exc.name is None or not exc.name.startswith("mcp"):
            raise
        sys.exit(
            "endpoints-submission-mcp needs the 'mcp' extra:"
            " pip install 'endpoints-submission-cli[mcp]'"
        )
    server.run("stdio")


if __name__ == "__main__":
    main()
