# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for the MCP server that exposes the CLI's read-only commands."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import click
import pytest

from endpoints_submission_cli import mcp_server
from endpoints_submission_cli.main import app
from endpoints_submission_cli.mcp_server import CliError

_FIXTURES = Path(__file__).resolve().parents[2] / "test_submissions"


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class TestCheckSubmission:
    """Runs the real CLI in a subprocess, as the server does."""

    def test_passing_submission(self) -> None:
        report = mcp_server.check_submission(str(_FIXTURES / "valid_standardized"))
        assert report["results"]
        assert not [r for r in report["results"] if r["severity"] == "error"]

    def test_failing_submission_is_a_result_not_an_error(self) -> None:
        """The CLI exits 1 on a failing submission; the report is still the answer."""
        report = mcp_server.check_submission(str(_FIXTURES / "invalid_submission"))
        assert [r for r in report["results"] if r["severity"] == "error"]

    def test_strict_is_passed_through(self, mocker) -> None:
        run = mocker.patch("subprocess.run", return_value=_completed(0, "{}"))
        mcp_server.check_submission("/some/path", strict=True)
        args = run.call_args.args[0]
        assert args[:3] == [sys.executable, "-m", "endpoints_submission_cli"]
        assert "--strict" in args and args[-1] == "--json"


@pytest.mark.unit
class TestRunCli:
    def test_cli_failure_carries_its_message(self, mocker) -> None:
        mocker.patch("subprocess.run", return_value=_completed(1, stderr="Auth error: no token"))
        with pytest.raises(CliError, match="Auth error: no token"):
            mcp_server.list_runs()

    def test_non_json_output_is_an_error(self, mocker) -> None:
        mocker.patch("subprocess.run", return_value=_completed(0, "not json"))
        with pytest.raises(CliError, match="printed no JSON"):
            mcp_server.list_submissions()

    def test_timeout_is_an_error(self, mocker) -> None:
        mocker.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cli", 300))
        with pytest.raises(CliError, match="timed out"):
            mcp_server.list_runs()

    @pytest.mark.parametrize(
        ("call", "expected"),
        [
            (lambda: mcp_server.get_run("r-1"), ["runs", "get", "--run-id", "r-1"]),
            (
                lambda: mcp_server.get_submission("s-1"),
                ["submissions", "get", "--submission-id", "s-1"],
            ),
        ],
    )
    def test_ids_are_passed_as_flags(self, mocker, call, expected: list[str]) -> None:
        run = mocker.patch("subprocess.run", return_value=_completed(0, "{}"))
        call()
        assert run.call_args.args[0][3:] == [*expected, "--json"]


@pytest.mark.unit
class TestTextCommands:
    def test_both_streams_are_returned_without_the_upgrade_notice(self, mocker) -> None:
        mocker.patch(
            "subprocess.run",
            return_value=_completed(
                0,
                stdout="",
                stderr="Run pinned: r-1\n\nA new version of endpoints-submission-cli is"
                " available: 1.0 → 1.1\n(pip install --upgrade endpoints-submission-cli)\n",
            ),
        )
        assert mcp_server.pin_run("r-1") == {"output": "Run pinned: r-1"}

    def test_commands_cannot_wait_on_a_prompt(self, mocker) -> None:
        run = mocker.patch("subprocess.run", return_value=_completed(0))
        mcp_server.withdraw_submission("s-1")
        kwargs = run.call_args.kwargs
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["env"]["COLUMNS"] == "200"

    def test_failure_reports_the_cli_message(self, mocker) -> None:
        mocker.patch(
            "subprocess.run",
            return_value=_completed(1, stderr="Error deleting run: 404 not found"),
        )
        with pytest.raises(CliError, match="404 not found"):
            mcp_server.delete_run("r-1")

    def test_provisional_needs_explicit_confirmation(self, mocker) -> None:
        run = mocker.patch("subprocess.run", return_value=_completed(0))
        args = {
            "division": "standardized",
            "scenario": "cop",
            "availability": "available",
            "run_ids": ["r-1"],
            "provisional": True,
        }
        with pytest.raises(CliError, match="confirm_public_provisional"):
            mcp_server.create_submission(**args)
        run.assert_not_called()

        mcp_server.create_submission(**args, confirm_public_provisional=True)
        argv = run.call_args.args[0]
        assert "--provisional" in argv and "--yes" in argv

    def test_create_run_dry_run_returns_the_payload(self, endpoints_run_folder: Path) -> None:
        """Real subprocess: --dry-run prints JSON, though the command has no --json."""
        payload = mcp_server.create_run(str(endpoints_run_folder), test=True, dry_run=True)
        assert payload["is_test"] is True


def _leaf_commands() -> dict[tuple[str, ...], click.Command]:
    def walk(cmd: click.Command, path: tuple[str, ...]):
        if isinstance(cmd, click.Group):
            for name, sub in cmd.commands.items():
                yield from walk(sub, (*path, name))
        else:
            yield path, cmd

    return dict(walk(app, ()))


#: Each command, and calls to its tools that between them set every option.
_COVERAGE: dict[tuple[str, ...], list[tuple[Callable[..., Any], dict[str, Any]]]] = {
    ("check-submission",): [(mcp_server.check_submission, {"path": "p", "strict": True})],
    ("install-skill",): [
        (mcp_server.install_skill, {"project": True, "force": True}),
        (mcp_server.install_skill, {"dest": "d"}),
    ],
    ("runs", "list"): [(mcp_server.list_runs, {})],
    ("runs", "get"): [
        (mcp_server.get_run, {"run_id": "r"}),
        (mcp_server.download_run, {"run_id": "r", "directory": "d"}),
    ],
    ("runs", "create"): [
        (
            mcp_server.create_run,
            {"path": "p", "expires_at": "e", "pinned": True, "test": True, "dry_run": True},
        )
    ],
    ("runs", "delete"): [(mcp_server.delete_run, {"run_id": "r"})],
    ("runs", "pin"): [(mcp_server.pin_run, {"run_id": "r"})],
    ("runs", "unpin"): [(mcp_server.unpin_run, {"run_id": "r"})],
    ("submissions", "list"): [(mcp_server.list_submissions, {})],
    ("submissions", "get"): [
        (mcp_server.get_submission, {"submission_id": "s"}),
        (mcp_server.download_submission, {"submission_id": "s", "directory": "d"}),
    ],
    ("submissions", "create"): [
        (
            mcp_server.create_submission,
            {
                "division": "standardized",
                "scenario": "cop",
                "availability": "preview",
                "run_ids": ["r1", "r2"],
                "shared_src": ["s"],
                "shared_docs": ["d"],
                "provisional": True,
                "confirm_public_provisional": True,
                "target_availability_date": "2026-01-01",
                "embargo_date": "2026-01-01T00:00:00",
                "dry_run": True,
                "test": True,
            },
        )
    ],
    ("submissions", "update"): [
        (
            mcp_server.update_submission,
            {
                "submission_id": "s",
                "run_ids": ["r"],
                "target_availability_date": "2026-01-01",
                "embargo_date": "2026-01-01T00:00:00",
            },
        )
    ],
    ("submissions", "remove-run"): [
        (mcp_server.remove_run_from_submission, {"submission_id": "s", "run_id": "r"})
    ],
    ("submissions", "withdraw"): [(mcp_server.withdraw_submission, {"submission_id": "s"})],
}

#: Options no tool sets, and why. Anything else a command grows must reach a tool.
_NOT_EXPOSED = {
    "token",  # auth comes from PRISM_USER_API_TOKEN in the server's environment
    "quiet",  # check-submission: hides INFO rows in the table; the tool returns JSON
    "output",  # check-submission: writes the JSON report the tool already returns
}


@pytest.mark.unit
class TestEveryCommandIsATool:
    """Fails when the CLI gains a command or an option that no tool can reach."""

    def test_every_command_has_tools(self) -> None:
        assert set(_COVERAGE) == set(_leaf_commands())

    @pytest.mark.parametrize("path", sorted(_COVERAGE), ids=" ".join)
    def test_every_option_is_reachable(self, mocker, path: tuple[str, ...]) -> None:
        command = _leaf_commands()[path]
        by_flag = {
            flag: param.name
            for param in command.params
            for flag in (*param.opts, *param.secondary_opts)
        }
        run = mocker.patch("subprocess.run", return_value=_completed(0, "{}"))
        reached: set[str | None] = set()
        for tool, kwargs in _COVERAGE[path]:
            tool(**kwargs)
            argv = run.call_args.args[0][3:]
            assert tuple(argv[: len(path)]) == path
            reached |= {by_flag[a] for a in argv[len(path) :] if a in by_flag}
            reached |= {"path"} if path == ("check-submission",) else set()

        expected = {p.name for p in command.params} - _NOT_EXPOSED
        assert expected <= reached, f"{' '.join(path)}: no tool sets {expected - reached}"


@pytest.mark.unit
class TestServer:
    def test_tools_are_annotated(self) -> None:
        tools = {t.name: t.annotations for t in anyio.run(mcp_server.build_server().list_tools)}
        assert len(tools) == 16
        read_only = {name for name, a in tools.items() if a and a.read_only_hint}
        assert read_only == {
            "check_submission",
            "list_runs",
            "get_run",
            "list_submissions",
            "get_submission",
        }
        destructive = {name for name, a in tools.items() if a and a.destructive_hint}
        assert destructive == {
            "update_submission",
            "remove_run_from_submission",
            "withdraw_submission",
            "delete_run",
        }
        # Every tool says whether it reaches PRISM.
        assert all(a is not None and a.open_world_hint is not None for a in tools.values())

    def test_cli_errors_reach_the_client(self, mocker) -> None:
        """Only a ToolError's message survives MCPServer; anything else becomes a crash."""
        mocker.patch("subprocess.run", return_value=_completed(1, stderr="No API token provided."))
        server = mcp_server.build_server()

        async def call() -> Any:
            return await server.call_tool("list_runs", {})

        with pytest.raises(Exception, match="No API token provided"):
            anyio.run(call)

    def test_missing_extra_is_explained(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
        with pytest.raises(SystemExit, match=r"endpoints-submission-cli\[mcp\]"):
            mcp_server.main()
