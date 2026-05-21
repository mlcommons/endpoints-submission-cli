# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for github_ops module (subprocess calls mocked)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from endpoints_submission_cli.exceptions import GitHubError
from endpoints_submission_cli.github_ops import (
    _parse_pr_number,
    checkout_pr,
    close_pr,
    commit_and_push,
    create_pr,
    get_target_repo,
)


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.stdout = stdout
    cp.returncode = returncode
    return cp


@pytest.mark.unit
class TestGetTargetRepo:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLPERF_SUBMISSION_REPO", raising=False)
        repo = get_target_repo()
        assert "/" in repo  # org/repo format

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLPERF_SUBMISSION_REPO", "myorg/myrepo")
        assert get_target_repo() == "myorg/myrepo"


@pytest.mark.unit
class TestCreatePr:
    def test_returns_url_and_number(self) -> None:
        url = "https://github.com/mlcommons/repo/pull/42"
        with patch("subprocess.run", return_value=_completed(stdout=url)):
            pr_url, pr_number = create_pr("sub-123", "branch-sub-123", "org/repo")
        assert pr_url == url
        assert pr_number == 42

    def test_bad_output_raises(self) -> None:
        with patch("subprocess.run", return_value=_completed(stdout="not a url")):
            with pytest.raises(GitHubError, match="Unexpected"):
                create_pr("sub-123", "branch-123", "org/repo")

    def test_subprocess_error_raises(self) -> None:
        err = subprocess.CalledProcessError(1, ["gh"], stderr="auth error")
        with patch("subprocess.run", side_effect=err), pytest.raises(GitHubError):
            create_pr("sub-123", "branch-123", "org/repo")

    def test_command_not_found_raises(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(GitHubError, match="not found"):
                create_pr("sub-123", "branch-123", "org/repo")


@pytest.mark.unit
class TestCheckoutPr:
    def test_success(self, tmp_path: Path) -> None:
        with patch("subprocess.run", return_value=_completed()):
            checkout_pr(42, tmp_path)

    def test_failure_raises(self, tmp_path: Path) -> None:
        err = subprocess.CalledProcessError(1, ["gh"], stderr="not found")
        with patch("subprocess.run", side_effect=err), pytest.raises(GitHubError):
            checkout_pr(42, tmp_path)


@pytest.mark.unit
class TestCommitAndPush:
    def test_success(self, tmp_path: Path) -> None:
        with patch("subprocess.run", return_value=_completed()):
            commit_and_push(tmp_path, "test commit")

    def test_git_add_failure_raises(self, tmp_path: Path) -> None:
        err = subprocess.CalledProcessError(1, ["git"], stderr="error")
        with patch("subprocess.run", side_effect=err), pytest.raises(GitHubError):
            commit_and_push(tmp_path, "test commit")


@pytest.mark.unit
class TestClosePr:
    def test_success(self) -> None:
        with patch("subprocess.run", return_value=_completed()):
            close_pr(42, "org/repo")

    def test_failure_raises(self) -> None:
        err = subprocess.CalledProcessError(1, ["gh"], stderr="auth error")
        with patch("subprocess.run", side_effect=err), pytest.raises(GitHubError):
            close_pr(42, "org/repo")


@pytest.mark.unit
class TestParsePrNumber:
    def test_standard_url(self) -> None:
        assert _parse_pr_number("https://github.com/org/repo/pull/123") == 123

    def test_trailing_slash(self) -> None:
        assert _parse_pr_number("https://github.com/org/repo/pull/99/") == 99

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(GitHubError):
            _parse_pr_number("https://github.com/org/repo/pull/abc")
