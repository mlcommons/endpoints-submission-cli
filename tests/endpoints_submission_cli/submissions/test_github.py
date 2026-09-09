# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for submissions.github module (subprocess calls mocked)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from endpoints_submission_cli.exceptions import GitHubError
from endpoints_submission_cli.submissions.github import (
    _parse_pr_number,
    check_prerequisites,
    checkout_pr,
    close_pr,
    commit_and_push,
    create_pr,
    get_target_repo,
    prepare_pr_branch_merge,
    update_pr_branch,
)


def _completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess:
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.stdout = stdout
    cp.stderr = stderr
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
class TestCheckPrerequisites:
    def test_success(self) -> None:
        with patch("subprocess.run", return_value=_completed()):
            ok, warning = check_prerequisites("org/repo")
        assert ok
        assert warning == ""

    def test_repo_unreachable_returns_false_with_warning(self) -> None:
        call_count = 0

        def _side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                return _completed(returncode=1, stdout="not found")
            return _completed()

        with patch("subprocess.run", side_effect=_side_effect):
            ok, warning = check_prerequisites("org/repo")
        assert not ok
        assert "org/repo" in warning

    def test_gh_not_installed_raises(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(GitHubError, match="not found"):
                check_prerequisites("org/repo")

    def test_auth_failure_raises(self) -> None:
        err = subprocess.CalledProcessError(1, ["gh"], stderr="not logged in")
        call_count = 0

        def _side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise err
            return _completed()

        with patch("subprocess.run", side_effect=_side_effect):
            with pytest.raises(GitHubError):
                check_prerequisites("org/repo")


@pytest.mark.unit
class TestPreparePrBranchMerge:
    def test_returns_repo_dir_and_org_dir(self, tmp_path: Path) -> None:
        submission_dir = tmp_path / "ORG"
        submission_dir.mkdir()
        work_dir = tmp_path / "work"
        (work_dir / "repo").mkdir(parents=True)

        with patch("subprocess.run", return_value=_completed()):
            repo_dir, org_dir = prepare_pr_branch_merge(
                submission_dir, "org/repo", work_dir, branch="submission-abc"
            )

        assert repo_dir == work_dir / "repo"
        assert org_dir == work_dir / "repo" / "ORG"

    def test_fresh_org_dir_copies_entire_tree(self, tmp_path: Path) -> None:
        submission_dir = tmp_path / "ORG"
        (submission_dir / "systems").mkdir(parents=True)
        work_dir = tmp_path / "work"
        (work_dir / "repo").mkdir(parents=True)

        with patch("subprocess.run", return_value=_completed()):
            _, org_dir = prepare_pr_branch_merge(
                submission_dir, "org/repo", work_dir, branch="sub-x"
            )

        assert org_dir.exists()

    def _fresh_build(self, root: Path, concurrencies=(16, 20)) -> Path:
        """A current-format builder output: §8.1's results/<system>/<model>/r<N>/."""
        for c in concurrencies:
            point = root / "results" / "H200x8" / "llama3_1-8b" / f"r{c}"
            point.mkdir(parents=True)
            (point / "point.yaml").write_text(f"concurrency: {c}\n")
            (point / "result_summary.json").write_text(json.dumps({"fresh": True}))
            (point / "system_desc.json").write_text(json.dumps({"from": "fresh"}))
            (point / "server_configs").mkdir()
            (point / "server_configs" / "engine.json").write_text("{}")
        impl = root / "src" / "trtllm"
        impl.mkdir(parents=True)
        (impl / "README.md").write_text("# trtllm\n")
        (root / "docs").mkdir()
        return root

    def _merged(self, tmp_path: Path, fresh: Path, repo_org: Path) -> Path:
        with patch("subprocess.run", return_value=_completed()):
            _, org_dir = prepare_pr_branch_merge(
                fresh, "org/repo", repo_org.parent.parent, branch="submission-abc"
            )
        return org_dir

    def test_merge_does_not_crash_on_a_v1_bundle(self, tmp_path: Path) -> None:
        """Regression: the merge walked v0.7's pareto/ tree and copied a top-level
        systems/ directory, so every amendment raised FileNotFoundError once the org
        directory already existed on the branch."""
        fresh = self._fresh_build(tmp_path / "ORG")
        repo_org = tmp_path / "work" / "repo" / "ORG"
        repo_org.mkdir(parents=True)

        org_dir = self._merged(tmp_path, fresh, repo_org)
        assert (org_dir / "results" / "H200x8" / "llama3_1-8b" / "r16").is_dir()

    def test_reviewer_edited_system_desc_is_preserved(self, tmp_path: Path) -> None:
        """Reviewers annotate system_desc.json on the branch; a rebuild must not undo it."""
        fresh = self._fresh_build(tmp_path / "ORG")
        repo_org = tmp_path / "work" / "repo" / "ORG"
        existing = repo_org / "results" / "H200x8" / "llama3_1-8b" / "r16"
        existing.mkdir(parents=True)
        (existing / "system_desc.json").write_text(json.dumps({"from": "reviewer"}))

        org_dir = self._merged(tmp_path, fresh, repo_org)
        merged = org_dir / "results" / "H200x8" / "llama3_1-8b" / "r16"
        assert json.loads((merged / "system_desc.json").read_text()) == {"from": "reviewer"}

    def test_measurement_files_are_replaced(self, tmp_path: Path) -> None:
        fresh = self._fresh_build(tmp_path / "ORG")
        repo_org = tmp_path / "work" / "repo" / "ORG"
        existing = repo_org / "results" / "H200x8" / "llama3_1-8b" / "r16"
        existing.mkdir(parents=True)
        (existing / "system_desc.json").write_text(json.dumps({"from": "reviewer"}))
        (existing / "result_summary.json").write_text(json.dumps({"fresh": False}))

        org_dir = self._merged(tmp_path, fresh, repo_org)
        merged = org_dir / "results" / "H200x8" / "llama3_1-8b" / "r16"
        assert json.loads((merged / "result_summary.json").read_text()) == {"fresh": True}

    def test_new_point_is_seeded_with_the_fresh_description(self, tmp_path: Path) -> None:
        """A point that has never been reviewed has no version worth preserving."""
        fresh = self._fresh_build(tmp_path / "ORG")
        repo_org = tmp_path / "work" / "repo" / "ORG"
        (repo_org / "results" / "H200x8" / "llama3_1-8b" / "r16").mkdir(parents=True)

        org_dir = self._merged(tmp_path, fresh, repo_org)
        seeded = org_dir / "results" / "H200x8" / "llama3_1-8b" / "r20" / "system_desc.json"
        assert json.loads(seeded.read_text()) == {"from": "fresh"}

    def test_points_dropped_from_the_build_are_removed(self, tmp_path: Path) -> None:
        """After remove-run, the branch must not keep the withdrawn point."""
        fresh = self._fresh_build(tmp_path / "ORG")
        repo_org = tmp_path / "work" / "repo" / "ORG"
        stale = repo_org / "results" / "H200x8" / "llama3_1-8b" / "r99"
        stale.mkdir(parents=True)
        (stale / "point.yaml").write_text("concurrency: 99\n")

        org_dir = self._merged(tmp_path, fresh, repo_org)
        assert not (org_dir / "results" / "H200x8" / "llama3_1-8b" / "r99").exists()

    def test_curves_dropped_from_the_build_are_removed(self, tmp_path: Path) -> None:
        fresh = self._fresh_build(tmp_path / "ORG")
        repo_org = tmp_path / "work" / "repo" / "ORG"
        stale = repo_org / "results" / "OldSystem" / "gpt-oss-120b" / "r8"
        stale.mkdir(parents=True)

        org_dir = self._merged(tmp_path, fresh, repo_org)
        assert not (org_dir / "results" / "OldSystem").exists()

    def test_shared_trees_are_replaced(self, tmp_path: Path) -> None:
        """src/ and docs/ are wholly builder-generated, so the fresh build wins."""
        fresh = self._fresh_build(tmp_path / "ORG")
        repo_org = tmp_path / "work" / "repo" / "ORG"
        old_impl = repo_org / "src" / "vllm"
        old_impl.mkdir(parents=True)
        (old_impl / "README.md").write_text("# stale\n")

        org_dir = self._merged(tmp_path, fresh, repo_org)
        assert (org_dir / "src" / "trtllm" / "README.md").is_file()
        assert not (org_dir / "src" / "vllm").exists()

    def test_point_subdirectories_are_replaced(self, tmp_path: Path) -> None:
        """server_configs/ is submitter-owned and comes from the fresh build."""
        fresh = self._fresh_build(tmp_path / "ORG")
        repo_org = tmp_path / "work" / "repo" / "ORG"
        old_cfg = repo_org / "results" / "H200x8" / "llama3_1-8b" / "r16" / "server_configs"
        old_cfg.mkdir(parents=True)
        (old_cfg / "stale.json").write_text("{}")

        org_dir = self._merged(tmp_path, fresh, repo_org)
        cfg = org_dir / "results" / "H200x8" / "llama3_1-8b" / "r16" / "server_configs"
        assert (cfg / "engine.json").is_file()
        assert not (cfg / "stale.json").exists()

    def test_clone_failure_raises(self, tmp_path: Path) -> None:
        err = subprocess.CalledProcessError(1, ["gh"], stderr="auth error")
        submission_dir = tmp_path / "ORG"
        submission_dir.mkdir()
        with patch("subprocess.run", side_effect=err):
            with pytest.raises(GitHubError):
                prepare_pr_branch_merge(
                    submission_dir, "org/repo", tmp_path / "work", branch="sub-x"
                )

    def test_fetch_failure_raises(self, tmp_path: Path) -> None:
        submission_dir = tmp_path / "ORG"
        submission_dir.mkdir()
        call_count = 0

        def _side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise subprocess.CalledProcessError(1, cmd, stderr="fetch error")
            return _completed()

        with patch("subprocess.run", side_effect=_side_effect):
            with pytest.raises(GitHubError):
                prepare_pr_branch_merge(
                    submission_dir, "org/repo", tmp_path / "work", branch="sub-x"
                )


@pytest.mark.unit
class TestUpdatePrBranch:
    def test_delegates_to_prepare_and_push(self, tmp_path: Path) -> None:
        submission_dir = tmp_path / "ORG"
        submission_dir.mkdir()
        fake_repo_dir = tmp_path / "repo"

        with patch(
            "endpoints_submission_cli.submissions.github.prepare_pr_branch_merge",
            return_value=(fake_repo_dir, fake_repo_dir / "ORG"),
        ) as mock_merge:
            with patch("endpoints_submission_cli.submissions.github.commit_and_push") as mock_push:
                update_pr_branch(
                    42, submission_dir, "org/repo", tmp_path / "work", "test msg", branch="sub-x"
                )

        mock_merge.assert_called_once_with(submission_dir, "org/repo", tmp_path / "work", "sub-x")
        mock_push.assert_called_once_with(fake_repo_dir, "test msg")

    def test_merge_failure_propagates(self, tmp_path: Path) -> None:
        submission_dir = tmp_path / "ORG"
        submission_dir.mkdir()
        with patch(
            "endpoints_submission_cli.submissions.github.prepare_pr_branch_merge",
            side_effect=GitHubError("clone failed"),
        ):
            with pytest.raises(GitHubError, match="clone failed"):
                update_pr_branch(
                    42, submission_dir, "org/repo", tmp_path / "work", "msg", branch="sub-x"
                )


@pytest.mark.unit
class TestParsePrNumber:
    def test_standard_url(self) -> None:
        assert _parse_pr_number("https://github.com/org/repo/pull/123") == 123

    def test_trailing_slash(self) -> None:
        assert _parse_pr_number("https://github.com/org/repo/pull/99/") == 99

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(GitHubError):
            _parse_pr_number("https://github.com/org/repo/pull/abc")
