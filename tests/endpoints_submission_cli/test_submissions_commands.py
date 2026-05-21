# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the submissions command group."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from endpoints_submission_cli.exceptions import APIError, GitHubError
from endpoints_submission_cli.main import app
from tests.endpoints_submission_cli.conftest import (
    RUN_ID,
    SUBMISSION_ID,
    SUBMISSION_OUT,
)

TOKEN = "mlc_test"
_TOKEN_ARGS = ["--token", TOKEN]

_PR_URL = "https://github.com/mlcommons/submissions/pull/42"
_PR_NUMBER = 42


def _run_app(*args: str) -> None:
    """Invoke the app, treating SystemExit(0) as success."""
    with pytest.raises(SystemExit) as exc_info:
        app(list(args), exit_on_error=False)
    assert exc_info.value.code == 0


@pytest.mark.unit
class TestSubmissionsList:
    def test_list_success(self) -> None:
        with patch("endpoints_submission_cli.api_client.list_submissions", return_value=[SUBMISSION_OUT]):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app("submissions", "list", *_TOKEN_ARGS)

    def test_list_json_flag(self, capsys: pytest.CaptureFixture) -> None:
        with patch("endpoints_submission_cli.api_client.list_submissions", return_value=[SUBMISSION_OUT]):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app("submissions", "list", "-j", *_TOKEN_ARGS)
        out = capsys.readouterr().out
        assert SUBMISSION_ID in out

    def test_list_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.list_submissions", side_effect=APIError("fail")):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                with pytest.raises(SystemExit) as exc_info:
                    app(["submissions", "list", *_TOKEN_ARGS], exit_on_error=False)
                assert exc_info.value.code == 1

    def test_list_no_token_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRISM_USER_API_TOKEN", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            app(["submissions", "list"], exit_on_error=False)
        assert exc_info.value.code == 1


@pytest.mark.unit
class TestSubmissionsGet:
    def test_get_success(self) -> None:
        with patch("endpoints_submission_cli.api_client.get_submission", return_value=SUBMISSION_OUT):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app("submissions", "get", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)

    def test_get_json_flag(self, capsys: pytest.CaptureFixture) -> None:
        with patch("endpoints_submission_cli.api_client.get_submission", return_value=SUBMISSION_OUT):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app("submissions", "get", "--submission-id", SUBMISSION_ID, "-j", *_TOKEN_ARGS)
        out = capsys.readouterr().out
        assert SUBMISSION_ID in out

    def test_get_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.get_submission", side_effect=APIError("404")):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                with pytest.raises(SystemExit) as exc_info:
                    app(
                        ["submissions", "get", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS],
                        exit_on_error=False,
                    )
                assert exc_info.value.code == 1


@pytest.mark.unit
class TestSubmissionsUpdate:
    def test_update_status(self) -> None:
        updated = {**SUBMISSION_OUT, "status": "REVIEW_PENDING"}
        with patch("endpoints_submission_cli.api_client.update_submission", return_value=updated) as mock_patch:
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app(
                    "submissions", "update",
                    "--submission-id", SUBMISSION_ID,
                    "--status", "REVIEW_PENDING",
                    *_TOKEN_ARGS,
                )
        mock_patch.assert_called_once_with(TOKEN, SUBMISSION_ID, {"status": "REVIEW_PENDING"})

    def test_update_pr_fields(self) -> None:
        updated = {**SUBMISSION_OUT, "pr_url": _PR_URL, "pr_number": _PR_NUMBER}
        with patch("endpoints_submission_cli.api_client.update_submission", return_value=updated) as mock_patch:
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app(
                    "submissions", "update",
                    "--submission-id", SUBMISSION_ID,
                    "--pr-url", _PR_URL,
                    "--pr-number", str(_PR_NUMBER),
                    *_TOKEN_ARGS,
                )
        call_kwargs = mock_patch.call_args[0][2]
        assert call_kwargs["pr_url"] == _PR_URL
        assert call_kwargs["pr_number"] == _PR_NUMBER

    def test_update_no_fields_is_noop(self) -> None:
        with patch("endpoints_submission_cli.api_client.update_submission") as mock_patch:
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                # "Nothing to update" → returns without exit 0 from command; cyclopts still exits 0
                _run_app("submissions", "update", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)
        mock_patch.assert_not_called()

    def test_update_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.update_submission", side_effect=APIError("500")):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                with pytest.raises(SystemExit) as exc_info:
                    app(
                        [
                            "submissions", "update",
                            "--submission-id", SUBMISSION_ID,
                            "--status", "REVIEW_PENDING",
                            *_TOKEN_ARGS,
                        ],
                        exit_on_error=False,
                    )
                assert exc_info.value.code == 1


@pytest.mark.unit
class TestSubmissionsWithdraw:
    def test_withdraw_success(self) -> None:
        withdrawn = {**SUBMISSION_OUT, "status": "WITHDRAWN", "pr_number": _PR_NUMBER}
        with patch("endpoints_submission_cli.api_client.withdraw_submission", return_value=withdrawn):
            with patch("endpoints_submission_cli.api_client.delete_submission_archive"):
                with patch("endpoints_submission_cli.github_ops.close_pr"):
                    with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                        with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                            _run_app("submissions", "withdraw", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)

    def test_withdraw_no_pr_number(self) -> None:
        withdrawn = {**SUBMISSION_OUT, "status": "WITHDRAWN", "pr_number": None}
        with patch("endpoints_submission_cli.api_client.withdraw_submission", return_value=withdrawn):
            with patch("endpoints_submission_cli.api_client.delete_submission_archive"):
                with patch("endpoints_submission_cli.github_ops.close_pr") as mock_close:
                    with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                        with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                            _run_app("submissions", "withdraw", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)
        mock_close.assert_not_called()

    def test_withdraw_pr_close_failure_is_warning(self) -> None:
        withdrawn = {**SUBMISSION_OUT, "status": "WITHDRAWN", "pr_number": _PR_NUMBER}
        with patch("endpoints_submission_cli.api_client.withdraw_submission", return_value=withdrawn):
            with patch("endpoints_submission_cli.api_client.delete_submission_archive"):
                with patch("endpoints_submission_cli.github_ops.close_pr", side_effect=GitHubError("closed")):
                    with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                        with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                            # PR close failure is non-fatal → should exit 0
                            _run_app("submissions", "withdraw", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)

    def test_withdraw_archive_failure_is_warning(self) -> None:
        withdrawn = {**SUBMISSION_OUT, "status": "WITHDRAWN", "pr_number": None}
        with patch("endpoints_submission_cli.api_client.withdraw_submission", return_value=withdrawn):
            with patch(
                "endpoints_submission_cli.api_client.delete_submission_archive",
                side_effect=APIError("blob error"),
            ):
                with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                    with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                        # Archive deletion failure is non-fatal → should exit 0
                        _run_app("submissions", "withdraw", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)

    def test_withdraw_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.withdraw_submission", side_effect=APIError("locked")):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                    with pytest.raises(SystemExit) as exc_info:
                        app(
                            ["submissions", "withdraw", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS],
                            exit_on_error=False,
                        )
                    assert exc_info.value.code == 1


def _make_fake_archive(tmp_path: Path) -> Path:
    """Return a tiny tar.gz for use as a fake downloaded archive."""
    import tarfile

    folder = tmp_path / "fake_run"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "system_info.json").write_text("{}")
    archive = tmp_path / "fake.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)
    return archive


@pytest.mark.unit
class TestSubmissionsCreate:
    def test_create_success(self, tmp_path: Path) -> None:
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.api_client.download_run_archive", return_value=fake_archive):
                with patch(
                    "endpoints_submission_cli.commands.submissions.build_submission_folder",
                    return_value=fake_sub_dir,
                ):
                    with patch("endpoints_submission_cli.commands.submissions._run_submission_checker"):
                        with patch(
                            "endpoints_submission_cli.api_client.create_submission",
                            return_value=SUBMISSION_OUT,
                        ) as mock_create:
                            with patch(
                                "endpoints_submission_cli.commands.submissions.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch("endpoints_submission_cli.api_client.upload_submission_archive"):
                                    with patch(
                                        "endpoints_submission_cli.github_ops.create_pr",
                                        return_value=(_PR_URL, _PR_NUMBER),
                                    ):
                                        with patch("endpoints_submission_cli.api_client.update_submission"):
                                            with patch(
                                                "endpoints_submission_cli.github_ops.get_target_repo",
                                                return_value="org/repo",
                                            ):
                                                _run_app(
                                                    "submissions", "create",
                                                    "--division", "standardized",
                                                    "--availability", "available",
                                                    "--run-ids", RUN_ID,
                                                    *_TOKEN_ARGS,
                                                )
        mock_create.assert_called_once()

    def test_create_download_failure_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.api_client.download_run_archive",
                side_effect=APIError("not found"),
            ):
                with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                    with pytest.raises(SystemExit) as exc_info:
                        app(
                            [
                                "submissions", "create",
                                "--division", "standardized",
                                "--availability", "available",
                                "--run-ids", RUN_ID,
                                *_TOKEN_ARGS,
                            ],
                            exit_on_error=False,
                        )
                    assert exc_info.value.code == 1

    def test_create_checker_failure_exits_1(self, tmp_path: Path) -> None:
        from endpoints_submission_cli.exceptions import SubmissionCheckError

        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.api_client.download_run_archive", return_value=fake_archive):
                with patch(
                    "endpoints_submission_cli.commands.submissions.build_submission_folder",
                    return_value=fake_sub_dir,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions._run_submission_checker",
                        side_effect=SubmissionCheckError("1 error"),
                    ):
                        with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                            with pytest.raises(SystemExit) as exc_info:
                                app(
                                    [
                                        "submissions", "create",
                                        "--division", "standardized",
                                        "--availability", "available",
                                        "--run-ids", RUN_ID,
                                        *_TOKEN_ARGS,
                                    ],
                                    exit_on_error=False,
                                )
                            assert exc_info.value.code == 1

    def test_create_api_error_exits_1(self, tmp_path: Path) -> None:
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.api_client.download_run_archive", return_value=fake_archive):
                with patch(
                    "endpoints_submission_cli.commands.submissions.build_submission_folder",
                    return_value=fake_sub_dir,
                ):
                    with patch("endpoints_submission_cli.commands.submissions._run_submission_checker"):
                        with patch(
                            "endpoints_submission_cli.api_client.create_submission",
                            side_effect=APIError("500"),
                        ):
                            with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                                with pytest.raises(SystemExit) as exc_info:
                                    app(
                                        [
                                            "submissions", "create",
                                            "--division", "standardized",
                                            "--availability", "available",
                                            "--run-ids", RUN_ID,
                                            *_TOKEN_ARGS,
                                        ],
                                        exit_on_error=False,
                                    )
                                assert exc_info.value.code == 1

    def test_create_upload_failure_rolls_back(self, tmp_path: Path) -> None:
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.api_client.download_run_archive", return_value=fake_archive):
                with patch(
                    "endpoints_submission_cli.commands.submissions.build_submission_folder",
                    return_value=fake_sub_dir,
                ):
                    with patch("endpoints_submission_cli.commands.submissions._run_submission_checker"):
                        with patch(
                            "endpoints_submission_cli.api_client.create_submission",
                            return_value=SUBMISSION_OUT,
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch(
                                    "endpoints_submission_cli.api_client.upload_submission_archive",
                                    side_effect=APIError("upload failed"),
                                ):
                                    with patch(
                                        "endpoints_submission_cli.api_client.withdraw_submission"
                                    ) as mock_withdraw:
                                        with patch(
                                            "endpoints_submission_cli.github_ops.get_target_repo",
                                            return_value="org/repo",
                                        ):
                                            with pytest.raises(SystemExit) as exc_info:
                                                app(
                                                    [
                                                        "submissions", "create",
                                                        "--division", "standardized",
                                                        "--availability", "available",
                                                        "--run-ids", RUN_ID,
                                                        *_TOKEN_ARGS,
                                                    ],
                                                    exit_on_error=False,
                                                )
                                            assert exc_info.value.code == 1
                                            mock_withdraw.assert_called_once_with(TOKEN, SUBMISSION_ID)

    def test_create_pr_failure_is_non_fatal(self, tmp_path: Path) -> None:
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.api_client.download_run_archive", return_value=fake_archive):
                with patch(
                    "endpoints_submission_cli.commands.submissions.build_submission_folder",
                    return_value=fake_sub_dir,
                ):
                    with patch("endpoints_submission_cli.commands.submissions._run_submission_checker"):
                        with patch(
                            "endpoints_submission_cli.api_client.create_submission",
                            return_value=SUBMISSION_OUT,
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch("endpoints_submission_cli.api_client.upload_submission_archive"):
                                    with patch(
                                        "endpoints_submission_cli.github_ops.create_pr",
                                        side_effect=GitHubError("gh not found"),
                                    ):
                                        with patch(
                                            "endpoints_submission_cli.github_ops.get_target_repo",
                                            return_value="org/repo",
                                        ):
                                            # PR failure is non-fatal → exits 0
                                            _run_app(
                                                "submissions", "create",
                                                "--division", "standardized",
                                                "--availability", "available",
                                                "--run-ids", RUN_ID,
                                                *_TOKEN_ARGS,
                                            )


@pytest.mark.unit
class TestSubmissionsAddRun:
    _NEW_RUN_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    def _sub_with_runs(self, *run_ids: str) -> dict:
        return {**SUBMISSION_OUT, "run_ids": list(run_ids)}

    def test_add_run_success(self, tmp_path: Path) -> None:
        sub_out = self._sub_with_runs(RUN_ID, self._NEW_RUN_ID)
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.api_client.add_run_to_submission", return_value=sub_out):
                with patch("endpoints_submission_cli.api_client.download_run_archive", return_value=fake_archive):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch("endpoints_submission_cli.commands.submissions._run_submission_checker"):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch("endpoints_submission_cli.api_client.upload_submission_archive"):
                                    with patch("endpoints_submission_cli.github_ops.checkout_pr"):
                                        with patch("endpoints_submission_cli.github_ops.commit_and_push"):
                                            with patch(
                                                "endpoints_submission_cli.github_ops.get_target_repo",
                                                return_value="org/repo",
                                            ):
                                                _run_app(
                                                    "submissions", "add-run",
                                                    "--submission-id", SUBMISSION_ID,
                                                    "--run-id", self._NEW_RUN_ID,
                                                    *_TOKEN_ARGS,
                                                )

    def test_add_run_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.api_client.add_run_to_submission",
                side_effect=APIError("conflict"),
            ):
                with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                    with pytest.raises(SystemExit) as exc_info:
                        app(
                            [
                                "submissions", "add-run",
                                "--submission-id", SUBMISSION_ID,
                                "--run-id", self._NEW_RUN_ID,
                                *_TOKEN_ARGS,
                            ],
                            exit_on_error=False,
                        )
                    assert exc_info.value.code == 1

    def test_add_run_download_failure_rolls_back(self, tmp_path: Path) -> None:
        sub_out = self._sub_with_runs(RUN_ID, self._NEW_RUN_ID)

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.api_client.add_run_to_submission", return_value=sub_out):
                with patch(
                    "endpoints_submission_cli.api_client.download_run_archive",
                    side_effect=APIError("not found"),
                ):
                    with patch(
                        "endpoints_submission_cli.api_client.remove_run_from_submission"
                    ) as mock_rollback:
                        with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                            with pytest.raises(SystemExit) as exc_info:
                                app(
                                    [
                                        "submissions", "add-run",
                                        "--submission-id", SUBMISSION_ID,
                                        "--run-id", self._NEW_RUN_ID,
                                        *_TOKEN_ARGS,
                                    ],
                                    exit_on_error=False,
                                )
                            assert exc_info.value.code == 1
                            mock_rollback.assert_called_once_with(TOKEN, SUBMISSION_ID, self._NEW_RUN_ID)

    def test_add_run_github_failure_is_warning(self, tmp_path: Path) -> None:
        sub_out = self._sub_with_runs(RUN_ID, self._NEW_RUN_ID)
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.api_client.add_run_to_submission", return_value=sub_out):
                with patch("endpoints_submission_cli.api_client.download_run_archive", return_value=fake_archive):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch("endpoints_submission_cli.commands.submissions._run_submission_checker"):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch("endpoints_submission_cli.api_client.upload_submission_archive"):
                                    with patch(
                                        "endpoints_submission_cli.github_ops.checkout_pr",
                                        side_effect=GitHubError("gh failed"),
                                    ):
                                        with patch(
                                            "endpoints_submission_cli.github_ops.get_target_repo",
                                            return_value="org/repo",
                                        ):
                                            # GitHub push failure is non-fatal → exits 0
                                            _run_app(
                                                "submissions", "add-run",
                                                "--submission-id", SUBMISSION_ID,
                                                "--run-id", self._NEW_RUN_ID,
                                                *_TOKEN_ARGS,
                                            )


@pytest.mark.unit
class TestSubmissionsRemoveRun:
    _REMOVED_RUN_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"

    def test_remove_run_success(self, tmp_path: Path) -> None:
        sub_out = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.api_client.remove_run_from_submission", return_value=sub_out
            ):
                with patch("endpoints_submission_cli.api_client.download_run_archive", return_value=fake_archive):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch("endpoints_submission_cli.commands.submissions._run_submission_checker"):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch("endpoints_submission_cli.api_client.upload_submission_archive"):
                                    with patch("endpoints_submission_cli.github_ops.checkout_pr"):
                                        with patch("endpoints_submission_cli.github_ops.commit_and_push"):
                                            with patch(
                                                "endpoints_submission_cli.github_ops.get_target_repo",
                                                return_value="org/repo",
                                            ):
                                                _run_app(
                                                    "submissions", "remove-run",
                                                    "--submission-id", SUBMISSION_ID,
                                                    "--run-id", self._REMOVED_RUN_ID,
                                                    *_TOKEN_ARGS,
                                                )

    def test_remove_run_no_runs_left(self) -> None:
        sub_out = {**SUBMISSION_OUT, "run_ids": []}

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.api_client.remove_run_from_submission", return_value=sub_out
            ):
                with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                    # Empty run list → early return, no rebuild → exits 0
                    _run_app(
                        "submissions", "remove-run",
                        "--submission-id", SUBMISSION_ID,
                        "--run-id", self._REMOVED_RUN_ID,
                        *_TOKEN_ARGS,
                    )

    def test_remove_run_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.api_client.remove_run_from_submission",
                side_effect=APIError("not found"),
            ):
                with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                    with pytest.raises(SystemExit) as exc_info:
                        app(
                            [
                                "submissions", "remove-run",
                                "--submission-id", SUBMISSION_ID,
                                "--run-id", self._REMOVED_RUN_ID,
                                *_TOKEN_ARGS,
                            ],
                            exit_on_error=False,
                        )
                    assert exc_info.value.code == 1

    def test_remove_run_download_failure_rolls_back(self, tmp_path: Path) -> None:
        sub_out = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}

        with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.api_client.remove_run_from_submission", return_value=sub_out
            ):
                with patch(
                    "endpoints_submission_cli.api_client.download_run_archive",
                    side_effect=APIError("not found"),
                ):
                    with patch(
                        "endpoints_submission_cli.api_client.add_run_to_submission"
                    ) as mock_rollback:
                        with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                            with pytest.raises(SystemExit) as exc_info:
                                app(
                                    [
                                        "submissions", "remove-run",
                                        "--submission-id", SUBMISSION_ID,
                                        "--run-id", self._REMOVED_RUN_ID,
                                        *_TOKEN_ARGS,
                                    ],
                                    exit_on_error=False,
                                )
                            assert exc_info.value.code == 1
                            mock_rollback.assert_called_once_with(TOKEN, SUBMISSION_ID, self._REMOVED_RUN_ID)
