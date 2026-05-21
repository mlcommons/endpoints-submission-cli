# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the submissions command group."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

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

_runner = CliRunner()


def _run_app(*args: str) -> None:
    """Invoke the app and assert exit code 0."""
    result = _runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output


@pytest.mark.unit
class TestSubmissionsList:
    def test_list_success(self) -> None:
        with patch("endpoints_submission_cli.api_client.list_submissions", return_value=[SUBMISSION_OUT]):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app("submissions", "list", *_TOKEN_ARGS)

    def test_list_json_flag(self) -> None:
        with patch("endpoints_submission_cli.api_client.list_submissions", return_value=[SUBMISSION_OUT]):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["submissions", "list", "-j", *_TOKEN_ARGS])
        assert result.exit_code == 0
        assert SUBMISSION_ID in result.output

    def test_list_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.list_submissions", side_effect=APIError("fail")):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["submissions", "list", *_TOKEN_ARGS])
        assert result.exit_code == 1

    def test_list_no_token_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRISM_USER_API_TOKEN", raising=False)
        result = _runner.invoke(app, ["submissions", "list"])
        assert result.exit_code == 1


@pytest.mark.unit
class TestSubmissionsGet:
    def test_get_success(self) -> None:
        with patch("endpoints_submission_cli.api_client.get_submission", return_value=SUBMISSION_OUT):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app("submissions", "get", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)

    def test_get_json_flag(self) -> None:
        with patch("endpoints_submission_cli.api_client.get_submission", return_value=SUBMISSION_OUT):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app, ["submissions", "get", "--submission-id", SUBMISSION_ID, "-j", *_TOKEN_ARGS]
                )
        assert result.exit_code == 0
        assert SUBMISSION_ID in result.output

    def test_get_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.get_submission", side_effect=APIError("404")):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app,
                    ["submissions", "get", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS],
                )
        assert result.exit_code == 1


@pytest.mark.unit
class TestSubmissionsUpdate:
    def test_update_run_ids(self) -> None:
        updated = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        with patch("endpoints_submission_cli.api_client.update_submission", return_value=updated) as mock_patch:
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app(
                    "submissions", "update",
                    "--submission-id", SUBMISSION_ID,
                    "--run-ids", RUN_ID,
                    *_TOKEN_ARGS,
                )
        mock_patch.assert_called_once_with(TOKEN, SUBMISSION_ID, {"run_ids": [RUN_ID]})

    def test_update_target_availability_date(self) -> None:
        updated = {**SUBMISSION_OUT, "target_availability_date": "2026-06-01"}
        with patch("endpoints_submission_cli.api_client.update_submission", return_value=updated) as mock_patch:
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app(
                    "submissions", "update",
                    "--submission-id", SUBMISSION_ID,
                    "--target-availability-date", "2026-06-01",
                    *_TOKEN_ARGS,
                )
        mock_patch.assert_called_once_with(
            TOKEN, SUBMISSION_ID, {"target_availability_date": "2026-06-01"}
        )

    def test_update_no_fields_is_noop(self) -> None:
        with patch("endpoints_submission_cli.api_client.update_submission") as mock_patch:
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                _run_app("submissions", "update", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)
        mock_patch.assert_not_called()

    def test_update_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.update_submission", side_effect=APIError("500")):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app,
                    [
                        "submissions", "update",
                        "--submission-id", SUBMISSION_ID,
                        "--run-ids", RUN_ID,
                        *_TOKEN_ARGS,
                    ],
                )
        assert result.exit_code == 1


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
                        _run_app("submissions", "withdraw", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)

    def test_withdraw_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.api_client.withdraw_submission", side_effect=APIError("locked")):
            with patch("endpoints_submission_cli.api_client.get_token", return_value=TOKEN):
                with patch("endpoints_submission_cli.github_ops.get_target_repo", return_value="org/repo"):
                    result = _runner.invoke(
                        app,
                        ["submissions", "withdraw", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS],
                    )
        assert result.exit_code == 1


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
    @pytest.fixture(autouse=True)
    def _mock_gh_prereqs(self) -> pytest.FixtureRequest:
        with patch(
            "endpoints_submission_cli.github_ops.check_prerequisites",
            return_value=(True, ""),
        ):
            yield

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
                                    with patch("endpoints_submission_cli.github_ops.prepare_submission_branch"):
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
                    result = _runner.invoke(
                        app,
                        [
                            "submissions", "create",
                            "--division", "standardized",
                            "--availability", "available",
                            "--run-ids", RUN_ID,
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 1

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
                            result = _runner.invoke(
                                app,
                                [
                                    "submissions", "create",
                                    "--division", "standardized",
                                    "--availability", "available",
                                    "--run-ids", RUN_ID,
                                    *_TOKEN_ARGS,
                                ],
                            )
        assert result.exit_code == 1

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
                                result = _runner.invoke(
                                    app,
                                    [
                                        "submissions", "create",
                                        "--division", "standardized",
                                        "--availability", "available",
                                        "--run-ids", RUN_ID,
                                        *_TOKEN_ARGS,
                                    ],
                                )
        assert result.exit_code == 1

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
                                            result = _runner.invoke(
                                                app,
                                                [
                                                    "submissions", "create",
                                                    "--division", "standardized",
                                                    "--availability", "available",
                                                    "--run-ids", RUN_ID,
                                                    *_TOKEN_ARGS,
                                                ],
                                            )
        assert result.exit_code == 1
        mock_withdraw.assert_called_once_with(TOKEN, SUBMISSION_ID)

    def test_create_pr_failure_rolls_back_and_exits_1(self, tmp_path: Path) -> None:
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
                                    with patch("endpoints_submission_cli.github_ops.prepare_submission_branch"):
                                        with patch(
                                            "endpoints_submission_cli.github_ops.create_pr",
                                            side_effect=GitHubError("gh not found"),
                                        ):
                                            with patch(
                                                "endpoints_submission_cli.github_ops.get_target_repo",
                                                return_value="org/repo",
                                            ):
                                                with patch(
                                                    "endpoints_submission_cli.api_client.withdraw_submission"
                                                ) as mock_withdraw:
                                                    result = _runner.invoke(
                                                        app,
                                                        [
                                                            "submissions", "create",
                                                            "--division", "standardized",
                                                            "--availability", "available",
                                                            "--run-ids", RUN_ID,
                                                            *_TOKEN_ARGS,
                                                        ],
                                                    )
        assert result.exit_code == 1
        mock_withdraw.assert_called_once_with(TOKEN, SUBMISSION_ID)


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
                    result = _runner.invoke(
                        app,
                        [
                            "submissions", "add-run",
                            "--submission-id", SUBMISSION_ID,
                            "--run-id", self._NEW_RUN_ID,
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 1

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
                            result = _runner.invoke(
                                app,
                                [
                                    "submissions", "add-run",
                                    "--submission-id", SUBMISSION_ID,
                                    "--run-id", self._NEW_RUN_ID,
                                    *_TOKEN_ARGS,
                                ],
                            )
        assert result.exit_code == 1
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
                    result = _runner.invoke(
                        app,
                        [
                            "submissions", "remove-run",
                            "--submission-id", SUBMISSION_ID,
                            "--run-id", self._REMOVED_RUN_ID,
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 1

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
                            result = _runner.invoke(
                                app,
                                [
                                    "submissions", "remove-run",
                                    "--submission-id", SUBMISSION_ID,
                                    "--run-id", self._REMOVED_RUN_ID,
                                    *_TOKEN_ARGS,
                                ],
                            )
        assert result.exit_code == 1
        mock_rollback.assert_called_once_with(TOKEN, SUBMISSION_ID, self._REMOVED_RUN_ID)
