# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the submissions command group."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from endpoints_submission_cli.exceptions import APIError, GitHubError, SubmissionCheckError
from endpoints_submission_cli.main import app
from endpoints_submission_cli.submissions.builder import PENDING_SUBMISSION_ID
from tests.endpoints_submission_cli.conftest import (
    RUN_ID,
    RUN_OUT,
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
        with patch(
            "endpoints_submission_cli.submissions.api.list_submissions",
            return_value=[SUBMISSION_OUT],
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app("submissions", "list", *_TOKEN_ARGS)

    def test_list_json_flag(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.list_submissions",
            return_value=[SUBMISSION_OUT],
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["submissions", "list", "-j", *_TOKEN_ARGS])
        assert result.exit_code == 0
        assert SUBMISSION_ID in result.output

    def test_list_api_error_exits_1(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.list_submissions",
            side_effect=APIError("fail"),
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["submissions", "list", *_TOKEN_ARGS])
        assert result.exit_code == 1

    def test_list_no_token_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRISM_USER_API_TOKEN", raising=False)
        result = _runner.invoke(app, ["submissions", "list"])
        assert result.exit_code == 1


@pytest.mark.unit
class TestSubmissionsGet:
    def test_get_success(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.get_submission", return_value=SUBMISSION_OUT
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app("submissions", "get", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)

    def test_get_json_flag(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.get_submission", return_value=SUBMISSION_OUT
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app,
                    ["submissions", "get", "--submission-id", SUBMISSION_ID, "-j", *_TOKEN_ARGS],
                )
        assert result.exit_code == 0
        assert SUBMISSION_ID in result.output

    def test_get_api_error_exits_1(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.get_submission", side_effect=APIError("404")
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app,
                    ["submissions", "get", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS],
                )
        assert result.exit_code == 1

    def test_get_download_to_saves_archive(self, tmp_path: Path) -> None:
        archive = tmp_path / f"{SUBMISSION_ID}.tar.gz"
        with patch(
            "endpoints_submission_cli.submissions.api.get_submission", return_value=SUBMISSION_OUT
        ):
            with patch(
                "endpoints_submission_cli.submissions.api.download_submission_archive",
                return_value=archive,
            ) as mock_dl:
                with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                    result = _runner.invoke(
                        app,
                        [
                            "submissions",
                            "get",
                            "--submission-id",
                            SUBMISSION_ID,
                            "--download-to",
                            str(tmp_path),
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(TOKEN, SUBMISSION_ID, tmp_path)
        assert "Archive saved to" in result.output

    def test_get_download_api_error_exits_1(self, tmp_path: Path) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.get_submission", return_value=SUBMISSION_OUT
        ):
            with patch(
                "endpoints_submission_cli.submissions.api.download_submission_archive",
                side_effect=APIError("download failed"),
            ):
                with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                    result = _runner.invoke(
                        app,
                        [
                            "submissions",
                            "get",
                            "--submission-id",
                            SUBMISSION_ID,
                            "--download-to",
                            str(tmp_path),
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 1


@pytest.mark.unit
class TestSubmissionsUpdate:
    _OLD_RUN_ID = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"

    @pytest.fixture(autouse=True)
    def _mock_gh_prereqs(self) -> pytest.FixtureRequest:
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch(
                    "endpoints_submission_cli.submissions.github.check_prerequisites",
                    return_value=(True, ""),
                )
            )
            # The pre-flight test/non-test consistency check reads each run.
            # RUN_OUT and SUBMISSION_OUT are both is_test=False, so they agree.
            stack.enter_context(
                patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT)
            )
            yield

    def test_update_run_ids(self, tmp_path: Path) -> None:
        """--run-ids triggers full rebuild pipeline; update_submission called with new run list."""
        current_sub = {**SUBMISSION_OUT, "run_ids": [self._OLD_RUN_ID], "pr_number": _PR_NUMBER}
        updated_sub = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")
        fake_repo_dir = tmp_path / "repo"

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.get_submission", return_value=current_sub
            ):
                with patch(
                    "endpoints_submission_cli.submissions.api.update_submission",
                    return_value=updated_sub,
                ) as mock_patch:
                    with patch(
                        "endpoints_submission_cli.runs.api.download_run_archive",
                        return_value=fake_archive,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.update.build_submission_folder",
                            return_value=fake_sub_dir,
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.update._run_submission_checker"
                            ):
                                with patch(
                                    "endpoints_submission_cli.commands.submissions.update.create_bundle_archive",
                                    return_value=fake_bundle,
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.api.upload_submission_archive"
                                    ):
                                        with patch(
                                            "endpoints_submission_cli.submissions.github.prepare_pr_branch_merge",
                                            return_value=(fake_repo_dir, fake_sub_dir),
                                        ):
                                            with patch(
                                                "endpoints_submission_cli.submissions.github.commit_and_push"
                                            ):
                                                with patch(
                                                    "endpoints_submission_cli.submissions.github.get_target_repo",
                                                    return_value="org/repo",
                                                ):
                                                    _run_app(
                                                        "submissions",
                                                        "update",
                                                        "--submission-id",
                                                        SUBMISSION_ID,
                                                        "--run-ids",
                                                        RUN_ID,
                                                        *_TOKEN_ARGS,
                                                    )
        mock_patch.assert_called_once_with(TOKEN, SUBMISSION_ID, {"run_ids": [RUN_ID]})

    def test_update_run_ids_merge_failure_rolls_back(self, tmp_path: Path) -> None:
        """PR branch merge failure rolls back the DB PATCH with original run IDs."""
        current_sub = {**SUBMISSION_OUT, "run_ids": [self._OLD_RUN_ID], "pr_number": _PR_NUMBER}
        updated_sub = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.get_submission", return_value=current_sub
            ):
                with patch(
                    "endpoints_submission_cli.submissions.api.update_submission",
                    return_value=updated_sub,
                ) as mock_patch:
                    with patch(
                        "endpoints_submission_cli.runs.api.download_run_archive",
                        return_value=fake_archive,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.update.build_submission_folder",
                            return_value=fake_sub_dir,
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.update._run_submission_checker"
                            ):
                                with patch(
                                    "endpoints_submission_cli.submissions.github.prepare_pr_branch_merge",
                                    side_effect=GitHubError("clone failed"),
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.github.get_target_repo",
                                        return_value="org/repo",
                                    ):
                                        result = _runner.invoke(
                                            app,
                                            [
                                                "submissions",
                                                "update",
                                                "--submission-id",
                                                SUBMISSION_ID,
                                                "--run-ids",
                                                RUN_ID,
                                                *_TOKEN_ARGS,
                                            ],
                                        )
        assert result.exit_code == 1
        assert mock_patch.call_count == 2
        mock_patch.assert_any_call(TOKEN, SUBMISSION_ID, {"run_ids": [RUN_ID]})
        mock_patch.assert_any_call(TOKEN, SUBMISSION_ID, {"run_ids": [self._OLD_RUN_ID]})

    def test_update_run_ids_download_failure_rolls_back(self, tmp_path: Path) -> None:
        """Download failure rolls back the DB PATCH with original run IDs."""
        current_sub = {**SUBMISSION_OUT, "run_ids": [self._OLD_RUN_ID]}
        updated_sub = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.get_submission", return_value=current_sub
            ):
                with patch(
                    "endpoints_submission_cli.submissions.api.update_submission",
                    return_value=updated_sub,
                ) as mock_patch:
                    with patch(
                        "endpoints_submission_cli.runs.api.download_run_archive",
                        side_effect=APIError("not found"),
                    ):
                        with patch(
                            "endpoints_submission_cli.submissions.github.get_target_repo",
                            return_value="org/repo",
                        ):
                            result = _runner.invoke(
                                app,
                                [
                                    "submissions",
                                    "update",
                                    "--submission-id",
                                    SUBMISSION_ID,
                                    "--run-ids",
                                    RUN_ID,
                                    *_TOKEN_ARGS,
                                ],
                            )
        assert result.exit_code == 1
        # First call: PATCH forward; second call: rollback with original run IDs
        assert mock_patch.call_count == 2
        mock_patch.assert_any_call(TOKEN, SUBMISSION_ID, {"run_ids": [RUN_ID]})
        mock_patch.assert_any_call(TOKEN, SUBMISSION_ID, {"run_ids": [self._OLD_RUN_ID]})

    def test_update_run_ids_no_change_applies_date_only(self) -> None:
        """When desired run list equals current, skip rebuild and apply date patch only."""
        current_sub = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        updated_sub = {**SUBMISSION_OUT, "target_availability_date": "2026-06-01"}

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.get_submission", return_value=current_sub
            ):
                with patch(
                    "endpoints_submission_cli.submissions.api.update_submission",
                    return_value=updated_sub,
                ) as mock_patch:
                    with patch(
                        "endpoints_submission_cli.submissions.github.get_target_repo",
                        return_value="org/repo",
                    ):
                        _run_app(
                            "submissions",
                            "update",
                            "--submission-id",
                            SUBMISSION_ID,
                            "--run-ids",
                            RUN_ID,
                            "--target-availability-date",
                            "2026-06-01",
                            *_TOKEN_ARGS,
                        )
        mock_patch.assert_called_once_with(
            TOKEN, SUBMISSION_ID, {"target_availability_date": "2026-06-01"}
        )

    def test_update_target_availability_date(self) -> None:
        """--target-availability-date only triggers a DB-only PATCH (no rebuild)."""
        updated = {**SUBMISSION_OUT, "target_availability_date": "2026-06-01"}
        with patch(
            "endpoints_submission_cli.submissions.api.update_submission", return_value=updated
        ) as mock_patch:
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app(
                    "submissions",
                    "update",
                    "--submission-id",
                    SUBMISSION_ID,
                    "--target-availability-date",
                    "2026-06-01",
                    *_TOKEN_ARGS,
                )
        mock_patch.assert_called_once_with(
            TOKEN, SUBMISSION_ID, {"target_availability_date": "2026-06-01"}
        )

    def test_update_no_fields_is_noop(self) -> None:
        with patch("endpoints_submission_cli.submissions.api.update_submission") as mock_patch:
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app("submissions", "update", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)
        mock_patch.assert_not_called()

    def test_update_api_error_exits_1(self) -> None:
        """Date-only PATCH API error exits with code 1."""
        with patch(
            "endpoints_submission_cli.submissions.api.update_submission",
            side_effect=APIError("500"),
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app,
                    [
                        "submissions",
                        "update",
                        "--submission-id",
                        SUBMISSION_ID,
                        "--target-availability-date",
                        "2026-06-01",
                        *_TOKEN_ARGS,
                    ],
                )
        assert result.exit_code == 1


@pytest.mark.unit
class TestSubmissionsWithdraw:
    def test_withdraw_success(self) -> None:
        withdrawn = {**SUBMISSION_OUT, "status": "WITHDRAWN", "pr_number": _PR_NUMBER}
        with patch(
            "endpoints_submission_cli.submissions.api.withdraw_submission", return_value=withdrawn
        ):
            with patch("endpoints_submission_cli.submissions.api.delete_submission_archive"):
                with patch("endpoints_submission_cli.submissions.github.close_pr"):
                    with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                        with patch(
                            "endpoints_submission_cli.submissions.github.get_target_repo",
                            return_value="org/repo",
                        ):
                            _run_app(
                                "submissions",
                                "withdraw",
                                "--submission-id",
                                SUBMISSION_ID,
                                *_TOKEN_ARGS,
                            )

    def test_withdraw_no_pr_number(self) -> None:
        withdrawn = {**SUBMISSION_OUT, "status": "WITHDRAWN", "pr_number": None}
        with patch(
            "endpoints_submission_cli.submissions.api.withdraw_submission", return_value=withdrawn
        ):
            with patch("endpoints_submission_cli.submissions.api.delete_submission_archive"):
                with patch("endpoints_submission_cli.submissions.github.close_pr") as mock_close:
                    with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                        with patch(
                            "endpoints_submission_cli.submissions.github.get_target_repo",
                            return_value="org/repo",
                        ):
                            _run_app(
                                "submissions",
                                "withdraw",
                                "--submission-id",
                                SUBMISSION_ID,
                                *_TOKEN_ARGS,
                            )
        mock_close.assert_not_called()

    def test_withdraw_pr_close_failure_is_warning(self) -> None:
        withdrawn = {**SUBMISSION_OUT, "status": "WITHDRAWN", "pr_number": _PR_NUMBER}
        with patch(
            "endpoints_submission_cli.submissions.api.withdraw_submission", return_value=withdrawn
        ):
            with patch("endpoints_submission_cli.submissions.api.delete_submission_archive"):
                with patch(
                    "endpoints_submission_cli.submissions.github.close_pr",
                    side_effect=GitHubError("closed"),
                ):
                    with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                        with patch(
                            "endpoints_submission_cli.submissions.github.get_target_repo",
                            return_value="org/repo",
                        ):
                            _run_app(
                                "submissions",
                                "withdraw",
                                "--submission-id",
                                SUBMISSION_ID,
                                *_TOKEN_ARGS,
                            )

    def test_withdraw_archive_failure_is_warning(self) -> None:
        withdrawn = {**SUBMISSION_OUT, "status": "WITHDRAWN", "pr_number": None}
        with patch(
            "endpoints_submission_cli.submissions.api.withdraw_submission", return_value=withdrawn
        ):
            with patch(
                "endpoints_submission_cli.submissions.api.delete_submission_archive",
                side_effect=APIError("blob error"),
            ):
                with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                    with patch(
                        "endpoints_submission_cli.submissions.github.get_target_repo",
                        return_value="org/repo",
                    ):
                        _run_app(
                            "submissions",
                            "withdraw",
                            "--submission-id",
                            SUBMISSION_ID,
                            *_TOKEN_ARGS,
                        )

    def test_withdraw_api_error_exits_1(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.withdraw_submission",
            side_effect=APIError("locked"),
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                with patch(
                    "endpoints_submission_cli.submissions.github.get_target_repo",
                    return_value="org/repo",
                ):
                    result = _runner.invoke(
                        app,
                        ["submissions", "withdraw", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS],
                    )
        assert result.exit_code == 1


_CREATE_LOCAL_BASE_ARGS = [
    "submissions",
    "create-local",
    "--division",
    "standardized",
    "--scenario",
    "cop",
    "--availability",
    "available",
]

_FAKE_BUNDLE = b"bundle"
_FAKE_RUN_PAYLOAD = {
    "benchmark_version": "abc123",
    "started_at": "2025-04-28T09:00:00+00:00",
    "finished_at": "2025-04-28T10:00:00+00:00",
    "system_info": {},
    "config": {},
    "result_summary": {},
}


def _make_submission_dir(tmp_path: Path, n_points: int = 2) -> Path:
    """Create a minimal assembled submission directory with n_points result dirs."""
    sub = tmp_path / "sub"
    for i in range(1, n_points + 1):
        model_dir = sub / "results" / "sys_a" / "Llama-3-8B"
        point_dir = model_dir / f"r{i * 4}"
        point_dir.mkdir(parents=True)
        # A point is identified by its result_summary.json.
        (point_dir / "result_summary.json").write_text("{}")
        # The system description is shared per system, not stored per point.
        (model_dir.parent / "system_desc_id.json").write_text("{}")
    return sub


@pytest.mark.unit
class TestSubmissionsCreateLocal:
    def _invoke(self, submission_dir: Path, *extra: str) -> object:
        return _runner.invoke(
            app,
            [
                *_CREATE_LOCAL_BASE_ARGS,
                "--path",
                str(submission_dir),
                *_TOKEN_ARGS,
                *extra,
            ],
        )

    def test_create_local_success(self, tmp_path: Path) -> None:
        sub = _make_submission_dir(tmp_path)
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(_FAKE_BUNDLE)

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.commands.submissions.create_local._run_submission_checker"
            ):
                with patch(
                    "endpoints_submission_cli.commands.submissions.create_local._parse_result_dir",
                    return_value=_FAKE_RUN_PAYLOAD,
                ):
                    with patch(
                        "endpoints_submission_cli.runs.api.create_run", return_value=RUN_OUT
                    ) as mock_create_run:
                        with patch(
                            "endpoints_submission_cli.commands.submissions.create_local.build_archive",
                            return_value=fake_bundle,
                        ):
                            with patch("endpoints_submission_cli.runs.api.upload_run_archive"):
                                with patch(
                                    "endpoints_submission_cli.submissions.api.create_submission",
                                    return_value=SUBMISSION_OUT,
                                ) as mock_create_sub:
                                    with patch(
                                        "endpoints_submission_cli.commands.submissions.create_local.create_bundle_archive",
                                        return_value=fake_bundle,
                                    ):
                                        with patch(
                                            "endpoints_submission_cli.submissions.api.upload_submission_archive"
                                        ):
                                            with patch(
                                                "endpoints_submission_cli.submissions.api.update_submission"
                                            ):
                                                result = self._invoke(sub)
        assert result.exit_code == 0, result.output
        assert SUBMISSION_ID in result.output
        assert mock_create_run.call_count == 2
        mock_create_sub.assert_called_once()
        meta = json.loads((sub / "cli_metadata.json").read_text())
        assert meta["command"] == "create-local"
        assert "cli_version" in meta and "created_at" in meta

    def test_create_local_test_flag_marks_runs_and_submission(self, tmp_path: Path) -> None:
        """--test must flag the runs it registers too, not just the submission."""
        sub = _make_submission_dir(tmp_path)
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(_FAKE_BUNDLE)
        C = "endpoints_submission_cli"

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(f"{C}._http.get_token", return_value=TOKEN))
            stack.enter_context(
                patch(f"{C}.commands.submissions.create_local._run_submission_checker")
            )
            # side_effect, not return_value: the payload is mutated downstream and a
            # shared dict would leak is_test into every other test in this module.
            stack.enter_context(
                patch(
                    f"{C}.commands.submissions.create_local._parse_result_dir",
                    side_effect=lambda *_a, **_k: dict(_FAKE_RUN_PAYLOAD),
                )
            )
            mock_create_run = stack.enter_context(
                patch(f"{C}.runs.api.create_run", return_value=RUN_OUT)
            )
            stack.enter_context(
                patch(
                    f"{C}.commands.submissions.create_local.build_archive",
                    return_value=fake_bundle,
                )
            )
            stack.enter_context(patch(f"{C}.runs.api.upload_run_archive"))
            mock_create_sub = stack.enter_context(
                patch(f"{C}.submissions.api.create_submission", return_value=SUBMISSION_OUT)
            )
            stack.enter_context(
                patch(
                    f"{C}.commands.submissions.create_local.create_bundle_archive",
                    return_value=fake_bundle,
                )
            )
            stack.enter_context(patch(f"{C}.submissions.api.upload_submission_archive"))
            stack.enter_context(patch(f"{C}.submissions.api.update_submission"))
            result = self._invoke(sub, "--test")

        assert result.exit_code == 0, result.output
        assert mock_create_run.call_count == 2
        for call in mock_create_run.call_args_list:
            assert call[0][1]["is_test"] is True
        assert mock_create_sub.call_args[0][1]["is_test"] is True

    def test_create_local_without_test_flag_leaves_runs_unflagged(self, tmp_path: Path) -> None:
        sub = _make_submission_dir(tmp_path)
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(_FAKE_BUNDLE)
        C = "endpoints_submission_cli"

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(f"{C}._http.get_token", return_value=TOKEN))
            stack.enter_context(
                patch(f"{C}.commands.submissions.create_local._run_submission_checker")
            )
            stack.enter_context(
                patch(
                    f"{C}.commands.submissions.create_local._parse_result_dir",
                    side_effect=lambda *_a, **_k: dict(_FAKE_RUN_PAYLOAD),
                )
            )
            mock_create_run = stack.enter_context(
                patch(f"{C}.runs.api.create_run", return_value=RUN_OUT)
            )
            stack.enter_context(
                patch(
                    f"{C}.commands.submissions.create_local.build_archive",
                    return_value=fake_bundle,
                )
            )
            stack.enter_context(patch(f"{C}.runs.api.upload_run_archive"))
            mock_create_sub = stack.enter_context(
                patch(f"{C}.submissions.api.create_submission", return_value=SUBMISSION_OUT)
            )
            stack.enter_context(
                patch(
                    f"{C}.commands.submissions.create_local.create_bundle_archive",
                    return_value=fake_bundle,
                )
            )
            stack.enter_context(patch(f"{C}.submissions.api.upload_submission_archive"))
            stack.enter_context(patch(f"{C}.submissions.api.update_submission"))
            result = self._invoke(sub)

        assert result.exit_code == 0, result.output
        for call in mock_create_run.call_args_list:
            assert "is_test" not in call[0][1]
        assert mock_create_sub.call_args[0][1]["is_test"] is False

    def test_create_local_dry_run(self, tmp_path: Path) -> None:
        sub = _make_submission_dir(tmp_path)
        with patch(
            "endpoints_submission_cli.commands.submissions.create_local._run_submission_checker"
        ):
            result = self._invoke(sub, "--dry-run")
        assert result.exit_code == 0
        assert "dry-run" in result.output

    def test_create_local_no_result_dirs_exits_1(self, tmp_path: Path) -> None:
        sub = tmp_path / "empty_sub"
        sub.mkdir()
        result = self._invoke(sub)
        assert result.exit_code == 1

    def test_create_local_checker_failure_exits_1(self, tmp_path: Path) -> None:
        sub = _make_submission_dir(tmp_path)
        with patch(
            "endpoints_submission_cli.commands.submissions.create_local._run_submission_checker",
            side_effect=SubmissionCheckError("1 error"),
        ):
            result = self._invoke(sub)
        assert result.exit_code == 1

    def test_create_local_run_upload_failure_rolls_back(self, tmp_path: Path) -> None:
        sub = _make_submission_dir(tmp_path)
        fake_bundle = tmp_path / "a.tar.gz"
        fake_bundle.write_bytes(_FAKE_BUNDLE)
        run1_out = {**RUN_OUT, "id": "aaaa-1111"}
        call_count = {"n": 0}

        def _create_run_side_effect(*_a, **_kw):
            return run1_out

        def _upload_side_effect(*_a, **_kw):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise APIError("upload failed")

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.commands.submissions.create_local._run_submission_checker"
            ):
                with patch(
                    "endpoints_submission_cli.commands.submissions.create_local._parse_result_dir",
                    return_value=_FAKE_RUN_PAYLOAD,
                ):
                    with patch(
                        "endpoints_submission_cli.runs.api.create_run",
                        side_effect=_create_run_side_effect,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.create_local.build_archive",
                            return_value=fake_bundle,
                        ):
                            with patch(
                                "endpoints_submission_cli.runs.api.upload_run_archive",
                                side_effect=_upload_side_effect,
                            ):
                                with patch("endpoints_submission_cli.runs.api.delete_run_archive"):
                                    with patch(
                                        "endpoints_submission_cli.runs.api.delete_run"
                                    ) as mock_delete:
                                        result = self._invoke(sub)
        assert result.exit_code == 1
        assert mock_delete.call_count >= 1

    def test_create_local_submission_upload_failure_withdraws(self, tmp_path: Path) -> None:
        sub = _make_submission_dir(tmp_path)
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(_FAKE_BUNDLE)

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.commands.submissions.create_local._run_submission_checker"
            ):
                with patch(
                    "endpoints_submission_cli.commands.submissions.create_local._parse_result_dir",
                    return_value=_FAKE_RUN_PAYLOAD,
                ):
                    with patch(
                        "endpoints_submission_cli.runs.api.create_run", return_value=RUN_OUT
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.create_local.build_archive",
                            return_value=fake_bundle,
                        ):
                            with patch("endpoints_submission_cli.runs.api.upload_run_archive"):
                                with patch(
                                    "endpoints_submission_cli.submissions.api.create_submission",
                                    return_value=SUBMISSION_OUT,
                                ):
                                    with patch(
                                        "endpoints_submission_cli.commands.submissions.create_local.create_bundle_archive",
                                        return_value=fake_bundle,
                                    ):
                                        with patch(
                                            "endpoints_submission_cli.submissions.api.upload_submission_archive",
                                            side_effect=APIError("upload failed"),
                                        ):
                                            with patch(
                                                "endpoints_submission_cli.submissions.api.withdraw_submission"
                                            ) as mock_withdraw:
                                                result = self._invoke(sub)
        assert result.exit_code == 1
        mock_withdraw.assert_called_once_with(TOKEN, SUBMISSION_ID)


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
            "endpoints_submission_cli.submissions.github.check_prerequisites",
            return_value=(True, ""),
        ):
            yield

    def test_create_success(self, tmp_path: Path) -> None:
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.create._run_submission_checker"
                        ):
                            with patch(
                                "endpoints_submission_cli.submissions.api.create_submission",
                                return_value=SUBMISSION_OUT,
                            ) as mock_create:
                                with patch(
                                    "endpoints_submission_cli.commands.submissions.create.create_bundle_archive",
                                    return_value=fake_bundle,
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.api.upload_submission_archive"
                                    ):
                                        with patch(
                                            "endpoints_submission_cli.submissions.github.prepare_submission_branch"
                                        ):
                                            with patch(
                                                "endpoints_submission_cli.submissions.github.create_pr",
                                                return_value=(_PR_URL, _PR_NUMBER),
                                            ):
                                                with patch(
                                                    "endpoints_submission_cli.submissions.api.update_submission"
                                                ):
                                                    with patch(
                                                        "endpoints_submission_cli.submissions.github.get_target_repo",
                                                        return_value="org/repo",
                                                    ):
                                                        _run_app(
                                                            "submissions",
                                                            "create",
                                                            "--division",
                                                            "standardized",
                                                            "--scenario",
                                                            "cop",
                                                            "--availability",
                                                            "available",
                                                            "--run-ids",
                                                            RUN_ID,
                                                            *_TOKEN_ARGS,
                                                        )
        mock_create.assert_called_once()
        # Without --test, the submission is created as a non-test entry.
        assert mock_create.call_args.args[1]["is_test"] is False
        # The bundle carries a cli_metadata.json marker identifying the build command.
        meta_path = fake_sub_dir / "cli_metadata.json"
        assert meta_path.is_file()
        meta = json.loads(meta_path.read_text())
        assert meta["command"] == "create"
        assert "cli_version" in meta and "created_at" in meta

    def test_create_rejects_mixed_test_and_real_runs(self, tmp_path: Path) -> None:
        """A run set that mixes test and real runs is refused before any download."""
        C = "endpoints_submission_cli"
        second = "cccccccc-cccc-cccc-cccc-cccccccccccc"

        def _get(_token: str, run_id: str) -> dict:
            return {**RUN_OUT, "id": run_id, "is_test": run_id == second}

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(f"{C}._http.get_token", return_value=TOKEN))
            stack.enter_context(patch(f"{C}.runs.api.get_run", side_effect=_get))
            mock_dl = stack.enter_context(patch(f"{C}.runs.api.download_run_archive"))
            mock_create = stack.enter_context(patch(f"{C}.submissions.api.create_submission"))
            result = _runner.invoke(
                app,
                [
                    "submissions",
                    "create",
                    "--division",
                    "standardized",
                    "--scenario",
                    "cop",
                    "--availability",
                    "available",
                    "--run-ids",
                    RUN_ID,
                    "--run-ids",
                    second,
                    *_TOKEN_ARGS,
                ],
            )

        assert result.exit_code == 1
        assert "cannot mix test runs with real ones" in result.output
        # Refused up front: nothing downloaded, nothing created.
        mock_dl.assert_not_called()
        mock_create.assert_not_called()

    def test_create_test_flag_sets_is_test(self, tmp_path: Path) -> None:
        import contextlib

        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")
        C = "endpoints_submission_cli"

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(f"{C}._http.get_token", return_value=TOKEN))
            # --test requires test runs; a non-test run here is now a hard error.
            stack.enter_context(
                patch(f"{C}.runs.api.get_run", return_value={**RUN_OUT, "is_test": True})
            )
            stack.enter_context(
                patch(f"{C}.runs.api.download_run_archive", return_value=fake_archive)
            )
            stack.enter_context(
                patch(
                    f"{C}.commands.submissions.create.build_submission_folder",
                    return_value=fake_sub_dir,
                )
            )
            stack.enter_context(patch(f"{C}.commands.submissions.create._run_submission_checker"))
            mock_create = stack.enter_context(
                patch(f"{C}.submissions.api.create_submission", return_value=SUBMISSION_OUT)
            )
            stack.enter_context(
                patch(
                    f"{C}.commands.submissions.create.create_bundle_archive",
                    return_value=fake_bundle,
                )
            )
            stack.enter_context(patch(f"{C}.submissions.api.upload_submission_archive"))
            stack.enter_context(patch(f"{C}.submissions.github.prepare_submission_branch"))
            stack.enter_context(
                patch(f"{C}.submissions.github.create_pr", return_value=(_PR_URL, _PR_NUMBER))
            )
            stack.enter_context(patch(f"{C}.submissions.api.update_submission"))
            stack.enter_context(
                patch(f"{C}.submissions.github.get_target_repo", return_value="org/repo")
            )
            _run_app(
                "submissions",
                "create",
                "--division",
                "standardized",
                "--scenario",
                "cop",
                "--availability",
                "available",
                "--run-ids",
                RUN_ID,
                "--test",
                *_TOKEN_ARGS,
            )

        mock_create.assert_called_once()
        assert mock_create.call_args.args[1]["is_test"] is True

    def test_create_download_failure_exits_1(self) -> None:
        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.runs.api.download_run_archive",
                side_effect=APIError("not found"),
            ):
                with patch(
                    "endpoints_submission_cli.submissions.github.get_target_repo",
                    return_value="org/repo",
                ):
                    result = _runner.invoke(
                        app,
                        [
                            "submissions",
                            "create",
                            "--division",
                            "standardized",
                            "--scenario",
                            "cop",
                            "--availability",
                            "available",
                            "--run-ids",
                            RUN_ID,
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 1

    def test_create_checker_failure_exits_1(self, tmp_path: Path) -> None:
        from endpoints_submission_cli.exceptions import SubmissionCheckError

        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.create._run_submission_checker",
                            side_effect=SubmissionCheckError("1 error"),
                        ):
                            with patch(
                                "endpoints_submission_cli.submissions.github.get_target_repo",
                                return_value="org/repo",
                            ):
                                result = _runner.invoke(
                                    app,
                                    [
                                        "submissions",
                                        "create",
                                        "--division",
                                        "standardized",
                                        "--scenario",
                                        "cop",
                                        "--availability",
                                        "available",
                                        "--run-ids",
                                        RUN_ID,
                                        *_TOKEN_ARGS,
                                    ],
                                )
        assert result.exit_code == 1

    def test_create_api_error_exits_1(self, tmp_path: Path) -> None:
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.create._run_submission_checker"
                        ):
                            with patch(
                                "endpoints_submission_cli.submissions.api.create_submission",
                                side_effect=APIError("500"),
                            ):
                                with patch(
                                    "endpoints_submission_cli.submissions.github.get_target_repo",
                                    return_value="org/repo",
                                ):
                                    result = _runner.invoke(
                                        app,
                                        [
                                            "submissions",
                                            "create",
                                            "--division",
                                            "standardized",
                                            "--scenario",
                                            "cop",
                                            "--availability",
                                            "available",
                                            "--run-ids",
                                            RUN_ID,
                                            *_TOKEN_ARGS,
                                        ],
                                    )
        assert result.exit_code == 1

    def test_create_upload_failure_rolls_back(self, tmp_path: Path) -> None:
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.create._run_submission_checker"
                        ):
                            with patch(
                                "endpoints_submission_cli.submissions.api.create_submission",
                                return_value=SUBMISSION_OUT,
                            ):
                                with patch(
                                    "endpoints_submission_cli.commands.submissions.create.create_bundle_archive",
                                    return_value=fake_bundle,
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.api.upload_submission_archive",
                                        side_effect=APIError("upload failed"),
                                    ):
                                        with patch(
                                            "endpoints_submission_cli.submissions.api.withdraw_submission"
                                        ) as mock_withdraw:
                                            with patch(
                                                "endpoints_submission_cli.submissions.github.get_target_repo",
                                                return_value="org/repo",
                                            ):
                                                result = _runner.invoke(
                                                    app,
                                                    [
                                                        "submissions",
                                                        "create",
                                                        "--division",
                                                        "standardized",
                                                        "--scenario",
                                                        "cop",
                                                        "--availability",
                                                        "available",
                                                        "--run-ids",
                                                        RUN_ID,
                                                        *_TOKEN_ARGS,
                                                    ],
                                                )
        assert result.exit_code == 1
        mock_withdraw.assert_called_once_with(TOKEN, SUBMISSION_ID)


@pytest.mark.unit
class TestSubmissionsAddRun:
    _NEW_RUN_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    @pytest.fixture(autouse=True)
    def _mock_gh_prereqs(self) -> pytest.FixtureRequest:
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch(
                    "endpoints_submission_cli.submissions.github.check_prerequisites",
                    return_value=(True, ""),
                )
            )
            # add-run now reads the submission and the incoming run up front to
            # confirm they agree on is_test; both fixtures are is_test=False.
            stack.enter_context(
                patch(
                    "endpoints_submission_cli.submissions.api.get_submission",
                    return_value=SUBMISSION_OUT,
                )
            )
            stack.enter_context(
                patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT)
            )
            yield

    def _sub_with_runs(self, *run_ids: str) -> dict:
        return {**SUBMISSION_OUT, "run_ids": list(run_ids)}

    def test_add_run_success(self, tmp_path: Path) -> None:
        sub_out = self._sub_with_runs(RUN_ID, self._NEW_RUN_ID)
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")
        fake_repo_dir = tmp_path / "repo"

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.add_run_to_submission",
                return_value=sub_out,
            ):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.add_run.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.add_run._run_submission_checker"
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.add_run.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch(
                                    "endpoints_submission_cli.submissions.api.upload_submission_archive"
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.github.prepare_pr_branch_merge",
                                        return_value=(fake_repo_dir, fake_sub_dir),
                                    ):
                                        with patch(
                                            "endpoints_submission_cli.submissions.github.commit_and_push"
                                        ):
                                            with patch(
                                                "endpoints_submission_cli.submissions.github.get_target_repo",
                                                return_value="org/repo",
                                            ):
                                                _run_app(
                                                    "submissions",
                                                    "add-run",
                                                    "--submission-id",
                                                    SUBMISSION_ID,
                                                    "--run-id",
                                                    self._NEW_RUN_ID,
                                                    *_TOKEN_ARGS,
                                                )

    def test_add_run_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.add_run_to_submission",
                side_effect=APIError("conflict"),
            ):
                with patch(
                    "endpoints_submission_cli.submissions.github.get_target_repo",
                    return_value="org/repo",
                ):
                    result = _runner.invoke(
                        app,
                        [
                            "submissions",
                            "add-run",
                            "--submission-id",
                            SUBMISSION_ID,
                            "--run-id",
                            self._NEW_RUN_ID,
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 1

    def test_add_run_download_failure_rolls_back(self, tmp_path: Path) -> None:
        sub_out = self._sub_with_runs(RUN_ID, self._NEW_RUN_ID)

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.add_run_to_submission",
                return_value=sub_out,
            ):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    side_effect=APIError("not found"),
                ):
                    with patch(
                        "endpoints_submission_cli.submissions.api.remove_run_from_submission"
                    ) as mock_rollback:
                        with patch(
                            "endpoints_submission_cli.submissions.github.get_target_repo",
                            return_value="org/repo",
                        ):
                            result = _runner.invoke(
                                app,
                                [
                                    "submissions",
                                    "add-run",
                                    "--submission-id",
                                    SUBMISSION_ID,
                                    "--run-id",
                                    self._NEW_RUN_ID,
                                    *_TOKEN_ARGS,
                                ],
                            )
        assert result.exit_code == 1
        mock_rollback.assert_called_once_with(TOKEN, SUBMISSION_ID, self._NEW_RUN_ID)

    def test_add_run_github_failure_is_warning(self, tmp_path: Path) -> None:
        """Push failure after successful merge and upload is a warning — exits 0."""
        sub_out = self._sub_with_runs(RUN_ID, self._NEW_RUN_ID)
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")
        fake_repo_dir = tmp_path / "repo"

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.add_run_to_submission",
                return_value=sub_out,
            ):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.add_run.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.add_run._run_submission_checker"
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.add_run.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch(
                                    "endpoints_submission_cli.submissions.api.upload_submission_archive"
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.github.prepare_pr_branch_merge",
                                        return_value=(fake_repo_dir, fake_sub_dir),
                                    ):
                                        with patch(
                                            "endpoints_submission_cli.submissions.github.commit_and_push",
                                            side_effect=GitHubError("push failed"),
                                        ):
                                            with patch(
                                                "endpoints_submission_cli.submissions.github.get_target_repo",
                                                return_value="org/repo",
                                            ):
                                                _run_app(
                                                    "submissions",
                                                    "add-run",
                                                    "--submission-id",
                                                    SUBMISSION_ID,
                                                    "--run-id",
                                                    self._NEW_RUN_ID,
                                                    *_TOKEN_ARGS,
                                                )

    def test_add_run_merge_failure_rolls_back(self, tmp_path: Path) -> None:
        """PR branch merge failure rolls back the run registration."""
        sub_out = self._sub_with_runs(RUN_ID, self._NEW_RUN_ID)
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.add_run_to_submission",
                return_value=sub_out,
            ):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.add_run.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.add_run._run_submission_checker"
                        ):
                            with patch(
                                "endpoints_submission_cli.submissions.github.prepare_pr_branch_merge",
                                side_effect=GitHubError("clone failed"),
                            ):
                                with patch(
                                    "endpoints_submission_cli.submissions.api.remove_run_from_submission"
                                ) as mock_rollback:
                                    with patch(
                                        "endpoints_submission_cli.submissions.github.get_target_repo",
                                        return_value="org/repo",
                                    ):
                                        result = _runner.invoke(
                                            app,
                                            [
                                                "submissions",
                                                "add-run",
                                                "--submission-id",
                                                SUBMISSION_ID,
                                                "--run-id",
                                                self._NEW_RUN_ID,
                                                *_TOKEN_ARGS,
                                            ],
                                        )
        assert result.exit_code == 1
        mock_rollback.assert_called_once_with(TOKEN, SUBMISSION_ID, self._NEW_RUN_ID)


@pytest.mark.unit
class TestSubmissionsRemoveRun:
    _REMOVED_RUN_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"

    @pytest.fixture(autouse=True)
    def _mock_gh_prereqs(self) -> pytest.FixtureRequest:
        with patch(
            "endpoints_submission_cli.submissions.github.check_prerequisites",
            return_value=(True, ""),
        ):
            yield

    def test_remove_run_success(self, tmp_path: Path) -> None:
        sub_out = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")
        fake_repo_dir = tmp_path / "repo"

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.remove_run_from_submission",
                return_value=sub_out,
            ):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.remove_run.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.remove_run._run_submission_checker"
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.remove_run.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch(
                                    "endpoints_submission_cli.submissions.api.upload_submission_archive"
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.github.prepare_pr_branch_merge",
                                        return_value=(fake_repo_dir, fake_sub_dir),
                                    ):
                                        with patch(
                                            "endpoints_submission_cli.submissions.github.commit_and_push"
                                        ):
                                            with patch(
                                                "endpoints_submission_cli.submissions.github.get_target_repo",
                                                return_value="org/repo",
                                            ):
                                                _run_app(
                                                    "submissions",
                                                    "remove-run",
                                                    "--submission-id",
                                                    SUBMISSION_ID,
                                                    "--run-id",
                                                    self._REMOVED_RUN_ID,
                                                    *_TOKEN_ARGS,
                                                )

    def test_remove_run_no_runs_left(self) -> None:
        sub_out = {**SUBMISSION_OUT, "run_ids": []}

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.remove_run_from_submission",
                return_value=sub_out,
            ):
                with patch(
                    "endpoints_submission_cli.submissions.github.get_target_repo",
                    return_value="org/repo",
                ):
                    _run_app(
                        "submissions",
                        "remove-run",
                        "--submission-id",
                        SUBMISSION_ID,
                        "--run-id",
                        self._REMOVED_RUN_ID,
                        *_TOKEN_ARGS,
                    )

    def test_remove_run_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.remove_run_from_submission",
                side_effect=APIError("not found"),
            ):
                with patch(
                    "endpoints_submission_cli.submissions.github.get_target_repo",
                    return_value="org/repo",
                ):
                    result = _runner.invoke(
                        app,
                        [
                            "submissions",
                            "remove-run",
                            "--submission-id",
                            SUBMISSION_ID,
                            "--run-id",
                            self._REMOVED_RUN_ID,
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 1

    def test_remove_run_download_failure_rolls_back(self, tmp_path: Path) -> None:
        sub_out = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.remove_run_from_submission",
                return_value=sub_out,
            ):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    side_effect=APIError("not found"),
                ):
                    with patch(
                        "endpoints_submission_cli.submissions.api.add_run_to_submission"
                    ) as mock_rollback:
                        with patch(
                            "endpoints_submission_cli.submissions.github.get_target_repo",
                            return_value="org/repo",
                        ):
                            result = _runner.invoke(
                                app,
                                [
                                    "submissions",
                                    "remove-run",
                                    "--submission-id",
                                    SUBMISSION_ID,
                                    "--run-id",
                                    self._REMOVED_RUN_ID,
                                    *_TOKEN_ARGS,
                                ],
                            )
        assert result.exit_code == 1
        mock_rollback.assert_called_once_with(TOKEN, SUBMISSION_ID, self._REMOVED_RUN_ID)

    def test_remove_run_merge_failure_rolls_back(self, tmp_path: Path) -> None:
        """PR branch merge failure rolls back the run removal."""
        sub_out = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.remove_run_from_submission",
                return_value=sub_out,
            ):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.remove_run.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.remove_run._run_submission_checker"
                        ):
                            with patch(
                                "endpoints_submission_cli.submissions.github.prepare_pr_branch_merge",
                                side_effect=GitHubError("clone failed"),
                            ):
                                with patch(
                                    "endpoints_submission_cli.submissions.api.add_run_to_submission"
                                ) as mock_rollback:
                                    with patch(
                                        "endpoints_submission_cli.submissions.github.get_target_repo",
                                        return_value="org/repo",
                                    ):
                                        result = _runner.invoke(
                                            app,
                                            [
                                                "submissions",
                                                "remove-run",
                                                "--submission-id",
                                                SUBMISSION_ID,
                                                "--run-id",
                                                self._REMOVED_RUN_ID,
                                                *_TOKEN_ARGS,
                                            ],
                                        )
        assert result.exit_code == 1
        mock_rollback.assert_called_once_with(TOKEN, SUBMISSION_ID, self._REMOVED_RUN_ID)


_PROMPT_TEXT = "Would you like to continue?"
_WARNING_TEXT = "publicly viewable on the visualizer"


@contextlib.contextmanager
def _patched_create(tmp_path: Path):
    """Patch out every side effect of `submissions create` past the confirmation prompt."""
    fake_sub_dir = tmp_path / "sub"
    fake_sub_dir.mkdir(exist_ok=True)
    # build_submission_folder returns the org dir; the submission level lives below it.
    (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir(exist_ok=True)
    fake_bundle = tmp_path / "bundle.tar.gz"
    fake_bundle.write_bytes(b"bundle")
    targets = [
        patch("endpoints_submission_cli._http.get_token", return_value=TOKEN),
        # The pre-flight test/non-test consistency check reads each run.
        patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT),
        patch(
            "endpoints_submission_cli.runs.api.download_run_archive",
            return_value=_make_fake_archive(tmp_path),
        ),
        patch(
            "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
            return_value=fake_sub_dir,
        ),
        patch("endpoints_submission_cli.commands.submissions.create._run_submission_checker"),
        patch(
            "endpoints_submission_cli.commands.submissions.create.create_bundle_archive",
            return_value=fake_bundle,
        ),
        patch("endpoints_submission_cli.submissions.api.upload_submission_archive"),
        patch("endpoints_submission_cli.submissions.api.update_submission"),
    ]
    with contextlib.ExitStack() as stack:
        for target in targets:
            stack.enter_context(target)
        yield stack.enter_context(
            patch(
                "endpoints_submission_cli.submissions.api.create_submission",
                return_value=SUBMISSION_OUT,
            )
        )


@pytest.mark.unit
class TestProvisionalConfirmation:
    """`--provisional` must be confirmed before anything is submitted."""

    _CREATE_ARGS = [
        "submissions",
        "create",
        "--division",
        "standardized",
        "--scenario",
        "cop",
        "--availability",
        "available",
        "--run-ids",
        RUN_ID,
        *_TOKEN_ARGS,
    ]

    def test_no_prompt_without_provisional(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, self._CREATE_ARGS)
        assert result.exit_code == 0, result.output
        assert _PROMPT_TEXT not in result.output
        assert mock_create.call_args[0][1]["early_publish"] is False

    def test_declining_prompt_exits_1_without_downloading(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, "--provisional"], input="n\n")
        assert result.exit_code == 1
        assert _WARNING_TEXT in result.output
        assert "peer review pending" in result.output
        mock_create.assert_not_called()

    def test_accepting_prompt_sets_flag(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, "--provisional"], input="y\n")
        assert result.exit_code == 0, result.output
        assert _PROMPT_TEXT in result.output
        assert mock_create.call_args[0][1]["early_publish"] is True

    def test_empty_answer_declines(self, tmp_path: Path) -> None:
        """The prompt defaults to "no" — a bare Enter must not publish."""
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, "--provisional"], input="\n")
        assert result.exit_code == 1
        mock_create.assert_not_called()

    def test_yes_flag_skips_prompt(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, "--provisional", "--yes"])
        assert result.exit_code == 0, result.output
        assert _PROMPT_TEXT not in result.output
        assert _WARNING_TEXT in result.output
        assert mock_create.call_args[0][1]["early_publish"] is True

    def test_dry_run_skips_prompt(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, "--provisional", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert _PROMPT_TEXT not in result.output
        mock_create.assert_not_called()

    def test_create_local_declining_prompt_exits_1(self, tmp_path: Path) -> None:
        sub = _make_submission_dir(tmp_path)
        with patch(
            "endpoints_submission_cli.commands.submissions.create_local._run_submission_checker"
        ) as mock_checker:
            with patch("endpoints_submission_cli.submissions.api.create_submission") as mock_create:
                result = _runner.invoke(
                    app,
                    [*_CREATE_LOCAL_BASE_ARGS, "--path", str(sub), *_TOKEN_ARGS, "--provisional"],
                    input="n\n",
                )
        assert result.exit_code == 1
        assert _WARNING_TEXT in result.output
        # Aborted before the checker ran, so nothing was registered either.
        mock_checker.assert_not_called()
        mock_create.assert_not_called()

    def test_create_local_accepting_prompt_sets_flag(self, tmp_path: Path) -> None:
        sub = _make_submission_dir(tmp_path)
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(_FAKE_BUNDLE)
        targets = [
            patch("endpoints_submission_cli._http.get_token", return_value=TOKEN),
            patch(
                "endpoints_submission_cli.commands.submissions.create_local._run_submission_checker"
            ),
            patch(
                "endpoints_submission_cli.commands.submissions.create_local._parse_result_dir",
                return_value=_FAKE_RUN_PAYLOAD,
            ),
            patch("endpoints_submission_cli.runs.api.create_run", return_value=RUN_OUT),
            patch(
                "endpoints_submission_cli.commands.submissions.create_local.build_archive",
                return_value=fake_bundle,
            ),
            patch("endpoints_submission_cli.runs.api.upload_run_archive"),
            patch(
                "endpoints_submission_cli.commands.submissions.create_local.create_bundle_archive",
                return_value=fake_bundle,
            ),
            patch("endpoints_submission_cli.submissions.api.upload_submission_archive"),
            patch("endpoints_submission_cli.submissions.api.update_submission"),
        ]
        with contextlib.ExitStack() as stack:
            for target in targets:
                stack.enter_context(target)
            mock_create = stack.enter_context(
                patch(
                    "endpoints_submission_cli.submissions.api.create_submission",
                    return_value=SUBMISSION_OUT,
                )
            )
            result = _runner.invoke(
                app,
                [*_CREATE_LOCAL_BASE_ARGS, "--path", str(sub), *_TOKEN_ARGS, "--provisional"],
                input="y\n",
            )
        assert result.exit_code == 0, result.output
        assert mock_create.call_args[0][1]["early_publish"] is True
