# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the runs command group."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from endpoints_submission_cli.exceptions import APIError
from endpoints_submission_cli.main import app
from tests.endpoints_submission_cli.conftest import RUN_ID, RUN_OUT, RUN_SUMMARY

TOKEN = "mlc_test"
_TOKEN_ARGS = ["--token", TOKEN]

_runner = CliRunner()


def _run_app(*args: str) -> None:
    """Invoke the app and assert exit code 0."""
    result = _runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output


@pytest.mark.unit
class TestRunsList:
    def test_list_success(self) -> None:
        with patch("endpoints_submission_cli.runs.api.list_runs", return_value=[RUN_SUMMARY]):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app("runs", "list", *_TOKEN_ARGS)

    def test_list_json_flag(self) -> None:
        with patch("endpoints_submission_cli.runs.api.list_runs", return_value=[RUN_SUMMARY]):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["runs", "list", "-j", *_TOKEN_ARGS])
        assert result.exit_code == 0
        assert RUN_ID in result.output

    def test_list_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.runs.api.list_runs", side_effect=APIError("fail")):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["runs", "list", *_TOKEN_ARGS])
        assert result.exit_code == 1

    def test_list_no_token_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRISM_USER_API_TOKEN", raising=False)
        result = _runner.invoke(app, ["runs", "list"])
        assert result.exit_code == 1


@pytest.mark.unit
class TestRunsCreate:
    def test_create_success(self, run_folder: Path) -> None:
        with patch(
            "endpoints_submission_cli.runs.api.create_run", return_value=RUN_OUT
        ) as mock_create:
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                with patch("endpoints_submission_cli.runs.api.upload_run_archive"):
                    _run_app("runs", "create", "--path", str(run_folder), *_TOKEN_ARGS)
        mock_create.assert_called_once()

    def test_create_rollback_on_upload_failure(self, run_folder: Path) -> None:
        with patch("endpoints_submission_cli.runs.api.create_run", return_value=RUN_OUT):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                with patch(
                    "endpoints_submission_cli.runs.api.upload_run_archive",
                    side_effect=APIError("upload failed"),
                ):
                    with patch("endpoints_submission_cli.runs.api.delete_run") as mock_delete:
                        result = _runner.invoke(
                            app, ["runs", "create", "--path", str(run_folder), *_TOKEN_ARGS]
                        )
        assert result.exit_code == 1
        mock_delete.assert_called_once_with(TOKEN, RUN_ID)

    def test_create_invalid_folder_exits_1(self, tmp_path: Path) -> None:
        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            result = _runner.invoke(
                app,
                ["runs", "create", "--path", str(tmp_path / "nonexistent"), *_TOKEN_ARGS],
            )
        assert result.exit_code == 1

    def test_create_api_error_exits_1(self, run_folder: Path) -> None:
        with patch("endpoints_submission_cli.runs.api.create_run", side_effect=APIError("500")):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app, ["runs", "create", "--path", str(run_folder), *_TOKEN_ARGS]
                )
        assert result.exit_code == 1

    def test_create_test_flag_sets_is_test(self, run_folder: Path) -> None:
        with patch(
            "endpoints_submission_cli.runs.api.create_run", return_value=RUN_OUT
        ) as mock_create:
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                with patch("endpoints_submission_cli.runs.api.upload_run_archive"):
                    _run_app("runs", "create", "--path", str(run_folder), "--test", *_TOKEN_ARGS)
        assert mock_create.call_args[0][1]["is_test"] is True

    def test_create_without_test_flag_omits_is_test(self, run_folder: Path) -> None:
        """Absent flag sends nothing, leaving the API's own default to apply."""
        with patch(
            "endpoints_submission_cli.runs.api.create_run", return_value=RUN_OUT
        ) as mock_create:
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                with patch("endpoints_submission_cli.runs.api.upload_run_archive"):
                    _run_app("runs", "create", "--path", str(run_folder), *_TOKEN_ARGS)
        assert "is_test" not in mock_create.call_args[0][1]

    def test_create_test_flag_visible_in_dry_run(self, run_folder: Path) -> None:
        result = _runner.invoke(
            app, ["runs", "create", "--path", str(run_folder), "--test", "--dry-run"]
        )
        assert result.exit_code == 0
        assert json.loads(result.output)["is_test"] is True


@pytest.mark.unit
class TestRunsGet:
    def test_get_json_flag(self) -> None:
        with patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app, ["runs", "get", "--run-id", RUN_ID, "-j", *_TOKEN_ARGS]
                )
        assert result.exit_code == 0
        assert RUN_ID in result.output
        # Raw JSON, not the table — a quoted key only the json.dumps path emits.
        assert '"benchmark_version":' in result.output

    def test_get_renders_table_by_default(self) -> None:
        with patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["runs", "get", "--run-id", RUN_ID, *_TOKEN_ARGS])
        assert result.exit_code == 0
        # A row label the table path emits and the JSON path never does.
        assert "Benchmark Version" in result.output
        assert '"benchmark_version":' not in result.output

    def test_get_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.runs.api.get_run", side_effect=APIError("404")):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["runs", "get", "--run-id", RUN_ID, *_TOKEN_ARGS])
        assert result.exit_code == 1

    def test_get_download_to_saves_archive(self, tmp_path: Path) -> None:
        archive = tmp_path / f"{RUN_ID}.tar.gz"
        with patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT):
            with patch(
                "endpoints_submission_cli.runs.api.download_run_archive",
                return_value=archive,
            ) as mock_dl:
                with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                    result = _runner.invoke(
                        app,
                        [
                            "runs",
                            "get",
                            "--run-id",
                            RUN_ID,
                            "--download-to",
                            str(tmp_path),
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(TOKEN, RUN_ID, tmp_path)
        assert "Archive saved to" in result.output

    def test_get_download_api_error_exits_1(self, tmp_path: Path) -> None:
        with patch("endpoints_submission_cli.runs.api.get_run", return_value=RUN_OUT):
            with patch(
                "endpoints_submission_cli.runs.api.download_run_archive",
                side_effect=APIError("download failed"),
            ):
                with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                    result = _runner.invoke(
                        app,
                        [
                            "runs",
                            "get",
                            "--run-id",
                            RUN_ID,
                            "--download-to",
                            str(tmp_path),
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 1


@pytest.mark.unit
class TestRunsDelete:
    def test_delete_success(self) -> None:
        with (
            patch("endpoints_submission_cli.runs.api.delete_run_archive"),
            patch("endpoints_submission_cli.runs.api.delete_run"),
            patch("endpoints_submission_cli._http.get_token", return_value=TOKEN),
        ):
            _run_app("runs", "delete", "--run-id", RUN_ID, *_TOKEN_ARGS)

    def test_delete_no_archive_is_silent(self) -> None:
        with (
            patch(
                "endpoints_submission_cli.runs.api.delete_run_archive",
                side_effect=APIError("API error 404: No archive uploaded yet"),
            ),
            patch("endpoints_submission_cli.runs.api.delete_run"),
            patch("endpoints_submission_cli._http.get_token", return_value=TOKEN),
        ):
            _run_app("runs", "delete", "--run-id", RUN_ID, *_TOKEN_ARGS)

    def test_delete_archive_non404_failure_is_warning(self) -> None:
        with (
            patch(
                "endpoints_submission_cli.runs.api.delete_run_archive",
                side_effect=APIError("API error 500: GCS unavailable"),
            ),
            patch("endpoints_submission_cli.runs.api.delete_run"),
            patch("endpoints_submission_cli._http.get_token", return_value=TOKEN),
        ):
            _run_app("runs", "delete", "--run-id", RUN_ID, *_TOKEN_ARGS)

    def test_delete_api_error_exits_1(self) -> None:
        with (
            patch("endpoints_submission_cli.runs.api.delete_run_archive"),
            patch(
                "endpoints_submission_cli.runs.api.delete_run",
                side_effect=APIError("run in submission"),
            ),
            patch("endpoints_submission_cli._http.get_token", return_value=TOKEN),
        ):
            result = _runner.invoke(app, ["runs", "delete", "--run-id", RUN_ID, *_TOKEN_ARGS])
        assert result.exit_code == 1


@pytest.mark.unit
class TestRunsPinUnpin:
    def test_pin_success(self) -> None:
        with patch("endpoints_submission_cli.runs.api.pin_run"):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app("runs", "pin", "--run-id", RUN_ID, *_TOKEN_ARGS)

    def test_unpin_success(self) -> None:
        with patch("endpoints_submission_cli.runs.api.unpin_run"):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app("runs", "unpin", "--run-id", RUN_ID, *_TOKEN_ARGS)

    def test_pin_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.runs.api.pin_run", side_effect=APIError("404")):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["runs", "pin", "--run-id", RUN_ID, *_TOKEN_ARGS])
        assert result.exit_code == 1

    def test_unpin_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli.runs.api.unpin_run", side_effect=APIError("404")):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["runs", "unpin", "--run-id", RUN_ID, *_TOKEN_ARGS])
        assert result.exit_code == 1
