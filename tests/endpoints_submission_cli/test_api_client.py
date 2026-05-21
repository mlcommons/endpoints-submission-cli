# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for api_client module (all HTTP calls mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from endpoints_submission_cli.api_client import (
    add_run_to_submission,
    create_run,
    create_submission,
    delete_run,
    delete_run_archive,
    delete_submission_archive,
    download_run_archive,
    get_run,
    get_submission,
    get_token,
    list_runs,
    list_submissions,
    pin_run,
    remove_run_from_submission,
    unpin_run,
    update_submission,
    upload_run_archive,
    upload_submission_archive,
    withdraw_submission,
)
from endpoints_submission_cli.exceptions import APIError, AuthError

RUN_ID = "d5d9873e-5eca-4f8d-a487-4be1cb8b440c"
SUBMISSION_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
TOKEN = "mlc_testtoken"


def _mock_response(status_code: int = 200, json_data: object = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_http_error(status_code: int) -> httpx.HTTPStatusError:
    request = MagicMock(spec=httpx.Request)
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = f"HTTP {status_code}"
    return httpx.HTTPStatusError("error", request=request, response=response)


@pytest.mark.unit
class TestGetToken:
    def test_explicit_token_returned(self) -> None:
        assert get_token("mlc_abc") == "mlc_abc"

    def test_env_var_used_when_no_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRISM_USER_API_TOKEN", "mlc_env")
        assert get_token(None) == "mlc_env"

    def test_explicit_takes_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRISM_USER_API_TOKEN", "mlc_env")
        assert get_token("mlc_explicit") == "mlc_explicit"

    def test_no_token_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRISM_USER_API_TOKEN", raising=False)
        with pytest.raises(AuthError):
            get_token(None)


@pytest.mark.unit
class TestListRuns:
    def test_returns_list(self) -> None:
        runs = [{"id": RUN_ID, "model": "llama", "concurrency": 4}]
        with patch("httpx.get", return_value=_mock_response(200, runs)):
            result = list_runs(TOKEN)
        assert result == runs

    def test_http_error_raises_api_error(self) -> None:
        with patch("httpx.get", side_effect=_mock_http_error(500)), pytest.raises(APIError):
            list_runs(TOKEN)

    def test_401_raises_auth_error(self) -> None:
        with patch("httpx.get", side_effect=_mock_http_error(401)):
            with pytest.raises(AuthError):
                list_runs(TOKEN)


@pytest.mark.unit
class TestCreateRun:
    def test_returns_run_out(self) -> None:
        payload = {"benchmark_version": "abc", "started_at": "2025-01-01T00:00:00"}
        out = {"id": RUN_ID, "user_id": "u_test"}
        with patch("httpx.post", return_value=_mock_response(201, out)):
            result = create_run(TOKEN, payload)
        assert result["id"] == RUN_ID

    def test_api_error_propagated(self) -> None:
        with patch("httpx.post", side_effect=_mock_http_error(422)):
            with pytest.raises(APIError):
                create_run(TOKEN, {})


@pytest.mark.unit
class TestGetRun:
    def test_returns_run_out(self) -> None:
        out = {"id": RUN_ID}
        with patch("httpx.get", return_value=_mock_response(200, out)):
            result = get_run(TOKEN, RUN_ID)
        assert result["id"] == RUN_ID


@pytest.mark.unit
class TestDeleteRun:
    def test_deletes_ok(self) -> None:
        with patch("httpx.delete", return_value=_mock_response(200, {})):
            delete_run(TOKEN, RUN_ID)

    def test_api_error_propagated(self) -> None:
        with patch("httpx.delete", side_effect=_mock_http_error(404)):
            with pytest.raises(APIError):
                delete_run(TOKEN, RUN_ID)


@pytest.mark.unit
class TestPinUnpin:
    def test_pin(self) -> None:
        with patch("httpx.patch", return_value=_mock_response(200, {})):
            pin_run(TOKEN, RUN_ID)

    def test_unpin(self) -> None:
        with patch("httpx.patch", return_value=_mock_response(200, {})):
            unpin_run(TOKEN, RUN_ID)


@pytest.mark.unit
class TestUploadRunArchive:
    def test_upload_ok(self, tmp_path: Path) -> None:
        archive = tmp_path / "run.tar.gz"
        archive.write_bytes(b"fake data")
        with patch("httpx.post", return_value=_mock_response(200, {})):
            upload_run_archive(TOKEN, RUN_ID, archive)

    def test_http_error_raises(self, tmp_path: Path) -> None:
        archive = tmp_path / "run.tar.gz"
        archive.write_bytes(b"fake data")
        with patch("httpx.post", side_effect=_mock_http_error(500)):
            with pytest.raises(APIError):
                upload_run_archive(TOKEN, RUN_ID, archive)


@pytest.mark.unit
class TestDeleteRunArchive:
    def test_delete_ok(self) -> None:
        resp = _mock_response(204)
        resp.json.side_effect = Exception("no body")
        resp.status_code = 204
        with patch("httpx.delete", return_value=resp):
            delete_run_archive(TOKEN, RUN_ID)


@pytest.mark.unit
class TestDownloadRunArchive:
    def test_download_creates_file(self, tmp_path: Path) -> None:
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.raise_for_status = MagicMock()
        mock_stream.iter_bytes = MagicMock(return_value=[b"data1", b"data2"])
        mock_stream.status_code = 200

        with patch("httpx.stream", return_value=mock_stream):
            result = download_run_archive(TOKEN, RUN_ID, tmp_path)

        assert result.exists()
        assert result.read_bytes() == b"data1data2"


@pytest.mark.unit
class TestListSubmissions:
    def test_returns_list(self) -> None:
        subs = [{"id": SUBMISSION_ID, "status": "COMPLIANCE_CHECKING"}]
        with patch("httpx.get", return_value=_mock_response(200, subs)):
            result = list_submissions(TOKEN)
        assert result == subs


@pytest.mark.unit
class TestCreateSubmission:
    def test_returns_submission_out(self) -> None:
        out = {"id": SUBMISSION_ID}
        with patch("httpx.post", return_value=_mock_response(201, out)):
            result = create_submission(TOKEN, {"division": "standardized", "run_ids": [RUN_ID]})
        assert result["id"] == SUBMISSION_ID


@pytest.mark.unit
class TestGetSubmission:
    def test_returns_submission_with_runs(self) -> None:
        out = {"id": SUBMISSION_ID, "runs": []}
        with patch("httpx.get", return_value=_mock_response(200, out)):
            result = get_submission(TOKEN, SUBMISSION_ID)
        assert result["id"] == SUBMISSION_ID


@pytest.mark.unit
class TestUpdateSubmission:
    def test_patch_ok(self) -> None:
        out = {"id": SUBMISSION_ID, "status": "REVIEW_PENDING"}
        with patch("httpx.patch", return_value=_mock_response(200, out)):
            result = update_submission(TOKEN, SUBMISSION_ID, {"status": "REVIEW_PENDING"})
        assert result["status"] == "REVIEW_PENDING"


@pytest.mark.unit
class TestWithdrawSubmission:
    def test_withdraw_ok(self) -> None:
        out = {"id": SUBMISSION_ID, "status": "WITHDRAWN"}
        with patch("httpx.delete", return_value=_mock_response(200, out)):
            result = withdraw_submission(TOKEN, SUBMISSION_ID)
        assert result["status"] == "WITHDRAWN"


@pytest.mark.unit
class TestAddRemoveRunSubmission:
    def test_add_run(self) -> None:
        out = {"id": SUBMISSION_ID, "run_ids": [RUN_ID]}
        with patch("httpx.post", return_value=_mock_response(200, out)):
            result = add_run_to_submission(TOKEN, SUBMISSION_ID, RUN_ID)
        assert RUN_ID in result["run_ids"]

    def test_remove_run(self) -> None:
        out = {"id": SUBMISSION_ID, "run_ids": []}
        with patch("httpx.delete", return_value=_mock_response(200, out)):
            result = remove_run_from_submission(TOKEN, SUBMISSION_ID, RUN_ID)
        assert result["run_ids"] == []


@pytest.mark.unit
class TestUploadDownloadSubmissionArchive:
    def test_upload_ok(self, tmp_path: Path) -> None:
        archive = tmp_path / "bundle.tar.gz"
        archive.write_bytes(b"bundle")
        with patch("httpx.post", return_value=_mock_response(200, {})):
            upload_submission_archive(TOKEN, SUBMISSION_ID, archive)

    def test_delete_ok(self) -> None:
        resp = _mock_response(204)
        resp.status_code = 204
        resp.json.side_effect = Exception("no body")
        with patch("httpx.delete", return_value=resp):
            delete_submission_archive(TOKEN, SUBMISSION_ID)
