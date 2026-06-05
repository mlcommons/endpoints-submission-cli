# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runs.api module (all HTTP calls mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from endpoints_submission_cli.exceptions import APIError, ArchiveError, AuthError
from endpoints_submission_cli.runs.api import (
    create_run,
    delete_run,
    delete_run_archive,
    download_run_archive,
    get_run,
    list_runs,
    pin_run,
    unpin_run,
    upload_run_archive,
)

RUN_ID = "d5d9873e-5eca-4f8d-a487-4be1cb8b440c"
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
    _SIGNED_RESP = {"upload_url": "https://storage.example.com/signed", "expires_in": 3600}

    def test_upload_ok(self, tmp_path: Path) -> None:
        archive = tmp_path / "run.tar.gz"
        archive.write_bytes(b"fake data")
        with patch("httpx.get", return_value=_mock_response(200, self._SIGNED_RESP)):
            with patch("httpx.put", return_value=_mock_response(200)):
                upload_run_archive(TOKEN, RUN_ID, archive)

    def test_http_error_on_sign_raises(self, tmp_path: Path) -> None:
        archive = tmp_path / "run.tar.gz"
        archive.write_bytes(b"fake data")
        with patch("httpx.get", side_effect=_mock_http_error(500)):
            with pytest.raises(APIError):
                upload_run_archive(TOKEN, RUN_ID, archive)

    def test_http_error_on_put_raises(self, tmp_path: Path) -> None:
        archive = tmp_path / "run.tar.gz"
        archive.write_bytes(b"fake data")
        with patch("httpx.get", return_value=_mock_response(200, self._SIGNED_RESP)):
            with patch("httpx.put", side_effect=_mock_http_error(403)):
                with pytest.raises(APIError):
                    upload_run_archive(TOKEN, RUN_ID, archive)

    def test_missing_file_raises_archive_error(self, tmp_path: Path) -> None:
        archive = tmp_path / "nonexistent.tar.gz"
        with patch("httpx.get", return_value=_mock_response(200, self._SIGNED_RESP)):
            with pytest.raises(ArchiveError):
                upload_run_archive(TOKEN, RUN_ID, archive)


@pytest.mark.unit
class TestDeleteRunArchive:
    def test_delete_ok(self) -> None:
        resp = _mock_response(204)
        resp.json.side_effect = Exception("no body")
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

        with patch("httpx.stream", return_value=mock_stream):
            result = download_run_archive(TOKEN, RUN_ID, tmp_path)

        assert result.exists()
        assert result.read_bytes() == b"data1data2"
