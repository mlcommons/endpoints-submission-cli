# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for submissions.api module (all HTTP calls mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from endpoints_submission_cli.exceptions import APIError, ArchiveError
from endpoints_submission_cli.submissions.api import (
    add_run_to_submission,
    create_submission,
    delete_submission_archive,
    get_submission,
    list_submissions,
    remove_run_from_submission,
    update_submission,
    upload_submission_archive,
    withdraw_submission,
)

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
class TestUploadDeleteSubmissionArchive:
    def test_upload_ok(self, tmp_path: Path) -> None:
        archive = tmp_path / "bundle.tar.gz"
        archive.write_bytes(b"bundle")
        with patch("httpx.post", return_value=_mock_response(200, {})):
            upload_submission_archive(TOKEN, SUBMISSION_ID, archive)

    def test_missing_file_raises_archive_error(self, tmp_path: Path) -> None:
        archive = tmp_path / "nonexistent.tar.gz"
        with pytest.raises(ArchiveError):
            upload_submission_archive(TOKEN, SUBMISSION_ID, archive)

    def test_delete_ok(self) -> None:
        resp = _mock_response(204)
        resp.json.side_effect = Exception("no body")
        with patch("httpx.delete", return_value=resp):
            delete_submission_archive(TOKEN, SUBMISSION_ID)
