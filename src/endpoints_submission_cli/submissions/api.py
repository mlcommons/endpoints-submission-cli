# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Submissions API client — all /submissions endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .._http import (
    _DOWNLOAD_TIMEOUT,
    _base_url,
    _delete,
    _get,
    _headers,
    _patch,
    _post,
    _put_to_signed_url,
    _raise_request,
    _raise_status,
)

__all__ = [
    "list_submissions",
    "create_submission",
    "get_submission",
    "update_submission",
    "withdraw_submission",
    "add_run_to_submission",
    "remove_run_from_submission",
    "upload_submission_archive",
    "delete_submission_archive",
    "download_submission_archive",
]


def list_submissions(token: str) -> list[dict[str, Any]]:
    """GET /submissions — list all submissions for the authenticated user."""
    result = _get("/submissions", token)
    return result  # type: ignore[return-value]


def create_submission(token: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST /submissions — create a new submission."""
    result = _post("/submissions", token, json=payload)
    return result  # type: ignore[return-value]


def get_submission(token: str, submission_id: str, include_runs: bool = True) -> dict[str, Any]:
    """GET /submissions/{submission_id} — fetch submission details."""
    result = _get(
        f"/submissions/{submission_id}",
        token,
        params={"include_runs": include_runs},
    )
    return result  # type: ignore[return-value]


def update_submission(
    token: str, submission_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """PATCH /submissions/{submission_id} — update submission fields."""
    result = _patch(f"/submissions/{submission_id}", token, json=payload)
    return result  # type: ignore[return-value]


def withdraw_submission(token: str, submission_id: str) -> dict[str, Any]:
    """DELETE /submissions/{submission_id} — withdraw the submission."""
    result = _delete(f"/submissions/{submission_id}", token)
    return result  # type: ignore[return-value]


def add_run_to_submission(
    token: str, submission_id: str, run_id: str
) -> dict[str, Any]:
    """POST /submissions/{submission_id}/runs/{run_id} — add a run to a submission."""
    result = _post(f"/submissions/{submission_id}/runs/{run_id}", token)
    return result  # type: ignore[return-value]


def remove_run_from_submission(
    token: str, submission_id: str, run_id: str
) -> dict[str, Any]:
    """DELETE /submissions/{submission_id}/runs/{run_id} — remove a run from a submission."""
    result = _delete(f"/submissions/{submission_id}/runs/{run_id}", token)
    return result  # type: ignore[return-value]


def upload_submission_archive(
    token: str, submission_id: str, archive_path: Path
) -> dict[str, Any]:
    """Upload a submission bundle via a server-issued signed URL.

    GET /submissions/{submission_id}/archive/upload-url → {"upload_url": "...", "expires_in": 3600}
    PUT <upload_url>                                    → streams file directly to object storage
    """
    result = _get(f"/submissions/{submission_id}/archive/upload-url", token)
    _put_to_signed_url(result["upload_url"], archive_path)
    return result


def delete_submission_archive(token: str, submission_id: str) -> None:
    """DELETE /submissions/{submission_id}/archive — remove the stored submission bundle."""
    _delete(f"/submissions/{submission_id}/archive", token)


def download_submission_archive(
    token: str, submission_id: str, dest_dir: Path
) -> Path:
    """GET /submissions/{submission_id}/archive — download submission bundle to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{submission_id}.tar.gz"
    try:
        with httpx.stream(
            "GET",
            f"{_base_url()}/submissions/{submission_id}/archive",
            headers=_headers(token),
            timeout=_DOWNLOAD_TIMEOUT,
        ) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
    except httpx.HTTPStatusError as exc:
        _raise_status(exc)
    except httpx.RequestError as exc:
        _raise_request(exc)
    return dest
