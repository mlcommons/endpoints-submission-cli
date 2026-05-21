# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""HTTP client for the MLPerf Submission API (httpx-based)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from .exceptions import APIError, AuthError

__all__ = [
    "get_token",
    "list_runs",
    "create_run",
    "get_run",
    "delete_run",
    "pin_run",
    "unpin_run",
    "upload_run_archive",
    "delete_run_archive",
    "download_run_archive",
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

_DEFAULT_BASE_URL = "http://localhost:8080"

# Per-operation idle timeouts (not wall-clock totals).
# write=300: each write syscall can stall up to 5 min — handles large uploads on slow links.
# read=120:  server must send the first response byte within 2 min after receiving the file.
_UPLOAD_TIMEOUT = httpx.Timeout(connect=10.0, write=300.0, read=120.0, pool=5.0)
_API_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=5.0)


def _base_url() -> str:
    return os.environ.get("MLPERF_API_BASE_URL", _DEFAULT_BASE_URL)


def get_token(explicit_token: str | None) -> str:
    """Resolve the API token from the explicit flag or PRISM_USER_API_TOKEN env var."""
    token = explicit_token or os.environ.get("PRISM_USER_API_TOKEN")
    if not token:
        raise AuthError(
            "No API token provided. Pass --token or set PRISM_USER_API_TOKEN."
        )
    return token


def _headers(token: str) -> dict[str, str]:
    return {"X-API-Key": token}


def _raise_status(exc: httpx.HTTPStatusError) -> None:
    if exc.response.status_code in (401, 403):
        raise AuthError(f"Authentication failed (HTTP {exc.response.status_code})") from exc
    body = exc.response.text[:500]
    raise APIError(f"API error {exc.response.status_code}: {body}") from exc


def _raise_request(exc: httpx.RequestError) -> None:
    raise APIError(f"Request failed: {type(exc).__name__}: {exc}") from exc


def _get(path: str, token: str, **kwargs: Any) -> Any:
    try:
        r = httpx.get(
            f"{_base_url()}{path}", headers=_headers(token), timeout=_API_TIMEOUT, **kwargs
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        _raise_status(exc)
    except httpx.RequestError as exc:
        _raise_request(exc)


def _post(path: str, token: str, **kwargs: Any) -> Any:
    try:
        r = httpx.post(
            f"{_base_url()}{path}", headers=_headers(token), timeout=_API_TIMEOUT, **kwargs
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        _raise_status(exc)
    except httpx.RequestError as exc:
        _raise_request(exc)


def _patch(path: str, token: str, **kwargs: Any) -> Any:
    try:
        r = httpx.patch(
            f"{_base_url()}{path}", headers=_headers(token), timeout=_API_TIMEOUT, **kwargs
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        _raise_status(exc)
    except httpx.RequestError as exc:
        _raise_request(exc)


def _delete(path: str, token: str, **kwargs: Any) -> Any:
    try:
        r = httpx.delete(
            f"{_base_url()}{path}", headers=_headers(token), timeout=_API_TIMEOUT, **kwargs
        )
        r.raise_for_status()
        # 204 No Content returns empty body
        if r.status_code == 204:
            return None
        return r.json()
    except httpx.HTTPStatusError as exc:
        _raise_status(exc)
    except httpx.RequestError as exc:
        _raise_request(exc)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def list_runs(token: str) -> list[dict[str, Any]]:
    """GET /runs — list all runs for the authenticated user."""
    result = _get("/runs", token)
    return result  # type: ignore[return-value]


def create_run(token: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST /runs — register a new run."""
    result = _post("/runs", token, json=payload)
    return result  # type: ignore[return-value]


def get_run(token: str, run_id: str) -> dict[str, Any]:
    """GET /runs/{run_id} — fetch full run details."""
    result = _get(f"/runs/{run_id}", token)
    return result  # type: ignore[return-value]


def delete_run(token: str, run_id: str) -> None:
    """DELETE /runs/{run_id} — delete the run record."""
    _delete(f"/runs/{run_id}", token)


def pin_run(token: str, run_id: str) -> None:
    """PATCH /runs/{run_id}/pin — pin a run to prevent expiry."""
    _patch(f"/runs/{run_id}/pin", token)


def unpin_run(token: str, run_id: str) -> None:
    """PATCH /runs/{run_id}/unpin — unpin a run to restore normal expiry."""
    _patch(f"/runs/{run_id}/unpin", token)


def upload_run_archive(token: str, run_id: str, archive_path: Path) -> dict[str, Any]:
    """POST /runs/{run_id}/archive — upload the run archive (multipart/form-data)."""
    try:
        with open(archive_path, "rb") as fh:
            r = httpx.post(
                f"{_base_url()}/runs/{run_id}/archive",
                headers={"X-API-Key": token},
                files={"archive": (archive_path.name, fh, "application/octet-stream")},
                timeout=_UPLOAD_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as exc:
        _raise_status(exc)
    except httpx.RequestError as exc:
        _raise_request(exc)
    return {}  # unreachable


def delete_run_archive(token: str, run_id: str) -> None:
    """DELETE /runs/{run_id}/archive — remove the stored archive."""
    _delete(f"/runs/{run_id}/archive", token)


def download_run_archive(token: str, run_id: str, dest_dir: Path) -> Path:
    """GET /runs/{run_id}/archive — download the run archive to dest_dir.

    Returns the path of the saved file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{run_id}.tar.gz"
    try:
        with httpx.stream(
            "GET",
            f"{_base_url()}/runs/{run_id}/archive",
            headers=_headers(token),
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


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------


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
    """POST /submissions/{submission_id}/archive — upload the submission bundle."""
    try:
        with open(archive_path, "rb") as fh:
            r = httpx.post(
                f"{_base_url()}/submissions/{submission_id}/archive",
                headers={"X-API-Key": token},
                files={"archive": (archive_path.name, fh, "application/octet-stream")},
                timeout=_UPLOAD_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as exc:
        _raise_status(exc)
    except httpx.RequestError as exc:
        _raise_request(exc)
    return {}  # unreachable


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
