# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Runs API client — all /runs endpoints."""

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
    "list_runs",
    "create_run",
    "get_run",
    "delete_run",
    "pin_run",
    "unpin_run",
    "upload_run_archive",
    "delete_run_archive",
    "download_run_archive",
]


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
    """Upload a run archive via a server-issued signed URL.

    GET /runs/{run_id}/archive/upload-url → {"upload_url": "...", "expires_in": 3600}
    PUT <upload_url>                      → streams file directly to object storage
    """
    result = _get(f"/runs/{run_id}/archive/upload-url", token)
    _put_to_signed_url(result["upload_url"], archive_path)
    return result


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
