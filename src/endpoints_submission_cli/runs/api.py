# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Runs API client — all /runs endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import httpx

from .._http import (
    _DOWNLOAD_TIMEOUT,
    _delete,
    _get,
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
    return cast(list[dict[str, Any]], _get("/runs", token))


def create_run(token: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST /runs — register a new run."""
    return cast(dict[str, Any], _post("/runs", token, json=payload))


def get_run(token: str, run_id: str) -> dict[str, Any]:
    """GET /runs/{run_id} — fetch full run details."""
    return cast(dict[str, Any], _get(f"/runs/{run_id}", token))


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
    result = cast(dict[str, Any], _get(f"/runs/{run_id}/archive/upload-url", token))
    _put_to_signed_url(result["upload_url"], archive_path)
    return result


def delete_run_archive(token: str, run_id: str) -> None:
    """DELETE /runs/{run_id}/archive — remove the stored archive."""
    _delete(f"/runs/{run_id}/archive", token)


def download_run_archive(token: str, run_id: str, dest_dir: Path) -> Path:
    """Download the run archive to dest_dir. Returns the saved file path.

    GET /runs/{run_id}/archive → {"download_url": "...", "expires_in": 300}
    GET <download_url>         → streams file directly from object storage
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{run_id}.tar.gz"
    result = cast(dict[str, Any], _get(f"/runs/{run_id}/archive", token))
    download_url = result["download_url"]
    try:
        with httpx.stream("GET", download_url, timeout=_DOWNLOAD_TIMEOUT) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
    except httpx.HTTPStatusError as exc:
        _raise_status(exc)
    except httpx.RequestError as exc:
        _raise_request(exc)
    return dest
