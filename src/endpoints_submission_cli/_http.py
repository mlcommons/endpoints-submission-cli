# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""HTTP transport layer for the MLPerf Submission API (httpx-based).

All HTTP plumbing lives here: timeouts, base-URL resolution, auth header,
error translators, and the four verb helpers used by the domain API modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NoReturn

import httpx

from .exceptions import APIError, AuthError

__all__ = [
    "get_token",
    "_base_url",
    "_headers",
    "_raise_status",
    "_raise_request",
    "_get",
    "_post",
    "_patch",
    "_delete",
    "_put_to_signed_url",
    "_UPLOAD_TIMEOUT",
    "_DOWNLOAD_TIMEOUT",
    "_API_TIMEOUT",
]

_DEFAULT_BASE_URL = "https://mlperf-endpoints-api-50577619532.us-central1.run.app"

# Per-operation idle timeouts (time without a byte transferred), not wall-clock totals.
# A slow-but-steady transfer never times out; only a stalled connection does.
# write: each chunk send; read: wait for server ack after full upload
_UPLOAD_TIMEOUT = httpx.Timeout(connect=10.0, write=300.0, read=120.0, pool=5.0)
# read: gap between received bytes; accommodates large archives on slow links
_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0)
_API_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=5.0)  # standard JSON calls


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


# Translates an httpx HTTP error (4xx/5xx response) into our own APIError/AuthError.
def _raise_status(exc: httpx.HTTPStatusError) -> NoReturn:
    if exc.response.status_code in (401, 403):
        raise AuthError(f"Authentication failed (HTTP {exc.response.status_code})") from exc
    body = exc.response.text[:500]  # truncate to avoid dumping full HTML/JSON error pages into logs
    raise APIError(f"API error {exc.response.status_code}: {body}") from exc


# Translates an httpx network/transport error (timeout, connection refused, DNS failure, etc.)
def _raise_request(exc: httpx.RequestError) -> NoReturn:
    raise APIError(f"Request failed: {type(exc).__name__}: {exc}") from exc


def _put_to_signed_url(url: str, archive_path: Path) -> None:
    """PUT a local file directly to a pre-signed URL (GCS/S3).

    No auth headers are sent — the signature is embedded in the URL.
    The file is streamed chunk-by-chunk to avoid loading it into memory.
    """
    try:
        with open(archive_path, "rb") as fh:
            r = httpx.put(
                url,
                content=fh,
                headers={"Content-Type": "application/octet-stream"},
                timeout=_UPLOAD_TIMEOUT,
            )
            r.raise_for_status()
    except OSError as exc:
        from .exceptions import ArchiveError
        raise ArchiveError(f"Failed to open archive for upload: {archive_path}") from exc
    except httpx.HTTPStatusError as exc:
        _raise_status(exc)
    except httpx.RequestError as exc:
        _raise_request(exc)


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
