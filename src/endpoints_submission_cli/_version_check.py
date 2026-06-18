# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Background PyPI version check with a 24-hour file cache.

Usage::

    from endpoints_submission_cli._version_check import register_upgrade_notice
    register_upgrade_notice()  # call once at CLI entry

The check never blocks the CLI: if the cache is stale a daemon thread fetches
the latest version from PyPI while the command runs, then an atexit handler
prints a one-line notice (to stderr) if a newer release exists.
"""

from __future__ import annotations

import atexit
import json
import threading
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from urllib.request import urlopen

__all__ = ["register_upgrade_notice"]

_PACKAGE = "endpoints-submission-cli"
_PYPI_URL = f"https://pypi.org/pypi/{_PACKAGE}/json"
_CACHE_PATH = Path.home() / ".cache" / _PACKAGE / "version_check.json"
_CACHE_TTL = 24 * 3600  # seconds
_TIMEOUT = 3.0  # PyPI request timeout


def _current_version() -> str | None:
    try:
        return _pkg_version(_PACKAGE)
    except PackageNotFoundError:
        return None


def _cached_latest() -> str | None:
    try:
        data = json.loads(_CACHE_PATH.read_text())
        if time.time() - data["ts"] < _CACHE_TTL:
            return str(data["latest"])
    except Exception:
        pass
    return None


def _fetch_latest() -> str | None:
    try:
        with urlopen(_PYPI_URL, timeout=_TIMEOUT) as resp:
            latest: str = json.loads(resp.read())["info"]["version"]
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_PATH.write_text(json.dumps({"latest": latest, "ts": time.time()}))
        except Exception:
            pass
        return latest
    except Exception:
        return None


def _is_newer(latest: str, current: str) -> bool:
    def _parts(v: str) -> tuple[int, ...]:
        try:
            return tuple(int(x) for x in v.split(".")[:3])
        except ValueError:
            return (0,)

    return _parts(latest) > _parts(current)


def register_upgrade_notice() -> None:
    """Register a background version check and an atexit upgrade notice.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    current = _current_version()
    if not current:
        return

    latest = _cached_latest()
    result: list[str | None] = [latest]
    thread: threading.Thread | None = None

    if latest is None:

        def _fetch() -> None:
            result[0] = _fetch_latest()

        thread = threading.Thread(target=_fetch, daemon=True)
        thread.start()

    def _on_exit() -> None:
        if thread is not None:
            thread.join(timeout=_TIMEOUT + 0.5)
        fetched = result[0]
        if fetched and _is_newer(fetched, current):
            # Import here to avoid a top-level Rich import cost on every invocation.
            from rich.console import Console

            Console(stderr=True).print(
                f"\n[dim]A new version of [bold]{_PACKAGE}[/bold] is available: "
                f"[yellow]{current}[/yellow] → [green]{fetched}[/green]  "
                f"([bold]pip install --upgrade {_PACKAGE}[/bold])[/dim]"
            )

    atexit.register(_on_exit)
