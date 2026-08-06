# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Resolve the installed CLI version for reporting to the API and bundle markers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

__all__ = ["cli_version"]

_PACKAGE = "endpoints-submission-cli"


def cli_version() -> str:
    """Return the installed CLI version, or ``"unknown"`` if it can't be resolved."""
    try:
        return _pkg_version(_PACKAGE)
    except PackageNotFoundError:
        return "unknown"
