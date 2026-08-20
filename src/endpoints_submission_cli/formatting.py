# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Shared value formatters for Rich table cells.

Every API response reaches the formatters as a plain ``dict``, so a field the
server did not send is indistinguishable from one it sent as null. These helpers
give both cases a single rendering (:data:`DASH`) while keeping real falsy values
— ``0`` and ``False`` — visible.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DASH", "fmt_bool", "fmt_dt", "fmt_int", "fmt_str"]

DASH = "—"


def fmt_dt(value: Any) -> str:
    """Render an ISO timestamp truncated to seconds.

    Args:
        value: An ISO-8601 timestamp string, or ``None`` if absent.

    Returns:
        The timestamp as ``YYYY-MM-DD HH:MM:SS``, or :data:`DASH` if missing.
    """
    if not value:
        return DASH
    return str(value).replace("T", " ").split(".")[0]


def fmt_bool(value: Any) -> str:
    """Render a tri-state boolean.

    ``None`` means the field was absent from the response — for example against an
    API server older than the field — and renders as :data:`DASH` rather than
    silently reading as ``No``.

    Args:
        value: A boolean, or ``None`` if absent.

    Returns:
        ``"Yes"``, ``"No"``, or :data:`DASH`.
    """
    if value is None:
        return DASH
    return "Yes" if value else "No"


def fmt_int(value: Any) -> str:
    """Render an integer field.

    ``0`` is a real value and must not collapse to a dash, which is why the
    ``value or DASH`` idiom cannot be used for counts such as ``reviewers_assigned``
    where zero is the most common reading.

    Args:
        value: An integer, or ``None`` if absent.

    Returns:
        The integer as a string, or :data:`DASH` if ``None``.
    """
    if value is None:
        return DASH
    return str(value)


def fmt_str(value: Any) -> str:
    """Render a string field.

    Args:
        value: A value to stringify, or ``None``/``""`` if absent.

    Returns:
        The value as a string, or :data:`DASH` if it is ``None`` or empty.
    """
    if value is None or value == "":
        return DASH
    return str(value)
