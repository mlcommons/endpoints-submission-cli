# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Suite-wide fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from submission_checker import checker, messages

#: The instant the suite checks submissions as of: inside cohort 2026-10-C1, which the
#: bundled seed set and approved drafters are valid for, so results don't drift with
#: the calendar.
PINNED_NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _strict_messages() -> Iterator[None]:
    """A check result whose message cannot render fails the test that produced it.

    Outside tests the catalog degrades to a generic message instead, so a wording
    mistake never stops a submission from being checked.
    """
    previous = messages.strict
    messages.strict = True
    try:
        yield
    finally:
        messages.strict = previous


@pytest.fixture(autouse=True)
def _pinned_submission_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check every submission as if sent at :data:`PINNED_NOW` (§4.6, §2.9.4 count from it)."""
    monkeypatch.setattr(checker, "_now", lambda: PINNED_NOW)
