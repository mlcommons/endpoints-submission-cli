# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Publication cohorts (§4.2) — identifiers and the arithmetic §4.6 needs.

MLCommons publishes results on a bi-weekly cadence, two cohorts per calendar month,
identified as ``YYYY-MM-C0`` and ``YYYY-MM-C1`` (§4.2). §4.6 then counts in cohorts:
a seed set is adoptable for "its publication cohort and the following three", which
only means something if a cohort has a well-defined successor. That successor lives
here rather than in the seed-set loader, because it is a property of the publication
calendar and the next rule to count in cohorts will want it too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["COHORT_RE", "Cohort", "adoption_window"]

#: A cohort identifier: calendar year-month plus the index within that month.
COHORT_RE = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-C(?P<index>[01])")

#: Cohorts published per calendar month (§4.2's 1st and 3rd Wednesday).
_COHORTS_PER_MONTH = 2


@dataclass(frozen=True, order=True)
class Cohort:
    """One publication cohort.

    Ordered by ``(year, month, index)``, so cohorts compare and sort chronologically.

    Attributes:
        year: Calendar year.
        month: Calendar month, 1–12.
        index: 0 for the first cohort of the month, 1 for the second.
    """

    year: int
    month: int
    index: int

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-C{self.index}"

    @classmethod
    def parse(cls, value: str) -> Cohort | None:
        """Parse ``YYYY-MM-C0`` / ``YYYY-MM-C1``, returning None when malformed.

        A month outside 1–12 is rejected: ``2026-13-C0`` matches the shape but names
        no cohort, and silently accepting it would put a submission in a window that
        cannot exist.
        """
        match = COHORT_RE.fullmatch(value.strip()) if value else None
        if match is None:
            return None
        month = int(match["month"])
        if not 1 <= month <= 12:
            return None
        return cls(year=int(match["year"]), month=month, index=int(match["index"]))

    def next(self) -> Cohort:
        """The cohort immediately following this one."""
        if self.index + 1 < _COHORTS_PER_MONTH:
            return Cohort(self.year, self.month, self.index + 1)
        if self.month == 12:
            return Cohort(self.year + 1, 1, 0)
        return Cohort(self.year, self.month + 1, 0)


def adoption_window(published: Cohort, length: int = 4) -> tuple[Cohort, ...]:
    """The cohorts during which a set published in *published* may be adopted.

    §4.6: "Each published seed set is available for adoption by new submissions for
    **four consecutive cohorts** — its publication cohort and the following three."

    Args:
        published: The cohort the set was published for.
        length: Window length in cohorts. Defaults to §4.6's four; a parameter so a
            working-group change is a call-site edit rather than a rewrite.

    Returns:
        The window in chronological order, starting with *published*.
    """
    window = [published]
    while len(window) < length:
        window.append(window[-1].next())
    return tuple(window)
