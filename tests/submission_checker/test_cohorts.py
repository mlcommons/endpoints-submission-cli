# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for publication-cohort identifiers and arithmetic (§4.2, §4.6)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from submission_checker.cohorts import PUBLICATION_TZ, Cohort, adoption_window


class TestParse:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("2026-10-C1", Cohort(2026, 10, 1)),
            ("2026-01-C0", Cohort(2026, 1, 0)),
            ("  2026-10-C1  ", Cohort(2026, 10, 1)),
        ],
    )
    def test_valid(self, text: str, expected: Cohort) -> None:
        assert Cohort.parse(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "2026-10",
            "2026-10-C2",  # only two cohorts per month
            "2026-13-C0",  # month out of range
            "2026-00-C0",
            "26-10-C1",
            "2026-10-c1",
            "not-a-cohort",
        ],
    )
    def test_rejected(self, text: str) -> None:
        assert Cohort.parse(text) is None

    def test_round_trips_through_str(self) -> None:
        for text in ("2026-01-C0", "2026-12-C1", "2030-06-C0"):
            assert str(Cohort.parse(text)) == text


class TestSuccession:
    def test_first_cohort_of_a_month_yields_the_second(self) -> None:
        assert str(Cohort(2026, 10, 0).next()) == "2026-10-C1"

    def test_second_cohort_rolls_into_the_next_month(self) -> None:
        assert str(Cohort(2026, 10, 1).next()) == "2026-11-C0"

    def test_december_rolls_the_year(self) -> None:
        assert str(Cohort(2026, 12, 1).next()) == "2027-01-C0"

    def test_succession_is_strictly_increasing(self) -> None:
        """Ordering must agree with succession, or a window can't be compared."""
        c = Cohort(2026, 11, 0)
        for _ in range(30):
            assert c < c.next()
            c = c.next()

    def test_two_cohorts_per_calendar_month(self) -> None:
        """§4.2's bi-weekly cadence: a year advances in exactly 24 steps."""
        c = Cohort(2026, 1, 0)
        for _ in range(24):
            c = c.next()
        assert str(c) == "2027-01-C0"


class TestAdoptionWindow:
    def test_four_consecutive_cohorts_by_default(self) -> None:
        """§4.6: the publication cohort and the following three."""
        window = adoption_window(Cohort(2026, 10, 1))
        assert [str(c) for c in window] == [
            "2026-10-C1",
            "2026-11-C0",
            "2026-11-C1",
            "2026-12-C0",
        ]

    def test_window_starts_at_the_publication_cohort(self) -> None:
        published = Cohort(2027, 3, 0)
        assert adoption_window(published)[0] == published

    def test_window_crosses_a_year_boundary(self) -> None:
        window = adoption_window(Cohort(2026, 12, 0))
        assert [str(c) for c in window] == [
            "2026-12-C0",
            "2026-12-C1",
            "2027-01-C0",
            "2027-01-C1",
        ]

    @pytest.mark.parametrize("length", [1, 2, 4, 8])
    def test_length_is_configurable(self, length: int) -> None:
        """A working-group change to the window is a call-site edit, not a rewrite."""
        assert len(adoption_window(Cohort(2026, 10, 1), length=length)) == length

    def test_window_is_contiguous(self) -> None:
        window = adoption_window(Cohort(2026, 11, 1), length=6)
        for earlier, later in zip(window, window[1:], strict=False):
            assert earlier.next() == later

    def test_two_sets_published_two_cohorts_apart_overlap_by_two(self) -> None:
        """§4.6: refresh every two cohorts, adoptable for four — so two sets overlap."""
        first = set(adoption_window(Cohort(2026, 10, 1)))
        second = set(adoption_window(Cohort(2026, 11, 1)))
        assert len(first & second) == 2


class TestCohortDates:
    """§4.2: C0 is the 1st Wednesday, C1 the 3rd, both at 8:00 AM Pacific."""

    @pytest.mark.parametrize(
        "cohort, expected",
        [
            # Submission Rules §4.2's worked example names these dates.
            ("2026-10-C1", datetime(2026, 10, 21, 8, tzinfo=PUBLICATION_TZ)),
            ("2026-11-C0", datetime(2026, 11, 4, 8, tzinfo=PUBLICATION_TZ)),
            ("2026-11-C1", datetime(2026, 11, 18, 8, tzinfo=PUBLICATION_TZ)),
            # A month that starts on a Wednesday.
            ("2026-07-C0", datetime(2026, 7, 1, 8, tzinfo=PUBLICATION_TZ)),
        ],
    )
    def test_publication_time(self, cohort: str, expected: datetime) -> None:
        assert Cohort.parse(cohort).publication_time() == expected

    @pytest.mark.parametrize(
        "when, expected",
        [
            (datetime(2026, 10, 12, tzinfo=timezone.utc), "2026-10-C1"),
            # Exactly at publication still makes that cohort; a minute later does not.
            (datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc), "2026-10-C1"),
            (datetime(2026, 10, 21, 15, 1, tzinfo=timezone.utc), "2026-11-C0"),
            # After the month's last cohort, the next month's first.
            (datetime(2026, 12, 30, tzinfo=timezone.utc), "2027-01-C0"),
        ],
    )
    def test_on_or_after(self, when: datetime, expected: str) -> None:
        assert str(Cohort.on_or_after(when)) == expected
