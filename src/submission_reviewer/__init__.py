# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Automated hints for the §9.2 manual review of an MLPerf Endpoints submission.

§9.1's checker (:mod:`submission_checker`) decides compliance. §9.2 lists what human
reviewers should look at because automation cannot decide it. This package does not
decide those items either: it runs heuristics over a submission and reports what looks
out of the ordinary, so a reviewer knows where to start. Every finding is a hint for a
person to confirm, never an objection, and the reviewer never fails a build.

Entry points:

- :func:`submission_reviewer.runner.review` reviews one submission directory.
- ``endpoints-submission-reviewer`` (:mod:`submission_reviewer.cli`) reviews every
  submission under a path and renders the result as a PR comment or JSON.
"""

from __future__ import annotations

from .findings import FOCUS_AREAS, CheckOutcome, Finding, FocusArea, Level, ReviewReport
from .runner import review

__all__ = [
    "FOCUS_AREAS",
    "CheckOutcome",
    "Finding",
    "FocusArea",
    "Level",
    "ReviewReport",
    "review",
]
