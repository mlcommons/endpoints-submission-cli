# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""What a review produces: findings, per-check outcomes, and the §9.2 focus areas.

These models are deliberately separate from :class:`submission_checker.models.CheckResult`.
A checker result is a compliance verdict worded through the checker's message catalog;
a reviewer finding is a heuristic hint whose wording carries its own evidence, so it
needs neither the catalog nor a pass/fail notion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

__all__ = [
    "FOCUS_AREAS",
    "CheckOutcome",
    "Finding",
    "FocusArea",
    "Level",
    "ReviewReport",
    "focus_area",
]


class Level(str, Enum):
    """How strongly a finding asks for a reviewer's attention.

    Attributes:
        ATTENTION: Unlikely to happen in a genuine run; the reviewer should look.
        NOTE: Unusual but often explainable; worth a glance.
        INFO: Context the reviewer may want, with no suggestion of a problem.
    """

    ATTENTION = "attention"
    NOTE = "note"
    INFO = "info"


#: Display order and badges, shared by the renderer and the CLI.
LEVEL_ORDER: dict[Level, int] = {Level.ATTENTION: 0, Level.NOTE: 1, Level.INFO: 2}
LEVEL_BADGE: dict[Level, str] = {Level.ATTENTION: "🔶", Level.NOTE: "🔹", Level.INFO: "ℹ️"}


@dataclass(frozen=True)
class FocusArea:
    """One bullet of rules §9.2, Manual Review Focus Areas.

    Attributes:
        key: Stable identifier, used by checks and in JSON output.
        title: Short heading for the PR comment.
        question: What the reviewer is asked to judge, paraphrasing §9.2.
        spec_ref: The rules sections that define the underlying requirement.
        manual_note: Shown when no automated check covers the area, to say why.
        scope: Shown when checks do cover it, to say what they cover and what remains
            the reviewer's to judge.
    """

    key: str
    title: str
    question: str
    spec_ref: str
    manual_note: str = ""
    scope: str = ""


#: Every §9.2 bullet, in the rules' order. An area with no registered check still
#: appears in the comment, under "Manual review only", so the comment doubles as the
#: reviewer's full checklist.
FOCUS_AREAS: tuple[FocusArea, ...] = (
    FocusArea(
        "pareto_shape",
        "Pareto curve plausibility",
        "Is the curve physically plausible: throughput rising with concurrency up to "
        "saturation, then plateauing or falling?",
        "§9.2, §5",
    ),
    FocusArea(
        "metric_distributions",
        "Metric distributions",
        "Do the metric distributions suggest artificial manipulation, such as uniform TTFT "
        "across very different concurrency levels?",
        "§9.2, §4.1",
    ),
    FocusArea(
        "ttft_fragments",
        "TTFT-triggering fragments",
        "Are the fragments that stop the TTFT clock genuine parts of the response?",
        "§9.2, §4.1",
        scope="The first streamed fragment of each response in events.jsonl, and the pause "
        "after it. Whether the content is genuine is for the reviewer.",
    ),
    FocusArea(
        "output_padding",
        "Duplicated or padded output",
        "Is content duplicated across response fields, or padded to inflate the official "
        "output-token count?",
        "§9.2, §2.8",
        scope="Output length across the curve and token-limit stops; with events.jsonl, "
        "duplicated fields, repetition and trailing filler in the sampled responses.",
    ),
    FocusArea(
        "offline_reordering",
        "Offline reordering",
        "Did Offline query reordering stay within dataset-pass boundaries?",
        "§9.2, §5.7.1",
        scope="Completion order across dataset passes of the dedicated Offline run, from "
        "events.jsonl. The client does not log pass boundaries, so they are inferred.",
    ),
    FocusArea(
        "warmup",
        "Warmup data",
        "Did warmup requests avoid every sample of the performance dataset?",
        "§9.2, §6.3.1",
        scope="The point.yaml warmup declarations; with events.jsonl, warmup prompts "
        "compared whole with the performance prompts. Truncated or derived samples are not "
        "detected.",
    ),
    FocusArea(
        "system_description",
        "System description",
        "Does the system description accurately reflect the hardware and software used?",
        "§9.2, §8.2",
        scope="Internal consistency only: placeholders, prose vs fields, src/ and "
        "system_power.json. Whether it matches the real hardware is for the reviewer.",
    ),
    FocusArea(
        "cross_submission",
        "Cross-submission consistency",
        "Are results consistent with other submissions on the same hardware platform?",
        "§9.2",
        "Needs other submissions' results; review manually.",
    ),
    FocusArea(
        "division",
        "Division eligibility",
        "Is the submission eligible for its division, especially Serviced API compliance "
        "and availability status?",
        "§9.2, §2, Submission Rules §7",
        scope="Declared division and status, the Preview window, the Serviced endpoint and "
        "disclosures. Whether the product qualifies is for the reviewer.",
    ),
    FocusArea(
        "updates",
        "Post-submission updates",
        "Are updates made after submission consistent with the original system configuration?",
        "§9.2, Submission Rules §6.3",
        scope="Compares the current files with the original submission (--base-ref). "
        "Whether the committee asked for each change is for the reviewer.",
    ),
    FocusArea(
        "speculative_decoding",
        "Speculative decoding",
        "Is the disclosed drafter and algorithm consistent with the benchmark's approved list?",
        "§9.2, §2.9.4",
        "The approved-drafter match is a §9.1 check; see the Check submission results.",
        scope="Lists what each point declares; the approved-list match is §9.1's check.",
    ),
)

_BY_KEY = {area.key: area for area in FOCUS_AREAS}


def focus_area(key: str) -> FocusArea:
    """Return the focus area named *key*.

    Raises:
        KeyError: If no §9.2 focus area has that key.
    """
    return _BY_KEY[key]


class Finding(BaseModel):
    """One thing a reviewer should look at.

    Attributes:
        check: The check that produced it, e.g. ``"pareto_shape"``.
        focus_area: The §9.2 focus area it informs (a :data:`FOCUS_AREAS` key).
        level: How strongly it asks for attention.
        title: One line naming what was observed.
        detail: The evidence, with the numbers that triggered it.
        curve: ``<system>/<model>`` of the Pareto frontier, when the finding has one.
        point: The point directory name (``r88``), when the finding is about one point.
        spec_ref: Rules sections that explain why it matters.
        evidence: Machine-readable values behind the finding, for JSON output.
    """

    model_config = ConfigDict(frozen=True)

    check: str
    focus_area: str
    level: Level
    title: str
    detail: str
    curve: str | None = None
    point: str | None = None
    spec_ref: str = ""
    evidence: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class CheckOutcome(BaseModel):
    """Whether a check ran, and why not when it did not.

    Attributes:
        check: The check's name.
        focus_area: The §9.2 focus area it informs.
        status: ``ran``; ``skipped`` when its inputs are absent; ``error`` when it raised.
        reason: Why it was skipped or what went wrong. For a check that ran, what it
            could not cover (some unreadable logs); empty when it covered everything.
    """

    model_config = ConfigDict(frozen=True)

    check: str
    focus_area: str
    status: Literal["ran", "skipped", "error"]
    reason: str = ""


class ReviewReport(BaseModel):
    """Everything one review of one submission produced.

    Attributes:
        submission: The submission directory, relative to where the review started.
        findings: Every finding, in check order.
        outcomes: One entry per registered check.
        notes: Problems reading the submission, such as a file that did not parse.
            They narrow what the checks could see; the §9.1 checker reports the
            underlying defects.
    """

    submission: str
    findings: list[Finding] = Field(default_factory=list)
    outcomes: list[CheckOutcome] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def counts(self) -> dict[str, int]:
        """Number of findings at each level."""
        return {level.value: sum(f.level is level for f in self.findings) for level in Level}
