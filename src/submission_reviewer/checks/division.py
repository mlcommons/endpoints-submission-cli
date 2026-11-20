# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: is the submission eligible for its division and availability status?

The rules call out "especially Serviced division API compliance and availability
status". Eligibility is a judgement about the product, but several prerequisites are
visible in the bundle:

- **Division declared consistently**: ``point.yaml`` and ``system_desc.json`` agree.
- **RDI division carries RDI status** (Submission Rules §7.2.5).
- **Preview window** (Submission Rules §7.3): ``target_availability_date`` within 180
  days of the cohort the submission targets, and a statement naming which components
  are not yet available (§7.3.4 asks for one "with specificity").
- **Serviced endpoint** (§2.3, Submission Rules §7.2.5): an ``endpoint_url`` any paying
  customer could reach, not a private, local or preview-tier host, and disclosure of
  pricing, rate limits and terms of service.
- **Available evidence**: an ``availability_url`` a reviewer can open.

§9.1 already rejects a missing ``availability_url`` (Available) or
``target_availability_date`` (Preview), so neither is repeated here.
"""

from __future__ import annotations

import datetime
import re

from submission_checker.cohorts import Cohort
from submission_checker.models.file import SystemDescription
from submission_checker.models.file.system import Division, SystemAvailabilityStatus

from ..context import INPUT_SYSTEM_DESCRIPTIONS, Curve, ReviewContext
from ..findings import Finding, Level
from ..text import UrlProblem, cohort_date, read_docs_text, url_problem
from .base import Check, finding_factory, points_label

__all__ = ["CHECK"]

_finding = finding_factory("division")

#: Submission Rules §7.3: Available within 180 days of first publication.
PREVIEW_WINDOW_DAYS = 180

#: Hostname labels suggesting a tier other than the production endpoint.
_NON_GA_HOST = re.compile(r"(^|[.-])(preview|beta|alpha|staging|stage|dev|test|internal)([.-]|$)")

#: What §2.3 asks a Serviced submission to disclose, and how to spot each in prose.
_SERVICED_DISCLOSURES: dict[str, re.Pattern[str]] = {
    "pricing": re.compile(
        r"pric|\$\s?\d|\busd\b|per (?:1m|1k|million|thousand) tokens", re.IGNORECASE
    ),
    "rate limits or quotas": re.compile(
        r"rate[- ]?limit|quota|requests per (?:minute|second)|\brpm\b|\btpm\b", re.IGNORECASE
    ),
    "terms of service permitting benchmarking": re.compile(
        r"terms of (?:service|use)|\btos\b|acceptable use", re.IGNORECASE
    ),
}

#: Words a §7.3.4 statement of what is not yet available would use. Not "launch":
#: every src/ README says how to launch the SUT.
_AVAILABILITY_STATEMENT = re.compile(
    r"availab|\bship|general(?:ly)? release|\bGA\b|orderable", re.IGNORECASE
)


def _division_mismatch(curve: Curve, desc: SystemDescription) -> list[Finding]:
    declared = desc.division.value.lower()
    differing = [
        f"{p.name} ({c.division})"
        for p, c in curve.configs()
        if c.division and c.division.strip().lower() != declared
    ]
    if not differing:
        return []
    return [
        _finding(
            Level.ATTENTION,
            "Division declared inconsistently",
            f"system_desc.json declares {desc.division.value}, but point.yaml declares a "
            f"different division at {points_label(differing)}.",
            curve=curve.label,
            spec_ref="§9.2, §8.2, §8.3",
        )
    ]


def _rdi_status(label: str, desc: SystemDescription) -> list[Finding]:
    if desc.division is not Division.RDI:
        return []
    if desc.system_availability_status is SystemAvailabilityStatus.RDI:
        return []
    return [
        _finding(
            Level.ATTENTION,
            "RDI division with a non-RDI status",
            f"The division is RDI but system_availability_status is "
            f"{desc.system_availability_status.value}. RDI submissions carry RDI status and "
            "cannot qualify as Available or Preview.",
            curve=label,
            spec_ref="§9.2, Submission Rules §7.2.5",
        )
    ]


def _target_cohort(curve: Curve) -> Cohort | None:
    cohorts = {c.target_cohort for _, c in curve.configs() if c.target_cohort}
    parsed = sorted(filter(None, (Cohort.parse(c) for c in cohorts)))
    return parsed[0] if parsed else None


def _preview(curve: Curve, desc: SystemDescription, docs: str) -> list[Finding]:
    if desc.system_availability_status is not SystemAvailabilityStatus.PREVIEW:
        return []
    out: list[Finding] = []
    label = curve.system_dir.name
    cohort = _target_cohort(curve)
    if cohort is not None and desc.target_availability_date:
        start = cohort_date(cohort)
        deadline = start + datetime.timedelta(days=PREVIEW_WINDOW_DAYS)
        target = datetime.date.fromisoformat(desc.target_availability_date)
        if target > deadline:
            out.append(
                _finding(
                    Level.ATTENTION,
                    "Preview availability date is outside the window",
                    f"target_availability_date {target} is {(target - start).days} days after "
                    f"the target cohort {cohort} ({start}). If first published in that "
                    f"cohort, the system must be Available by {deadline}.",
                    curve=label,
                    spec_ref="§9.2, Submission Rules §7.3.1, §7.3.4",
                    days=(target - start).days,
                )
            )
        elif target <= start:
            out.append(
                _finding(
                    Level.NOTE,
                    "Preview availability date has already passed",
                    f"target_availability_date {target} is on or before the target cohort "
                    f"{cohort} ({start}); if the system is available, it should be submitted "
                    "as Available.",
                    curve=label,
                    spec_ref="§9.2, Submission Rules §7.3",
                )
            )
    notes = " ".join(
        str(text)
        for node in desc.node_types
        for text in (node.hw_notes, node.sw_notes)
        if isinstance(text, str)
    )
    if not _AVAILABILITY_STATEMENT.search(notes + " " + docs):
        out.append(
            _finding(
                Level.NOTE,
                "No statement of what is not yet available",
                "Nothing in hw_notes, sw_notes or docs/ says which components are not yet "
                "available. §7.3.4 asks Preview submitters to name them with specificity.",
                curve=label,
                spec_ref="§9.2, Submission Rules §7.3.4",
            )
        )
    return out


def _serviced(curve: Curve, desc: SystemDescription, docs: str) -> list[Finding]:
    if desc.division is not Division.SERVICED:
        return []
    label = curve.system_dir.name
    out: list[Finding] = []
    url = (desc.endpoint_url or "").strip()
    problem = url_problem(url) if url else "is missing"
    if problem is not None:
        level = Level.ATTENTION if problem in ("is missing", UrlProblem.PRIVATE) else Level.NOTE
        out.append(
            _finding(
                level,
                "Serviced endpoint URL is not a public endpoint",
                f"endpoint_url “{url}” {problem}. A Serviced result must be measured on the "
                "endpoint any paying customer uses in production.",
                curve=label,
                spec_ref="§9.2, §2.3, Submission Rules §7.2.5",
            )
        )
    else:
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
        if _NON_GA_HOST.search(host):
            out.append(
                _finding(
                    Level.NOTE,
                    "Serviced endpoint looks like a non-production tier",
                    f"The endpoint host “{host}” suggests a preview, beta or staging tier. "
                    "Serviced results need the full GA production endpoint.",
                    curve=label,
                    spec_ref="§9.2, Submission Rules §7.2.5",
                )
            )
    text = (
        docs
        + " "
        + " ".join(
            str(v)
            for v in (desc.node_config, desc.config_summary_notes)
            + tuple(n.sw_notes for n in desc.node_types)
            if isinstance(v, str)
        )
    )
    missing = [name for name, pattern in _SERVICED_DISCLOSURES.items() if not pattern.search(text)]
    if missing:
        out.append(
            _finding(
                Level.NOTE,
                "Serviced disclosures not found",
                f"No mention of {', '.join(missing)} in docs/, the src/ READMEs or the "
                "system notes. §2.3 asks Serviced submitters to disclose them.",
                curve=label,
                spec_ref="§9.2, §2.3",
                missing=", ".join(missing),
            )
        )
    return out


def _availability_url(label: str, desc: SystemDescription) -> list[Finding]:
    if desc.system_availability_status is not SystemAvailabilityStatus.AVAILABLE:
        return []
    problem = url_problem(desc.availability_url)
    if problem is None:
        return []
    return [
        _finding(
            Level.NOTE,
            "Availability URL can't be checked",
            f"availability_url “{desc.availability_url}” {problem}. Available status needs "
            "public evidence of shipment that a reviewer can open.",
            curve=label,
            spec_ref="§9.2, Submission Rules §7.2.1",
        )
    ]


def run(ctx: ReviewContext) -> list[Finding]:
    """Return the division and availability findings, once per system."""
    out: list[Finding] = []
    docs: str | None = None
    seen: set[str] = set()
    for curve in ctx.curves:
        desc = curve.system_desc
        if desc is None:
            continue
        out.extend(_division_mismatch(curve, desc))
        system = curve.system_dir.name
        if system in seen:
            continue
        seen.add(system)
        needs_docs = desc.division is Division.SERVICED or (
            desc.system_availability_status is SystemAvailabilityStatus.PREVIEW
        )
        if needs_docs and docs is None:
            docs = read_docs_text(ctx.root)
        out.extend(_rdi_status(system, desc))
        out.extend(_preview(curve, desc, docs or ""))
        out.extend(_serviced(curve, desc, docs or ""))
        out.extend(_availability_url(system, desc))
    return out


CHECK = Check(
    name="division",
    focus_area="division",
    requires=frozenset({INPUT_SYSTEM_DESCRIPTIONS}),
    run=run,
)
