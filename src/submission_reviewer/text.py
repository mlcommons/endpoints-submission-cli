# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Text heuristics shared by the metadata checks.

The §8.2 and §8.3 disclosures are mostly free text, so the metadata checks read them
with small, deliberately conservative patterns: a placeholder value, a URL that no
reviewer outside the submitter could open, an accelerator named in prose that the
structured fields do not list. Every helper here errs toward *not* matching, because a
false hint costs a reviewer more time than a missed one.
"""

from __future__ import annotations

import datetime
import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit

from submission_checker.cohorts import Cohort

__all__ = [
    "UrlProblem",
    "accelerator_counts",
    "accelerator_families",
    "cohort_date",
    "is_placeholder",
    "name_key",
    "normalize",
    "read_docs_text",
    "url_problem",
]

_PLACEHOLDER = re.compile(
    r"^\s*(?:tbd|tba|todo|n/?a|none|null|unknown|x{2,}|\?+|-+|\.{3}|placeholder|"
    r"fill(?: me| in)?|<[^>]*>|lorem ipsum.*)\s*$",
    re.IGNORECASE,
)


def is_placeholder(value: object) -> bool:
    """Whether *value* is a stand-in such as ``TBD``, ``N/A`` or ``<fill in>``."""
    return isinstance(value, str) and _PLACEHOLDER.match(value) is not None


def normalize(text: str) -> str:
    """Lower-case *text* and collapse every run of non-alphanumerics to one space."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def name_key(text: str) -> str:
    """A comparison key for a framework or directory name.

    Drops a trailing version or parenthetical (``vLLM 0.4.2``, ``vLLM (TPU)``) and every
    non-alphanumeric, so ``TensorRT-LLM`` and ``tensorrt-llm`` compare equal.
    """
    head = re.split(r"[\s(]", text.strip(), maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]", "", head.lower())


# ── Accelerators ──────────────────────────────────────────────────────────────

#: Accelerator product families as they appear in prose. ``\b`` keeps ``B200`` from
#: matching inside ``GB200``; underscores are word characters, so a system name such
#: as ``sys_gb300_72gpu`` does not match at all, which is intended: only prose counts.
_FAMILY = (
    r"GB\d{3}|[HB]\d{3}[A-Z]?|A\d{2,3}|L\d{2}S?|MI\d{3}[A-Z]?|Gaudi\s?\d?|"
    r"TPU\s?v?\d+[a-z]?|Trainium\s?\d?|Inferentia\s?\d?"
)
_FAMILY_RE = re.compile(rf"\b({_FAMILY})\b", re.IGNORECASE)
_COUNT_RE = re.compile(
    rf"\b(\d+)\s*[x×]\s*(?:NVIDIA\s+|AMD\s+|Intel\s+|Google\s+)?({_FAMILY})\b", re.IGNORECASE
)


def _family_key(match: str) -> str:
    return re.sub(r"\s+", "", match.upper())


def accelerator_families(text: str | None) -> set[str]:
    """Accelerator families named in *text*, e.g. ``{"H100"}`` for ``8x H100 SXM``."""
    if not text:
        return set()
    return {_family_key(m) for m in _FAMILY_RE.findall(text)}


def accelerator_counts(text: str | None) -> list[tuple[int, str]]:
    """``(count, family)`` pairs written as ``8x H100`` in *text*."""
    if not text:
        return []
    return [(int(n), _family_key(f)) for n, f in _COUNT_RE.findall(text)]


# ── URLs ──────────────────────────────────────────────────────────────────────

#: Hosts reserved for documentation (RFC 2606) or obviously not real.
_PLACEHOLDER_HOST = re.compile(
    r"(^|\.)(example\.(com|org|net)|invalid|test|localhost)$|your[-_]|<|>", re.IGNORECASE
)
_INTERNAL_HOST = re.compile(r"(^|\.)(local|internal|lan|intranet|corp|home\.arpa)$", re.IGNORECASE)


class UrlProblem:
    """What :func:`url_problem` can report, as short phrases for a finding."""

    PLACEHOLDER = "is a placeholder"
    NOT_A_URL = "is not an http(s) URL"
    PRIVATE = "points at a private or local host"
    NOT_HTTPS = "is not https"


def url_problem(value: str | None) -> str | None:
    """Return why *value* is not a URL a reviewer could open, or ``None`` if it looks fine.

    The order matters: a placeholder is reported as one even when it is also http, so
    the finding names the more useful problem.
    """
    if value is None or not value.strip():
        return None
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return UrlProblem.NOT_A_URL
    host = (parts.hostname or "").lower()
    if parts.scheme not in ("http", "https") or not host:
        return UrlProblem.PLACEHOLDER if is_placeholder(value) else UrlProblem.NOT_A_URL
    if _PLACEHOLDER_HOST.search(host):
        return UrlProblem.PLACEHOLDER
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (address.is_private or address.is_loopback or address.is_link_local):
        return UrlProblem.PRIVATE
    if address is None and ("." not in host or _INTERNAL_HOST.search(host)):
        return UrlProblem.PRIVATE
    if parts.scheme == "http":
        return UrlProblem.NOT_HTTPS
    return None


# ── Cohorts ───────────────────────────────────────────────────────────────────

_WEDNESDAY = 2


def cohort_date(cohort: Cohort) -> datetime.date:
    """The publication date of *cohort*: §4.2's first (C0) or third (C1) Wednesday."""
    first = datetime.date(cohort.year, cohort.month, 1)
    offset = (_WEDNESDAY - first.weekday()) % 7
    return first + datetime.timedelta(days=offset + 14 * cohort.index)


# ── Submission docs ───────────────────────────────────────────────────────────

_DOC_SUFFIXES = {".md", ".txt", ".adoc", ".rst", ".json", ".yaml", ".yml", ".html"}
_DOC_BUDGET = 2_000_000  # characters; docs are prose, this is far beyond any real set


def read_docs_text(root: Path) -> str:
    """The concatenated text of a submission's ``docs/`` and ``src/*/README.md`` files.

    Bounded by a character budget so a submission that ships large data files under
    ``docs/`` cannot make the review slow.
    """
    files = sorted(p for p in (root / "docs").rglob("*") if p.suffix.lower() in _DOC_SUFFIXES)
    files += sorted((root / "src").glob("*/README.md"))
    chunks: list[str] = []
    used = 0
    for path in files:
        if used >= _DOC_BUDGET or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[: _DOC_BUDGET - used]
        except OSError:
            continue
        chunks.append(text)
        used += len(text)
    return "\n".join(chunks)
