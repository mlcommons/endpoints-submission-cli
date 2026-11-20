# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""What every check shares: the :class:`Check` record, point metrics, and formatting."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..context import Point, ReviewContext
from ..findings import Finding, Level

__all__ = [
    "Check",
    "FindingFactory",
    "PointMetrics",
    "finding_factory",
    "fmt",
    "metrics",
    "pct",
    "points_label",
]

Evidence = float | int | str | bool | None


class FindingFactory:
    """Builds the findings of one check, so each call names only what varies."""

    def __init__(self, check: str, focus_area: str) -> None:
        self.check = check
        self.focus_area = focus_area

    def __call__(
        self,
        level: Level,
        title: str,
        detail: str,
        *,
        curve: str | None = None,
        point: str | None = None,
        spec_ref: str = "",
        **evidence: Evidence,
    ) -> Finding:
        """Return a :class:`Finding` attributed to this check."""
        return Finding(
            check=self.check,
            focus_area=self.focus_area,
            level=level,
            title=title,
            detail=detail,
            curve=curve,
            point=point,
            spec_ref=spec_ref,
            evidence=evidence,
        )


def finding_factory(check: str, focus_area: str | None = None) -> FindingFactory:
    """A :class:`FindingFactory` for *check*, whose focus area defaults to its name."""
    return FindingFactory(check, focus_area or check)


@dataclass(frozen=True)
class Check:
    """One registered review check.

    Attributes:
        name: Stable identifier, recorded on its findings and outcome.
        focus_area: The §9.2 focus area it informs (a ``FOCUS_AREAS`` key).
        requires: Input kinds it needs (see :mod:`submission_reviewer.context`). The
            runner skips the check, and says why, when one is missing.
        run: Produces the findings for one submission.
    """

    name: str
    focus_area: str
    requires: frozenset[str]
    run: Callable[[ReviewContext], list[Finding]]


@dataclass(frozen=True)
class PointMetrics:
    """The aggregate metrics of one point, in the units the rules use.

    Every field but ``concurrency`` and ``system_tps`` is ``None`` when the summary does
    not report it, or reports a non-positive value that no heuristic could use.

    Attributes:
        point: The point the values came from.
        concurrency: Target concurrency.
        system_tps: Output tokens per second, derived as §9.1 does.
        tps_per_user: ``1000 / tpot_p90_ms`` (§4.1).
        ttft_p50_ms: Median TTFT.
        ttft_p90_ms: P90 TTFT.
        tpot_p50_ms: Median TPOT.
        implied_concurrency: ``latency.total / duration_ns``: the mean number of
            requests in flight, by Little's law.
    """

    point: Point
    concurrency: int
    system_tps: float
    tps_per_user: float | None
    ttft_p50_ms: float | None
    ttft_p90_ms: float | None
    tpot_p50_ms: float | None
    implied_concurrency: float | None

    @property
    def name(self) -> str:
        """The point directory's name."""
        return self.point.name


def _positive(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) and value > 0 else None


def metrics(point: Point) -> PointMetrics:
    """Return *point*'s metrics. The point must have a loaded summary."""
    s = point.summary
    if s is None:
        raise ValueError(f"{point.name} has no loaded result summary")
    tpot_p90 = _positive(s.tpot_p90_ms)
    tpot_p50_raw = s.tpot.percentiles.get("50")
    latency_total = _positive(s.latency.total)
    return PointMetrics(
        point=point,
        concurrency=point.concurrency,
        system_tps=s.system_tps,
        tps_per_user=1000.0 / tpot_p90 if tpot_p90 is not None else None,
        ttft_p50_ms=_positive(s.ttft_p50_ms),
        ttft_p90_ms=_positive(s.ttft_p90_ms),
        tpot_p50_ms=_positive(tpot_p50_raw / 1e6 if tpot_p50_raw is not None else None),
        implied_concurrency=(
            latency_total / s.duration_ns if latency_total and s.duration_ns > 0 else None
        ),
    )


def fmt(value: float, digits: int = 1) -> str:
    """Format a measurement with thousands separators, e.g. ``12,345.6``."""
    return f"{value:,.{digits}f}"


def pct(fraction: float) -> str:
    """Format a fraction as a whole percentage, e.g. ``0.237`` as ``24%``."""
    return f"{fraction:.0%}"


def points_label(names: list[str], limit: int = 6) -> str:
    """Join point names, eliding the middle of a long list."""
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[: limit - 1]) + f", … ({len(names) - limit + 1} more)"
