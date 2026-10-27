"""Region boundary computation — §5 Pareto Collection Methodology.

Implements the reference algorithm from §5.5 verbatim, using Python's built-in
``round()`` (banker's / round-half-to-even) for all boundary calculations.

§5.5 requires submitters to "use the reference implementation in the MLCommons
Endpoints repository to compute their boundaries and validate their submitted
points" — this module is that implementation, so it must match the spec exactly.
Appendix B's 14-row quick-reference table is reproduced as test vectors in
``tests/submission_checker/test_regions.py``.

An optional ``c_min_override`` selects the lower boundary (§5.4). Without it,
Ultra Low coverage retains the legacy 1–32 range, independently of ``low_latency``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..messages import Invalid

__all__ = [
    "AGENTIC_MIN_DURATION_MS",
    "MARGIN_SATISFIES_HIGH_CONCURRENCY",
    "MIN_DURATION_MS",
    "REGION_NAMES",
    "SUBMITTERS_CHOICE",
    "ULTRA_LOW_CONCURRENCY_MAX",
    "RegionBounds",
    "Regions",
    "classify_concurrency",
    "compute_regions",
    "covered_region",
    "min_duration_ms",
    "validate_c_min_override",
]

#: Default Ultra Low coverage limit and maximum permitted override (§5.4).
ULTRA_LOW_CONCURRENCY_MAX = 32

#: Region names the algorithm can emit, in ascending concurrency order.
REGION_NAMES = ("low_latency", "low_concurrency", "med_concurrency", "high_concurrency", "margin")

#: A declaration-only region value (§8.3). The algorithm never emits it: a
#: submitter's-choice point still falls inside one of the computed regions.
SUBMITTERS_CHOICE = "submitters_choice"

#: Whether a point in the 10 % margin counts towards High Concurrency coverage.
#: §5.4 says "the margin does not affect the required point distribution", and a
#: margin point sits above the submitter's own declared ``C_max``, so it does not.
#: Kept as a constant so the working group's ruling flips one line.
MARGIN_SATISFIES_HIGH_CONCURRENCY = False


@dataclass(frozen=True)
class RegionBounds:
    """Inclusive ``[start, end]`` concurrency range for one region.

    Attributes:
        start: Lowest valid concurrency level (inclusive).
        end: Highest valid concurrency level (inclusive).
    """

    start: int
    end: int

    def contains(self, concurrency: int) -> bool:
        """Return True if *concurrency* falls within this region."""
        return self.start <= concurrency <= self.end

    def __str__(self) -> str:
        return f"{self.start}–{self.end}"


@dataclass(frozen=True)
class Regions:
    """All region boundaries for a given ``(C_max, C_min)`` pair.

    Attributes:
        low_latency: ``1`` to ``C_min`` from the reference algorithm.
            Coverage uses :attr:`ultra_low_concurrency` instead.
        low_concurrency: First logarithmic concurrency region.
        med_concurrency: Second logarithmic concurrency region.
        high_concurrency: Third logarithmic concurrency region, ending at ``C_max``.
        margin: The §5.4 10 % extension above ``C_max``. A distinct region in v1.0 —
            it used to be folded into the end of the high region.
        c_min_overridden: Whether the lower boundary was explicitly declared.
    """

    low_latency: RegionBounds
    low_concurrency: RegionBounds
    med_concurrency: RegionBounds
    high_concurrency: RegionBounds
    margin: RegionBounds
    c_min_overridden: bool = False

    @property
    def ultra_low_concurrency(self) -> RegionBounds:
        """Ultra Low coverage and accuracy range, preserving defaults without an override."""
        return (
            self.low_latency
            if self.c_min_overridden
            else RegionBounds(1, ULTRA_LOW_CONCURRENCY_MAX)
        )


#: Minimum steady-state duration per region for non-agentic benchmarks (§6.2's
#: Legacy LLM table; example values pending ratification).
MIN_DURATION_MS: dict[str, int] = {
    "low_latency": 600_000,
    "low_concurrency": 1_200_000,
    "med_concurrency": 1_200_000,
    "high_concurrency": 1_200_000,
    # A margin point is an addition above C_max; hold it to the same bar as the
    # high region it extends.
    "margin": 1_200_000,
}

#: §6.2's table for agentic benchmarks: the same 600 s at Ultra Low Concurrency, and
#: 2,400 s everywhere above it.
AGENTIC_MIN_DURATION_MS: dict[str, int] = {
    "low_latency": 600_000,
    "low_concurrency": 2_400_000,
    "med_concurrency": 2_400_000,
    "high_concurrency": 2_400_000,
    "margin": 2_400_000,
}


def min_duration_ms(region: str, concurrency: int, *, agentic: bool) -> int:
    """§6.2's minimum steady-state duration for a point, in milliseconds.

    Both §6.2 tables key their 600 s row to Ultra Low Concurrency as 1–32, not to the
    computed ``low_latency`` band (1 to ``C_min``), so a point at or below
    :data:`ULTRA_LOW_CONCURRENCY_MAX` takes it even when ``C_min`` is lower and the
    point falls in Low Concurrency.
    """
    table = AGENTIC_MIN_DURATION_MS if agentic else MIN_DURATION_MS
    if concurrency <= ULTRA_LOW_CONCURRENCY_MAX:
        return table["low_latency"]
    return table.get(region, 0)


def validate_c_min_override(value: object) -> int:
    """Validate an explicitly supplied JSON integer; omission is handled by the caller."""
    if type(value) is not int or not 1 <= value <= ULTRA_LOW_CONCURRENCY_MAX:
        raise Invalid("c-min-override", "invalid", value=value)
    return value


def compute_regions(c_max: int, c_min: int, *, c_min_overridden: bool = False) -> Regions:
    """Compute region boundaries from a frontier's ``C_max`` and ``C_min``.

    The reference algorithm from §5.5, using banker's rounding (Python's built-in
    ``round()``).

    Args:
        c_max: Maximum Supported Concurrency, declared in the system description.
            Must be greater than 32.
        c_min: Lower boundary, derived from the points or explicitly overridden (§5.4).
            Must be between 1 and 32 inclusive.
        c_min_overridden: Whether an explicit override selects Ultra Low coverage 1–C_min.

    Returns:
        A :class:`Regions` instance with inclusive ``[start, end]`` boundaries.

    Raises:
        Invalid: If *c_max* is not greater than 32, or *c_min* is outside 1–32.

    Example::

        regions = compute_regions(1024, 16)
        print(regions.low_concurrency)  # 17–26
    """
    if not 1 <= c_min <= ULTRA_LOW_CONCURRENCY_MAX:
        raise Invalid(
            "region-computation", "c-min-range", limit=ULTRA_LOW_CONCURRENCY_MAX, c_min=c_min
        )
    if c_max <= ULTRA_LOW_CONCURRENCY_MAX:
        raise Invalid(
            "region-computation", "c-max-too-low", limit=ULTRA_LOW_CONCURRENCY_MAX, c_max=c_max
        )

    interval = math.log2(c_max - c_min) / 3

    low_conc_end = round(c_min + 2**interval)
    med_conc_end = round(c_min + 2 ** (2 * interval))
    margin_end = math.ceil(1.10 * c_max)

    return Regions(
        low_latency=RegionBounds(1, c_min),
        low_concurrency=RegionBounds(c_min + 1, low_conc_end),
        med_concurrency=RegionBounds(low_conc_end + 1, med_conc_end),
        high_concurrency=RegionBounds(med_conc_end + 1, c_max),
        margin=RegionBounds(c_max + 1, margin_end),
        c_min_overridden=c_min_overridden,
    )


def classify_concurrency(concurrency: int, regions: Regions) -> str | None:
    """Return the region name for *concurrency*, or ``None`` if out of range.

    Regions are tested in ascending order. A rounding collision can make a region
    zero-width (``start > end``); §5.4 says such a region "has zero width and a
    single valid concurrency level at the boundary value", which
    :meth:`RegionBounds.contains` yields naturally because the neighbouring region
    still covers that value.

    Args:
        concurrency: The concurrency level to classify.
        regions: Pre-computed boundaries from :func:`compute_regions`.

    Returns:
        One of :data:`REGION_NAMES`, or ``None`` when the concurrency exceeds the
        10 % margin — meaning it is out of range for submission purposes and the
        caller should flag the point.
    """
    if regions.low_latency.contains(concurrency):
        return "low_latency"
    if regions.low_concurrency.contains(concurrency):
        return "low_concurrency"
    if regions.med_concurrency.contains(concurrency):
        return "med_concurrency"
    if regions.high_concurrency.contains(concurrency):
        return "high_concurrency"
    if regions.margin.contains(concurrency):
        return "margin"
    return None


def covered_region(concurrency: int, regions: Regions) -> str | None:
    """Return the region *concurrency* counts towards for §9.1 coverage, or ``None``.

    Differs from :func:`classify_concurrency` only in the margin: a margin point sits
    above the submitter's own declared ``C_max``, and §5.4 says "the margin does not
    affect the required point distribution", so by default it satisfies no region. See
    :data:`MARGIN_SATISFIES_HIGH_CONCURRENCY`.

    Args:
        concurrency: The concurrency level to attribute.
        regions: Pre-computed boundaries from :func:`compute_regions`.

    Returns:
        One of :data:`REGION_NAMES` excluding ``"margin"``, or ``None`` when the point
        contributes to no region's coverage.
    """
    region = classify_concurrency(concurrency, regions)
    if region == "margin":
        return "high_concurrency" if MARGIN_SATISFIES_HIGH_CONCURRENCY else None
    return region
