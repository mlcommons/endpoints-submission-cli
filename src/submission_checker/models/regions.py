"""Region boundary computation — §5 Pareto Collection Methodology.

Implements the reference algorithm from §5.5 verbatim, using Python's built-in
``round()`` (banker's / round-half-to-even) for all boundary calculations.

§5.5 requires submitters to "use the reference implementation in the MLCommons
Endpoints repository to compute their boundaries and validate their submitted
points" — this module is that implementation, so it must match the spec exactly.
Appendix B's 14-row quick-reference table is reproduced as test vectors in
``tests/submission_checker/test_regions.py``.

Note §5.4's Worked Example C contradicts both this algorithm and Appendix B (it
prints a Low Concurrency range starting at ``C_min`` rather than ``C_min + 1``, and
a Medium Concurrency end of 116 rather than 117). The algorithm and Appendix B
agree with each other; the worked example is wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
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
]

#: Upper bound of the Ultra Low Concurrency region (§5.4), fixed for all submissions.
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
        low_latency: ``1`` to ``C_min`` — the Ultra Low Concurrency point's band.
        low_concurrency: First logarithmic concurrency region.
        med_concurrency: Second logarithmic concurrency region.
        high_concurrency: Third logarithmic concurrency region, ending at ``C_max``.
        margin: The §5.4 10 % extension above ``C_max``. A distinct region in v1.0 —
            it used to be folded into the end of the high region.
    """

    low_latency: RegionBounds
    low_concurrency: RegionBounds
    med_concurrency: RegionBounds
    high_concurrency: RegionBounds
    margin: RegionBounds


#: Minimum steady-state duration per region (§6.2, illustrative — section is WIP).
MIN_DURATION_MS: dict[str, int] = {
    "low_latency": 600_000,
    "low_concurrency": 1_200_000,
    "med_concurrency": 1_200_000,
    "high_concurrency": 1_200_000,
    # A margin point is an addition above C_max; hold it to the same bar as the
    # high region it extends.
    "margin": 1_200_000,
}


def compute_regions(c_max: int, c_min: int) -> Regions:
    """Compute region boundaries from a curve's ``C_max`` and ``C_min``.

    The reference algorithm from §5.5, using banker's rounding (Python's built-in
    ``round()``).

    Args:
        c_max: Maximum Supported Concurrency, declared in the system description.
            Must be greater than 32.
        c_min: Minimum concurrency, **derived from the submission's points** (§5.4)
            rather than declared. Must be between 1 and 32 inclusive.

    Returns:
        A :class:`Regions` instance with inclusive ``[start, end]`` boundaries.

    Raises:
        ValueError: If *c_max* is not greater than 32, or *c_min* is outside 1–32.

    Example::

        regions = compute_regions(1024, 16)
        print(regions.low_concurrency)  # 17–26
    """
    if not 1 <= c_min <= ULTRA_LOW_CONCURRENCY_MAX:
        raise ValueError(
            f"Minimum concurrency must be between 1 and {ULTRA_LOW_CONCURRENCY_MAX} "
            f"(inclusive), got {c_min}"
        )
    if c_max <= ULTRA_LOW_CONCURRENCY_MAX:
        raise ValueError(
            f"Maximum Supported Concurrency must be > {ULTRA_LOW_CONCURRENCY_MAX}, got {c_max}"
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
    )


def classify_concurrency(concurrency: int, regions: Regions) -> str | None:
    """Return the region name for *concurrency*, or ``None`` if out of range.

    Regions are tested in ascending order. A rounding collision can make a region
    zero-width (``start > end``); §5.5 says such a region "has zero width and a
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
