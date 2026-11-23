"""Per-point run requirements — §6.2 Minimum Run Duration.

§6.2 keys its thresholds on the point's **concurrency**, "independently of its §5.4
region or ``c_min_override``": concurrency 24 takes the 1–32 row even when
``c_min_override=16`` places it in Low Concurrency. Nothing here therefore reads the
frontier's region boundaries, and the check runs even when those boundaries cannot
be computed.

§6.2 still carries "(Example values — subject to ratification.)". The tables below
encode its current text verbatim, so that a ratified change is a one-line edit.
"""

from __future__ import annotations

__all__ = [
    "AGENTIC_MIN_DURATION_MS",
    "LOW_CONCURRENCY_TIER",
    "MIN_DURATION_MS",
    "RUN_REQUIREMENT_TIER_MAX",
    "UPPER_CONCURRENCY_TIER",
    "concurrency_tier",
    "min_duration_ms",
]

#: Highest concurrency in §6.2's (and §6.4's) lower row, inclusive. Deliberately
#: separate from ``regions.ULTRA_LOW_CONCURRENCY_MAX``: they share a value today, but
#: one is a region default and the other a run-requirement boundary.
RUN_REQUIREMENT_TIER_MAX = 32

#: §6.2's row labels, as its tables print them.
LOW_CONCURRENCY_TIER = f"1–{RUN_REQUIREMENT_TIER_MAX}"
UPPER_CONCURRENCY_TIER = f">{RUN_REQUIREMENT_TIER_MAX}"

#: §6.2's Legacy LLM (non-agentic) table: minimum steady-state duration in ms.
MIN_DURATION_MS: dict[str, int] = {
    LOW_CONCURRENCY_TIER: 600_000,
    UPPER_CONCURRENCY_TIER: 1_200_000,
}

#: §6.2's table for all agentic benchmarks.
AGENTIC_MIN_DURATION_MS: dict[str, int] = {
    LOW_CONCURRENCY_TIER: 600_000,
    UPPER_CONCURRENCY_TIER: 2_400_000,
}


def concurrency_tier(concurrency: int) -> str:
    """Return the §6.2 row label (``"1–32"`` or ``">32"``) for *concurrency*."""
    return (
        LOW_CONCURRENCY_TIER if concurrency <= RUN_REQUIREMENT_TIER_MAX else UPPER_CONCURRENCY_TIER
    )


def min_duration_ms(concurrency: int, *, agentic: bool) -> int:
    """§6.2's minimum steady-state duration for a point, in milliseconds.

    Args:
        concurrency: The point's target concurrency. A dedicated Offline run's
            concurrency is the dataset cardinality (§5.7.1), and §6.2 applies to it
            unchanged.
        agentic: Whether the point belongs to an agentic benchmark.
    """
    table = AGENTIC_MIN_DURATION_MS if agentic else MIN_DURATION_MS
    return table[concurrency_tier(concurrency)]
