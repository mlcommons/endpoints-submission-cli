"""Tests for region boundary computation (§5.5 reference algorithm).

§5.5 requires submitters to use *this* implementation to compute their boundaries,
so these are the highest-value tests in the suite: whatever they assert becomes
normative for every submission. Appendix B's 14-row quick-reference table is
reproduced verbatim below and every column is asserted.
"""

import math

import pytest

from submission_checker.models import RegionBounds, classify_concurrency, compute_regions
from submission_checker.models.regions import (
    MARGIN_SATISFIES_HIGH_CONCURRENCY,
    ULTRA_LOW_CONCURRENCY_MAX,
    covered_region,
)

#: Appendix B, verbatim — (C_max, C_min, low, med, high, margin), each an inclusive pair.
APPENDIX_B = [
    (64, 2, (3, 6), (7, 18), (19, 64), (65, 71)),
    (128, 2, (3, 7), (8, 27), (28, 128), (129, 141)),
    (256, 2, (3, 8), (9, 42), (43, 256), (257, 282)),
    (256, 8, (9, 14), (15, 47), (48, 256), (257, 282)),
    (512, 8, (9, 16), (17, 71), (72, 512), (513, 564)),
    (1024, 8, (9, 18), (19, 109), (110, 1024), (1025, 1127)),
    (512, 16, (17, 24), (25, 79), (80, 512), (513, 564)),
    (1024, 16, (17, 26), (27, 117), (118, 1024), (1025, 1127)),
    (2048, 16, (17, 29), (30, 176), (177, 2048), (2049, 2253)),
    (1024, 32, (33, 42), (43, 131), (132, 1024), (1025, 1127)),
    (2048, 32, (33, 45), (46, 192), (193, 2048), (2049, 2253)),
    (4096, 32, (33, 48), (49, 287), (288, 4096), (4097, 4506)),
    (8192, 32, (33, 52), (53, 437), (438, 8192), (8193, 9012)),
    (16384, 32, (33, 57), (58, 676), (677, 16384), (16385, 18023)),
]


@pytest.mark.parametrize("c_max, c_min, low, med, high, margin", APPENDIX_B)
def test_region_boundaries_match_appendix_b(c_max, c_min, low, med, high, margin):
    r = compute_regions(c_max, c_min)
    assert (r.low_latency.start, r.low_latency.end) == (1, c_min)
    assert (r.low_concurrency.start, r.low_concurrency.end) == low
    assert (r.med_concurrency.start, r.med_concurrency.end) == med
    assert (r.high_concurrency.start, r.high_concurrency.end) == high
    assert (r.margin.start, r.margin.end) == margin


def test_v07_boundaries_are_the_c_min_32_case():
    """The v0.7 fixed boundaries are what v1.0 yields at C_min = 32.

    Asserted on its own because the whole fixture corpus was built against them:
    if this drifts, ~340 layout-coupled tests are testing a different spec.
    """
    r = compute_regions(1024, 32)
    assert (r.low_concurrency.start, r.low_concurrency.end) == (33, 42)
    assert (r.med_concurrency.start, r.med_concurrency.end) == (43, 131)
    assert (r.high_concurrency.start, r.high_concurrency.end) == (132, 1024)


@pytest.mark.parametrize("c_max, c_min", [(c[0], c[1]) for c in APPENDIX_B])
def test_regions_are_contiguous(c_max, c_min):
    r = compute_regions(c_max, c_min)
    assert r.low_latency.end + 1 == r.low_concurrency.start
    assert r.low_concurrency.end + 1 == r.med_concurrency.start
    assert r.med_concurrency.end + 1 == r.high_concurrency.start
    assert r.high_concurrency.end + 1 == r.margin.start


def test_low_latency_starts_at_one_and_ends_at_c_min():
    for c_min in (1, 4, 16, 32):
        r = compute_regions(1024, c_min)
        assert r.low_latency.start == 1
        assert r.low_latency.end == c_min


def test_high_concurrency_ends_at_c_max_and_margin_extends_it():
    """§5.4 split the 10% margin out of the high region; it is its own band now."""
    for c_max in (64, 512, 2048):
        r = compute_regions(c_max, 16)
        assert r.high_concurrency.end == c_max
        assert r.margin.start == c_max + 1
        assert r.margin.end == math.ceil(c_max * 1.10)


def test_c_max_must_exceed_ultra_low_maximum():
    with pytest.raises(ValueError):
        compute_regions(ULTRA_LOW_CONCURRENCY_MAX, 16)
    with pytest.raises(ValueError):
        compute_regions(1, 1)


def test_c_min_must_be_within_ultra_low_band():
    """§5.4: C_min is the Ultra Low Concurrency point, so it cannot exceed 32."""
    with pytest.raises(ValueError):
        compute_regions(1024, ULTRA_LOW_CONCURRENCY_MAX + 1)
    with pytest.raises(ValueError):
        compute_regions(1024, 0)


def test_zero_width_region_is_covered_by_its_neighbour():
    """§5.5: a rounding collision leaves a region with a single valid level.

    ``start > end`` makes ``contains`` false for every value, so the collided level
    is classified by the next region up rather than vanishing from the space.
    """
    r = compute_regions(33, 32)
    assert r.low_concurrency.start == 33
    # log2(1)/3 = 0, so both boundaries round to 33 and med_concurrency is 34–33.
    assert r.med_concurrency.start > r.med_concurrency.end
    assert classify_concurrency(33, r) == "low_concurrency"
    for c in range(1, r.margin.end + 1):
        assert classify_concurrency(c, r) is not None


def test_classify_low_latency():
    r = compute_regions(1024, 32)
    assert classify_concurrency(1, r) == "low_latency"
    assert classify_concurrency(32, r) == "low_latency"


def test_classify_concurrency_regions():
    r = compute_regions(1024, 32)  # low=33-42, med=43-131, high=132-1024
    assert classify_concurrency(33, r) == "low_concurrency"
    assert classify_concurrency(42, r) == "low_concurrency"
    assert classify_concurrency(43, r) == "med_concurrency"
    assert classify_concurrency(131, r) == "med_concurrency"
    assert classify_concurrency(132, r) == "high_concurrency"
    assert classify_concurrency(1024, r) == "high_concurrency"


def test_classify_above_c_max_is_margin():
    r = compute_regions(1024, 32)  # margin = 1025–1127
    assert classify_concurrency(1025, r) == "margin"
    assert classify_concurrency(1127, r) == "margin"


def test_classify_out_of_range_returns_none():
    r = compute_regions(1024, 32)
    assert classify_concurrency(1128, r) is None
    assert classify_concurrency(9999, r) is None


def test_margin_point_does_not_cover_high_concurrency():
    """§5.4: "the margin does not affect the required point distribution"."""
    r = compute_regions(1024, 32)
    assert classify_concurrency(1100, r) == "margin"
    expected = "high_concurrency" if MARGIN_SATISFIES_HIGH_CONCURRENCY else None
    assert covered_region(1100, r) == expected
    # Everything below C_max attributes exactly as it classifies.
    assert covered_region(1024, r) == "high_concurrency"
    assert covered_region(32, r) == "low_latency"


def test_region_bounds_contains():
    b = RegionBounds(33, 42)
    assert b.contains(33)
    assert b.contains(42)
    assert not b.contains(32)
    assert not b.contains(43)
