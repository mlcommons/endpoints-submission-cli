"""Tests for region boundary computation (§5.5 reference algorithm)."""

import math

import pytest

from submission_checker.models import RegionBounds, classify_concurrency, compute_regions

# Reference table for the m=32 offset — (M, LT_end, MT_end).
# With m=32 the amended algorithm reproduces the pre-amendment fixed-offset boundaries.
APPENDIX_B = [
    (64, 35, 42),
    (128, 37, 53),
    (256, 38, 69),
    (512, 40, 93),  # Appendix B: MT = 41–93 (algorithm gives round(93.30) = 93)
    (1024, 42, 131),
    (2048, 45, 192),
    (4096, 48, 287),
    (8192, 52, 437),
    (16384, 57, 676),
]


@pytest.mark.parametrize("M, expected_lt_end, expected_mt_end", APPENDIX_B)
def test_region_boundaries_match_appendix_b(M, expected_lt_end, expected_mt_end):
    r = compute_regions(32, M)
    assert r.low_throughput.end == expected_lt_end, f"M={M}: LT end"
    assert r.med_throughput.end == expected_mt_end, f"M={M}: MT end"


def test_amended_examples_use_m_as_offset():
    """PR #63 worked examples — regions offset from declared m, not a fixed 32."""
    # Example B: m=1, M=256
    r = compute_regions(1, 256)
    assert (r.low_latency.start, r.low_latency.end) == (1, 1)
    assert (r.low_throughput.start, r.low_throughput.end) == (2, 7)
    assert (r.med_throughput.start, r.med_throughput.end) == (8, 41)
    assert r.high_throughput.start == 42
    # Example C: m=16, M=1024
    # NB: the exact reference algorithm gives med end = round(16 + 100.53) = 117.
    # The PR's worked example shows 116 from rounding an intermediate (2^6.652≈100.4);
    # we follow the algorithm, not the hand-computed illustration.
    r = compute_regions(16, 1024)
    assert (r.low_latency.start, r.low_latency.end) == (1, 16)
    assert (r.low_throughput.start, r.low_throughput.end) == (17, 26)
    assert (r.med_throughput.start, r.med_throughput.end) == (27, 117)
    assert r.high_throughput.start == 118


def test_low_latency_spans_1_to_m():
    for m, M in [(1, 256), (16, 1024), (32, 8192), (64, 4096)]:
        r = compute_regions(m, M)
        assert r.low_latency.start == 1
        assert r.low_latency.end == m


def test_high_throughput_ends_at_margin():
    for M in [64, 512, 2048]:
        r = compute_regions(32, M)
        assert r.high_throughput.end == math.ceil(M * 1.10)


def test_regions_are_contiguous():
    r = compute_regions(32, 512)
    assert r.low_latency.end + 1 == r.low_throughput.start
    assert r.low_throughput.end + 1 == r.med_throughput.start
    assert r.med_throughput.end + 1 == r.high_throughput.start


def test_M_must_be_greater_than_m():
    with pytest.raises(ValueError):
        compute_regions(32, 32)  # M == m
    with pytest.raises(ValueError):
        compute_regions(64, 32)  # M < m


def test_m_must_be_at_least_1():
    with pytest.raises(ValueError):
        compute_regions(0, 256)


def test_classify_low_latency():
    r = compute_regions(32, 1024)
    assert classify_concurrency(1, r) == "low_latency"
    assert classify_concurrency(32, r) == "low_latency"


def test_classify_throughput_regions():
    r = compute_regions(32, 1024)  # LT=33-42, MT=43-131, HT=132-1024
    assert classify_concurrency(33, r) == "low_throughput"
    assert classify_concurrency(42, r) == "low_throughput"
    assert classify_concurrency(43, r) == "med_throughput"
    assert classify_concurrency(131, r) == "med_throughput"
    assert classify_concurrency(132, r) == "high_throughput"
    assert classify_concurrency(1024, r) == "high_throughput"


def test_classify_above_m_still_high_throughput():
    r = compute_regions(32, 1024)
    # high_throughput now extends to ceil(1024 * 1.10) = 1127
    assert classify_concurrency(1025, r) == "high_throughput"
    assert classify_concurrency(1126, r) == "high_throughput"
    assert classify_concurrency(1127, r) == "high_throughput"


def test_classify_out_of_range_returns_none():
    r = compute_regions(32, 1024)
    assert classify_concurrency(9999, r) is None


def test_region_bounds_contains():
    b = RegionBounds(33, 42)
    assert b.contains(33)
    assert b.contains(42)
    assert not b.contains(32)
    assert not b.contains(43)
