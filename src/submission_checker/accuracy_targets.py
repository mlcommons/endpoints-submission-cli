"""Accuracy quality targets ported from the MLPerf Inference submission checker.

Thresholds mirror mlcommons/inference:
  tools/submission/submission_checker/constants.py  (v6.0 "accuracy-target")

Each model entry maps lowercase metric name → (lower_bound, upper_bound).
upper_bound is None when the inference spec sets no upper limit for that metric.

min_queries mirrors the "dataset-size" section of the same constants file, which
defines the minimum number of samples required for a valid accuracy run.
"""

from __future__ import annotations

import re

__all__ = ["get_thresholds"]


# (model-keyword fragments, {metric → (lower, upper | None)}, min_queries)
# More-specific entries must come before less-specific ones.
_TARGETS: list[tuple[frozenset[str], dict[str, tuple[float, float | None]], int]] = [
    # deepseek-r1  — golden fp32 exact_match 81.3582 (gate ≥ 0.99×).
    # Only exact_match is gated: results.json's accuracy_scores expose a single scalar
    # `score` (== exact_match). The canonical spec also bounds TOKENS_PER_SAMPLE
    # (0.9–1.1 × 3886.2274), but that metric lives only in the deepseek_eval file, not
    # in results.json, so it is intentionally NOT gated here.
    (
        frozenset({"deepseek", "r1"}),
        {
            "exact_match": (81.3582 * 0.99, None),
        },
        4388,
    ),
    # gpt-oss-120b  — golden fp32 exact_match 83.13; upstream tokens_per_sample upper
    # bound is still a placeholder (constants.py:215), so it is intentionally omitted.
    # min_queries uses the accuracy-sample-count (4395), not the perf count (6396).
    (
        frozenset({"gptoss", "120b"}),
        {
            "exact_match": (83.13 * 0.99, None),
        },
        4395,
    ),
    # llama3.1-405b  — inference uses ROUGEL as primary; also exact_match + tokens
    (
        frozenset({"llama3", "405b"}),
        {
            "rougel": (21.6666 * 0.99, None),
            "exact_match": (90.1335 * 0.99, None),
            "tokens_per_sample": (684.68 * 0.9, 684.68 * 1.1),
        },
        8313,
    ),
    # llama2-70b  (using -99 / 1 % delta tier)
    (
        frozenset({"llama2", "70b"}),
        {
            "rouge1": (44.4312 * 0.99, None),
            "rouge2": (22.0352 * 0.99, None),
            "rougel": (28.6162 * 0.99, None),
            "tokens_per_sample": (294.45 * 0.9, 294.45 * 1.1),
        },
        24576,
    ),
    # mixtral-8x7b
    (
        frozenset({"mixtral", "8x7b"}),
        {
            "rouge1": (45.5989 * 0.99, None),
            "rouge2": (23.3526 * 0.99, None),
            "rougel": (30.4608 * 0.99, None),
            "tokens_per_sample": (144.84 * 0.9, 145.9 * 1.1),
        },
        15000,
    ),
    # llama3.1-8b  (covers Instruct and other fine-tunes of the 8B base)
    (
        frozenset({"llama3", "8b"}),
        {
            "rouge1": (38.7792 * 0.99, None),
            "rouge2": (15.9075 * 0.99, None),
            "rougel": (24.4957 * 0.99, None),
            "rougelsum": (35.793 * 0.99, None),
            "gen_len": (8167644 * 0.9, 8167644 * 1.1),
        },
        13368,
    ),
]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def get_thresholds(
    model: str,
) -> tuple[dict[str, tuple[float, float | None]], int] | None:
    """Return ``(metric_thresholds, min_queries)`` for *model*, or None if unknown.

    Args:
        model: Model directory name (e.g. ``"Llama-3_1-8B-Instruct"``).

    Returns:
        A 2-tuple of:
          - mapping of lowercase metric name → ``(lower_bound, upper_bound)``
            where ``upper_bound`` is ``None`` when the spec sets no upper limit.
          - minimum required number of accuracy samples.
        Returns ``None`` when no entry matches.
    """
    haystack = _normalize(model)
    for fragments, thresholds, min_queries in _TARGETS:
        if all(f in haystack for f in fragments):
            return thresholds, min_queries
    return None
