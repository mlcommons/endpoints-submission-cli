# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Accuracy targets for the agentic benchmarks (§3.2, §4.3).

§3.2 makes the reference implementation the authority for per-model specifications —
"canonical weights, dataset, chat template, server parameters, accuracy target" — and
these come from its Agentic Inference example:

    mlcommons/endpoints  examples/10_Agentic_Inference/README.md  § Accuracy

They are kept apart from :mod:`submission_checker.accuracy_targets` because they are
gated differently, not merely valued differently. That module folds every dataset of a
point into one sample-weighted score per metric and gates each point against it. The
agentic benchmarks gate three quantities that do not reduce that way:

* **Inline accuracy** — per point, from the ``agentic_combined`` performance dataset.
* **SWE-bench accuracy** — **mean-of-N across points**, not per point: "average one
  SWE-bench accuracy result from each of the four mandatory regions (``N = 4``), then
  compare that mean with the model-specific SWE-bench threshold". This is §4.3's
  multi-turn branch: "The arithmetic mean of the ``N`` required accuracy results MUST
  meet the quality threshold; individual results need not."
* **OSL per-turn mean** — a *range*, and not an accuracy metric at all: it is read from
  ``result_summary.json``. The README is emphatic about which field, because the
  windowed one has the same name.

Folding the first two together with a weighted mean, as the single-turn gate would,
gives a number with no meaning under either rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "INLINE_DATASET",
    "OSL_FULL_RUN_FIELD",
    "SWEBENCH_DATASET",
    "SWEBENCH_MEAN_OF_N",
    "AgenticTargets",
    "get_agentic_targets",
]

#: Performance dataset carrying the inline accuracy score
#: (``accuracy_config.eval_method: agentic_inference_inline``).
INLINE_DATASET = "agentic_combined"

#: Accuracy dataset scored by ``swe_bench_scorer``.
SWEBENCH_DATASET = "swe_bench"

#: §4.3's multi-turn mean-of-N, over the four mandatory regions.
SWEBENCH_MEAN_OF_N = 4

#: Where the OSL per-turn mean is read from. The README: "Read from
#: ``output_sequence_lengths_full_run.output_sequence_lengths.avg`` (the full-run,
#: all-turns mean), **not** the windowed ``output_sequence_lengths.avg``."
OSL_FULL_RUN_FIELD = "output_sequence_lengths_full_run"


@dataclass(frozen=True)
class AgenticTargets:
    """One agentic model's accuracy gates.

    Every field is ``None`` where the reference implementation records the threshold as
    TBD. A model with no published thresholds is still a *recognised* model — it is the
    gate that cannot run, which is a different report from an unknown model.
    """

    #: Model name as §3.2 publishes it, for messages.
    name: str
    #: Inline accuracy floor, per point, on a 0–100 scale.
    inline_min: float | None
    #: SWE-bench floor for the mean of the N results, on a 0–100 scale.
    swebench_min: float | None
    #: Inclusive ``(low, high)`` bounds for the full-run OSL per-turn mean, in tokens.
    osl_range: tuple[float, float] | None

    @property
    def published(self) -> bool:
        """Whether any threshold is published for this model yet."""
        return self.inline_min is not None or self.swebench_min is not None


# (model-keyword fragments, targets). Reference means from the README, for context:
# Kimi K3 inline 58.9, OSL 472, SWE-bench 94.83; Qwen inline 56.43, OSL 383,
# SWE-bench 71.7.
_AGENTIC_TARGETS: list[tuple[frozenset[str], AgenticTargets]] = [
    (
        frozenset({"kimi", "k3"}),
        AgenticTargets(
            name="Kimi K3",
            inline_min=58.32,
            swebench_min=93.5,
            osl_range=(425.0, 520.0),
        ),
    ),
    (
        frozenset({"qwen3", "35b"}),
        AgenticTargets(
            name="Qwen3.6-35B-A3B",
            inline_min=55.86,
            swebench_min=69.0,
            osl_range=(344.0, 422.0),
        ),
    ),
    # "The DSV4 accuracy thresholds and SWE-bench evaluation policy are TBD."
    (
        frozenset({"deepseek", "v4"}),
        AgenticTargets(
            name="DSV4",
            inline_min=None,
            swebench_min=None,
            osl_range=None,
        ),
    ),
]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def get_agentic_targets(model: str) -> AgenticTargets | None:
    """Return the agentic targets for *model*, or ``None`` when it is not one.

    Args:
        model: Model directory name (§8.1), e.g. ``"kimi-k3"``.
    """
    haystack = _normalize(model)
    for fragments, targets in _AGENTIC_TARGETS:
        if all(f in haystack for f in fragments):
            return targets
    return None
