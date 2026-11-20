# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""The registered review checks, one module per §9.2 focus area.

A check is a :class:`~submission_reviewer.checks.base.Check`: a name, the focus area it
informs, the inputs it needs, and a function from a
:class:`~submission_reviewer.context.ReviewContext` to findings. To add one, write a
module exposing ``CHECK`` and append it to :data:`ALL_CHECKS`; the comment picks it up
from its focus area.
"""

from __future__ import annotations

from . import (
    division,
    metric_distributions,
    offline_reordering,
    output_padding,
    pareto_shape,
    specdec_info,
    system_description,
    ttft_fragments,
    updates,
    warmup,
    warmup_prompts,
)
from .base import Check

__all__ = ["ALL_CHECKS", "Check"]

#: Every check, in the order their findings are reported.
ALL_CHECKS: tuple[Check, ...] = (
    pareto_shape.CHECK,
    metric_distributions.CHECK,
    ttft_fragments.CHECK,
    output_padding.AGGREGATE_CHECK,
    output_padding.TEXT_CHECK,
    offline_reordering.CHECK,
    warmup.CHECK,
    warmup_prompts.CHECK,
    system_description.CHECK,
    division.CHECK,
    specdec_info.CHECK,
    updates.CHECK,
)
