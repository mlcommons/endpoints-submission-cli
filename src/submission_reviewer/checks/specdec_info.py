# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: is the disclosed drafter and algorithm consistent with the approved list?

Matching the drafter against the benchmark's approved list is §9.1's
``approved-drafter`` check, so this one does not repeat it. It gives the reviewer the
context §9.2 asks them to weigh for a Methodology objection:

- which points declare speculative decoding, and what they declare (INFO);
- a drafter identity (model ID or checksum) that differs between points of one curve,
  which §9.1's configuration-consistency row does not cover (NOTE).
"""

from __future__ import annotations

import json

from ..context import INPUT_POINT_CONFIGS, ReviewContext
from ..findings import Finding, Level
from .base import Check, finding_factory, points_label

__all__ = ["CHECK"]

_finding = finding_factory("specdec_info", "speculative_decoding")

#: Keys that identify the drafter itself, rather than how it was configured per point.
_IDENTITY_KEYS = (
    "drafter",
    "drafter_model_id",
    "drafter_checksum",
    "model_id",
    "checksum",
    "weight_checksum",
)
#: Keys shown first when summarizing a declaration.
_SUMMARY_KEYS = (*_IDENTITY_KEYS, "algorithm", "method", "precision", "num_speculative_tokens")


def _summary(declaration: dict[str, object], limit: int = 5) -> str:
    keys = [k for k in _SUMMARY_KEYS if k in declaration]
    keys += [k for k in declaration if k not in keys]
    parts = []
    for key in keys[:limit]:
        value = str(declaration[key])
        parts.append(f"{key}={value if len(value) <= 40 else value[:39] + '…'}")
    return ", ".join(parts) or "(empty declaration)"


def run(ctx: ReviewContext) -> list[Finding]:
    """Return the speculative-decoding context for every frontier in *ctx*."""
    out: list[Finding] = []
    for curve in ctx.curves:
        declared = [
            (p.name, c.speculative_decoding)
            for p, c in curve.configs()
            if c.speculative_decoding is not None
        ]
        if not declared:
            continue
        groups: dict[str, list[str]] = {}
        for name, block in declared:
            groups.setdefault(json.dumps(block, sort_keys=True, default=str), []).append(name)
        lines = [
            f"{points_label(names)}: {_summary(json.loads(key))}" for key, names in groups.items()
        ]
        out.append(
            _finding(
                Level.INFO,
                f"Speculative decoding on {len(declared)} point(s)",
                "; ".join(lines) + ". Whether the drafter is on the approved list is checked "
                "by §9.1 (approved-drafter); see the Check submission results.",
                curve=curve.label,
                spec_ref="§9.2, §2.9.4",
                points=len(declared),
            )
        )
        identities = {
            json.dumps({k: block.get(k) for k in _IDENTITY_KEYS if k in block}, sort_keys=True)
            for _, block in declared
        }
        identities.discard("{}")
        if len(identities) > 1:
            out.append(
                _finding(
                    Level.NOTE,
                    "Drafter differs between points",
                    f"The points of this curve declare {len(identities)} different drafter "
                    "identities. One result should use one model configuration; confirm each "
                    "drafter is approved for this benchmark.",
                    curve=curve.label,
                    spec_ref="§9.2, §2.9.4",
                )
            )
    return out


CHECK = Check(
    name="specdec_info",
    focus_area="speculative_decoding",
    requires=frozenset({INPUT_POINT_CONFIGS}),
    run=run,
)
