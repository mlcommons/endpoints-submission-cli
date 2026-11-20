# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: did Offline query reordering stay within dataset-pass boundaries?

A dedicated Offline run may reorder its queries, but "Reordering MUST NOT cross a
dataset-pass boundary" (§5.7.1): sorting across passes gathers the repeated instances
of one sample into a batch. The client issues the passes in order, so the order in
which queries *complete* shows how the system scheduled them. From ``events.jsonl``:

- A query's pass is its issue position divided by the pass size (``num_turns`` of the
  performance ``phase_start``, i.e. ``n_samples_from_dataset``).
- **Cross-pass share**: of the completions of pass N, the share that finished before
  the median completion of pass N-1. Within-pass reordering keeps it near zero, apart
  from the queries in flight at a boundary; sorting across passes pushes it toward half.
- **Same-sample neighbours**: of consecutive completions, the share that are the same
  dataset sample from different passes. Shuffled passes almost never produce that;
  sorting across passes does all the time. The sample is identified by the client's
  ``sample_idx_map.json`` when present, otherwise by the prompt itself.

§5.7's open item 5 asks the client to make pass boundaries explicit in the log; until
it does, this inference is the only automated view.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from ..context import INPUT_LOGS, ReviewContext
from ..findings import Finding, Level
from ..logs import PointLog, SampleRecord
from .base import Check, finding_factory, pct

__all__ = ["CHECK", "pass_metrics"]

_finding = finding_factory("offline_reordering")
_SPEC = "§9.2, §5.7.1"


def _identity(sample: SampleRecord, log: PointLog) -> str | None:
    index = log.dataset_index.get(sample.uuid)
    return str(index) if index is not None else sample.prompt_key


def pass_metrics(log: PointLog) -> tuple[int, float, float] | None:
    """``(passes, cross_pass_share, same_sample_share)`` for an Offline log.

    ``None`` when the log has no performance phase with a pass size, or fewer than two
    passes completed.
    """
    phases = log.phase_of("performance")
    if not phases or log.phases[phases[0]].num_turns <= 0:
        return None
    size = log.phases[phases[0]].num_turns
    issued = [s for s in log.issued("performance") if s.phase == phases[0]]
    pass_of = {s.uuid: s.order // size for s in issued}
    done = [s for s in log.completions("performance") if s.uuid in pass_of]
    passes = len({pass_of[s.uuid] for s in done})
    if passes < 2:
        return None

    positions: dict[int, list[int]] = defaultdict(list)
    for position, s in enumerate(done):
        positions[pass_of[s.uuid]].append(position)
    early = total = 0
    for p in sorted(positions)[1:]:
        if p - 1 not in positions:
            continue
        median_before = statistics.median(positions[p - 1])
        early += sum(pos < median_before for pos in positions[p])
        total += len(positions[p])
    cross = early / total if total else 0.0

    same = 0
    for a, b in zip(done, done[1:], strict=False):
        ia, ib = _identity(a, log), _identity(b, log)
        if ia is not None and ia == ib and pass_of[a.uuid] != pass_of[b.uuid]:
            same += 1
    same_share = same / (len(done) - 1) if len(done) > 1 else 0.0
    return passes, cross, same_share


def run(ctx: ReviewContext) -> list[Finding]:
    """Return the pass-boundary findings for every dedicated Offline point with a log."""
    t = ctx.thresholds.logs
    out: list[Finding] = []
    for curve in ctx.curves:
        for point in curve.points:
            if not point.is_dedicated_offline:
                continue
            log = ctx.logs.get(point.dir)
            if log is None:
                continue
            result = pass_metrics(log)
            if result is None:
                continue
            passes, cross, same = result
            if cross <= t.offline_cross_pass_note and same <= t.offline_same_sample_attention:
                continue
            level = (
                Level.ATTENTION
                if cross > t.offline_cross_pass_attention or same > t.offline_same_sample_attention
                else Level.NOTE
            )
            out.append(
                _finding(
                    level,
                    "Offline queries reordered across dataset passes",
                    f"Over {passes} passes, {pct(cross)} of each pass's completions finished "
                    f"before the median completion of the previous pass, and {pct(same)} of "
                    "consecutive completions were the same dataset sample from different "
                    "passes. Reordering within a pass keeps both near zero; ask how the "
                    "Offline queue was scheduled.",
                    curve=curve.label,
                    point=point.name,
                    spec_ref=_SPEC,
                    passes=passes,
                    cross_pass_share=round(cross, 4),
                    same_sample_share=round(same, 4),
                )
            )
    return out


CHECK = Check(
    name="offline_reordering",
    focus_area="offline_reordering",
    requires=frozenset({INPUT_LOGS}),
    run=run,
)
