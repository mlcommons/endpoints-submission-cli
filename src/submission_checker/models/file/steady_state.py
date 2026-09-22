# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Steady-state reporting basis — §4.4's `steady_state` block.

Until v0.7 a point's metrics were averaged over the whole post-``TEST_STARTED`` run,
which still contained the ramp-up that inflates the TTFT tail and the drain that
deflates throughput. v1.0 makes **the detected steady-state window the official
result**, with the whole-run ``total`` metrics kept as supplementary.

The checker cannot run the detector — that is a post-processing step over the durable
event log, and ``events.jsonl`` is not part of the submitted bundle. What it can do is
validate the block the submitter reports: that its vocabulary is the spec's, that the
status and shape agree with each other, and that the window it claims is long enough
to be the official result under §6.2.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "GATING_STATES",
    "MIN_TREND_N",
    "OFFICIAL_STATUS",
    "STEADY_STATE_STATUSES",
    "STEADY_STATE_VERDICTS",
    "SteadyState",
    "SteadyStateWindow",
]

#: §4.4's trend-test floor: a window must span at least this many super-passes.
MIN_TREND_N = 4

#: §4.4 coverage statuses, in descending order of confidence.
STEADY_STATE_STATUSES = (
    "windowable",
    "insufficient_duration",
    "insufficient_passes",
    "partial_dataset",
)

#: The only status under which steady-state metrics are the official result (§4.4).
OFFICIAL_STATUS = "windowable"

#: Detector verdicts (§4.4's shape table).
STEADY_STATE_VERDICTS = (
    "STEADY STATE",
    "drifting_up",
    "drifting_down",
    "anomaly",
    "not found",
)

#: Per-metric gating states. §4.4 gates on TTFT and TPOT at P50/P90.
GATING_STATES = ("Plateau", "Drifting Up", "Drifting Down")


class SteadyStateWindow(BaseModel):
    """The detected window's extent (§4.4).

    Attributes:
        super_pass_start: First super-pass in the window, inclusive.
        super_pass_end: Last super-pass in the window, inclusive.
        super_pass_size: Queries per super-pass actually used — one full dataset pass
            unless the benchmark definition specifies otherwise.
        n_samples: Completed queries inside the window.
        duration_s: The window's **issue-time** span. §4.4 measures §6.2's minimum run
            duration against this, not against wall-clock: the drain is excluded by
            defining the window on issue time, so wall-clock would overstate it.
    """

    model_config = ConfigDict(extra="allow")

    super_pass_start: int | None = None
    super_pass_end: int | None = None
    super_pass_size: int | None = None
    n_samples: int | None = None
    duration_s: float | None = None

    @property
    def super_passes(self) -> int | None:
        """How many super-passes the window spans, or None when unstated."""
        if self.super_pass_start is None or self.super_pass_end is None:
            return None
        return self.super_pass_end - self.super_pass_start + 1


class SteadyState(BaseModel):
    """A point's §4.4 reporting block.

    Attributes:
        status: Coverage classification — one of :data:`STEADY_STATE_STATUSES`.
        verdict: The detector's shape verdict — one of :data:`STEADY_STATE_VERDICTS`.
        window: The detected window's extent.
        state: Per-gating-metric stability, e.g. ``{"tpot_p90": "Plateau"}``.
        anomaly: Present only on a confirmed level shift (§4.4's staircase case).
    """

    model_config = ConfigDict(extra="allow")

    status: str | None = None
    verdict: str | None = None
    window: SteadyStateWindow = Field(default_factory=SteadyStateWindow)
    state: dict[str, str] = Field(default_factory=dict)
    anomaly: object | None = None

    @property
    def is_official(self) -> bool:
        """True when §4.4 makes the windowed metrics this point's official result.

        Only ``windowable`` qualifies. Every other status falls back to ``total``,
        with the steady-state numbers reported as low-confidence.
        """
        return self.status == OFFICIAL_STATUS

    @property
    def drifting_metrics(self) -> list[str]:
        """Gating metrics that are drifting rather than flat.

        §4.4 reports a drifting metric as a range or slope and "never as a point
        estimate", so a consumer of the reported percentile needs to know.
        """
        return sorted(name for name, state in self.state.items() if state.startswith("Drifting"))
