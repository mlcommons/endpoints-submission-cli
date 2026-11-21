# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Steady-state reporting basis — §4.4's `steady_state` block.

Until v0.7 a point's metrics were averaged over the whole post-``TEST_STARTED`` run,
which still contained the ramp-up that inflates the TTFT tail and the drain that
deflates throughput. v1.0 makes **the detected steady-state window the official
result**, with the whole-run ``total`` metrics kept as supplementary.

The checker cannot run the detector — that is a post-processing step over the durable
event log, and ``events.jsonl`` is not part of the submitted bundle. The reference
client runs it when ``settings.steady_state.enabled`` is set and writes its verdict to
``result_summary.json``. :func:`from_client_verdict` reads that verdict into §4.4's
terms, so the checker never relies on a block copied by hand into ``point.yaml``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "GATING_STATES",
    "MIN_TREND_N",
    "OFFICIAL_STATUS",
    "STEADY_STATE_STATUSES",
    "STEADY_STATE_VERDICTS",
    "ClientSteadyState",
    "SteadyState",
    "SteadyStateWindow",
    "from_client_verdict",
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


#: The client's per-metric trend (``global_trend``) in §4.4's gating-state terms.
#: ``insufficient`` (too few super-passes to test) has no §4.4 state and is left out.
_TREND_STATES = {"steady": "Plateau", "up": "Drifting Up", "down": "Drifting Down"}


class ClientWindow(BaseModel):
    """The client's ``steady_state.window``. ``sp_hi`` is exclusive."""

    model_config = ConfigDict(extra="allow")

    sp_lo: int
    sp_hi: int
    n_samples: int
    start_ns: int
    end_ns: int


class ClientAnomaly(BaseModel):
    """The client's ``steady_state.anomaly``."""

    model_config = ConfigDict(extra="allow")

    detected: bool = False


class ClientSteadyState(BaseModel):
    """The ``steady_state`` block the reference client writes to ``result_summary.json``.

    Only the fields the checker reads are declared. ``window`` is the reported plateau;
    when ``found`` is false it is the longest candidate the client rejected as too
    short, or ``None`` when there was no plateau at all.
    """

    model_config = ConfigDict(extra="allow")

    found: bool
    superpass_size: int
    n_super_passes: int
    warmup: int = 0
    window: ClientWindow | None = None
    anomaly: ClientAnomaly = Field(default_factory=ClientAnomaly)
    global_trend: dict[str, str] = Field(default_factory=dict)


def from_client_verdict(raw: dict[str, Any], min_duration_s: float | None) -> SteadyState:
    """Read the client's verdict into a §4.4 block.

    ``status`` follows §4.4's status table. A run with no full super-pass after warmup
    is ``partial_dataset``. A run with no accepted window of at least
    :data:`MIN_TREND_N` super-passes is ``insufficient_passes``, unless the client
    rejected a long enough window as too short, which is ``insufficient_duration``. An
    accepted window shorter than *min_duration_s* (§6.2) is also
    ``insufficient_duration``. Otherwise the point is ``windowable``. With
    *min_duration_s* unknown, the duration test is skipped.

    ``verdict`` follows §4.4's shape table: ``not found`` when the client accepted no
    window, then ``anomaly`` (the first plateau stays the result), then ``drifting_up``
    / ``drifting_down`` from the client's trend after the window, else
    ``STEADY STATE``.

    Raises:
        pydantic.ValidationError: *raw* is not the client's block.
    """
    client = ClientSteadyState.model_validate(raw)
    window = SteadyStateWindow()
    passes = 0
    if client.window is not None:
        cw = client.window
        passes = cw.sp_hi - cw.sp_lo
        window = SteadyStateWindow(
            super_pass_start=cw.sp_lo,
            super_pass_end=cw.sp_hi - 1,
            super_pass_size=client.superpass_size,
            n_samples=cw.n_samples,
            duration_s=(cw.end_ns - cw.start_ns) / 1e9,
        )
    state = {
        metric: _TREND_STATES[trend]
        for metric, trend in sorted(client.global_trend.items())
        if trend in _TREND_STATES
    }

    if client.n_super_passes - client.warmup < 1:
        status = "partial_dataset"
    elif passes < MIN_TREND_N:
        status = "insufficient_passes"
    elif not client.found or (
        min_duration_s is not None and (window.duration_s or 0.0) < min_duration_s
    ):
        status = "insufficient_duration"
    else:
        status = OFFICIAL_STATUS

    if not client.found:
        verdict = "not found"
    elif client.anomaly.detected:
        verdict = "anomaly"
    elif "Drifting Up" in state.values():
        verdict = "drifting_up"
    elif "Drifting Down" in state.values():
        verdict = "drifting_down"
    else:
        verdict = "STEADY STATE"

    return SteadyState(
        status=status,
        verdict=verdict,
        window=window,
        state=state,
        anomaly=raw.get("anomaly") if client.anomaly.detected else None,
    )
