"""Point result model — cross-file validation combining point config and result summary."""

from __future__ import annotations

import math
from pathlib import Path

__all__ = ["MIN_QUERY_COUNT", "PointResult"]

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationInfo, model_validator

from ..file.point_config import PointConfig
from ..file.point_summary import PointSummary
from ..regions import MIN_DURATION_MS, classify_concurrency
from ..results import CheckResult, err, ok, warn

_TPS_TOLERANCE = 0.01  # 1% relative tolerance for stored-vs-derived comparisons

# Dataset → minimum completed query count (§6.4).
# Values equal the full dataset size; every sample must be run for a valid submission.
MIN_QUERY_COUNT: dict[str, int] = {
    "open_orca": 24576,
    "cnn_dailymail": 13368,
    "aime25": 30,
    "gpqa": 198,
    "livecodebench": 880,
    "shopify_product_catalogue": 48289,
    "shopify_product_catalogue_8k": 8000,
    "mlperf_gpt_oss_performance": 6396,
    "mlperf_gpt_oss_accuracy": 4395,
    "math500": 500,
}


class PointResult(BaseModel):
    """Paired config and result summary for one measurement point."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    config: PointConfig
    summary: PointSummary
    yaml_path: Path

    @model_validator(mode="after")
    def _check_point_duration(self, info: ValidationInfo) -> PointResult:
        """§6.2: warn when the measured duration is below the per-region minimum.

        §4.4 changed what "measured duration" means. The minimum is now checked
        against the steady-state window's **issue-time span**, not wall-clock: the
        window excludes the drain by construction, so wall-clock overstates it and a
        point could clear §6.2 on time its official metrics never covered. Points
        reporting no window fall back to whole-run duration, which is the pre-1.0
        basis and what §4.4 calls the fallback result.
        """
        regions = (info.context or {}).get("regions")
        summary_path: Path | None = (info.context or {}).get("summary_path")
        if regions is None:
            return self
        c = self.config.concurrency
        region = classify_concurrency(c, regions)
        if region is None:
            return self  # already flagged by concurrency-in-range
        min_ms = MIN_DURATION_MS.get(region, 0)
        duration_ms, basis = self._duration_basis()
        if duration_ms < min_ms:
            self._check_results.append(
                warn(
                    "point-duration",
                    f"Point {c} ({region}): {basis} duration {duration_ms:.0f} ms"
                    f" < minimum {min_ms} ms",
                    summary_path,
                    "#11",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "point-duration",
                    f"Point {c}: {basis} duration {duration_ms:.0f} ms meets minimum for {region}",
                    summary_path,
                    "#11",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_steady_state_basis(self, info: ValidationInfo) -> PointResult:
        """§4.4: record which basis supplies this point's official result.

        Steady-state metrics are official only where ``status`` is ``windowable``;
        every other status falls back to whole-run ``total`` with the windowed numbers
        reported as low-confidence. That fallback is not a rejection — §4.4 keeps
        ``total`` as a valid basis — but it is a material difference in what the
        published number means, so it is surfaced rather than passed over.

        A ``drifting_up`` / ``drifting_down`` verdict is flagged separately: §4.4 says
        such a metric is reported "as drift (range/slope), never as a point estimate",
        which a percentile read out of ``result_summary.json`` silently is.
        """
        path: Path | None = (info.context or {}).get("summary_path")
        block = self.config.steady_state
        c = self.config.concurrency
        if block is None:
            self._check_results.append(
                warn(
                    "steady-state-basis",
                    f"Point {c}: no steady_state block; metrics are whole-run, which §4.4"
                    " makes the fallback rather than the official basis",
                    path,
                    "#4.4",
                )
            )
            return self

        if block.is_official:
            self._check_results.append(
                ok(
                    "steady-state-basis",
                    f"Point {c}: official result is the steady-state window"
                    f" ({block.window.super_passes or '?'} super-passes)",
                    path,
                    "#4.4",
                )
            )
        else:
            self._check_results.append(
                warn(
                    "steady-state-basis",
                    f"Point {c}: status {block.status!r} — official result falls back to"
                    " whole-run `total`; steady-state numbers are low-confidence (§4.4)",
                    path,
                    "#4.4",
                )
            )

        if block.verdict in ("drifting_up", "drifting_down"):
            self._check_results.append(
                warn(
                    "steady-state-basis",
                    f"Point {c}: verdict {block.verdict!r} — §4.4 reports a drifting gating"
                    f" metric as a range or slope, never as a point estimate"
                    + (f" ({', '.join(block.drifting_metrics)})" if block.drifting_metrics else ""),
                    path,
                    "#4.4",
                )
            )
        return self

    @model_validator(mode="after")
    def _check_min_query_count(self, info: ValidationInfo) -> PointResult:
        """§12: n_samples_completed must meet the dataset's minimum query count (§6.4).

        Skipped when the dataset is not in MIN_QUERY_COUNT (unknown datasets are
        not yet mapped; add them as the spec is ratified).
        """
        summary_path: Path | None = (info.context or {}).get("summary_path")
        dataset = self.config.dataset
        min_queries = MIN_QUERY_COUNT.get(dataset)
        if min_queries is None:
            return self
        completed = self.summary.n_samples_completed
        if completed < min_queries:
            self._check_results.append(
                err(
                    "min-query-count",
                    f"Dataset '{dataset}': completed {completed} < minimum {min_queries} (§6.4)",
                    summary_path,
                    "#12",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "min-query-count",
                    f"Dataset '{dataset}': completed {completed} ≥ minimum {min_queries}",
                    summary_path,
                    "#12",
                )
            )
        return self

    def _duration_basis(self) -> tuple[float, str]:
        """Return ``(duration in ms, what it measures)`` for the §6.2 comparison.

        §4.4 measures §6.2 against the steady-state window's issue-time span. A point
        that reports no window — or one whose window states no ``duration_s`` — falls
        back to the whole run, which is §4.4's own fallback basis.
        """
        block = self.config.steady_state
        if block is not None and block.window.duration_s is not None:
            return block.window.duration_s * 1000.0, "steady-state window"
        return self.summary.duration_ms, "whole-run"

    # ------------------------------------------------------------------
    # Metric-consistency sub-checks (called from _check_metric_consistency)
    # ------------------------------------------------------------------

    def _check_duration_positive(self, s: PointSummary, path: Path | None) -> None:
        """Emit ok/err for the duration_ns > 0 invariant (§14)."""
        if s.duration_ns <= 0:
            self._check_results.append(
                err(
                    "metric-consistency-duration",
                    f"duration_ns is not positive: {s.duration_ns}",
                    path,
                    "#14",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "metric-consistency-duration",
                    f"duration_ns={s.duration_ns:.0f} ns",
                    path,
                    "#14",
                )
            )

    def _check_sample_accounting(self, s: PointSummary, path: Path | None) -> None:
        """Emit ok/err for the completed + failed == issued invariant (§14).

        Skipped when n_samples_issued is 0 (point tool did not track issued count).
        """
        if s.n_samples_issued > 0:
            accounted = s.n_samples_completed + s.n_samples_failed
            if accounted != s.n_samples_issued:
                self._check_results.append(
                    err(
                        "metric-consistency-accounting",
                        f"completed ({s.n_samples_completed}) + failed ({s.n_samples_failed})"
                        f" = {accounted} ≠ issued ({s.n_samples_issued})",
                        path,
                        "#14",
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "metric-consistency-accounting",
                        f"Sample accounting consistent: {s.n_samples_issued} issued",
                        path,
                        "#14",
                    )
                )

    def _check_output_tokens_nonnegative(self, s: PointSummary, path: Path | None) -> None:
        """Emit ok/err for the total_output_tokens >= 0 invariant (§14)."""
        if s.total_output_tokens < 0:
            self._check_results.append(
                err(
                    "metric-consistency-output-tokens",
                    f"total_output_tokens is negative: {s.total_output_tokens}",
                    path,
                    "#14",
                )
            )
        else:
            self._check_results.append(
                ok(
                    "metric-consistency-output-tokens",
                    f"total_output_tokens={s.total_output_tokens}",
                    path,
                    "#14",
                )
            )

    def _check_system_tps_derivable(self, s: PointSummary, path: Path | None) -> None:
        """§9.1: system_tps must equal total_output_tokens / elapsed_duration_seconds.

        If the result log also stores a system_tps field, verify it matches the
        derived value within _TPS_TOLERANCE. Always emits ok when consistent.
        """
        derived = s.system_tps
        stored = (s.model_extra or {}).get("system_tps")
        if stored is not None:
            rel_err = abs(float(stored) - derived) / max(abs(derived), 1e-9)
            if rel_err > _TPS_TOLERANCE:
                self._check_results.append(
                    err(
                        "metric-consistency-system-tps",
                        f"stored system_tps {stored:.3f} ≠ derived {derived:.3f}"
                        f" (rel err {rel_err:.1%})",
                        path,
                        "#9.1",
                    )
                )
                return
        self._check_results.append(
            ok(
                "metric-consistency-system-tps",
                f"system_tps={derived:.3f} tok/s"
                f" ({s.total_output_tokens} tokens / {s.elapsed_duration_seconds:.1f}s)",
                path,
                "#9.1",
            )
        )

    def _check_tpot_p90(self, s: PointSummary, path: Path | None) -> None:
        """§9.1: TPOT P90 must be present, finite, and strictly positive.

        §9.1 words this as "the valid per-response TPOT distribution must be non-empty
        with a finite, strictly positive P90". A checker reading only the summary can
        never see the distribution, so what is actually verifiable is the *reported*
        percentile — the message says so rather than implying the samples were audited.
        """
        p90 = s.tpot_p90_ms
        if p90 is None:
            self._check_results.append(
                err(
                    "metric-consistency-tpot-p90",
                    "No TPOT P90 reported (result_summary.json has no tpot.percentiles['90']);"
                    " tps_per_user is defined as 1000 / tpot_p90_ms and has no other source",
                    path,
                    "#9.1",
                )
            )
            return
        if not math.isfinite(p90) or p90 <= 0:
            self._check_results.append(
                err(
                    "metric-consistency-tpot-p90",
                    f"Reported TPOT P90 is {p90}, which is not finite and strictly positive",
                    path,
                    "#9.1",
                )
            )
            return
        self._check_results.append(
            ok(
                "metric-consistency-tpot-p90",
                f"Reported TPOT P90 = {p90:.4f} ms (distribution itself is not verifiable"
                " from the summary)",
                path,
                "#9.1",
            )
        )

    def _check_tps_per_user(self, s: PointSummary, concurrency: int, path: Path | None) -> None:
        """§9.1: ``tps_per_user = 1000 / tpot_p90_ms``.

        v0.7 defined this as ``system_tps / concurrency``; v1.0 redefines it as the
        per-user token rate implied by the P90 time per output token, which is a
        latency the user actually experiences rather than an average over the batch.

        If the result log stores a ``tps_per_user`` field, it is verified against the
        derived value within ``_TPS_TOLERANCE``.
        """
        p90 = s.tpot_p90_ms
        if p90 is None or not math.isfinite(p90) or p90 <= 0:
            return  # already reported by metric-consistency-tpot-p90
        derived = 1000.0 / p90
        stored = (s.model_extra or {}).get("tps_per_user")
        if stored is not None:
            rel_err = abs(float(stored) - derived) / max(abs(derived), 1e-9)
            if rel_err > _TPS_TOLERANCE:
                self._check_results.append(
                    err(
                        "metric-consistency-tps-per-user",
                        f"stored tps_per_user {stored:.4f} ≠ derived 1000 / tpot_p90_ms"
                        f" {derived:.4f} (rel err {rel_err:.1%})",
                        path,
                        "#9.1",
                    )
                )
                return
        self._check_results.append(
            ok(
                "metric-consistency-tps-per-user",
                f"tps_per_user={derived:.4f} tok/s/user (1000 / tpot_p90_ms {p90:.4f})",
                path,
                "#9.1",
            )
        )

    @model_validator(mode="after")
    def _check_metric_consistency(self, info: ValidationInfo) -> PointResult:
        """§14 + §9.1: validate point-log accounting invariants and tps derivability."""
        summary_path: Path | None = (info.context or {}).get("summary_path")
        s = self.summary
        self._check_duration_positive(s, summary_path)
        self._check_sample_accounting(s, summary_path)
        self._check_output_tokens_nonnegative(s, summary_path)
        self._check_system_tps_derivable(s, summary_path)
        self._check_tpot_p90(s, summary_path)
        self._check_tps_per_user(s, self.config.concurrency, summary_path)
        return self
