"""Point result model — cross-file validation combining point config and result summary."""

from __future__ import annotations

import math
from pathlib import Path

__all__ = ["MIN_QUERY_COUNT", "PointResult"]

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationInfo, model_validator

from ..file.point_config import OFFLINE_DEDICATED, PointConfig
from ..file.point_summary import PointSummary
from ..regions import classify_concurrency, min_duration_ms
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

        Agentic points take §6.2's agentic table (2,400 s above Ultra Low Concurrency).
        The point's own load pattern decides; a frontier whose points disagree is
        reported by ``benchmark-type-consistency``.
        """
        regions = (info.context or {}).get("regions")
        summary_path: Path | None = (info.context or {}).get("summary_path")
        if regions is None:
            return self
        c = self.config.concurrency
        region = classify_concurrency(c, regions)
        if region is None:
            return self  # already flagged by concurrency-in-range
        min_ms = min_duration_ms(region, c, agentic=self.config.is_agentic)
        duration_ms, basis = self._duration_basis()
        if duration_ms < min_ms:
            self._check_results.append(
                warn(
                    "point-duration",
                    "warn",
                    summary_path,
                    c=c,
                    region=region,
                    basis=basis,
                    duration_ms=duration_ms,
                    min_ms=min_ms,
                )
            )
        else:
            self._check_results.append(
                ok(
                    "point-duration",
                    "pass",
                    summary_path,
                    c=c,
                    basis=basis,
                    duration_ms=duration_ms,
                    region=region,
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
            self._check_results.append(warn("steady-state-basis", "warn", path, c=c))
            return self

        if block.is_official:
            self._check_results.append(
                ok("steady-state-basis", "pass", path, c=c, value=block.window.super_passes or "?")
            )
        else:
            self._check_results.append(
                warn("steady-state-basis", "warn-2", path, c=c, status=block.status)
            )

        if block.verdict in ("drifting_up", "drifting_down"):
            self._check_results.append(
                warn(
                    "steady-state-basis",
                    "drifting-metrics" if block.drifting_metrics else "drifting",
                    path,
                    c=c,
                    verdict=block.verdict,
                    metrics=", ".join(block.drifting_metrics),
                )
            )
        return self

    @model_validator(mode="after")
    def _check_min_query_count(self, info: ValidationInfo) -> PointResult:
        """§6.4: n_samples_completed must meet the dataset's minimum query count.

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
                    "fail",
                    summary_path,
                    dataset=dataset,
                    completed=completed,
                    min_queries=min_queries,
                )
            )
        else:
            self._check_results.append(
                ok(
                    "min-query-count",
                    "pass",
                    summary_path,
                    dataset=dataset,
                    completed=completed,
                    min_queries=min_queries,
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
        """Emit ok/err for the duration_ns > 0 invariant (§9.1)."""
        if s.duration_ns <= 0:
            self._check_results.append(
                err("metric-consistency-duration", "fail", path, duration_ns=s.duration_ns)
            )
        else:
            self._check_results.append(
                ok("metric-consistency-duration", "pass", path, duration_ns=s.duration_ns)
            )

    def _check_sample_accounting(self, s: PointSummary, path: Path | None) -> None:
        """Emit ok/err for the completed + failed == issued invariant (§9.1).

        Skipped when n_samples_issued is 0 (point tool did not track issued count).
        """
        if s.n_samples_issued > 0:
            accounted = s.n_samples_completed + s.n_samples_failed
            if accounted != s.n_samples_issued:
                self._check_results.append(
                    err(
                        "metric-consistency-accounting",
                        "fail",
                        path,
                        n_samples_completed=s.n_samples_completed,
                        n_samples_failed=s.n_samples_failed,
                        accounted=accounted,
                        n_samples_issued=s.n_samples_issued,
                    )
                )
            else:
                self._check_results.append(
                    ok(
                        "metric-consistency-accounting",
                        "pass",
                        path,
                        n_samples_issued=s.n_samples_issued,
                    )
                )

    def _check_output_tokens_nonnegative(self, s: PointSummary, path: Path | None) -> None:
        """Emit ok/err for the total_output_tokens >= 0 invariant (§9.1)."""
        if s.total_output_tokens < 0:
            self._check_results.append(
                err(
                    "metric-consistency-output-tokens",
                    "fail",
                    path,
                    total_output_tokens=s.total_output_tokens,
                )
            )
        else:
            self._check_results.append(
                ok(
                    "metric-consistency-output-tokens",
                    "pass",
                    path,
                    total_output_tokens=s.total_output_tokens,
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
                        "fail",
                        path,
                        stored=stored,
                        derived=derived,
                        rel_err=rel_err,
                    )
                )
                return
        self._check_results.append(
            ok(
                "metric-consistency-system-tps",
                "pass",
                path,
                derived=derived,
                total_output_tokens=s.total_output_tokens,
                elapsed_duration_seconds=s.elapsed_duration_seconds,
            )
        )

    def _check_tpot_p90(self, s: PointSummary, path: Path | None) -> None:
        """§9.1: TPOT P90 must be present, finite, and strictly positive.

        §9.1 words this as "the valid per-response TPOT distribution must be non-empty
        with a finite, strictly positive P90". A checker reading only the summary can
        never see the distribution, so what is actually verifiable is the *reported*
        percentile — the message says so rather than implying the samples were audited.

        A **dedicated** Offline run is exempt. §5.7.1: "Throughput is the only metric of
        interest" and "Latency metrics do not apply"; §6.5 exempts it from streaming
        "since it reports only ``system_tps``", so it has no TPOT to report. An elected
        point is not exempt — §5.7.2 keeps its latency metrics "defined and reported as
        for any other fixed-concurrency point".
        """
        if self.config.offline == OFFLINE_DEDICATED:
            self._check_results.append(ok("metric-consistency-tpot-p90", "exempt", path))
            return
        p90 = s.tpot_p90_ms
        if p90 is None:
            self._check_results.append(err("metric-consistency-tpot-p90", "fail", path))
            return
        if not math.isfinite(p90) or p90 <= 0:
            self._check_results.append(err("metric-consistency-tpot-p90", "fail-2", path, p90=p90))
            return
        self._check_results.append(ok("metric-consistency-tpot-p90", "pass", path, p90=p90))

    def _check_tps_per_user(self, s: PointSummary, concurrency: int, path: Path | None) -> None:
        """§9.1: ``tps_per_user = 1000 / tpot_p90_ms``.

        v0.7 defined this as ``system_tps / concurrency``; v1.0 redefines it as the
        per-user token rate implied by the P90 time per output token, which is a
        latency the user actually experiences rather than an average over the batch.

        If the result log stores a ``tps_per_user`` field, it is verified against the
        derived value within ``_TPS_TOLERANCE``.

        Not applied to a dedicated Offline run, for the reason given in
        :meth:`_check_tpot_p90`: a per-user rate is a latency metric §5.7.1 says does
        not apply to it.
        """
        if self.config.offline == OFFLINE_DEDICATED:
            return
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
                        "fail",
                        path,
                        stored=stored,
                        derived=derived,
                        rel_err=rel_err,
                    )
                )
                return
        self._check_results.append(
            ok("metric-consistency-tps-per-user", "pass", path, derived=derived, p90=p90)
        )

    def _check_agentic_metrics(self, s: PointSummary, path: Path | None) -> None:
        """§9.1: reported agentic metrics are derivable from their §4 definitions.

        Only applies where the summary reports the inputs — a single-turn benchmark
        reports neither, and §9.1's row is about agentic metrics that *are* reported.
        """
        derived = s.e2e_avg_interactivity
        stored = (s.model_extra or {}).get("e2e_avg_interactivity")
        if derived is None:
            if stored is not None:
                self._check_results.append(
                    err("agentic-metric-consistency", "fail", path, stored=stored)
                )
            return
        if stored is not None:
            rel_err = abs(float(stored) - derived) / max(abs(derived), 1e-9)
            if rel_err > _TPS_TOLERANCE:
                self._check_results.append(
                    err(
                        "agentic-metric-consistency",
                        "fail-2",
                        path,
                        stored=float(stored),
                        derived=derived,
                        rel_err=rel_err,
                    )
                )
                return
        self._check_results.append(ok("agentic-metric-consistency", "pass", path, derived=derived))

    @model_validator(mode="after")
    def _check_metric_consistency(self, info: ValidationInfo) -> PointResult:
        """§9.1: validate point-log accounting invariants and tps derivability."""
        summary_path: Path | None = (info.context or {}).get("summary_path")
        s = self.summary
        self._check_duration_positive(s, summary_path)
        self._check_sample_accounting(s, summary_path)
        self._check_output_tokens_nonnegative(s, summary_path)
        self._check_system_tps_derivable(s, summary_path)
        self._check_tpot_p90(s, summary_path)
        self._check_tps_per_user(s, self.config.concurrency, summary_path)
        self._check_agentic_metrics(s, summary_path)
        return self
