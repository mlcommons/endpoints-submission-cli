# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""The agentic accuracy gates (§3.2, §4.3).

Three quantities aggregated three ways — inline per point, SWE-bench as a mean across
points, OSL as a per-point range read from the summary — so most of what is asserted
here is that each one is gated on its own terms and not folded into the others.

Thresholds come from mlcommons/endpoints
``examples/10_Agentic_Inference/README.md`` § Accuracy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from submission_checker.agentic_targets import get_agentic_targets
from submission_checker.models import AccuracyResult, PointConfig, Severity

from .conftest import _config, _model_ctx, _summary


def _agentic_config(concurrency: int = 64) -> PointConfig:
    return _config(concurrency=concurrency, lp_type="agentic_inference")


def _accuracy(inline: float | None = None, swebench: float | None = None) -> AccuracyResult:
    root: dict[str, dict[str, object]] = {}
    if inline is not None:
        root["agentic_combined"] = {"score": inline, "num_samples": 613}
    if swebench is not None:
        root["swe_bench"] = {"score": swebench, "num_samples": 200}
    return AccuracyResult(root)


def _ctx(tmp_path: Path, model_name: str = "kimi-k3", **kwargs):
    points = kwargs.pop("valid_points", None) or [
        (tmp_path / f"p{c}.yaml", _agentic_config(c)) for c in (16, 64)
    ]
    return _model_ctx(tmp_path, model_name=model_name, valid_points=points, **kwargs)


def _hits(ctx, rule: str):
    return [r for r in ctx._check_results if r.rule == rule]


def _errors(ctx, rule: str):
    return [r for r in _hits(ctx, rule) if r.severity == Severity.ERROR]


@pytest.mark.unit
class TestTargetLookup:
    @pytest.mark.parametrize(
        ("model", "name"),
        [
            ("kimi-k3", "Kimi K3"),
            ("Kimi-K3-NVFP4", "Kimi K3"),
            ("qwen3.6-35b-a3b", "Qwen3.6-35B-A3B"),
            ("Qwen3.6-35B-A3B-FP8", "Qwen3.6-35B-A3B"),
            ("deepseek-v4.1-flash", "DSV4"),
        ],
    )
    def test_recognised(self, model: str, name: str) -> None:
        targets = get_agentic_targets(model)
        assert targets is not None and targets.name == name

    @pytest.mark.parametrize("model", ["llama3.1-8b", "gpt-oss-120b", "deepseek-r1"])
    def test_single_turn_models_are_not_agentic_targets(self, model: str) -> None:
        """deepseek-r1 must not collide with the deepseek-v4 entry."""
        assert get_agentic_targets(model) is None

    def test_dsv4_thresholds_are_unpublished(self) -> None:
        """The README records every DSV4 threshold as TBD."""
        targets = get_agentic_targets("deepseek-v4.1-flash")
        assert targets is not None and not targets.published


@pytest.mark.unit
class TestGateScope:
    def test_single_turn_curve_is_untouched(self, tmp_path: Path) -> None:
        pts = [(tmp_path / "p.yaml", _config(concurrency=64))]
        ctx = _model_ctx(tmp_path, model_name="kimi-k3", valid_points=pts)
        assert not _hits(ctx, "agentic-accuracy-inline")

    def test_unknown_agentic_model_warns(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path, model_name="some-new-model")
        hits = _hits(ctx, "agentic-accuracy")
        assert hits and hits[0].severity == Severity.WARNING

    def test_dsv4_warns_that_thresholds_are_tbd(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path, model_name="deepseek-v4.1-flash")
        hits = _hits(ctx, "agentic-accuracy")
        assert hits and hits[0].severity == Severity.WARNING
        assert "TBD" in hits[0].message
        assert not _hits(ctx, "agentic-accuracy-inline")

    def test_single_turn_gate_stands_down(self, tmp_path: Path) -> None:
        """§15's gate would fold inline and SWE-bench into one weighted mean."""
        ctx = _ctx(tmp_path, accuracy_by_point={64: _accuracy(inline=60.0, swebench=95.0)})
        assert not _hits(ctx, "accuracy-gate")


@pytest.mark.unit
class TestInlineAccuracy:
    """Per point: "Every … submitted Pareto point must satisfy all of the thresholds"."""

    def test_above_threshold_passes(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path, accuracy_by_point={16: _accuracy(inline=58.9)})
        assert not _errors(ctx, "agentic-accuracy-inline")

    def test_below_threshold_errors(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path, accuracy_by_point={16: _accuracy(inline=58.0)})
        assert _errors(ctx, "agentic-accuracy-inline")

    def test_every_point_is_gated_not_just_their_mean(self, tmp_path: Path) -> None:
        """One failing point fails, even where the mean clears — this is not mean-of-N."""
        ctx = _ctx(
            tmp_path,
            accuracy_by_point={16: _accuracy(inline=70.0), 64: _accuracy(inline=50.0)},
        )
        assert _errors(ctx, "agentic-accuracy-inline")

    def test_fractional_scores_are_rescaled(self, tmp_path: Path) -> None:
        """Scorers report 0–1; the README's thresholds are percentages."""
        ctx = _ctx(tmp_path, accuracy_by_point={16: _accuracy(inline=0.589)})
        assert not _errors(ctx, "agentic-accuracy-inline")

    def test_qwen_uses_its_own_threshold(self, tmp_path: Path) -> None:
        """56.0 clears Qwen's 55.86 and fails Kimi's 58.32."""
        for model, fails in (("qwen3.6-35b-a3b", False), ("kimi-k3", True)):
            ctx = _ctx(tmp_path, model_name=model, accuracy_by_point={16: _accuracy(inline=56.0)})
            assert bool(_errors(ctx, "agentic-accuracy-inline")) is fails

    def test_absent_warns(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path, accuracy_by_point={16: _accuracy(swebench=95.0)})
        hits = _hits(ctx, "agentic-accuracy-inline")
        assert hits and hits[0].severity == Severity.WARNING


@pytest.mark.unit
class TestSweBenchMeanOfN:
    """§4.3 multi-turn: the mean must clear; "individual results need not"."""

    def _four(self, tmp_path: Path, scores: list[float]):
        return _ctx(
            tmp_path,
            accuracy_by_point={
                c: _accuracy(swebench=s) for c, s in zip((16, 64, 256, 1024), scores, strict=True)
            },
        )

    def test_mean_clears_though_one_point_does_not(self, tmp_path: Path) -> None:
        """90 is below Kimi's 93.5; the mean of the four is 94.5, so it passes."""
        ctx = self._four(tmp_path, [90.0, 96.0, 96.0, 96.0])
        assert not _errors(ctx, "agentic-accuracy-swebench")

    def test_mean_below_threshold_errors_though_one_point_clears(self, tmp_path: Path) -> None:
        ctx = self._four(tmp_path, [96.0, 90.0, 90.0, 90.0])
        assert _errors(ctx, "agentic-accuracy-swebench")

    def test_short_set_is_gated_and_flagged(self, tmp_path: Path) -> None:
        """Three strong points must not pass unremarked — §4.3 averages four."""
        ctx = _ctx(tmp_path, accuracy_by_point={c: _accuracy(swebench=96.0) for c in (16, 64)})
        hits = _hits(ctx, "agentic-accuracy-swebench")
        assert hits and hits[0].severity == Severity.WARNING
        assert "mean of 2" in hits[0].message

    def test_absent_warns(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path, accuracy_by_point={16: _accuracy(inline=60.0)})
        hits = _hits(ctx, "agentic-accuracy-swebench")
        assert hits and hits[0].severity == Severity.WARNING


@pytest.mark.unit
class TestOslRange:
    """Read from the full-run block, explicitly not the windowed one."""

    def _ctx_with_osl(self, tmp_path: Path, avg: float | None, windowed: float | None = None):
        config = _agentic_config(64)
        summary = _summary()
        if avg is not None:
            object.__setattr__(
                summary,
                "output_sequence_lengths_full_run",
                {"output_sequence_lengths": {"avg": avg}},
            )
        if windowed is not None:
            object.__setattr__(summary.output_sequence_lengths, "avg", windowed)
        return _ctx(
            tmp_path,
            valid_points=[(tmp_path / "p.yaml", config)],
            loaded_points=[(config, summary)],
            accuracy_by_point={64: _accuracy(inline=60.0, swebench=95.0)},
        )

    def test_inside_range_passes(self, tmp_path: Path) -> None:
        ctx = self._ctx_with_osl(tmp_path, 472.0)
        assert not _errors(ctx, "agentic-osl-range")

    @pytest.mark.parametrize("avg", [424.0, 521.0])
    def test_outside_range_errors(self, tmp_path: Path, avg: float) -> None:
        assert _errors(self._ctx_with_osl(tmp_path, avg), "agentic-osl-range")

    @pytest.mark.parametrize("avg", [425.0, 520.0])
    def test_bounds_are_inclusive(self, tmp_path: Path, avg: float) -> None:
        assert not _errors(self._ctx_with_osl(tmp_path, avg), "agentic-osl-range")

    def test_windowed_value_is_not_used_as_a_fallback(self, tmp_path: Path) -> None:
        """The two blocks share a name and differ in value; gating the wrong one silently
        would be worse than reporting the field as absent."""
        ctx = self._ctx_with_osl(tmp_path, None, windowed=472.0)
        hits = _hits(ctx, "agentic-osl-range")
        assert hits and hits[0].severity == Severity.WARNING
        assert not _errors(ctx, "agentic-osl-range")
