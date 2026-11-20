# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""The runner, submission discovery, thresholds, and the corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from submission_reviewer import review
from submission_reviewer.checks import ALL_CHECKS, Check
from submission_reviewer.context import INPUT_SUMMARIES, find_submissions
from submission_reviewer.findings import FOCUS_AREAS
from submission_reviewer.thresholds import ThresholdsError, load_thresholds

from .conftest import NEEDS_EXTRA_INPUT, clean_curve, titled, with_point, write_submission

_CORPUS = Path(__file__).resolve().parents[2] / "test_submissions"


# ── Runner ────────────────────────────────────────────────────────────────────


def test_missing_summaries_skip_the_checks_and_say_why(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", clean_curve()[:3])
    for summary in root.rglob("result_summary.json"):
        summary.unlink()
    report = review(root)
    assert report.findings == []
    statuses = {o.check: o for o in report.outcomes}
    for name in ("pareto_shape", "metric_distributions"):
        assert statuses[name].status == "skipped"
        assert "result_summary.json" in statuses[name].reason
    # The metadata checks read point.yaml and system_desc.json, which are still there.
    assert statuses["warmup"].status == "ran"
    assert statuses["system_description"].status == "ran"
    assert len(report.notes) == 3


def test_unparseable_summary_leaves_the_rest_reviewed(tmp_path: Path) -> None:
    specs = with_point(clean_curve(), 256, system_tps=5000.0)
    root = write_submission(tmp_path / "sub", specs)
    (root / "results/sys_test/llama3_1-8b/r16/result_summary.json").write_text("{not json")
    report = review(root)
    assert any("r16/result_summary.json" in n for n in report.notes)
    assert titled(report, "dips")


def test_a_failing_check_does_not_hide_the_others(tmp_path: Path) -> None:
    def boom(ctx):
        raise RuntimeError("broken heuristic")

    broken = Check("broken", "warmup", frozenset({INPUT_SUMMARIES}), boom)
    root = write_submission(tmp_path / "sub", with_point(clean_curve(), 256, system_tps=5000.0))
    report = review(root, checks=[broken, *ALL_CHECKS])
    statuses = {o.check: (o.status, o.reason) for o in report.outcomes}
    assert statuses["broken"] == ("error", "RuntimeError: broken heuristic")
    assert statuses["pareto_shape"][0] == "ran"
    assert titled(report, "dips")


def test_every_check_names_a_known_focus_area() -> None:
    keys = {area.key for area in FOCUS_AREAS}
    assert {c.focus_area for c in ALL_CHECKS} <= keys
    assert len({c.name for c in ALL_CHECKS}) == len(ALL_CHECKS)


# ── Discovery ─────────────────────────────────────────────────────────────────


def test_find_submissions_at_every_level(tmp_path: Path) -> None:
    sub_a = write_submission(tmp_path / "OrgA" / "abc", clean_curve()[:2])
    sub_b = write_submission(tmp_path / "OrgB" / "def", clean_curve()[:2])
    (tmp_path / ".git" / "x" / "results").mkdir(parents=True)

    assert find_submissions(tmp_path) == [sub_a, sub_b]
    assert find_submissions(tmp_path / "OrgA") == [sub_a]
    assert find_submissions(sub_a) == [sub_a]
    assert find_submissions(tmp_path / ".git" / "x" / "results") == []


# ── Thresholds ────────────────────────────────────────────────────────────────


def test_override_replaces_only_the_keys_it_names(tmp_path: Path) -> None:
    default = load_thresholds()
    override = tmp_path / "t.yaml"
    override.write_text("pareto:\n  dip_tolerance: 0.5\n")
    loaded = load_thresholds(override)
    assert loaded.pareto.dip_tolerance == 0.5
    assert loaded.pareto.superlinear_tolerance == default.pareto.superlinear_tolerance
    assert loaded.distributions == default.distributions


def test_override_applies_to_the_review(tmp_path: Path) -> None:
    override = tmp_path / "t.yaml"
    override.write_text("pareto:\n  dip_tolerance: 0.5\n")
    root = write_submission(tmp_path / "sub", with_point(clean_curve(), 256, system_tps=5000.0))
    assert titled(review(root), "dips")
    assert not titled(review(root, load_thresholds(override)), "dips")


@pytest.mark.parametrize(
    "text", ["pareto:\n  no_such_key: 1\n", "pareto:\n  dip_tolerance: lots\n", "- a list\n"]
)
def test_bad_override_is_rejected(tmp_path: Path, text: str) -> None:
    override = tmp_path / "t.yaml"
    override.write_text(text)
    with pytest.raises(ThresholdsError):
        load_thresholds(override)


# ── Corpus ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("submission", sorted(p.name for p in _CORPUS.iterdir() if p.is_dir()))
def test_corpus_reviews_without_errors(submission: str) -> None:
    report = review(_CORPUS / submission)
    assert {o.status for o in report.outcomes if o.check not in NEEDS_EXTRA_INPUT} == {"ran"}
    assert report.notes == []


def test_corpus_ttft_collapse_is_flagged() -> None:
    # sub_j's TTFT p50 falls from 5,158 ms at r4096 to 398 ms at r8192.
    (found,) = titled(review(_CORPUS / "sub_j"), "TTFT falls")
    assert found.point == "r8192"


def test_corpus_prose_contradiction_is_flagged() -> None:
    # sub_j declares 72 NVIDIA GB300s; its node_config says "8x H100 test node".
    (found,) = titled(review(_CORPUS / "sub_j"), "Prose contradicts")
    assert "names H100, but accelerator_info lists GB300" in found.detail


def test_corpus_system_size_mismatch_is_flagged() -> None:
    # sub_f's system_size says "1 node"; it declares two.
    (found,) = titled(review(_CORPUS / "sub_f"), "system_size disagrees")
    assert "system_node_ensemble_total is 2" in found.detail


def test_corpus_superlinear_step_is_flagged() -> None:
    # valid_standardized is synthetic: 800 → 1,500 tokens/s for concurrency 16 → 20.
    (found,) = titled(review(_CORPUS / "valid_standardized"), "faster than concurrency")
    assert found.point == "r20"
