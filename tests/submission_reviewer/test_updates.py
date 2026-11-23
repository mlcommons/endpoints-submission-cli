# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2 post-submission updates, compared through git history."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from submission_reviewer import Level, review
from submission_reviewer.cli import main
from submission_reviewer.history import GitError, changed_files, extract_base, find_original

from .conftest import MODEL, SYSTEM, checked, clean_curve, titled, write_submission

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

_CURVE = Path("results") / SYSTEM / MODEL


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Reviewer Test",
            "-c",
            "user.email=reviewer@test.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        env={**os.environ, **(env or {})},
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, message: str, at: int | None = None) -> str:
    """Commit everything; *at* sets the commit time in seconds since the epoch."""
    dates = {"GIT_AUTHOR_DATE": f"@{at} +0000", "GIT_COMMITTER_DATE": f"@{at} +0000"}
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message, env=dates if at is not None else None)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the user's and the system's git configuration out of the tests."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, Path, str]:
    """A review-PR checkout: ``ACME/abc/`` committed once as the original submission."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "submission/abc")
    root = write_submission(repo / "ACME" / "abc", clean_curve())
    return repo, root, _commit(repo, "Original submission")


def _edit_json(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _edit_yaml(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    data = yaml.safe_load(path.read_text())
    edit(data)
    path.write_text(yaml.safe_dump(data))


def _each_desc(root: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    for path in (root / _CURVE).glob("r*/system_desc.json"):
        _edit_json(path, edit)


def _updates(report):
    return checked(report, "updates")


# ── Running or not ────────────────────────────────────────────────────────────


def test_unreadable_original_files_are_reported(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "submission/abc")
    root = write_submission(repo / "ACME" / "abc", clean_curve())
    summary = root / _CURVE / "r64" / "result_summary.json"
    good = summary.read_text()
    summary.write_text("{not json")
    base = _commit(repo, "Original submission")
    summary.write_text(good)
    _commit(repo, "Repair r64")
    report = review(root, base_ref=base)
    assert any(
        n.startswith(f"In the original submission (`{base}`)") and "r64/result_summary.json" in n
        for n in report.notes
    )


def test_skipped_without_a_base_ref(repo) -> None:
    _, root, _ = repo
    (outcome,) = [o for o in review(root).outcomes if o.check == "updates"]
    assert outcome.status == "skipped"
    assert "--base-ref" in outcome.reason


def test_no_updates_no_findings(repo) -> None:
    _, root, base = repo
    report = review(root, base_ref=base)
    assert _updates(report) == []
    assert report.notes == []
    (outcome,) = [o for o in report.outcomes if o.check == "updates"]
    assert outcome.status == "ran"


def test_unknown_ref_is_a_note_and_skips(repo) -> None:
    _, root, _ = repo
    report = review(root, base_ref="no-such-ref")
    assert any("'no-such-ref' is not a commit" in n for n in report.notes)
    (outcome,) = [o for o in report.outcomes if o.check == "updates"]
    assert outcome.status == "skipped"


def test_option_like_ref_is_rejected(repo) -> None:
    _, root, _ = repo
    with pytest.raises(GitError, match="invalid revision"):
        changed_files(root, "--output=/tmp/x")


def test_submission_absent_from_the_base(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("review repo\n")
    empty = _commit(repo, "Empty review repo")
    root = write_submission(repo / "ACME" / "abc", clean_curve())
    _commit(repo, "Submission")
    report = review(root, base_ref=empty)
    assert any("holds no results" in n for n in report.notes)


# ── What changed ──────────────────────────────────────────────────────────────


def test_performance_improvement_asks_for_attention(repo) -> None:
    repo_dir, root, base = repo

    def faster(summary: dict[str, Any]) -> None:
        summary["output_sequence_lengths"]["total"] *= 1.2
        summary["system_tps"] *= 1.2

    _edit_json(root / _CURVE / "r64" / "result_summary.json", faster)
    _commit(repo_dir, "Rerun r64")
    (found,) = titled(review(root, base_ref=base), "Performance improved")
    assert found.level is Level.ATTENTION
    assert found.detail.startswith("r64 (system_tps 6,243.9 → 7,492.7 (+20.0%))")


def test_interactivity_improvement_asks_for_attention(repo) -> None:
    repo_dir, root, _ = repo
    summary = root / _CURVE / "r64" / "result_summary.json"

    def agentic(seconds: float) -> Callable[[dict[str, Any]], None]:
        def edit(body: dict[str, Any]) -> None:
            body.update(output_tokens_per_turn_total=10_000, e2e_turn_time_seconds_total=seconds)

        return edit

    _edit_json(summary, agentic(100.0))
    base = _commit(repo_dir, "Report the agentic totals")
    # Only the per-turn time changes: throughput, TPOT and TTFT stay as they were.
    _edit_json(summary, agentic(50.0))
    _commit(repo_dir, "Halve the turn time")
    (found,) = titled(review(root, base_ref=base), "Performance improved")
    assert "e2e_avg_interactivity 100.0 → 200.0 (+100.0%)" in found.detail


def test_regression_is_a_note(repo) -> None:
    repo_dir, root, base = repo

    def slower(summary: dict[str, Any]) -> None:
        summary["ttft"]["percentiles"]["90"] *= 1.5

    _edit_json(root / _CURVE / "r16" / "result_summary.json", slower)
    _commit(repo_dir, "Rerun r16")
    (found,) = titled(review(root, base_ref=base), "Performance changed")
    assert found.level is Level.NOTE
    assert "ttft_p90_ms" in found.detail and "+50.0%" in found.detail


def test_changes_below_the_threshold_are_ignored(repo) -> None:
    repo_dir, root, base = repo

    def nudge(summary: dict[str, Any]) -> None:
        summary["output_sequence_lengths"]["total"] *= 1.005

    _edit_json(root / _CURVE / "r64" / "result_summary.json", nudge)
    _commit(repo_dir, "Tiny change")
    assert titled(review(root, base_ref=base), "Performance") == []


def test_hardware_software_and_cosmetic_edits_are_graded(repo) -> None:
    repo_dir, root, base = repo

    def edit(desc: dict[str, Any]) -> None:
        desc["node_types"][0]["accelerator_info"][0]["accelerator_model_name"] = "NVIDIA H200"
        desc["node_types"][0]["driver"] = "560.10"
        desc["system_name"] = "ACME H100x8 Node rev B"

    _each_desc(root, edit)
    _commit(repo_dir, "Update system description")
    report = review(root, base_ref=base)
    (system,) = titled(report, "System changed since submission")
    (stack,) = titled(report, "Software stack or availability changed")
    (other,) = titled(report, "Other system description edits")
    assert (system.level, stack.level, other.level) == (Level.ATTENTION, Level.NOTE, Level.INFO)
    assert "accelerator_model_name`: NVIDIA H100 SXM5 80GB → NVIDIA H200" in system.detail
    assert "`node_types[0].driver`: 1.0 → 560.10" in stack.detail
    assert "`system_name`" in other.detail
    assert system.curve == SYSTEM


def test_concurrency_bounds_are_compared_on_every_curve(repo) -> None:
    repo_dir, root, _ = repo
    second = root / "results" / SYSTEM / "other-model"
    shutil.copytree(root / _CURVE, second)
    base = _commit(repo_dir, "Add a second curve on the same system")

    def raise_cmax(desc: dict[str, Any]) -> None:
        desc["max_supported_concurrency"] *= 2

    for path in second.glob("r*/system_desc.json"):
        _edit_json(path, raise_cmax)
    _commit(repo_dir, "Raise C_max on the second curve")
    (found,) = titled(review(root, base_ref=base), "Declared concurrency bounds changed")
    assert (found.level, found.curve) == (Level.ATTENTION, f"{SYSTEM}/other-model")
    assert "max_supported_concurrency" in found.detail


def test_points_added_and_removed(repo) -> None:
    repo_dir, root, base = repo
    shutil.copytree(root / _CURVE / "r1024", root / _CURVE / "r512")
    _edit_yaml(root / _CURVE / "r512" / "point.yaml", lambda c: c.update(concurrency=512))
    shutil.rmtree(root / _CURVE / "r1")
    _commit(repo_dir, "Replace r1 with r512")
    report = review(root, base_ref=base)
    added = titled(report, "Points added")
    removed = titled(report, "Points removed")
    assert [(f.level, f.evidence["added"]) for f in added] == [(Level.NOTE, "r512")]
    assert [(f.level, f.evidence["removed"]) for f in removed] == [(Level.INFO, "r1")]


def test_run_configuration_changes(repo) -> None:
    repo_dir, root, base = repo

    def reseed(config: dict[str, Any]) -> None:
        config["runtime_settings"]["runtime"]["model_seed"] = 1
        config["steady_state"]["window"]["n_samples"] = 1  # rerun noise, not reported

    def retune(desc: dict[str, Any]) -> None:
        desc["config_summary"]["tensor_parallel"] = 4
        desc["config_summary"]["data_parallel"] = 2
        desc["tps_utilization"] = 0.1  # per-point by definition, not reported

    _edit_yaml(root / _CURVE / "r64" / "point.yaml", reseed)
    _edit_json(root / _CURVE / "r256" / "system_desc.json", retune)
    _commit(repo_dir, "Reconfigure")
    (found,) = titled(review(root, base_ref=base), "Run configuration changed")
    assert "r64: `point.yaml runtime_settings.runtime.model_seed`" in found.detail
    assert "r256: `system_desc config_summary.data_parallel`: 1 → 2" in found.detail
    assert "steady_state" not in found.detail
    assert "tps_utilization" not in found.detail


def test_power_descriptor_change(repo) -> None:
    repo_dir, root, base = repo

    def edit(power: dict[str, Any]) -> None:
        power["node_sets"][0]["components"]["accelerator"]["tdp_per_unit"]["value_w"] = 600

    _edit_json(root / "results" / SYSTEM / "system_power.json", edit)
    _commit(repo_dir, "Fix TDP")
    (found,) = titled(review(root, base_ref=base), "Power descriptor changed")
    assert "tdp_per_unit.value_w`: 700.0 → 600.0" in found.detail


def test_code_changes_and_overview(repo) -> None:
    repo_dir, root, base = repo
    (root / "src" / "vllm" / "launch.sh").write_text("vllm serve --tp 8\n")
    (root / "docs" / "notes.md").write_text("More detail.\n")
    _commit(repo_dir, "Add launch script")
    report = review(root, base_ref=base)
    (overview,) = titled(report, "Files changed")
    assert "2 file(s) differ" in overview.detail
    assert "docs/ 1, src/ 1" in overview.detail
    (code,) = titled(report, "Implementation code changed")
    assert (code.level, code.detail.startswith("A src/vllm/launch.sh")) == (Level.NOTE, True)


def test_gitattributes_cannot_hide_the_original(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    root = write_submission(repo / "ACME" / "abc", clean_curve())
    (repo / ".gitattributes").write_text("*.json export-ignore\n*.yaml export-ignore\n")
    base = _commit(repo, "Original with export-ignore")

    def faster(summary: dict[str, Any]) -> None:
        summary["output_sequence_lengths"]["total"] *= 1.2

    _edit_json(root / _CURVE / "r64" / "result_summary.json", faster)
    _commit(repo, "Rerun")
    assert titled(review(root, base_ref=base), "Performance improved")


def test_extract_base_reads_only_the_checked_files(repo, tmp_path: Path) -> None:
    repo_dir, root, base = repo
    (root / "src" / "vllm" / "big.bin").write_bytes(b"\0" * 1024)
    _commit(repo_dir, "binary")
    dest = tmp_path / "extract"
    dest.mkdir()
    snapshot = extract_base(root, "HEAD", dest)
    assert snapshot.root == dest / "ACME" / "abc"
    extracted = {p.name for p in dest.rglob("*") if p.is_file()}
    assert extracted == {
        "point.yaml",
        "system_desc.json",
        "result_summary.json",
        "system_power.json",
    }
    assert snapshot.commit != base


def test_cli_base_ref(repo, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_dir, root, base = repo
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    _each_desc(root, lambda d: d.update(division="RDI"))
    _commit(repo_dir, "Change division")
    result = CliRunner().invoke(main, [str(repo_dir), "--base-ref", base])
    assert result.exit_code == 0, result.output
    assert "System changed since submission" in result.output
    assert "`division`: Standardized → RDI" in result.output


# ── Finding the original in a review PR (--pr-base) ───────────────────────────


@pytest.fixture
def review_repo(tmp_path: Path) -> tuple[Path, str]:
    """A review repo whose main branch the PRs are opened against."""
    repo = tmp_path / "review"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("Review repository\n")
    base = _commit(repo, "Initial commit")
    _git(repo, "checkout", "-q", "-b", "submission/abc12345")
    return repo, base


def test_original_is_the_add_commit(review_repo) -> None:
    repo, base = review_repo
    write_submission(repo / "ACME" / "abc", clean_curve())
    original = _commit(repo, "Add submission abc12345")
    (repo / "ACME" / "abc" / "docs" / "fix.md").write_text("fix\n")
    _commit(repo, "Update submission abc12345")
    assert find_original(repo, base) == original


def test_original_spans_batched_add_commits(review_repo) -> None:
    repo, base = review_repo
    root = write_submission(repo / "ACME" / "abc", clean_curve()[:3])
    shutil.move(str(root / "src"), str(repo / "held-src"))
    _commit(repo, "Add submission abc12345")
    shutil.move(str(repo / "held-src"), str(root / "src"))
    last_batch = _commit(repo, "Add submission abc12345")

    # A sync on a later run, under the same subject, is not part of the original.
    (root / "docs" / "new.md").write_text("new\n")
    future = int(_git(repo, "log", "-1", "--format=%ct")) + 86_400
    _commit(repo, "Add submission abc12345", at=future)
    (root / "docs" / "edit.md").write_text("web UI edit\n")
    _commit(repo, "Update README via web UI")
    assert find_original(repo, base) == last_batch


def test_original_after_a_placeholder(review_repo) -> None:
    repo, base = review_repo
    day = 1_790_000_000
    (repo / ".submission").write_text("submission_id: abc12345\n")
    _commit(repo, "Placeholder for abc12345 (source archive pending)", at=day)
    root = write_submission(repo / "ACME" / "abc", clean_curve()[:3])
    shutil.move(str(root / "src"), str(repo / "held-src"))
    _commit(repo, "Update submission abc12345", at=day + 86_400)
    shutil.move(str(repo / "held-src"), str(root / "src"))
    synced = _commit(repo, "Update submission abc12345", at=day + 86_405)  # second batch
    (root / "docs" / "later.md").write_text("later\n")
    # The next day's sync has the same subject, but is a separate push.
    _commit(repo, "Update submission abc12345", at=day + 2 * 86_400)
    assert find_original(repo, base) == synced


def test_unknown_first_commit_is_the_original(review_repo) -> None:
    repo, base = review_repo
    write_submission(repo / "ACME" / "abc", clean_curve())
    first = _commit(repo, "Hand-made submission")
    (repo / "ACME" / "abc" / "docs" / "x.md").write_text("x\n")
    _commit(repo, "Hand-made submission")
    assert find_original(repo, base) == first


def test_pr_without_commits(review_repo) -> None:
    repo, base = review_repo
    with pytest.raises(GitError, match="no commits"):
        find_original(repo, base)


def test_cli_pr_base(review_repo, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, base = review_repo
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    root = write_submission(repo / "ACME" / "abc", clean_curve())
    _commit(repo, "Add submission abc12345")

    def faster(summary: dict[str, Any]) -> None:
        summary["output_sequence_lengths"]["total"] *= 1.3

    _edit_json(root / _CURVE / "r256" / "result_summary.json", faster)
    _commit(repo, "Update submission abc12345")
    result = CliRunner().invoke(main, [str(repo), "--pr-base", base])
    assert result.exit_code == 0, result.output
    assert "Performance improved after submission" in result.output
    assert "r256 (system_tps" in result.output


def test_cli_base_ref_and_pr_base_are_exclusive(repo) -> None:
    repo_dir, _, base = repo
    result = CliRunner().invoke(main, [str(repo_dir), "--pr-base", base, "--base-ref", base])
    assert result.exit_code == 2
    assert "not both" in result.output
