# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""The per-request log parser and the checks that read ``events.jsonl``.

``data/events_reference.jsonl`` was written by the reference client's own encoder
(``inference_endpoint.core.record.EventRecord`` through ``msgspec.json.Encoder`` with
``EventType.encode_hook``, as its ``JSONLWriter`` does, msgspec 0.20.0), so the parser
test pins the real wire format. The heuristic tests build logs of the same shape with
:func:`_record`.
"""

from __future__ import annotations

import gzip
import json
import random
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from submission_reviewer import Level, review
from submission_reviewer.checks.offline_reordering import pass_metrics
from submission_reviewer.checks.ttft_fragments import is_trivial
from submission_reviewer.cli import main
from submission_reviewer.logs import LogError, LogStore, parse_events

from .conftest import MODEL, SYSTEM, PointSpec, checked, clean_curve, titled, write_submission

_REFERENCE = Path(__file__).parent / "data" / "events_reference.jsonl"
_CURVE = Path("results") / SYSTEM / MODEL


# ── Parser, against the client's real encoding ────────────────────────────────


def test_reference_log_parses() -> None:
    log = parse_events(_REFERENCE)
    assert [(p.type, p.num_turns) for p in log.phases] == [("warmup", 2), ("performance", 2)]
    assert log.bad_lines == 0

    w1, w2 = log.issued("warmup")
    p1, p2, p3 = log.issued("performance")
    assert (w1.salted, w2.salted) == (True, False)
    assert w1.prompt_key == p1.prompt_key  # the salt is stripped before hashing
    assert log.prompt_examples[w1.prompt_key or ""] == "Summarize: the cat sat."

    assert [s.uuid for s in log.completions("performance")] == ["p2", "p1", "p3"]
    assert (p2.first_fragment, p1.first_fragment, p3.first_fragment) == ("\n", "The", None)
    assert (p2.output, p2.reasoning) == ("\nThe dog ran.", "Short.")
    assert (p1.t_first, p1.t_second, p1.t_last_chunk, p1.later_chunks) == (30, 31, 33, 2)
    assert p3.finish_reason == "tool_calls"
    # Warmup responses are never logged, so warmup samples never complete.
    assert w1.completed is None and w1.t_first == 10


def test_gzip_log(tmp_path: Path) -> None:
    gz = tmp_path / "events.jsonl.gz"
    gz.write_bytes(gzip.compress(_REFERENCE.read_bytes()))
    assert len(parse_events(gz).samples) == 5


def test_text_budget(tmp_path: Path) -> None:
    log = parse_events(_REFERENCE, keep_texts=1, max_chars=4)
    p2, p1 = log.completions("performance")[:2]
    assert (p2.output, p1.output) == ("\nThe", None)
    # The end of the response is kept whole, reasoning first.
    assert (p2.tail, p1.tail) == ("Short.\nThe dog ran.", None)


@pytest.mark.parametrize(
    "content", ["", "garbage\n" * 5, '{"foo": 1}\n' * 20], ids=["empty", "garbage", "unrelated"]
)
def test_a_log_without_sample_events_is_rejected(tmp_path: Path, content: str) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text(content)
    with pytest.raises(LogError):
        parse_events(log)


def test_not_an_events_log(tmp_path: Path) -> None:
    junk = tmp_path / "events.jsonl"
    junk.write_text("not json\n" * 20)
    with pytest.raises(LogError, match="20 of 20 lines"):
        parse_events(junk)


def test_store_records_unreadable_logs_once(tmp_path: Path) -> None:
    junk = tmp_path / "events.jsonl"
    junk.write_text("garbage\n" * 20)
    store = LogStore({tmp_path: junk})
    assert store.get(tmp_path) is None and store.get(tmp_path) is None
    assert len(store.problems) == 1


@pytest.mark.parametrize(
    ("text", "trivial"),
    [("\n", True), (" .", True), ("\x00", True), ("The", False), ("<think>", False), ("1", False)],
)
def test_is_trivial(text: str, trivial: bool) -> None:
    assert is_trivial(text) is trivial


# ── Synthetic logs ────────────────────────────────────────────────────────────


def _record(kind: str, ts: int, uuid: str = "", data: Any = None, **extra: Any) -> str:
    return json.dumps(
        {
            "event_type": kind,
            "timestamp_ns": ts,
            "sample_uuid": uuid,
            "conversation_id": "",
            "turn": None,
            "data": data,
            **extra,
        }
    )


def _prompt(text: str) -> list[Any]:
    return ["PromptData", text, None, None, None, None, None, None]


def _response(output: Any, reasoning: Any = None) -> list[Any]:
    return ["TextModelOutput", output, reasoning, None]


def _answer(i: int) -> str:
    return f"Answer {i}: " + " ".join(f"word{i}_{j}" for j in range(40))


def build_log(
    n: int = 50,
    *,
    warmup: list[str] | None = None,
    first: Any = None,
    reasoning: Any = None,
    text: Any = None,
    gap: int | None = None,
    passes: int = 1,
    completion: list[int] | None = None,
) -> list[str]:
    """A performance phase of *n* samples per pass, with optional warmup.

    *first*, *reasoning* and *text* are callables of the sample number returning the
    first fragment, the reasoning trace and the answer text. *gap* adds per-chunk events
    with that pause after the first chunk. *completion* orders the completions by
    issue position (default: issue order).
    """
    lines = []
    ts = 0
    if warmup:
        lines.append(
            _record("session.phase_start", ts, data=["PhaseData", "warmup", True, len(warmup), 0])
        )
        for i, prompt in enumerate(warmup):
            lines.append(_record("sample.issued", ts + i, f"w{i}", _prompt(prompt)))
    lines.append(_record("session.phase_start", 100, data=["PhaseData", "performance", True, n, 0]))
    lines.append(_record("session.start_performance_tracking", 100))
    total = n * passes
    for k in range(total):
        lines.append(
            _record("sample.issued", 1000 + k, f"p{k}", _prompt(f"Dataset prompt {k % n}"))
        )
    for k in completion if completion is not None else range(total):
        i = k % n
        t0 = 10_000 * (k + 1)
        lines.append(_record("sample.recv_first", t0, f"p{k}"))
        if gap is not None:
            for c in range(10):
                lines.append(_record("sample.recv_non_first", t0 + gap + c, f"p{k}"))
        head = first(i) if first else ""
        body = text(i) if text else _answer(i)
        out = [head, body] if head or first else ["", body]
        reason = reasoning(i) if reasoning else None
        lines.append(
            _record(
                "sample.complete",
                t0 + 100,
                f"p{k}",
                _response(out, ["", reason] if reason else None),
                finish_reason="stop",
            )
        )
    return lines


def _put(
    root: Path, concurrency: int, lines: list[str], extra: dict[str, Any] | None = None
) -> None:
    point = root / _CURVE / f"r{concurrency}"
    (point / "events.jsonl").write_text("\n".join(lines) + "\n")
    if extra is not None:
        (point / "sample_idx_map.json").write_text(json.dumps(extra))


def _matching_warmup(count: int):
    def edit(config: dict[str, Any], concurrency: int) -> None:
        config["warmup"].update(requests_issued=count, requests_completed=count)

    return edit


@pytest.fixture
def logged(tmp_path: Path):
    """Write the clean curve, put *lines* in the logs of the given points, review."""

    def _run(logs: dict[int, list[str]], specs: list[PointSpec] | None = None, **kwargs: Any):
        kwargs.setdefault("edit_config", _matching_warmup(0))
        root = write_submission(tmp_path / "sub", specs or clean_curve(), **kwargs)
        for concurrency, lines in logs.items():
            _put(root, concurrency, lines)
        return review(root)

    return _run


def _no_warmup(config: dict[str, Any], concurrency: int) -> None:
    config["warmup"].update(duration_s=0, requests_issued=0, requests_completed=0)


def test_clean_logs_have_no_findings(logged) -> None:
    every_point = {spec.concurrency: build_log() for spec in clean_curve()}
    report = logged(every_point, edit_config=_no_warmup)
    assert report.findings == []
    outcomes = {o.check: (o.status, o.reason) for o in report.outcomes}
    for name in ("ttft_fragments", "output_padding_text", "warmup_prompts"):
        assert outcomes[name] == ("ran", "")
    # The clean curve has no dedicated Offline point to check.
    assert outcomes["offline_reordering"] == ("skipped", "no point it applies to")


# ── TTFT fragments ────────────────────────────────────────────────────────────


def test_trivial_first_fragments(logged) -> None:
    newline_mostly = build_log(first=lambda i: "\n" if i % 10 < 6 else "Answer")
    newline_rarely = build_log(first=lambda i: " " if i % 10 == 0 else "Answer")
    report = logged({64: newline_mostly, 256: newline_rarely}, edit_config=_no_warmup)
    (found,) = checked(report, "ttft_fragments")
    assert found.level is Level.ATTENTION
    assert "r64 60% of 50, r256 10% of 50" in found.detail
    assert "`'\\n'` ×30" in found.detail


def test_offline_points_are_not_checked_for_ttft(logged) -> None:
    offline = PointSpec(5000, 9000.0, 5.0, 60.0, offline="dedicated")
    report = logged(
        {5000: build_log(first=lambda i: "\n")},
        specs=[*clean_curve(), offline],
        edit_config=_no_warmup,
    )
    assert checked(report, "ttft_fragments") == []


def test_pause_after_the_first_fragment(logged) -> None:
    report = logged({64: build_log(gap=50), 256: build_log(gap=1)}, edit_config=_no_warmup)
    (found,) = titled(report, "Long pause")
    assert found.level is Level.NOTE
    assert found.detail.startswith("r64 50.0× (50 responses)")


# ── Output padding ────────────────────────────────────────────────────────────


def test_output_length_out_of_line(run_review) -> None:
    from .conftest import summary_for, with_point

    n = summary_for(clean_curve()[4])["n_samples_completed"]
    report = run_review(with_point(clean_curve(), 256, overrides={"n_samples_completed": n // 2}))
    (found,) = titled(report, "Output length out of line")
    assert "r256 500.0 (+100%)" in found.detail


def test_token_limit_stops(run_review) -> None:
    from .conftest import with_point

    counts = {"finish_reason_counts": {"stop": 10, "length": 90}}
    (found,) = titled(run_review(with_point(clean_curve(), 64, overrides=counts)), "token limit")
    assert "r64 90%" in found.detail


def _long_answer(i: int) -> str:
    return " ".join(f"step{i}_{j}" for j in range(150))


def test_duplicated_fields(logged) -> None:
    # The visible output repeats the reasoning trace word for word.
    copied = build_log(reasoning=_long_answer, text=_long_answer)
    report = logged({64: copied}, edit_config=_no_warmup)
    (found,) = titled(report, "duplicated across response fields")
    assert found.level is Level.ATTENTION


def test_distinct_reasoning_is_fine(logged) -> None:
    thinking = build_log(reasoning=lambda i: "Let me think. " + _answer(i + 1000))
    assert checked(logged({64: thinking}, edit_config=_no_warmup), "output_padding_text") == []


def test_repetitive_responses(logged) -> None:
    loop = build_log(text=lambda i: "the quick brown fox jumps over the lazy dog " * 20)
    (found,) = titled(logged({64: loop}, edit_config=_no_warmup), "Repetitive")
    assert found.level is Level.ATTENTION


def test_trailing_filler(logged) -> None:
    padded = build_log(text=lambda i: _answer(i) + ("." * 60 if i < 5 else ""))
    (found,) = titled(logged({64: padded}, edit_config=_no_warmup), "filler")
    assert (found.level, "r64 10% of 50" in found.detail) == (Level.ATTENTION, True)


def test_filler_past_the_text_budget(logged) -> None:
    # Kept texts are cut at max_text_chars (20,000); the filler is beyond that.
    long = build_log(text=lambda i: _answer(i) * 100 + ("." * 60 if i < 5 else ""))
    (found,) = titled(logged({64: long}, edit_config=_no_warmup), "filler")
    assert "r64 10% of 50" in found.detail


def test_a_run_at_the_text_budget_is_not_filler(logged) -> None:
    # Only the truncated prefix ends in a run of one character; the response does not.
    cut = build_log(text=lambda i: "a" * 25_000 + " and then the real ending.")
    assert titled(logged({64: cut}, edit_config=_no_warmup), "filler") == []


# ── Offline reordering ────────────────────────────────────────────────────────


def _offline_specs() -> list[PointSpec]:
    return [*clean_curve(), PointSpec(5000, 15000.0, 5.0, 60.0, offline="dedicated")]


def test_reordering_within_passes_is_fine(logged) -> None:
    rng = random.Random(0)
    order = []
    for p in range(3):
        block = list(range(p * 40, (p + 1) * 40))
        rng.shuffle(block)
        order += block
    lines = build_log(40, passes=3, completion=order)
    passes, cross, same = pass_metrics(_parse(lines))
    assert (passes, cross, same) == (3, 0.0, 0.0)
    report = logged({5000: lines}, specs=_offline_specs(), edit_config=_no_warmup)
    assert checked(report, "offline_reordering") == []


def test_sorting_across_passes(logged) -> None:
    by_sample = sorted(range(120), key=lambda k: (k % 40, k // 40))
    report = logged(
        {5000: build_log(40, passes=3, completion=by_sample)},
        specs=_offline_specs(),
        edit_config=_no_warmup,
    )
    (found,) = checked(report, "offline_reordering")
    assert (found.level, found.point) == (Level.ATTENTION, "r5000")
    assert found.evidence["passes"] == 3
    assert float(found.evidence["same_sample_share"] or 0) > 0.6


def test_sample_index_map_identifies_samples(tmp_path: Path) -> None:
    by_sample = sorted(range(80), key=lambda k: (k % 40, k // 40))
    lines = build_log(40, passes=2, completion=by_sample)
    # Make every prompt unique, so only sample_idx_map.json can match the passes.
    lines = [line.replace("Dataset prompt", f"Prompt {i}") for i, line in enumerate(lines)]
    (tmp_path / "events.jsonl").write_text("\n".join(lines))
    assert pass_metrics(parse_events(tmp_path / "events.jsonl"))[2] == 0.0
    index = {"performance": {f"p{k}": k % 40 for k in range(80)}}
    (tmp_path / "sample_idx_map.json").write_text(json.dumps(index))
    # Two passes sorted by sample pair up every sample: half the neighbours match.
    assert pass_metrics(parse_events(tmp_path / "events.jsonl"))[2] > 0.4


def _parse(lines: list[str]):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        path.write_text("\n".join(lines))
        return parse_events(path)


# ── Warmup prompts ────────────────────────────────────────────────────────────


def test_warmup_reusing_performance_prompts(logged) -> None:
    warm = ["Dataset prompt 1", "Dataset prompt 2", "synthetic", "synthetic"]
    report = logged({64: build_log(warmup=warm)}, edit_config=_matching_warmup(4))
    (found,) = checked(report, "warmup_prompts")
    assert found.level is Level.ATTENTION
    assert "r64 (2 of 4 warmup requests, 2 prompts)" in found.detail
    assert "'Dataset prompt 1'" in found.detail


def test_salted_warmup_reuse_is_context(logged) -> None:
    warm = ["[0123456789abcdef] Dataset prompt 1", "[fedcba9876543210] Dataset prompt 2"]
    report = logged({64: build_log(warmup=warm)}, edit_config=_matching_warmup(2))
    (found,) = checked(report, "warmup_prompts")
    assert (found.level, found.title) == (Level.INFO, "Salted warmup reused performance prompts")


def test_declared_warmup_missing_from_the_log(logged) -> None:
    (found,) = checked(logged({64: build_log()}, edit_config=_matching_warmup(4)), "warmup_prompts")
    assert found.title == "Declared warmup is not in the log"


def test_warmup_count_differs(logged) -> None:
    report = logged({64: build_log(warmup=["a", "b"])}, edit_config=_matching_warmup(4))
    (found,) = checked(report, "warmup_prompts")
    assert "r64 (4 declared, 2 logged)" in found.detail


# ── Finding the logs ──────────────────────────────────────────────────────────


def test_logs_dir_mirror(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", clean_curve(), edit_config=_no_warmup)
    mirror = tmp_path / "logs" / SYSTEM / MODEL / "r64"
    mirror.mkdir(parents=True)
    (mirror / "events.jsonl").write_text("\n".join(build_log(first=lambda i: "\n")))
    assert checked(review(root), "ttft_fragments") == []
    assert checked(review(root, logs_dir=tmp_path / "logs"), "ttft_fragments")


def _mirror(tmp_path: Path, concurrency: int, lines: list[str]) -> Path:
    """A --logs-dir holding *lines* as the stored log of one point."""
    mirror = tmp_path / "logs" / SYSTEM / MODEL / f"r{concurrency}"
    mirror.mkdir(parents=True)
    (mirror / "events.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path / "logs"


def test_stored_logs_win_over_logs_in_the_submission(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", clean_curve(), edit_config=_no_warmup)
    _put(root, 64, build_log())  # a clean log committed in the PR
    logs = _mirror(tmp_path, 64, build_log(first=lambda i: "\n"))
    report = review(root, logs_dir=logs)
    assert checked(report, "ttft_fragments")  # found in the stored log
    assert any(
        "`" + f"{SYSTEM}/{MODEL}/r64/events.jsonl` in the submission differs" in n
        for n in report.notes
    )


def test_a_submission_log_without_a_stored_one_is_not_used(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", clean_curve(), edit_config=_no_warmup)
    _put(root, 64, build_log())
    _put(root, 256, build_log(first=lambda i: "\n"))  # only in the PR
    report = review(root, logs_dir=_mirror(tmp_path, 64, build_log()))
    assert checked(report, "ttft_fragments") == []
    assert any("r256/events.jsonl` in the submission is not used" in n for n in report.notes)
    assert "no log for r1, r4, r16, r256, r1024" in _outcome(report, "ttft_fragments").reason


def test_identical_logs_need_no_note(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", clean_curve(), edit_config=_no_warmup)
    lines = build_log()
    _put(root, 64, lines)
    report = review(root, logs_dir=_mirror(tmp_path, 64, lines))
    assert not any("events.jsonl" in n for n in report.notes)
    assert _outcome(report, "ttft_fragments").status == "ran"


def test_unreadable_log_is_a_note(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", clean_curve(), edit_config=_no_warmup)
    (root / _CURVE / "r64" / "events.jsonl").write_text("garbage\n" * 50)
    report = review(root)
    assert any("`r64`: events.jsonl" in n and "not JSON" in n for n in report.notes)


def _outcome(report, check: str):
    (outcome,) = [o for o in report.outcomes if o.check == check]
    return outcome


def test_only_unusable_logs_skip_the_log_checks(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", clean_curve(), edit_config=_no_warmup)
    (root / _CURVE / "r64" / "events.jsonl").write_text("")  # no sample events
    outcome = _outcome(review(root), "ttft_fragments")
    assert outcome.status == "skipped"
    assert "unreadable log for r64 (see the notes)" in outcome.reason
    assert "no log for r1, r4, r16, r256, r1024" in outcome.reason


def test_missing_logs_are_partial_coverage(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", clean_curve(), edit_config=_no_warmup)
    _put(root, 64, build_log())
    outcome = _outcome(review(root), "ttft_fragments")
    points = len(clean_curve())
    assert outcome.status == "ran"
    assert outcome.reason == f"1 of {points} points evaluated; no log for r1, r4, r16, r256, r1024"


def test_offline_point_without_a_log_is_not_checked(tmp_path: Path) -> None:
    root = write_submission(tmp_path / "sub", _offline_specs(), edit_config=_no_warmup)
    _put(root, 64, build_log())
    outcome = _outcome(review(root), "offline_reordering")
    assert (outcome.status, outcome.reason) == ("skipped", "no log for r5000")


def test_cli_logs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    root = write_submission(tmp_path / "ACME" / "abc", clean_curve(), edit_config=_no_warmup)
    logs = tmp_path / "logs"
    shutil.copytree(root / "results", logs, ignore=shutil.ignore_patterns("*.json", "*.yaml"))
    (logs / SYSTEM / MODEL / "r64" / "events.jsonl").write_text(
        "\n".join(build_log(first=lambda i: "\n"))
    )
    result = CliRunner().invoke(main, [str(tmp_path / "ACME"), "--logs-dir", str(logs)])
    assert result.exit_code == 0, result.output
    assert "Responses start with meaningless fragments" in result.output
