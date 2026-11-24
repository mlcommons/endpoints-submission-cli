# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Read the reference client's per-request log (``events.jsonl``) into a compact digest.

Several §9.2 questions can only be answered from individual requests: which fragment
stopped the TTFT clock, what the responses contain, in which order Offline queries
completed, and what warmup sent. The reference client (``mlcommons/endpoints``) logs all
of it to ``events.jsonl``, one ``EventRecord`` per line, encoded by msgspec:

- ``event_type`` is ``<category>.<value>``: ``session.phase_start``,
  ``session.start_performance_tracking``, ``sample.issued``, ``sample.recv_first``,
  ``sample.recv_non_first``, ``sample.complete``, ...
- ``data`` is a tagged array with trailing defaults omitted:
  ``["PhaseData", phase_type, drain_after, num_turns, num_trajectories]``,
  ``["PromptData", text, token_ids, messages, ...]``,
  ``["TextModelOutput", output, reasoning, tool_calls]``.
- A streamed ``output`` or ``reasoning`` is an array whose entry 0 is the first delta's
  contribution and whose remaining entries are the tail. The first delta is what
  stopped the TTFT clock (``openai/accumulator.py``), and it carries reasoning before
  content.
- Warmup samples are issued and logged, but their responses are not.

The logs run to hundreds of megabytes, so they are streamed line by line and reduced
to a :class:`PointLog` digest: per-sample timings, a prompt hash and the first fragment
for every sample, and the full text for a bounded number of responses only. A point's
log is parsed at most once, on first use, and shared by every check.
"""

from __future__ import annotations

import filecmp
import gzip
import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from .text import quote

__all__ = [
    "LOG_NAMES",
    "SAMPLE_INDEX_MAP",
    "TAIL_CHARS",
    "LogError",
    "LogStore",
    "Phase",
    "PointLog",
    "SampleRecord",
    "parse_events",
]

#: File names the log may have, in lookup order.
LOG_NAMES = ("events.jsonl", "events.jsonl.gz")
#: The client's ``{phase_name: {sample_uuid: dataset_index}}`` map, beside the log.
SAMPLE_INDEX_MAP = "sample_idx_map.json"
#: How much of each kept response's end is kept untruncated, for the checks that look
#: at how a response ends.
TAIL_CHARS = 1000

#: §6.3.1 salting: the client prepends ``[<16 hex digits>] `` to a salted prompt.
SALT_RE = re.compile(r"^\[[0-9a-f]{16}\] ")

#: More unreadable lines than this fraction means the file is not an events log.
_MAX_BAD_FRACTION = 0.01


class LogError(ValueError):
    """The file cannot be read as an events log."""


@dataclass(frozen=True)
class Phase:
    """One ``session.phase_start`` announcement.

    Attributes:
        type: ``performance``, ``accuracy`` or ``warmup``.
        num_turns: Samples in one pass over the dataset.
    """

    type: str
    num_turns: int


@dataclass
class SampleRecord:
    """What the log says about one sample.

    Timestamps are the client's monotonic nanoseconds; only differences mean anything.

    Attributes:
        uuid: The sample's ID.
        phase: Index into :attr:`PointLog.phases`; -1 before any phase started.
        order: Issue order within its phase, from 0.
        prompt_key: Hash of the prompt with any salt stripped; equal prompts share it.
        salted: Whether the prompt carried a §6.3.1 salt.
        t_issued: When it was issued.
        t_first: When the first fragment arrived.
        t_second: When the second streamed chunk arrived (needs ``stream_all_chunks``).
        t_last_chunk: When the last streamed chunk arrived.
        later_chunks: How many chunks arrived after the first.
        completed: Completion order within the log, from 0; ``None`` if never completed.
        first_fragment: The text that stopped the TTFT clock; ``None`` when the
            response was not streamed, or started with a tool call.
        output: The visible output, when its text was kept.
        reasoning: The reasoning trace, when its text was kept.
        tail: The last :data:`TAIL_CHARS` characters of the whole response, reasoning
            then output, before any truncation; kept with the texts.
        finish_reason: As reported by the endpoint.
    """

    uuid: str
    phase: int
    order: int
    prompt_key: str | None = None
    salted: bool = False
    t_issued: int | None = None
    t_first: int | None = None
    t_second: int | None = None
    t_last_chunk: int | None = None
    later_chunks: int = 0
    completed: int | None = None
    first_fragment: str | None = None
    output: str | None = None
    reasoning: str | None = None
    tail: str | None = None
    finish_reason: str | None = None


@dataclass
class PointLog:
    """The digest of one point's ``events.jsonl``.

    Attributes:
        path: The file read.
        phases: Every phase, in the order they started.
        samples: Every sample, by UUID.
        prompt_examples: The first 120 characters of a prompt, by prompt key, for
            warmup prompts (to show what overlapped).
        dataset_index: ``sample_idx_map.json``'s performance map, when present.
        lines: Lines read.
        bad_lines: Lines that were not a JSON object.
    """

    path: Path
    phases: list[Phase] = field(default_factory=list)
    samples: dict[str, SampleRecord] = field(default_factory=dict)
    prompt_examples: dict[str, str] = field(default_factory=dict)
    dataset_index: dict[str, int] = field(default_factory=dict)
    lines: int = 0
    bad_lines: int = 0

    def phase_of(self, kind: str) -> list[int]:
        """Indexes of the phases of *kind*."""
        return [i for i, p in enumerate(self.phases) if p.type == kind]

    def issued(self, kind: str) -> list[SampleRecord]:
        """Samples of every phase of *kind*, in issue order."""
        wanted = set(self.phase_of(kind))
        return sorted(
            (s for s in self.samples.values() if s.phase in wanted),
            key=lambda s: (s.phase, s.order),
        )

    def completions(self, kind: str) -> list[SampleRecord]:
        """Completed samples of every phase of *kind*, in completion order."""
        done = [s for s in self.issued(kind) if s.completed is not None]
        return sorted(done, key=lambda s: s.completed or 0)


def _open(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _fields(data: Any, tag: str) -> list[Any] | None:
    """The fields of a msgspec tagged array, or ``None`` when *data* is not one."""
    if isinstance(data, list) and data and data[0] == tag:
        return data[1:]
    return None


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(v for v in value if isinstance(v, str))
    return ""


def _prompt_key(fields: list[Any]) -> tuple[str | None, bool, str]:
    """``(key, salted, example)`` for a PromptData payload."""
    text = fields[0] if len(fields) > 0 else None
    token_ids = fields[1] if len(fields) > 1 else None
    messages = fields[2] if len(fields) > 2 else None
    if isinstance(text, str):
        salted = SALT_RE.match(text) is not None
        plain = SALT_RE.sub("", text, count=1)
        return hashlib.sha1(plain.encode()).hexdigest(), salted, plain[:120]
    for value in (token_ids, messages):
        if value is not None:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
            return hashlib.sha1(encoded.encode()).hexdigest(), False, encoded[:120]
    return None, False, ""


def _first_fragment(output: Any, reasoning: Any) -> str | None:
    """The first delta's text, or ``None`` when neither field was streamed."""
    streamed = [f for f in (reasoning, output) if isinstance(f, list)]
    if not streamed:
        return None
    first = "".join(f[0] for f in streamed if f and isinstance(f[0], str))
    return first or None  # empty: the first delta was a tool call


def _records(handle: IO[str], log: PointLog) -> Iterator[dict[str, Any]]:
    for line in handle:
        log.lines += 1
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            log.bad_lines += 1
            continue
        if isinstance(record, dict):
            yield record
        else:
            log.bad_lines += 1


def parse_events(path: Path, keep_texts: int = 2000, max_chars: int = 20_000) -> PointLog:
    """Read *path* into a :class:`PointLog`.

    Args:
        path: An ``events.jsonl`` file, optionally gzip-compressed.
        keep_texts: Keep the full text of this many performance responses, the first
            to complete; the rest keep only their first fragment.
        max_chars: Truncate each kept text to this many characters. The end of the
            response is kept separately, in :attr:`SampleRecord.tail`.

    Raises:
        LogError: If the file cannot be read, or too much of it is not JSON.
    """
    log = PointLog(path=path)
    phase = -1
    performance: set[int] = set()
    issued_in_phase: dict[int, int] = {}
    completed = 0
    kept = 0
    try:
        with _open(path) as handle:
            for record in _records(handle, log):
                kind = record.get("event_type")
                data = record.get("data")
                ts = record.get("timestamp_ns")
                ts = ts if isinstance(ts, int) else None
                uuid = record.get("sample_uuid") or ""

                if kind == "session.phase_start":
                    fields = _fields(data, "PhaseData") or []
                    phase_type = str(fields[0]) if fields else "unknown"
                    turns = fields[2] if len(fields) > 2 and isinstance(fields[2], int) else 0
                    log.phases.append(Phase(type=phase_type, num_turns=turns))
                    phase = len(log.phases) - 1
                    if phase_type == "performance":
                        performance.add(phase)
                    continue
                if not uuid:
                    continue

                if kind == "sample.issued":
                    order = issued_in_phase.get(phase, 0)
                    issued_in_phase[phase] = order + 1
                    new = SampleRecord(uuid=uuid, phase=phase, order=order, t_issued=ts)
                    prompt = _fields(data, "PromptData")
                    if prompt is not None:
                        key, salted, example = _prompt_key(prompt)
                        new.prompt_key, new.salted = key, salted
                        if key and phase >= 0 and log.phases[phase].type == "warmup":
                            log.prompt_examples.setdefault(key, example)
                    log.samples[uuid] = new
                    continue

                sample = log.samples.get(uuid)
                if sample is None:
                    continue
                if kind == "sample.recv_first":
                    sample.t_first = ts
                elif kind == "sample.recv_non_first":
                    sample.later_chunks += 1
                    if sample.t_second is None:
                        sample.t_second = ts
                    sample.t_last_chunk = ts
                elif kind == "sample.complete":
                    sample.completed = completed
                    completed += 1
                    finish = record.get("finish_reason")
                    sample.finish_reason = finish if isinstance(finish, str) else None
                    out = _fields(data, "TextModelOutput")
                    if out is None:
                        continue
                    output = out[0] if len(out) > 0 else ""
                    reasoning = out[1] if len(out) > 1 else None
                    sample.first_fragment = _first_fragment(output, reasoning)
                    if sample.phase in performance and kept < keep_texts:
                        full_output, full_reasoning = _text(output), _text(reasoning)
                        sample.output = full_output[:max_chars]
                        sample.reasoning = full_reasoning[:max_chars] or None
                        sample.tail = (full_reasoning[-TAIL_CHARS:] + full_output)[-TAIL_CHARS:]
                        kept += 1
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise LogError(f"{path.name} could not be read: {exc}") from exc

    if log.lines and log.bad_lines > max(10, _MAX_BAD_FRACTION * log.lines):
        raise LogError(f"{path.name}: {log.bad_lines} of {log.lines} lines are not JSON objects")
    if not log.samples:
        # An empty file or unrelated JSON would otherwise pass as a log with nothing odd.
        raise LogError(f"{path.name} has no sample events; it is not an events log")
    log.dataset_index = _read_index_map(path.parent / SAMPLE_INDEX_MAP)
    return log


def _read_index_map(path: Path) -> dict[str, int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    performance = data.get("performance") if isinstance(data, dict) else None
    if not isinstance(performance, dict):
        return {}
    return {k: v for k, v in performance.items() if isinstance(v, int)}


class LogStore:
    """Finds and parses each point's events log once, on first use.

    Args:
        paths: The log file for each point directory that has one.
        keep_texts: See :func:`parse_events`.
        max_chars: See :func:`parse_events`.
    """

    def __init__(
        self, paths: dict[Path, Path] | None = None, keep_texts: int = 2000, max_chars: int = 20_000
    ) -> None:
        self.paths = dict(paths or {})
        self.keep_texts = keep_texts
        self.max_chars = max_chars
        self.problems: list[str] = []
        #: Point directories whose log was asked for since the last :meth:`track`,
        #: whether or not one was found.
        self.requested: set[Path] = set()
        self._cache: dict[Path, PointLog | None] = {}

    def __bool__(self) -> bool:
        return bool(self.paths)

    def has(self, point_dir: Path) -> bool:
        """Whether a log was found for *point_dir*."""
        return point_dir in self.paths

    def get(self, point_dir: Path) -> PointLog | None:
        """The parsed log of *point_dir*, or ``None`` if it has none or it is unreadable.

        An unreadable log is recorded in :attr:`problems` once.
        """
        self.requested.add(point_dir)
        if point_dir not in self.paths:
            return None
        if point_dir not in self._cache:
            try:
                self._cache[point_dir] = parse_events(
                    self.paths[point_dir], self.keep_texts, self.max_chars
                )
            except LogError as exc:
                self.problems.append(f"`{point_dir.name}`: {exc}; its log checks were skipped.")
                self._cache[point_dir] = None
        return self._cache[point_dir]

    def track(self) -> None:
        """Start recording, in :attr:`requested`, which logs are asked for."""
        self.requested = set()

    def missing(self, point_dirs: set[Path]) -> set[Path]:
        """Those of *point_dirs* for which no log was found."""
        return {d for d in point_dirs if d not in self.paths}

    def unreadable(self, point_dirs: set[Path]) -> set[Path]:
        """Those of *point_dirs* whose log was found but could not be parsed."""
        return {d for d in point_dirs if d in self._cache and self._cache[d] is None}

    @staticmethod
    def find(
        point_dirs: list[tuple[Path, Path]], logs_dir: Path | None
    ) -> tuple[dict[Path, Path], list[str]]:
        """Locate the log of each point.

        Without *logs_dir*, a point's log is the one in its own directory. With it, the
        logs come from *logs_dir* only: those are the stored run archives, while a log
        committed in the PR is whatever the submitter put there, and must not stand in
        for the stored one. A committed log that is not used is reported in a note,
        unless it is identical to the stored one.

        Args:
            point_dirs: ``(point_dir, path relative to results/)`` pairs.
            logs_dir: A directory mirroring ``results/`` (``<system>/<model>/r<N>/``)
                that holds the logs, for submissions that do not ship them.

        Returns:
            The log of each point that has one, and the notes.
        """
        found: dict[Path, Path] = {}
        notes: list[str] = []
        for point_dir, rel in point_dirs:
            shipped = _log_in(point_dir)
            if logs_dir is None:
                if shipped is not None:
                    found[point_dir] = shipped
                continue
            stored = _log_in(logs_dir / rel)
            if stored is not None:
                found[point_dir] = stored
            if shipped is None:
                continue
            where = quote(f"{rel.as_posix()}/{shipped.name}")
            if stored is None:
                notes.append(
                    f"{where} in the submission is not used: --logs-dir has no log for this "
                    "point, and only --logs-dir logs are read when it is given."
                )
            elif shipped.name != stored.name or not filecmp.cmp(shipped, stored, shallow=False):
                notes.append(
                    f"{where} in the submission differs from the log in --logs-dir; the "
                    "--logs-dir log is the one checked."
                )
        return found, notes


def _log_in(directory: Path) -> Path | None:
    """The events log in *directory*, if there is one."""
    return next((directory / n for n in LOG_NAMES if (directory / n).is_file()), None)
