# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""A lenient, read-only view of one submission for the review checks.

The §9.1 checker must report every defect, so it validates strictly. The reviewer runs
on submissions that may also fail §9.1, and a hint about the parts that did load is
still useful there. So a file that does not load is recorded as a note and its point
is left out of the checks that need it, instead of stopping the review.

File loading reuses :mod:`submission_checker.models.loader` and the tree walk reuses
:mod:`submission_checker.layout`, so the reviewer reads a submission exactly as the
checker does.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from submission_checker import layout
from submission_checker.models.file import (
    PointConfig,
    PointSummary,
    SystemDescription,
    SystemPower,
)
from submission_checker.models.file.point_config import OFFLINE_DEDICATED
from submission_checker.models.loader import (
    load_point_config,
    load_result_summary,
    load_system_description,
    load_system_power,
)

from . import history
from .logs import LogStore
from .thresholds import Thresholds

__all__ = [
    "INPUT_BASE",
    "INPUT_LOGS",
    "INPUT_POINT_CONFIGS",
    "INPUT_SUMMARIES",
    "INPUT_SYSTEM_DESCRIPTIONS",
    "Curve",
    "Point",
    "ReviewContext",
    "build_context",
    "find_submissions",
]

#: The checks' input kinds. A check names the ones it needs in ``requires``.
INPUT_SUMMARIES = "summaries"
INPUT_POINT_CONFIGS = "point_configs"
INPUT_SYSTEM_DESCRIPTIONS = "system_descriptions"
#: The original submission, read from git (``--base-ref``).
INPUT_BASE = "base"
#: At least one point's per-request log (``events.jsonl``).
INPUT_LOGS = "logs"

#: How deep below the starting path to look for ``<org>/<submission_id>/results/``.
_SEARCH_DEPTH = 2


@dataclass(frozen=True)
class Point:
    """One Pareto point (``results/<system>/<model>/r<N>/``) as far as it loaded.

    Attributes:
        dir: The point directory.
        concurrency: From ``point.yaml``, or from the ``r<N>`` name when that did not load.
        config: The parsed ``point.yaml``, or ``None``.
        summary: The parsed ``result_summary.json``, or ``None``.
        system_desc: The parsed ``system_desc.json``, or ``None``.
    """

    dir: Path
    concurrency: int
    config: PointConfig | None
    summary: PointSummary | None
    system_desc: SystemDescription | None

    @property
    def name(self) -> str:
        """The point directory's name, e.g. ``r88``."""
        return self.dir.name

    @property
    def is_dedicated_offline(self) -> bool:
        """Whether this is a dedicated Offline run (§5.7), not a fixed-concurrency point."""
        return self.config is not None and self.config.offline == OFFLINE_DEDICATED


@dataclass(frozen=True)
class Curve:
    """One Pareto frontier: one system and one benchmark model (§8.5).

    Attributes:
        label: ``<system>/<model>``, as shown in findings.
        dir: The ``results/<system>/<model>/`` directory.
        points: Every point directory, ordered by concurrency.
        system_power: The system's ``system_power.json``, or ``None`` when absent or
            unreadable. Shared by every curve of the same system.
    """

    label: str
    dir: Path
    points: tuple[Point, ...]
    system_power: SystemPower | None = None

    @property
    def system_dir(self) -> Path:
        """The ``results/<system>/`` directory."""
        return self.dir.parent

    @property
    def system_desc(self) -> SystemDescription | None:
        """The first loaded ``system_desc.json`` of the curve.

        §9.1's ``system-description-consistency`` already reports points that disagree
        on the system fields, so one description stands for the system. Per-point
        fields (parallelism, ``batch``) are read from each point instead.
        """
        return next((p.system_desc for p in self.points if p.system_desc is not None), None)

    def configs(self) -> list[tuple[Point, PointConfig]]:
        """Every point whose ``point.yaml`` loaded, with its config."""
        return [(p, p.config) for p in self.points if p.config is not None]

    def measured(self) -> list[Point]:
        """Fixed-concurrency points with a loaded summary, ordered by concurrency.

        A dedicated Offline point is left out: its concurrency is the dataset size
        (§5.7.1), so it does not belong on the load-versus-throughput curve.
        """
        return [p for p in self.points if p.summary is not None and not p.is_dedicated_offline]


@dataclass(frozen=True)
class ReviewContext:
    """Everything the checks may read about one submission.

    Attributes:
        root: The ``<submission_id>/`` directory.
        label: How the submission is named in the report.
        curves: Every Pareto frontier under ``results/``.
        thresholds: The heuristics' tolerances.
        notes: What could not be read, in the order it was met.
        inputs: The input kinds available, e.g. :data:`INPUT_SUMMARIES`.
        base: The original submission, loaded from ``base_ref``, when one was given and
            could be read. Its own ``base`` is always ``None``.
        base_ref: The revision the original was read from, as given.
        changed_files: ``(status, path)`` for every file that differs from ``base_ref``,
            path relative to :attr:`root` (see :func:`~submission_reviewer.history.changed_files`).
        logs: The per-request logs found for the points, parsed on first use.
    """

    root: Path
    label: str
    curves: tuple[Curve, ...]
    thresholds: Thresholds
    notes: tuple[str, ...]
    inputs: frozenset[str]
    base: ReviewContext | None = None
    base_ref: str | None = None
    changed_files: tuple[tuple[str, str], ...] = ()
    logs: LogStore = field(default_factory=LogStore)


def find_submissions(path: Path) -> list[Path]:
    """Return every submission directory at or below *path*.

    A submission directory holds ``results/``. *path* may be the submission itself,
    its organization directory, or a checkout holding ``<org>/<submission_id>/``, as a
    review PR does. Hidden directories such as ``.git`` are not searched.
    """
    if (path / layout.RESULTS_DIR).is_dir():
        return [path]
    found: list[Path] = []
    level = [path]
    for _ in range(_SEARCH_DEPTH):
        children = [
            child
            for parent in level
            for child in sorted(parent.iterdir())
            if child.is_dir() and not child.name.startswith(".")
        ]
        found.extend(c for c in children if (c / layout.RESULTS_DIR).is_dir())
        level = [c for c in children if not (c / layout.RESULTS_DIR).is_dir()]
    return found


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_point(point_dir: Path, root: Path, notes: list[str]) -> Point:
    def note(filename: str, what: str) -> None:
        notes.append(f"`{_rel(point_dir / filename, root)}` {what}")

    yaml_path = point_dir / layout.POINT_YAML
    config, _ = load_point_config(yaml_path, {"yaml_path": yaml_path})
    if config is None:
        note(layout.POINT_YAML, "did not load; the point's concurrency is read from its name.")

    summary_path = point_dir / layout.RESULT_SUMMARY_JSON
    summary, _ = load_result_summary(summary_path)
    if summary is None:
        note(layout.RESULT_SUMMARY_JSON, "did not load; the point is left out of metric checks.")

    desc_path = point_dir / layout.SYSTEM_DESC_JSON
    system_desc, _ = load_system_description(desc_path)
    if system_desc is None and desc_path.exists():
        note(layout.SYSTEM_DESC_JSON, "did not load.")

    concurrency = (
        config.concurrency if config is not None else layout.parse_point_dir(point_dir.name)
    )
    return Point(
        dir=point_dir,
        concurrency=concurrency or 0,
        config=config,
        summary=summary,
        system_desc=system_desc,
    )


def _load_power(system_dir: Path, root: Path, notes: list[str]) -> SystemPower | None:
    path = system_dir / layout.SYSTEM_POWER_JSON
    if not path.exists():
        return None  # optional outside Standardized; §9.1 reports a missing one
    power, _ = load_system_power(path)
    if power is None:
        notes.append(f"`{_rel(path, root)}` did not load; it is left out of hardware checks.")
    return power


def _load_base(
    root: Path, thresholds: Thresholds, base_ref: str, notes: list[str]
) -> tuple[ReviewContext | None, tuple[tuple[str, str], ...]]:
    """Load the original submission from *base_ref*, or record why it could not be."""
    try:
        with tempfile.TemporaryDirectory(prefix="submission-reviewer-") as tmp:
            snapshot = history.extract_base(root, base_ref, Path(tmp))
            # Every model is loaded eagerly, so the context outlives the directory.
            base = build_context(snapshot.root, thresholds, label=f"{base_ref}")
        changes = tuple(history.changed_files(root, base_ref))
    except history.GitError as exc:
        notes.append(f"The original submission could not be read from `{base_ref}`: {exc}.")
        return None, ()
    return base, changes


def build_context(
    root: Path,
    thresholds: Thresholds,
    label: str | None = None,
    base_ref: str | None = None,
    logs_dir: Path | None = None,
) -> ReviewContext:
    """Load the submission at *root* for review.

    Args:
        root: A ``<submission_id>/`` directory, holding ``results/``.
        thresholds: The tolerances the checks apply.
        label: How to name the submission in the report; defaults to *root*'s name.
        base_ref: A git revision holding the original submission, to compare against.
        logs_dir: A directory mirroring ``results/`` that holds the per-request logs,
            for a submission that does not ship them in its point directories.
    """
    notes: list[str] = []
    curves: list[Curve] = []
    powers: dict[Path, SystemPower | None] = {}
    for system_dir, model_dir in layout.iter_curves(root / layout.RESULTS_DIR):
        point_dirs = layout.iter_point_dirs(model_dir)
        if not point_dirs:
            continue
        if system_dir not in powers:
            powers[system_dir] = _load_power(system_dir, root, notes)
        points = tuple(_load_point(d, root, notes) for d in point_dirs)
        curves.append(
            Curve(
                label=f"{system_dir.name}/{model_dir.name}",
                dir=model_dir,
                points=points,
                system_power=powers[system_dir],
            )
        )

    all_points = [p for c in curves for p in c.points]
    inputs: set[str] = set()
    if any(p.summary is not None for p in all_points):
        inputs.add(INPUT_SUMMARIES)
    if any(p.config is not None for p in all_points):
        inputs.add(INPUT_POINT_CONFIGS)
    if any(p.system_desc is not None for p in all_points):
        inputs.add(INPUT_SYSTEM_DESCRIPTIONS)

    results = root / layout.RESULTS_DIR
    log_paths = LogStore.find([(p.dir, p.dir.relative_to(results)) for p in all_points], logs_dir)
    logs = LogStore(
        log_paths,
        keep_texts=thresholds.logs.max_text_samples,
        max_chars=thresholds.logs.max_text_chars,
    )
    if logs:
        inputs.add(INPUT_LOGS)

    base: ReviewContext | None = None
    changes: tuple[tuple[str, str], ...] = ()
    if base_ref:
        base, changes = _load_base(root, thresholds, base_ref, notes)
        if base is not None:
            inputs.add(INPUT_BASE)

    return ReviewContext(
        root=root,
        label=label or root.name,
        curves=tuple(curves),
        thresholds=thresholds,
        notes=tuple(notes),
        inputs=frozenset(inputs),
        base=base,
        base_ref=base_ref,
        changed_files=changes,
        logs=logs,
    )
