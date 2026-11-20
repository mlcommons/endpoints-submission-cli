# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: did warmup requests avoid every sample of the performance dataset?

§6.3.1 prohibits warmup on any performance-dataset sample, "direct use, subsets,
truncations, or any query whose content was derived from" one, and requires salting
whenever the client does use the performance dataset. The reference client's default
warmup *does* draw from the performance dataset, so this is a live question for most
submissions.

Without the per-request log only the §6.3.3 declaration can be read, so this check
looks at what each point's ``warmup`` block says:

- **Performance dataset named as the warmup source** without salting. When the source
  also names a different split (validation, train, ...) it is a note asking the reviewer
  to confirm the split is disjoint; otherwise it asks for attention.
- **Salted warmup on the performance dataset**, as context: §6.3.1's v0.7 allowance,
  which also requires KV-cache reuse to be disabled.
- **Retained logs that cannot be found**: ``logs_retained`` with no ``link_logs``, or a
  link no reviewer can open (§6.3.2 asks for the logs to be available for inspection).
- **Copied declarations**: every point declaring exactly the same warmup, although
  §6.3.3 asks for each point's own procedure.

Findings are grouped per curve, so a problem repeated on every point is one row.
"""

from __future__ import annotations

import re

from submission_checker.models.file import PointConfig

from ..context import INPUT_POINT_CONFIGS, Curve, Point, ReviewContext
from ..findings import Finding, Level
from ..text import normalize, url_problem
from .base import Check, finding_factory, points_label

__all__ = ["CHECK"]

_finding = finding_factory("warmup")

#: Words that name a dataset split other than the performance set itself.
_OTHER_SPLIT = re.compile(
    r"\b(validation|val|dev|train|training|calibration|accuracy|held ?out|holdout)\b"
)

_SPEC = "§9.2, §6.3.1"


def _names_performance_dataset(config: PointConfig) -> bool:
    source = normalize(config.warmup.data_source) if config.warmup else ""
    names = {normalize(n) for n in (config.dataset, config.dataset_name or "") if n}
    return any(len(name) >= 3 and name in source for name in names)


def _salted(config: PointConfig) -> bool:
    block = config.runtime_settings.warmup
    return block is not None and block.salt is True


def _sources(entries: list[tuple[Point, PointConfig]]) -> str:
    distinct = sorted({c.warmup.data_source for _, c in entries if c.warmup})
    shown = "; ".join(f"“{s}”" for s in distinct[:3])
    return shown + (f" and {len(distinct) - 3} more" if len(distinct) > 3 else "")


def _dataset_findings(curve: Curve, active: list[tuple[Point, PointConfig]]) -> list[Finding]:
    unsalted_direct: list[tuple[Point, PointConfig]] = []
    unsalted_split: list[tuple[Point, PointConfig]] = []
    salted: list[tuple[Point, PointConfig]] = []
    for point, config in active:
        if not _names_performance_dataset(config):
            continue
        assert config.warmup is not None
        if _salted(config):
            salted.append((point, config))
        elif _OTHER_SPLIT.search(normalize(config.warmup.data_source)):
            unsalted_split.append((point, config))
        else:
            unsalted_direct.append((point, config))

    out: list[Finding] = []
    if unsalted_direct:
        out.append(
            _finding(
                Level.ATTENTION,
                "Warmup declares the performance dataset",
                f"{points_label([p.name for p, _ in unsalted_direct])} declare warmup data "
                f"{_sources(unsalted_direct)}, which names the performance dataset, and "
                "warmup salting is not enabled. §6.3.1 bars performance samples from warmup "
                "unless salted; ask for the warmup logs to confirm.",
                curve=curve.label,
                spec_ref=_SPEC,
                points=",".join(p.name for p, _ in unsalted_direct),
            )
        )
    if unsalted_split:
        out.append(
            _finding(
                Level.NOTE,
                "Warmup uses a split of the performance dataset",
                f"{points_label([p.name for p, _ in unsalted_split])} declare warmup data "
                f"{_sources(unsalted_split)}. It names the performance dataset together with "
                "another split; confirm that split shares no samples with the performance set.",
                curve=curve.label,
                spec_ref=_SPEC,
                points=",".join(p.name for p, _ in unsalted_split),
            )
        )
    if salted:
        out.append(
            _finding(
                Level.INFO,
                "Salted warmup on the performance dataset",
                f"{points_label([p.name for p, _ in salted])} warm up on the performance "
                "dataset with salting enabled, which §6.3.1 allows only together with KV-cache "
                "reuse disabled.",
                curve=curve.label,
                spec_ref="§6.3.1",
                points=",".join(p.name for p, _ in salted),
            )
        )
    return out


def _log_findings(curve: Curve, active: list[tuple[Point, PointConfig]]) -> list[Finding]:
    no_link: list[str] = []
    bad_link: dict[str, list[str]] = {}
    for point, config in active:
        assert config.warmup is not None
        if config.warmup.logs_retained is True and not (config.warmup.link_logs or "").strip():
            no_link.append(point.name)
        problem = url_problem(config.warmup.link_logs)
        if problem is not None:
            bad_link.setdefault(problem, []).append(point.name)

    out: list[Finding] = []
    if no_link:
        out.append(
            _finding(
                Level.NOTE,
                "Warmup logs retained, but no link",
                f"{points_label(no_link)} declare logs_retained without link_logs, so a "
                "reviewer has no way to inspect them (§6.3.2).",
                curve=curve.label,
                spec_ref="§9.2, §6.3.2",
                points=",".join(no_link),
            )
        )
    for problem, names in bad_link.items():
        out.append(
            _finding(
                Level.NOTE,
                "Warmup log link can't be opened",
                f"link_logs {problem} at {points_label(names)}, so a reviewer cannot "
                "inspect the warmup logs (§6.3.2).",
                curve=curve.label,
                spec_ref="§9.2, §6.3.2",
                points=",".join(names),
            )
        )
    return out


def _copied(curve: Curve, active: list[tuple[Point, PointConfig]]) -> list[Finding]:
    if len(active) < 3 or len({c.concurrency for _, c in active}) < 3:
        return []
    declarations = {
        c.warmup.model_dump_json(exclude={"initialization_steps"}) for _, c in active if c.warmup
    }
    if len(declarations) != 1:
        return []
    return [
        _finding(
            Level.NOTE,
            "Identical warmup on every point",
            f"All {len(active)} points declare exactly the same warmup (duration, requests, "
            "concurrency, data source). §6.3.3 asks for each point's own procedure; confirm "
            "the values were measured per point.",
            curve=curve.label,
            spec_ref="§9.2, §6.3.3",
        )
    ]


def run(ctx: ReviewContext) -> list[Finding]:
    """Return the warmup-declaration findings for every frontier in *ctx*."""
    out: list[Finding] = []
    for curve in ctx.curves:
        active = [
            (p, c) for p, c in curve.configs() if c.warmup is not None and not c.warmup.is_disabled
        ]
        out.extend(_dataset_findings(curve, active))
        out.extend(_log_findings(curve, active))
        out.extend(_copied(curve, active))
    return out


CHECK = Check(
    name="warmup", focus_area="warmup", requires=frozenset({INPUT_POINT_CONFIGS}), run=run
)
