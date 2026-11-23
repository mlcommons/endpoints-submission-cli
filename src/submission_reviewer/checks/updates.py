# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: are post-submission updates consistent with the original system configuration?

Submission Rules §6.3 lets a submitter update a submission during review only when the
committee asks, to supply missing information or fix a material flaw, and requires
"any improvement to performance metrics" to be justified. This check compares the
current files with the original submission (``--base-ref``) and reports:

- **System changes** in ``system_desc.json``. Hardware, division and ``C_max`` changes
  ask for attention: the result would no longer describe the system first submitted.
  Software-stack and availability changes are notes; names, notes and links are info.
- **Metric changes** per point. An improvement asks for attention, because §6.3 needs
  it justified; a regression is a note.
- **Run-configuration changes** in ``point.yaml`` or the per-point deployment fields.
- **Points or curves added or removed.**
- **Power-descriptor changes**, which move the published TPS/kW.
- **Code changes** under ``src/``, and an overview of every changed file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from submission_checker.models.file import SystemDescription

from ..context import INPUT_BASE, Curve, Point, ReviewContext
from ..findings import Finding, Level
from .base import Check, PointMetrics, finding_factory, fmt, metrics, points_label

__all__ = ["CHECK"]

_finding = finding_factory("updates")
_SPEC = "§9.2, Submission Rules §6.3"

#: §8.2 fields that vary per point; compared per point, not as system changes.
_PER_POINT_DESC = frozenset(
    {
        "tps_utilization",
        "tensor_parallel",
        "pipeline_parallel",
        "expert_parallel",
        "data_parallel",
        "disaggregated",
        "batch",
        "config_summary",
        "config_summary_notes",
    }
)
#: system_desc.json fields that belong to one frontier (§5.4, §8.2), not to the
#: hardware, so they are compared on every curve rather than once per system.
_FRONTIER_DESC = frozenset({"max_supported_concurrency", "c_min_override"})
#: Changing these means the result describes a different system or curve.
_SYSTEM_IDENTITY = frozenset(
    {
        "accelerator_model_name",
        "accelerators_per_node",
        "accelerator_memory_capacity",
        "accelerator_memory_type",
        "accelerator_interconnect",
        "accelerator_host_interconnect",
        "host_processor_model_name",
        "host_processors_per_node",
        "host_processor_core_count",
        "host_processor_vcpu_count",
        "host_memory_capacity",
        "host_memory_configuration",
        "host_network_card_count",
        "host_networking",
        "host_storage_capacity",
        "host_storage_type",
        "other_hardware",
        "cooling",
        "number_of_nodes",
        "system_size",
        "system_node_ensemble_count",
        "system_node_ensemble_total",
        "division",
        "max_supported_concurrency",
        "c_min_override",
    }
)
#: Software stack and availability: plausible corrections, but worth a look.
_SYSTEM_STACK = frozenset(
    {
        "serving_framework",
        "inference_backend",
        "driver",
        "operating_system",
        "filesystem",
        "container_link",
        "other_software_stack",
        "node_config",
        "endpoint_url",
        "system_availability_status",
        "availability_url",
        "target_availability_date",
    }
)
#: point.yaml blocks rewritten by any rerun; their changes say nothing on their own.
_RERUN_NOISE = ("steady_state",)

#: Changes listed per finding before the rest are counted.
_MAX_LISTED = 8
_VALUE_CHARS = 50


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    """``{"a.b[0].c": leaf}`` for every leaf of a JSON-like value."""
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        out = {}
        for i, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def _leaf_key(path: str) -> str:
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _show(value: object) -> str:
    if value is None:
        return "(absent)"
    text = str(value)
    return text if len(text) <= _VALUE_CHARS else text[: _VALUE_CHARS - 1] + "…"


def _diff(old: Mapping[str, object], new: Mapping[str, object]) -> list[tuple[str, str]]:
    """``(path, "old → new")`` for every leaf that differs."""
    a, b = _flatten(old), _flatten(new)
    return [
        (path, f"{_show(a.get(path))} → {_show(b.get(path))}")
        for path in sorted(set(a) | set(b))
        if a.get(path) != b.get(path)
    ]


def _listing(changes: Iterable[tuple[str, str]]) -> str:
    items = [f"`{path}`: {change}" for path, change in changes]
    shown = "; ".join(items[:_MAX_LISTED])
    return shown + (f"; and {len(items) - _MAX_LISTED} more" if len(items) > _MAX_LISTED else "")


def _desc_dump(desc: SystemDescription, frontier: bool) -> dict[str, object]:
    """The fields compared once per system, or with *frontier* those of one curve."""
    if frontier:
        return desc.model_dump(mode="json", include=set(_FRONTIER_DESC))
    return desc.model_dump(mode="json", exclude=set(_PER_POINT_DESC | _FRONTIER_DESC))


def _system_changes(
    where: str, old: SystemDescription, new: SystemDescription, *, frontier: bool = False
) -> list[Finding]:
    groups: dict[Level, list[tuple[str, str]]] = {
        Level.ATTENTION: [],
        Level.NOTE: [],
        Level.INFO: [],
    }
    for path, change in _diff(_desc_dump(old, frontier), _desc_dump(new, frontier)):
        key = _leaf_key(path)
        level = (
            Level.ATTENTION
            if key in _SYSTEM_IDENTITY
            else Level.NOTE
            if key in _SYSTEM_STACK
            else Level.INFO
        )
        groups[level].append((path, change))
    titles = {
        Level.ATTENTION: (
            "System changed since submission",
            "The result would no longer describe the system first submitted; confirm the "
            "committee asked for this correction.",
        ),
        Level.NOTE: (
            "Software stack or availability changed",
            "Confirm the change corrects the disclosure rather than the system that ran.",
        ),
        Level.INFO: ("Other system description edits", ""),
    }
    if frontier:
        titles[Level.ATTENTION] = (
            "Declared concurrency bounds changed",
            "They decide the curve's regions (§5.4); confirm the committee asked for this "
            "correction.",
        )
    out = []
    for level, changes in groups.items():
        if changes:
            title, why = titles[level]
            out.append(
                _finding(
                    level,
                    title,
                    _listing(changes) + (f". {why}" if why else "."),
                    curve=where,
                    spec_ref=_SPEC,
                    count=len(changes),
                )
            )
    return out


def _metric_changes(
    old: PointMetrics, new: PointMetrics, threshold: float
) -> tuple[list[str], list[str]]:
    """Changes beyond *threshold* as (improvements, regressions), each worded."""
    better: list[str] = []
    worse: list[str] = []
    for name, a, b, higher_is_better in (
        ("system_tps", old.system_tps, new.system_tps, True),
        ("tps_per_user", old.tps_per_user, new.tps_per_user, True),
        ("ttft_p90_ms", old.ttft_p90_ms, new.ttft_p90_ms, False),
        (
            "e2e_avg_interactivity",
            old.e2e_avg_interactivity,
            new.e2e_avg_interactivity,
            True,
        ),
    ):
        if a is None or b is None or a <= 0:
            continue
        change = b / a - 1
        if abs(change) <= threshold:
            continue
        text = f"{name} {fmt(a)} → {fmt(b)} ({change:+.1%})"
        (better if (change > 0) == higher_is_better else worse).append(text)
    return better, worse


def _point_changes(curve: Curve, base: Curve, threshold: float) -> list[Finding]:
    old_points = {p.name: p for p in base.points}
    improved: list[str] = []
    regressed: list[str] = []
    reconfigured: list[str] = []
    for point in curve.points:
        before = old_points.get(point.name)
        if before is None:
            continue
        if point.summary is not None and before.summary is not None:
            better, worse = _metric_changes(metrics(before), metrics(point), threshold)
            if better:
                improved.append(f"{point.name} ({', '.join(better + worse)})")
            elif worse:
                regressed.append(f"{point.name} ({', '.join(worse)})")
        config_changes = _config_changes(before, point)
        if config_changes:
            reconfigured.append(f"{point.name}: {_listing(config_changes)}")

    out: list[Finding] = []
    if improved:
        out.append(
            _finding(
                Level.ATTENTION,
                "Performance improved after submission",
                f"{points_label(improved, limit=8)}. Submission Rules §6.3 requires any "
                "improvement to performance metrics to be justified to the review committee.",
                curve=curve.label,
                spec_ref=_SPEC,
                points=len(improved),
            )
        )
    if regressed:
        out.append(
            _finding(
                Level.NOTE,
                "Performance changed after submission",
                f"{points_label(regressed, limit=8)}.",
                curve=curve.label,
                spec_ref=_SPEC,
                points=len(regressed),
            )
        )
    if reconfigured:
        out.append(
            _finding(
                Level.NOTE,
                "Run configuration changed",
                " | ".join(reconfigured[:_MAX_LISTED])
                + (
                    f" | and {len(reconfigured) - _MAX_LISTED} more"
                    if len(reconfigured) > _MAX_LISTED
                    else ""
                )
                + ". A rerun with a different configuration is a different measurement.",
                curve=curve.label,
                spec_ref=_SPEC,
                points=len(reconfigured),
            )
        )
    return out


def _config_changes(before: Point, after: Point) -> list[tuple[str, str]]:
    changes: list[tuple[str, str]] = []
    if before.config is not None and after.config is not None:
        changes += [
            (f"point.yaml {path}", change)
            for path, change in _diff(
                before.config.model_dump(mode="json"), after.config.model_dump(mode="json")
            )
            if not path.startswith(_RERUN_NOISE)
        ]
    if before.system_desc is not None and after.system_desc is not None:
        old = before.system_desc.model_dump(mode="json", include=set(_PER_POINT_DESC))
        new = after.system_desc.model_dump(mode="json", include=set(_PER_POINT_DESC))
        changes += [
            (f"system_desc {path}", change)
            for path, change in _diff(old, new)
            if not path.startswith("tps_utilization")
        ]
    return changes


def _membership(curve_label: str, before: set[str], after: set[str], what: str) -> list[Finding]:
    out: list[Finding] = []
    added, removed = sorted(after - before), sorted(before - after)
    if added:
        out.append(
            _finding(
                Level.NOTE,
                f"{what.capitalize()} added since submission",
                f"Added: {points_label(added)}. New measurements during review need the "
                "committee's request.",
                curve=curve_label,
                spec_ref=_SPEC,
                added=",".join(added),
            )
        )
    if removed:
        out.append(
            _finding(
                Level.INFO,
                f"{what.capitalize()} removed since submission",
                f"Removed: {points_label(removed)}.",
                curve=curve_label,
                spec_ref="§9.2, Submission Rules §6.11",
                removed=",".join(removed),
            )
        )
    return out


def _power_changes(system: str, base: Curve, curve: Curve) -> list[Finding]:
    old, new = base.system_power, curve.system_power
    if old is None and new is None:
        return []
    changes = _diff(
        old.model_dump(mode="json") if old else {}, new.model_dump(mode="json") if new else {}
    )
    if not changes:
        return []
    return [
        _finding(
            Level.NOTE,
            "Power descriptor changed",
            _listing(changes) + ". This moves the published TPS/kW (§4.5).",
            curve=system,
            spec_ref="§9.2, §4.5.2",
            count=len(changes),
        )
    ]


def _file_changes(ctx: ReviewContext) -> list[Finding]:
    if not ctx.changed_files:
        return []
    by_area: dict[str, list[str]] = {}
    for status, path in ctx.changed_files:
        area = path.split("/", 1)[0] if "/" in path else "(top level)"
        by_area.setdefault(area, []).append(f"{status} {path}")
    overview = ", ".join(f"{area}/ {len(files)}" for area, files in sorted(by_area.items()))
    out = [
        _finding(
            Level.INFO,
            "Files changed since submission",
            f"{len(ctx.changed_files)} file(s) differ from `{ctx.base_ref}`: {overview}.",
            spec_ref=_SPEC,
            count=len(ctx.changed_files),
        )
    ]
    code = by_area.get("src", [])
    if code:
        out.append(
            _finding(
                Level.NOTE,
                "Implementation code changed",
                f"{points_label(code, limit=10)}. Confirm the result was produced by the "
                "updated code, or that the change only documents what ran.",
                spec_ref="§9.2, §8.1",
                count=len(code),
            )
        )
    return out


def run(ctx: ReviewContext) -> list[Finding]:
    """Compare the submission with its original (``ctx.base``)."""
    base = ctx.base
    if base is None:
        return []
    threshold = ctx.thresholds.updates.metric_change
    out: list[Finding] = []
    old_curves = {c.label: c for c in base.curves}
    new_curves = {c.label: c for c in ctx.curves}
    out.extend(_membership(ctx.label, set(old_curves), set(new_curves), "curves"))

    seen_systems: set[str] = set()
    for label, curve in new_curves.items():
        before = old_curves.get(label)
        if before is None:
            continue
        out.extend(
            _membership(
                label, {p.name for p in before.points}, {p.name for p in curve.points}, "points"
            )
        )
        out.extend(_point_changes(curve, before, threshold))
        if before.system_desc is not None and curve.system_desc is not None:
            out.extend(_system_changes(label, before.system_desc, curve.system_desc, frontier=True))
        system = curve.system_dir.name
        if system in seen_systems:
            continue
        seen_systems.add(system)
        if before.system_desc is not None and curve.system_desc is not None:
            out.extend(_system_changes(system, before.system_desc, curve.system_desc))
        out.extend(_power_changes(system, before, curve))
    out.extend(_file_changes(ctx))
    return out


CHECK = Check(name="updates", focus_area="updates", requires=frozenset({INPUT_BASE}), run=run)
