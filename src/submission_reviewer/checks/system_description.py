# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2: does the system description accurately reflect the hardware and software used?

Accuracy needs a person who knows the hardware; what a tool can find is the places
where the description disagrees with itself or with the rest of the submission:

- **Placeholders**: ``TBD``, ``N/A`` or empty values in fields a reproducer needs, and
  links to placeholder or private hosts.
- **Prose that contradicts the structured fields**: an accelerator named in
  ``node_config`` or the notes that ``accelerator_info`` does not list, or a count such
  as ``8x H100`` that matches no node type.
- **Framework without code**: a ``serving_framework`` with no matching ``src/<impl>/``.
- **Parallelism beyond the hardware**: ``TP × PP × DP`` larger than the accelerators
  the system declares, or a ``config_summary`` text naming different degrees.
- **``system_size``** naming accelerator or node counts the node types do not add up to.
- **``system_power.json``** describing different accelerators, counts or nodes than
  ``system_desc.json``. §9.1 already reconciles cooling and node-set IDs.

Each system is reported once, however many model curves it has.
"""

from __future__ import annotations

import math
import re

from submission_checker.models.file import SystemDescription, SystemPower

from ..context import INPUT_SYSTEM_DESCRIPTIONS, Curve, ReviewContext
from ..findings import Finding, Level
from ..text import (
    accelerator_counts,
    accelerator_families,
    is_placeholder,
    name_key,
    url_problem,
)
from .base import Check, finding_factory, points_label

__all__ = ["CHECK"]

_finding = finding_factory("system_description")
_SPEC = "§9.2, §8.2"

#: Fields a reproducer needs; empty or placeholder values are reported.
_REQUIRED_TOP = ("system_name", "serving_framework", "node_config")
_REQUIRED_NODE = (
    "host_processor_model_name",
    "host_memory_capacity",
    "driver",
    "operating_system",
    "container_link",
)
_REQUIRED_ACCELERATOR = ("accelerator_model_name", "accelerator_memory_capacity")
#: Free-form fields: empty is fine, a placeholder is not.
_OPTIONAL_TEXT = ("hw_notes", "sw_notes", "other_hardware", "other_software_stack")
#: Prose scanned for accelerator names that contradict ``accelerator_info``.
_PROSE_TOP = ("node_config", "config_summary_notes")
_PROSE_NODE = ("hw_notes",)

#: Framework spellings that name the same project.
_FRAMEWORK_ALIASES = {"trtllm": "tensorrtllm", "tensorrt": "tensorrtllm"}

_DEGREE_RE = re.compile(r"\b(TP|PP|EP|DP)\s*[=:]?\s*(\d+)\b", re.IGNORECASE)
_SIZE_ACCEL_RE = re.compile(
    r"(\d+)\s*(?:x\s*)?(?:accelerators?|gpus?|chips?|npus?|tpus?|xpus?|hpus?)\b", re.IGNORECASE
)
_SIZE_NODES_RE = re.compile(r"(\d+)\s*nodes?\b", re.IGNORECASE)


def _accelerators_per_node(desc: SystemDescription) -> dict[int, int]:
    """Accelerators per node, by ``system_node_ensemble_id`` (index when absent)."""
    out: dict[int, int] = {}
    for index, node in enumerate(desc.node_types):
        counts = [a.accelerators_per_node for a in node.accelerator_info]
        if counts and all(c is not None for c in counts):
            key = (
                node.system_node_ensemble_id if node.system_node_ensemble_id is not None else index
            )
            out[key] = sum(c for c in counts if c is not None)
    return out


def _node_counts(desc: SystemDescription) -> list[int] | None:
    """How many nodes of each node type, in ``node_types`` order, or ``None`` if unknown.

    ``number_of_nodes`` is optional. One missing count is the remainder of
    ``system_node_ensemble_total`` (the whole total for a single node type); with more
    missing, or no usable total, the counts are unknown rather than guessed.
    """
    counts = [
        n.number_of_nodes if n.number_of_nodes is not None and n.number_of_nodes > 0 else None
        for n in desc.node_types
    ]
    missing = [i for i, c in enumerate(counts) if c is None]
    if not missing:
        return [c for c in counts if c is not None]
    total = desc.system_node_ensemble_total
    if len(missing) > 1 or not total or total <= 0:
        return None
    remainder = total - sum(c for c in counts if c is not None)
    if remainder <= 0:
        return None
    counts[missing[0]] = remainder
    return [c for c in counts if c is not None]


def _total_accelerators(desc: SystemDescription) -> int | None:
    per_node = _accelerators_per_node(desc)
    nodes = _node_counts(desc)
    if len(per_node) != len(desc.node_types) or nodes is None:
        return None
    total = 0
    for index, node in enumerate(desc.node_types):
        key = node.system_node_ensemble_id if node.system_node_ensemble_id is not None else index
        total += per_node[key] * nodes[index]
    return total


def _family_counts(desc: SystemDescription) -> dict[str, tuple[set[int], int | None]]:
    """For each accelerator family, its counts per node and its system total.

    The counts per node are those of every node type that has the family; the total is
    ``None`` when the node counts are unknown. A family with any unknown
    ``accelerators_per_node`` is left out, so its prose is not compared.
    """
    nodes = _node_counts(desc)
    per_node: dict[str, list[int]] = {}
    totals: dict[str, int] = {}
    unknown: set[str] = set()
    for index, node in enumerate(desc.node_types):
        in_node: dict[str, int] = {}
        for acc in node.accelerator_info:
            for family in accelerator_families(acc.accelerator_model_name):
                if acc.accelerators_per_node is None:
                    unknown.add(family)
                else:
                    in_node[family] = in_node.get(family, 0) + acc.accelerators_per_node
        for family, count in in_node.items():
            per_node.setdefault(family, []).append(count)
            if nodes is not None:
                totals[family] = totals.get(family, 0) + count * nodes[index]
    return {
        family: (set(counts), totals.get(family))
        for family, counts in per_node.items()
        if family not in unknown
    }


def _count_fits(count: int, per_node: set[int], total: int | None) -> bool:
    """Whether prose stating *count* accelerators of a family can be right."""
    if count in per_node or count == total:
        return True
    # Without node counts the total is unknown; any whole number of nodes may be meant.
    return total is None and any(count % c == 0 for c in per_node if c > 0)


def _declared_families(desc: SystemDescription) -> set[str]:
    return {
        family
        for node in desc.node_types
        for acc in node.accelerator_info
        for family in accelerator_families(acc.accelerator_model_name)
    }


def _placeholders(label: str, desc: SystemDescription) -> list[Finding]:
    problems: list[str] = []
    for name in _REQUIRED_TOP:
        value = getattr(desc, name, None)
        if value is None or not str(value).strip() or is_placeholder(value):
            problems.append(f"`{name}` is {_describe(value)}")
    for i, node in enumerate(desc.node_types):
        where = f"node_types[{i}]" if len(desc.node_types) > 1 else "node"
        for name in _REQUIRED_NODE:
            value = getattr(node, name, None)
            if value is None or not str(value).strip() or is_placeholder(value):
                problems.append(f"{where} `{name}` is {_describe(value)}")
        for name in _OPTIONAL_TEXT:
            value = getattr(node, name, None)
            if is_placeholder(value):
                problems.append(f"{where} `{name}` is {_describe(value)}")
        for j, acc in enumerate(node.accelerator_info):
            for name in _REQUIRED_ACCELERATOR:
                value = getattr(acc, name, None)
                if value is None or not str(value).strip() or is_placeholder(value):
                    problems.append(f"{where} accelerator_info[{j}] `{name}` is {_describe(value)}")
    if not problems:
        return []
    return [
        _finding(
            Level.NOTE,
            "Missing or placeholder details",
            "; ".join(problems) + ". §8.2 asks for enough detail to reproduce the system.",
            curve=label,
            spec_ref=_SPEC,
            count=len(problems),
        )
    ]


def _describe(value: object) -> str:
    if value is None:
        return "missing"
    if not str(value).strip():
        return "empty"
    return f"“{value}”"


def _links(label: str, desc: SystemDescription, curve: Curve) -> list[Finding]:
    problems: list[str] = []
    for node in desc.node_types:
        problem = url_problem(node.container_link)
        if problem:
            problems.append(f"container_link ({node.container_link}) {problem}")
    config = next((c for _, c in curve.configs()), None)
    if config is not None:
        for name in ("link_to_model", "link_to_model_transformation", "dataset_link"):
            value = getattr(config, name, None)
            problem = url_problem(value)
            if problem:
                problems.append(f"point.yaml {name} ({value}) {problem}")
    if not problems:
        return []
    return [
        _finding(
            Level.NOTE,
            "Links a reviewer can't open",
            "; ".join(problems) + ".",
            curve=label,
            spec_ref="§9.2, §2.7",
            count=len(problems),
        )
    ]


def _prose(label: str, desc: SystemDescription) -> list[Finding]:
    declared = _declared_families(desc)
    if not declared:
        return []
    allowed = _family_counts(desc)
    texts: list[tuple[str, str]] = [
        (name, getattr(desc, name)) for name in _PROSE_TOP if isinstance(getattr(desc, name), str)
    ]
    for node in desc.node_types:
        texts += [(n, getattr(node, n)) for n in _PROSE_NODE if isinstance(getattr(node, n), str)]

    problems: list[str] = []
    for field, text in texts:
        foreign = accelerator_families(text) - declared
        if foreign:
            problems.append(
                f"`{field}` (“{text}”) names {', '.join(sorted(foreign))}, but accelerator_info "
                f"lists {', '.join(sorted(declared))}"
            )
            continue
        for count, family in accelerator_counts(text):
            if family not in allowed:
                continue
            per_node, total = allowed[family]
            if not _count_fits(count, per_node, total):
                expected = sorted(per_node | ({total} if total else set()))
                problems.append(
                    f"`{field}` (“{text}”) says {count}× {family}, but the node types declare "
                    f"{' or '.join(str(e) for e in expected)} {family}"
                )
    if not problems:
        return []
    return [
        _finding(
            Level.ATTENTION,
            "Prose contradicts the hardware fields",
            "; ".join(problems) + ". One of them does not describe the system that ran.",
            curve=label,
            spec_ref=_SPEC,
        )
    ]


def _framework(label: str, desc: SystemDescription, ctx: ReviewContext) -> list[Finding]:
    framework = desc.serving_framework
    src = ctx.root / "src"
    if not framework or is_placeholder(framework) or not src.is_dir():
        return []
    impls = sorted(d.name for d in src.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not impls:
        return []  # §9.1's src-readme reports an empty src/
    key = name_key(framework)
    key = _FRAMEWORK_ALIASES.get(key, key)
    for impl in impls:
        impl_key = _FRAMEWORK_ALIASES.get(name_key(impl), name_key(impl))
        if len(key) >= 3 and len(impl_key) >= 3 and (key in impl_key or impl_key in key):
            return []
    return [
        _finding(
            Level.NOTE,
            "Serving framework has no matching code",
            f"serving_framework is “{framework}”, but src/ holds {', '.join(impls)}. "
            "Confirm the shipped implementation is the one that ran.",
            curve=label,
            spec_ref="§9.2, §8.1",
        )
    ]


def _degree(desc: SystemDescription, name: str) -> int | None:
    value = getattr(desc, name, None)
    summary = desc.config_summary
    if value is None and summary is not None and not isinstance(summary, str):
        value = getattr(summary, name, None)
    return value if isinstance(value, int) else None


def _disaggregated(desc: SystemDescription) -> bool:
    summary = desc.config_summary
    flag = desc.disaggregated
    if flag is None and summary is not None and not isinstance(summary, str):
        flag = summary.disaggregated
    return bool(flag)


def _parallelism(label: str, curve: Curve) -> list[Finding]:
    oversized: list[str] = []
    text_mismatch: list[str] = []
    for point in curve.points:
        desc = point.system_desc
        if desc is None:
            continue
        degrees = {
            "TP": _degree(desc, "tensor_parallel"),
            "PP": _degree(desc, "pipeline_parallel"),
            "EP": _degree(desc, "expert_parallel"),
            "DP": _degree(desc, "data_parallel"),
        }
        total = _total_accelerators(desc)
        known = [d for d in (degrees["TP"], degrees["PP"], degrees["DP"]) if d]
        if known and total and not _disaggregated(desc):
            engaged = math.prod(known)
            if engaged > total:
                oversized.append(f"{point.name} (TP×PP×DP = {engaged} on {total})")
        if isinstance(desc.config_summary, str):
            for name, value in _DEGREE_RE.findall(desc.config_summary):
                declared = degrees.get(name.upper())
                if declared is not None and declared != int(value):
                    text_mismatch.append(
                        f"{point.name} ({name.upper()}{value} in text, {declared} in fields)"
                    )

    out: list[Finding] = []
    if oversized:
        out.append(
            _finding(
                Level.ATTENTION,
                "Parallelism exceeds the declared accelerators",
                f"{points_label(oversized)}. The deployment needs more accelerators than "
                "the system declares.",
                curve=label,
                spec_ref="§9.2, §8.2, §4.5.3",
            )
        )
    if text_mismatch:
        out.append(
            _finding(
                Level.NOTE,
                "config_summary disagrees with the parallelism fields",
                f"{points_label(text_mismatch)}.",
                curve=label,
                spec_ref=_SPEC,
            )
        )
    return out


def _system_size(label: str, desc: SystemDescription) -> list[Finding]:
    size = desc.system_size or ""
    problems: list[str] = []
    stated = sorted(int(n) for n in _SIZE_ACCEL_RE.findall(size))
    per_type = _accelerators_per_node(desc)
    total = _total_accelerators(desc)
    if stated and per_type:
        options = [sorted(per_type.values())]
        if total:
            options.append([total])
        if stated not in options:
            problems.append(
                f"names {' + '.join(map(str, stated))} accelerators, but the node types declare "
                f"{' + '.join(str(v) for v in sorted(per_type.values()))} per node"
                + (f" ({total} in total)" if total else "")
            )
    nodes = [int(n) for n in _SIZE_NODES_RE.findall(size)]
    if nodes and desc.system_node_ensemble_total and sum(nodes) != desc.system_node_ensemble_total:
        problems.append(
            f"names {sum(nodes)} node(s), but system_node_ensemble_total is "
            f"{desc.system_node_ensemble_total}"
        )
    if not problems:
        return []
    return [
        _finding(
            Level.NOTE,
            "system_size disagrees with the node types",
            f"system_size “{size}” " + "; ".join(problems) + ".",
            curve=label,
            spec_ref=_SPEC,
        )
    ]


def _power(label: str, desc: SystemDescription, power: SystemPower) -> list[Finding]:
    nodes = {
        (n.system_node_ensemble_id if n.system_node_ensemble_id is not None else i): n
        for i, n in enumerate(desc.node_types)
    }
    per_node = _accelerators_per_node(desc)
    attention: list[str] = []
    notes: list[str] = []
    for node_set in power.node_sets:
        node = nodes.get(node_set.system_node_ensemble_id)
        if node is None:
            continue  # §9.1's power-descriptor warns about unknown ensemble IDs
        where = f"node set {node_set.node_set_id}"
        components = node_set.components
        accelerator = components.accelerator if components else None
        if accelerator is not None:
            desc_count = per_node.get(node_set.system_node_ensemble_id)
            if desc_count is not None and accelerator.count_per_node != desc_count:
                attention.append(
                    f"{where}: {accelerator.count_per_node} accelerators per node in "
                    f"system_power.json, {desc_count} in system_desc.json"
                )
            power_families = accelerator_families(accelerator.model)
            desc_families = {
                f
                for a in node.accelerator_info
                for f in accelerator_families(a.accelerator_model_name)
            }
            if power_families and desc_families and not power_families & desc_families:
                attention.append(
                    f"{where}: accelerator “{accelerator.model}” in system_power.json, "
                    f"{', '.join(sorted(desc_families))} in system_desc.json"
                )
        cpu = components.cpu if components else None
        if (
            cpu is not None
            and node.host_processors_per_node is not None
            and cpu.count_per_node != node.host_processors_per_node
        ):
            notes.append(
                f"{where}: {cpu.count_per_node} processors per node in system_power.json, "
                f"{node.host_processors_per_node} in system_desc.json"
            )
        if node.number_of_nodes is not None and node_set.nodes_provisioned != node.number_of_nodes:
            notes.append(
                f"{where}: {node_set.nodes_provisioned} nodes provisioned in system_power.json, "
                f"{node.number_of_nodes} in system_desc.json"
            )

    out: list[Finding] = []
    if attention:
        out.append(
            _finding(
                Level.ATTENTION,
                "Power descriptor describes different accelerators",
                "; ".join(attention) + ". Power normalization (§4.5) uses system_power.json, "
                "so the published TPS/kW may not be for the system that ran.",
                curve=label,
                spec_ref="§9.2, §4.5.2",
            )
        )
    if notes:
        out.append(
            _finding(
                Level.NOTE,
                "Power descriptor and system description differ",
                "; ".join(notes) + ".",
                curve=label,
                spec_ref="§9.2, §4.5.2",
            )
        )
    return out


def run(ctx: ReviewContext) -> list[Finding]:
    """Return the system-description findings, once per system."""
    out: list[Finding] = []
    seen: set[str] = set()
    for curve in ctx.curves:
        desc = curve.system_desc
        if desc is None:
            continue
        system = curve.system_dir.name
        # Per-point parallelism is a per-curve question; the rest describe the system.
        out.extend(_parallelism(curve.label, curve))
        if system in seen:
            continue
        seen.add(system)
        out.extend(_placeholders(system, desc))
        out.extend(_links(system, desc, curve))
        out.extend(_prose(system, desc))
        out.extend(_framework(system, desc, ctx))
        out.extend(_system_size(system, desc))
        if curve.system_power is not None:
            out.extend(_power(system, desc, curve.system_power))
    return out


CHECK = Check(
    name="system_description",
    focus_area="system_description",
    requires=frozenset({INPUT_SYSTEM_DESCRIPTIONS}),
    run=run,
)
