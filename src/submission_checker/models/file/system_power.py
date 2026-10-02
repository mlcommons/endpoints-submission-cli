# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Provisioned power — the ``system_power.json`` descriptor of Appendix E.

§4.5 normalises throughput by the system's **provisioned** power: what the system is
built to draw, not what a given measurement point actually drew. §4.5.3 is explicit
that the denominator is therefore constant across a submission's whole Pareto curve.

Appendix E (policies PR #126) is the normative schema. The descriptor records *how*
provisioned power was established — per homogeneous set of nodes, with every power
figure carrying its own source — and §4.5.2's model turns it into one number:

.. code-block:: text

    System Power     = Major + Other + Published_node_power + Scale_out_switch_power
    Major            = Σ component_sum sets: nodes × (CPU + Accelerator + Scale-up)
                       + scale-out NIC power, where counted
    Other            = overhead_fraction × Major     (0.30 liquid, 0.50 air)
    Published        = Σ published_system sets: nodes × published node power
                       + Σ node_scaling sets: P_rack × (Y / N)

Two terms sit outside the overhead base. A published node figure already carries the
node's own cooling and power-supply overhead, and rack switch power is wall power; the
overhead fraction applied to either would count it twice.

This module parses the structure — required fields, types, and the shape of a sourced
value — and does the E.5 arithmetic. The E.7 rules that need the system description
(``cooling`` agreement, core counts for D.2 defaults) are applied by the checker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...power_defaults import (
    PowerDefault,
    accelerator_default,
    cpu_default,
    nic_default,
    switch_default,
)

__all__ = [
    "AIR_COOLED_OVERHEAD",
    "LIQUID_COOLED_OVERHEAD",
    "Components",
    "NodeSet",
    "PowerComputation",
    "ScaleOut",
    "SourcedValue",
    "SystemPower",
    "overhead_for_cooling",
]

#: Watts per kilowatt — §4.5.3's normalisation is expressed in kW.
_W_PER_KW = 1000.0

#: §4.5.2's overhead fractions, selected by ``cooling``.
LIQUID_COOLED_OVERHEAD = 0.30
AIR_COOLED_OVERHEAD = 0.50
_OVERHEAD = {"liquid": LIQUID_COOLED_OVERHEAD, "air": AIR_COOLED_OVERHEAD}

#: E.1's permitted source types. A submitter's own assertion is deliberately absent.
SourceType = Literal["vendor_spec", "publication", "public_statement", "mlc_default"]
_APPENDIX_D = frozenset({"D.1", "D.2", "D.3", "D.4"})

#: Tolerance when comparing a submitter's ``computed`` block with the recomputation.
#: Intermediate sums are not rounded (§4.5.2), so this only absorbs float noise and a
#: submitter writing a whole number of watts.
_COMPUTED_TOL_W = 0.5
_KW_TOL = 0.005


def overhead_for_cooling(cooling: str | None) -> float | None:
    """§4.5.2's overhead fraction for a §8.2 ``cooling`` description, if it names one.

    §8.2's ``cooling`` field is free text, so this matches on the words. Liquid is
    checked first: "liquid-cooled with air-cooled PSUs" is a liquid-cooled system.
    Passive cooling names neither fraction, so it returns ``None``.
    """
    if not cooling:
        return None
    text = cooling.lower()
    if "liquid" in text or "water" in text or "immersion" in text:
        return LIQUID_COOLED_OVERHEAD
    if "air" in text:
        return AIR_COOLED_OVERHEAD
    return None


class SourcedValue(BaseModel):
    """E.1: a power figure and the public, verifiable reference behind it.

    Exactly one of ``value_w``, ``value_kw`` or ``value_pj`` is set. ``source`` is a
    URL for the three verifiable types, and the Appendix D subsection for
    ``mlc_default`` — "which is its own reference and needs no URL".
    """

    model_config = ConfigDict(extra="forbid")

    value_w: float | None = Field(default=None, ge=0)
    value_kw: float | None = Field(default=None, ge=0)
    value_pj: float | None = Field(default=None, ge=0)
    source_type: SourceType
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _one_value_and_a_real_source(self) -> SourcedValue:
        values = [v for v in (self.value_w, self.value_kw, self.value_pj) if v is not None]
        if len(values) != 1:
            raise ValueError("exactly one of value_w, value_kw or value_pj is required")
        if self.source_type == "mlc_default":
            if self.source not in _APPENDIX_D:
                raise ValueError(
                    f"an mlc_default source names its Appendix D subsection"
                    f" ({', '.join(sorted(_APPENDIX_D))}), not {self.source!r}"
                )
        elif not self.source.startswith(("https://", "http://")):
            raise ValueError(
                f"a {self.source_type} source must be a resolvable URL, not {self.source!r}"
            )
        return self

    @property
    def watts(self) -> float | None:
        """The value in watts, or ``None`` for an energy-per-bit figure."""
        if self.value_w is not None:
            return self.value_w
        if self.value_kw is not None:
            return self.value_kw * _W_PER_KW
        return None

    @property
    def is_default(self) -> bool:
        """True for an Appendix D value, which sets the estimated-power tag."""
        return self.source_type == "mlc_default"


class _Open(BaseModel):
    """Appendix E objects tolerate keys the schema does not name — ``notes`` and the like."""

    model_config = ConfigDict(extra="allow")


class Processor(_Open):
    """E.3.1's ``cpu`` block. Counts are what is provisioned, not what the chassis holds."""

    model: str | None = None
    count_per_node: int = Field(ge=0)
    tdp_per_unit: SourcedValue | None = None


class Accelerator(_Open):
    """E.3.1's ``accelerator`` block."""

    model: str | None = None
    count_per_node: int = Field(ge=0)
    tdp_per_unit: SourcedValue | None = None
    below_rated_tdp: dict[str, object] | None = None


class ScaleUpNetwork(_Open):
    """E.3.1's ``scale_up_network`` block, costed by one of three methods."""

    method: Literal["declared_tdp", "bandwidth_estimate", "none"]
    switch_count: int | None = Field(default=None, ge=0)
    tdp_per_switch: SourcedValue | None = None
    aggregate_bandwidth_tbps: float | None = Field(default=None, ge=0)
    energy_per_bit_pj: SourcedValue | None = None


class Components(_Open):
    """E.3.1: a node's power built from its parts, per node."""

    cpu: Processor | None = None
    accelerator: Accelerator | None = None
    combined_cpu_accelerator: SourcedValue | None = None
    scale_up_network: ScaleUpNetwork | None = None


class NodeSet(_Open):
    """E.3: one set of identical nodes, and how its power was established."""

    node_set_id: int
    system_node_ensemble_id: int
    nodes_provisioned: int = Field(gt=0)
    power_method: Literal["component_sum", "published_system", "node_scaling"]
    published_power: SourcedValue | None = None
    nodes_in_published_rack: int | None = None
    components: Components | None = None


class Nics(_Open):
    """E.4's scale-out adapters."""

    count_per_node: int = Field(ge=0)
    bandwidth_per_nic_gbps: float = Field(ge=0)
    counted: bool
    tdp_per_nic: SourcedValue | None = None
    excluded_from_published_power: str | None = None


class Switch(_Open):
    """One scale-out switch model in E.4's ``switches`` array."""

    model: str
    count: int = Field(gt=0)
    bandwidth_tbps: float = Field(ge=0)
    power_per_switch: SourcedValue | None = None


class ScaleOut(_Open):
    """E.4: the scale-out fabric. Everything but ``present`` is conditional on it."""

    present: bool
    cabling: Literal["passive", "active_optical"] | None = None
    required_bandwidth_tbps: float | None = Field(default=None, ge=0)
    nics: Nics | None = None
    switches: list[Switch] | None = None


class Computed(_Open):
    """E.5's derived arithmetic, as a submitter wrote it. The checker's values govern."""

    major_components_w: float | None = None
    overhead_fraction: float | None = None
    other_components_w: float | None = None
    published_node_power_w: float | None = None
    scale_out_switch_power_w: float | None = None
    total_system_power_w: float | None = None


@dataclass
class PowerComputation:
    """The checker's own E.5 figures, and what it had to say while getting them.

    Attributes:
        provisioned_power_kw: §4.5.3's denominator, rounded to two decimal places, or
            ``None`` where the descriptor does not determine it.
        estimated: Appendix D values, supplied or auto-populated, that reach
            ``provisioned_power_kw`` — each one sets the "MLC Estimated Power" tag.
        problems: E.7 rejections and other §4.5.2 MUST violations.
        warnings: Findings that do not reject the descriptor.
    """

    major_components_w: float = 0.0
    overhead_fraction: float = 0.0
    other_components_w: float = 0.0
    published_node_power_w: float = 0.0
    scale_out_switch_power_w: float = 0.0
    provisioned_power_kw: float | None = None
    estimated: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_system_power_w(self) -> float:
        """E.5's ``total_system_power_w``: the four terms, unrounded."""
        return (
            self.major_components_w
            + self.other_components_w
            + self.published_node_power_w
            + self.scale_out_switch_power_w
        )


class _Tally:
    """Collects one figure's worth of watts, defaults and gaps while summing."""

    def __init__(self) -> None:
        self.estimated: list[str] = []
        self.gaps: list[str] = []
        self.problems: list[str] = []

    def watts(
        self, label: str, value: SourcedValue | None, fallback: PowerDefault | None = None
    ) -> float | None:
        """*value* in watts, or the Appendix D *fallback* where it is absent.

        A value present but given in pJ is a problem, not a gap: it names no power. An
        absent value with no fallback is a gap — fatal only where it reaches the total.
        """
        if value is not None:
            watts = value.watts
            if watts is None:
                self.problems.append(f"{label} is an energy figure; give value_w or value_kw")
                return None
            if value.is_default:
                self.estimated.append(f"{label} ({value.source} default)")
            return watts
        if fallback is not None:
            self.estimated.append(f"{label} (absent; auto-populated from {fallback.source})")
            return fallback.watts
        self.gaps.append(f"{label} is absent and Appendix D has no default for it")
        return None


class SystemPower(_Open):
    """Parsed contents of a system's ``system_power.json`` (Appendix E.2).

    ``computed``, ``provisioned_power_kw`` and ``mlc_estimated_power`` are the
    checker's fields. A submitter may fill them in; :meth:`compute` recomputes them,
    and a disagreement in ``computed`` is an E.7 rejection. ``mlc_estimated_power`` is
    ignored, as E.2 says.
    """

    system_desc_id: str = Field(min_length=1)
    cooling: Literal["liquid", "air"]
    node_sets: list[NodeSet] = Field(min_length=1)
    scale_out: ScaleOut
    declared_provisioned_power: SourcedValue | None = None
    computed: Computed | None = None
    provisioned_power_kw: float | None = None
    mlc_estimated_power: bool | None = None
    notes: str | None = None

    @property
    def overhead_fraction(self) -> float:
        """§4.5.2's overhead fraction for the declared ``cooling``."""
        return _OVERHEAD[self.cooling]

    def compute(self, cores_by_ensemble: dict[int, int] | None = None) -> PowerComputation:
        """E.5's arithmetic, with Appendix D filling what the descriptor leaves absent.

        Args:
            cores_by_ensemble: ``host_processor_core_count`` per
                ``system_node_ensemble_id`` from the system description, which D.2's
                CPU default needs. A CPU whose model name states its cores does not.
        """
        cores = cores_by_ensemble or {}
        out = PowerComputation(overhead_fraction=self.overhead_fraction)
        tally = _Tally()

        ids = [s.node_set_id for s in self.node_sets]
        if len(set(ids)) != len(ids):
            out.problems.append("node_set_id values must be unique within the file")

        for node_set in self.node_sets:
            self._add_node_set(node_set, cores, out, tally)
        self._add_scale_out(out, tally)

        out.other_components_w = out.overhead_fraction * out.major_components_w
        out.problems.extend(tally.problems)

        declared = self.declared_provisioned_power
        if declared is not None:
            # The declared figure governs, so the component block beneath it is only a
            # cross-check: its defaults do not tag the result and its gaps do not
            # reject it (E.1, E.6.2).
            watts = declared.watts
            if watts is None:
                out.problems.append("declared_provisioned_power must give value_kw or value_w")
            else:
                out.provisioned_power_kw = round(watts / _W_PER_KW, 2)
                if declared.is_default:
                    out.estimated.append("declared_provisioned_power (mlc_default)")
        else:
            out.problems.extend(tally.gaps)
            out.estimated.extend(tally.estimated)
            if not out.problems:
                out.provisioned_power_kw = round(out.total_system_power_w / _W_PER_KW, 2)

        # Only a complete recomputation can disagree with the submitter's figures.
        if not out.problems and not tally.gaps:
            out.problems.extend(self._computed_disagreements(out))
        # A descriptor E.7 rejects normalises nothing, even where a figure was reached.
        if out.problems:
            out.provisioned_power_kw = None
        return out

    def _add_node_set(
        self,
        node_set: NodeSet,
        cores: dict[int, int],
        out: PowerComputation,
        tally: _Tally,
    ) -> None:
        label = f"node_sets[{node_set.node_set_id}]"
        y = node_set.nodes_provisioned
        if node_set.power_method == "component_sum":
            if node_set.components is None:
                out.problems.append(f"{label}: component_sum requires components")
                return
            per_node = self._per_node_w(
                node_set.components, cores.get(node_set.system_node_ensemble_id), label, tally
            )
            if per_node is not None:
                out.major_components_w += y * per_node
            return

        if node_set.published_power is None:
            out.problems.append(f"{label}: {node_set.power_method} requires published_power")
            return
        published = tally.watts(f"{label}.published_power", node_set.published_power)
        if published is None:
            return
        if node_set.power_method == "published_system":
            out.published_node_power_w += y * published
            return
        n = node_set.nodes_in_published_rack
        if n is None or n <= y:
            out.problems.append(
                f"{label}: node_scaling requires nodes_in_published_rack greater than"
                f" nodes_provisioned ({y}), got {n}"
            )
            return
        out.published_node_power_w += published * (y / n)

    @staticmethod
    def _per_node_w(
        components: Components, cores: int | None, label: str, tally: _Tally
    ) -> float | None:
        """One node's CPU + accelerator + scale-up, in watts."""
        label = f"{label}.components"
        parts: list[float | None] = []
        cpu, acc = components.cpu, components.accelerator
        if components.combined_cpu_accelerator is not None:
            if (cpu and cpu.tdp_per_unit) or (acc and acc.tdp_per_unit):
                tally.problems.append(
                    f"{label}: combined_cpu_accelerator is mutually exclusive with"
                    " cpu.tdp_per_unit and accelerator.tdp_per_unit"
                )
            parts.append(
                tally.watts(
                    f"{label}.combined_cpu_accelerator", components.combined_cpu_accelerator
                )
            )
        else:
            if cpu is None or acc is None:
                tally.problems.append(
                    f"{label}: needs cpu and accelerator, or combined_cpu_accelerator"
                )
                return None
            cpu_w = tally.watts(
                f"{label}.cpu.tdp_per_unit", cpu.tdp_per_unit, cpu_default(cpu.model, cores)
            )
            acc_w = tally.watts(
                f"{label}.accelerator.tdp_per_unit",
                acc.tdp_per_unit,
                accelerator_default(acc.model),
            )
            parts.append(None if cpu_w is None else cpu.count_per_node * cpu_w)
            parts.append(None if acc_w is None else acc.count_per_node * acc_w)

        up = components.scale_up_network
        if up is None:
            tally.problems.append(
                f"{label}: scale_up_network is required; use method none where there is no"
                " scale-up switch"
            )
            return None
        parts.append(SystemPower._scale_up_w(up, f"{label}.scale_up_network", tally))
        if any(p is None for p in parts):
            return None
        return sum(p for p in parts if p is not None)

    @staticmethod
    def _scale_up_w(up: ScaleUpNetwork, label: str, tally: _Tally) -> float | None:
        if up.method == "none":
            return 0.0
        if up.method == "declared_tdp":
            if up.switch_count is None:
                tally.problems.append(f"{label}: declared_tdp requires switch_count")
                return None
            per_switch = tally.watts(f"{label}.tdp_per_switch", up.tdp_per_switch)
            return None if per_switch is None else up.switch_count * per_switch
        # bandwidth_estimate. The worked examples (C.8, C.9, E.6.2) read the aggregate
        # as terabytes per second — 14.4 TB/s × 8 bit/B × 5 pJ/bit = 576 W — despite
        # the field's `_tbps` suffix, so that is how it is read here.
        if up.aggregate_bandwidth_tbps is None:
            tally.problems.append(f"{label}: bandwidth_estimate requires aggregate_bandwidth_tbps")
            return None
        energy = up.energy_per_bit_pj
        if energy is None:
            tally.gaps.append(
                f"{label}.energy_per_bit_pj is absent; D.1's references depend on the link"
                " protocol, so there is no default to apply"
            )
            return None
        if energy.value_pj is None:
            tally.problems.append(f"{label}.energy_per_bit_pj must give value_pj")
            return None
        if energy.is_default:
            tally.estimated.append(f"{label}.energy_per_bit_pj ({energy.source} default)")
        return up.aggregate_bandwidth_tbps * 8.0 * energy.value_pj

    def _add_scale_out(self, out: PowerComputation, tally: _Tally) -> None:
        fabric = self.scale_out
        if not fabric.present:
            return
        missing = [
            name
            for name in ("cabling", "required_bandwidth_tbps", "nics", "switches")
            if getattr(fabric, name) is None
        ]
        if missing:
            out.problems.append(f"scale_out.present is true but {', '.join(missing)} is missing")
            return
        assert fabric.nics is not None and fabric.switches is not None
        assert fabric.required_bandwidth_tbps is not None

        nodes = sum(s.nodes_provisioned for s in self.node_sets)
        nics = fabric.nics
        by_formula = any(s.power_method == "component_sum" for s in self.node_sets)
        if nics.counted:
            if not by_formula and not nics.excluded_from_published_power:
                out.problems.append(
                    "scale_out.nics.counted is true, but node power comes from a published"
                    " specification, which §4.5.2 assumes includes the adapters. Set"
                    " counted to false, or evidence the exclusion in"
                    " excluded_from_published_power"
                )
            per_nic = tally.watts("scale_out.nics.tdp_per_nic", nics.tdp_per_nic, nic_default())
            if per_nic is not None:
                out.major_components_w += nodes * nics.count_per_node * per_nic
        elif by_formula:
            out.problems.append(
                "scale_out.nics.counted is false, but node power is built with the MLC"
                " formula, which has no NIC term — §4.5.2 says the NICs MUST be included"
            )

        offered = 0.0
        for index, switch in enumerate(fabric.switches):
            label = f"scale_out.switches[{index}].power_per_switch"
            watts = tally.watts(
                label, switch.power_per_switch, switch_default(switch.model, fabric.cabling)
            )
            if watts is not None:
                out.scale_out_switch_power_w += switch.count * watts
            offered += switch.count * switch.bandwidth_tbps
        if offered < fabric.required_bandwidth_tbps:
            out.problems.append(
                f"scale-out switches offer {offered:g} Tb/s, below the required"
                f" {fabric.required_bandwidth_tbps:g} Tb/s"
            )

        from_nics = nodes * nics.count_per_node * nics.bandwidth_per_nic_gbps / 1000.0
        if abs(from_nics - fabric.required_bandwidth_tbps) > 1e-6:
            out.warnings.append(
                f"scale_out.required_bandwidth_tbps is {fabric.required_bandwidth_tbps:g},"
                f" but {nodes} nodes × {nics.count_per_node} NICs ×"
                f" {nics.bandwidth_per_nic_gbps:g} Gb/s is {from_nics:g} Tb/s"
            )

    def _computed_disagreements(self, out: PowerComputation) -> list[str]:
        """E.7: a submitter-supplied figure that disagrees with the recomputation."""
        problems: list[str] = []
        stated = self.computed
        if stated is not None:
            ours = {
                "major_components_w": out.major_components_w,
                "other_components_w": out.other_components_w,
                "published_node_power_w": out.published_node_power_w,
                "scale_out_switch_power_w": out.scale_out_switch_power_w,
                "total_system_power_w": out.total_system_power_w,
            }
            for name, value in ours.items():
                theirs = getattr(stated, name)
                if theirs is not None and abs(theirs - value) > _COMPUTED_TOL_W:
                    problems.append(f"computed.{name} is {theirs:g}, recomputed {value:g}")
            if (
                stated.overhead_fraction is not None
                and abs(stated.overhead_fraction - out.overhead_fraction) > 1e-9
            ):
                problems.append(
                    f"computed.overhead_fraction is {stated.overhead_fraction:g}, but"
                    f" cooling {self.cooling!r} fixes it at {out.overhead_fraction:g}"
                )
        kw = out.provisioned_power_kw
        stated_kw = self.provisioned_power_kw
        if stated_kw is not None and kw is not None and abs(stated_kw - kw) > _KW_TOL:
            problems.append(f"provisioned_power_kw is {stated_kw:g}, recomputed {kw:.2f}")
        return problems
