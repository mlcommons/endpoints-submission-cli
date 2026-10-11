# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Provisioned power — the ``system_power.json`` descriptor of Appendix E.

§4.5 normalises throughput by the system's **provisioned** power: what the system is
built to draw, not what a given measurement point actually drew. Provisioned power is
fixed per system; §4.5.3 then scales it per measurement point by the whole nodes that
point engages, which is why :class:`PowerComputation` keeps each node set's share
``P_s`` and the scale-out switch power ``S`` apart rather than only their sum.

Appendix E (policies PR #126, at ``0e83c26``) is the normative schema. The descriptor records *how*
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

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...messages import Invalid, fragment
from ...power_defaults import (
    ETHERNET_SWITCH_TBPS_PER_DIRECTION,
    ETHERNET_SWITCH_W,
    NVLINK_PJ_PER_BIT,
    SUBSECTIONS,
    PowerDefault,
    accelerator_default,
    cpu_default,
    cpu_default_candidates,
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
    "SetPower",
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
            raise Invalid("power-descriptor", "value-count")
        if self.source_type == "mlc_default":
            if self.source not in _APPENDIX_D:
                raise Invalid(
                    "power-descriptor",
                    "mlc-default-source",
                    subsections=", ".join(sorted(_APPENDIX_D)),
                    source=self.source,
                )
        elif not self.source.startswith(("https://", "http://")):
            raise Invalid(
                "power-descriptor",
                "source-not-url",
                source_type=self.source_type,
                source=self.source,
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


@dataclass(frozen=True)
class SetPower:
    """One node set's share of provisioned power — §4.5.3's ``P_s`` and ``N_s``.

    Attributes:
        node_set_id: The set's E.3 identifier.
        system_node_ensemble_id: The §8.2 node type it describes, which is what a
            point's ``nodes_used`` names.
        nodes_provisioned: ``N_s``.
        power_w: ``P_s``. For a ``component_sum`` set this is its components and its
            share of counted scale-out NICs, with the overhead fraction applied; for a
            published set, the published figure (plus any NICs evidenced as excluded
            from it). Scale-out switch power is not in it — that is §4.5.3's ``S``.
        accelerators_per_node: ``accelerator.count_per_node`` where the descriptor
            states it, else ``None`` for the checker to fill from §8.2.
    """

    node_set_id: int
    system_node_ensemble_id: int
    nodes_provisioned: int
    power_w: float
    accelerators_per_node: int | None


@dataclass
class PowerComputation:
    """The checker's own E.5 figures, and what it had to say while getting them.

    Attributes:
        provisioned_power_kw: The system's full provisioned power, rounded to two
            decimal places, or ``None`` where the descriptor does not determine it.
            Each point's denominator is scaled from it (:meth:`point_power_kw`).
        sets: Each node set's ``P_s`` and ``N_s``, for §4.5.3's per-point scaling.
        declared: True where ``declared_provisioned_power`` governs, so the total
            does not decompose into the per-set shares.
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
    sets: list[SetPower] = field(default_factory=list)
    declared: bool = False
    estimated: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def point_power_kw(self, nodes_used: Mapping[int, int] | None = None) -> float | None:
        """§4.5.3's ``point_power_kw`` for a point engaging *nodes_used*.

        .. code-block:: text

            point_power_kw = Σ_s P_s × (Y_s / N_s)  +  S × (Σ Y_s / Σ N_s)

        Args:
            nodes_used: ``Y_s`` keyed by ``node_set_id``. A set it does not name is
                taken as fully engaged, ``Y_s = N_s`` — §4.5.3's conservative default
                for a point that declares nothing.

        A ``declared_provisioned_power`` is one figure for the whole system and does
        not split into per-set shares. For one set §4.5.3's homogeneous form applies
        exactly, ``provisioned_power_kw × (Y / N)``; for several, the rules give no
        decomposition, so the declared figure is scaled by the *largest* engaged
        fraction of any set — the conservative reading, and the same answer wherever
        the sets are engaged evenly.
        """
        kw = self.provisioned_power_kw
        if kw is None:
            return None
        used = nodes_used or {}
        fractions = {
            s.node_set_id: used.get(s.node_set_id, s.nodes_provisioned) / s.nodes_provisioned
            for s in self.sets
        }
        if not fractions or all(f == 1 for f in fractions.values()):
            return kw
        if self.declared:
            return round(kw * max(fractions.values()), 2)
        engaged = sum(s.nodes_provisioned * fractions[s.node_set_id] for s in self.sets)
        provisioned = sum(s.nodes_provisioned for s in self.sets)
        watts = sum(s.power_w * fractions[s.node_set_id] for s in self.sets)
        watts += self.scale_out_switch_power_w * engaged / provisioned
        return round(watts / _W_PER_KW, 2)

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
        # mlc_default values Appendix D does not state, and ones it may state but
        # which the descriptor carries too little to confirm. Like gaps, they matter
        # only where they reach the total (E.1, E.6.2).
        self.misquoted: list[str] = []
        self.unconfirmed: list[str] = []

    def watts(
        self,
        label: str,
        value: SourcedValue | None,
        fallback: PowerDefault | None = None,
        *,
        kind: str | None = None,
        defaults: tuple[PowerDefault, ...] | None = (),
        determined: bool = True,
        subject: str = "",
    ) -> float | None:
        """*value* in watts, or the Appendix D *fallback* where it is absent.

        A value present but given in pJ is a problem, not a gap: it names no power. An
        absent value with no fallback is a gap — fatal only where it reaches the total.

        A value sourced as ``mlc_default`` is held to Appendix D (:meth:`appendix_d`):
        *kind* is its :data:`~submission_checker.power_defaults.SUBSECTIONS` key, or
        ``None`` where Appendix D has no default for the field, and *defaults* the rows
        that could apply to *subject* — ``determined`` where exactly one does — or
        ``None`` where the descriptor cannot pin the default to one figure.
        """
        if value is not None:
            watts = value.watts
            if watts is None:
                self.problems.append(
                    fragment("power-descriptor", "energy-figure-give-value", label=label)
                )
                return None
            if value.is_default:
                self.estimated.append(f"{label} ({value.source} default)")
                self.appendix_d(
                    label,
                    value,
                    watts,
                    kind=kind,
                    expected=None if defaults is None else tuple(d.watts for d in defaults),
                    determined=determined,
                    subject=subject,
                )
            return watts
        if fallback is not None:
            self.estimated.append(f"{label} (absent; auto-populated from {fallback.source})")
            return fallback.watts
        self.gaps.append(fragment("power-descriptor", "absent-appendix-d-has", label=label))
        return None

    def appendix_d(
        self,
        label: str,
        value: SourcedValue,
        amount: float,
        *,
        kind: str | None,
        expected: tuple[float, ...] | None,
        determined: bool = True,
        subject: str = "",
        unit: str = "W",
    ) -> None:
        """Hold an ``mlc_default`` *value* to the Appendix D figure it claims to be.

        E.1 makes the default "its own reference", so it must come from the
        subsection Appendix D defines for the field and equal that subsection's
        figure for this component. *expected* of ``None`` checks the subsection only,
        for a default the descriptor cannot pin to one figure.
        """
        subsection = SUBSECTIONS.get(kind) if kind is not None else None
        if subsection is None:
            self.misquoted.append(
                fragment("power-descriptor", "mlc-default-no-subsection", label=label)
            )
            return
        if value.source != subsection:
            self.misquoted.append(
                fragment(
                    "power-descriptor",
                    "mlc-default-wrong-subsection",
                    label=label,
                    source=value.source,
                    subsection=subsection,
                )
            )
            return
        if expected is None:
            return
        if not expected:
            self.misquoted.append(
                fragment(
                    "power-descriptor",
                    "mlc-default-no-row",
                    label=label,
                    subsection=subsection,
                    subject=subject,
                )
            )
            return
        if not any(math.isclose(amount, e, rel_tol=1e-9, abs_tol=1e-6) for e in expected):
            self.misquoted.append(
                fragment(
                    "power-descriptor",
                    "mlc-default-value",
                    label=label,
                    value=amount,
                    unit=unit,
                    subsection=subsection,
                    expected=" or ".join(f"{e:g} {unit}" for e in expected),
                    subject=subject,
                )
            )
            return
        if not determined:
            self.unconfirmed.append(
                fragment(
                    "power-descriptor",
                    "mlc-default-unconfirmed",
                    label=label,
                    value=amount,
                    unit=unit,
                    subsection=subsection,
                    subject=subject,
                )
            )


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
            out.problems.append(fragment("power-descriptor", "node-set-id-values"))

        # Each set's own major and published watts, kept apart for §4.5.3's P_s.
        major: dict[int, float] = {}
        published: dict[int, float] = {}
        for node_set in self.node_sets:
            set_major, set_published = self._add_node_set(node_set, cores, out, tally)
            major[node_set.node_set_id] = set_major
            published[node_set.node_set_id] = set_published
        nic_per_node_w = self._add_scale_out(out, tally)
        for node_set in self.node_sets:
            if self._carries_nics(node_set):
                major[node_set.node_set_id] += node_set.nodes_provisioned * nic_per_node_w

        out.major_components_w = sum(major.values())
        out.published_node_power_w = sum(published.values())
        out.other_components_w = out.overhead_fraction * out.major_components_w
        out.sets = [
            SetPower(
                node_set_id=s.node_set_id,
                system_node_ensemble_id=s.system_node_ensemble_id,
                nodes_provisioned=s.nodes_provisioned,
                power_w=(1 + out.overhead_fraction) * major[s.node_set_id]
                + published[s.node_set_id],
                accelerators_per_node=(
                    s.components.accelerator.count_per_node
                    if s.components is not None and s.components.accelerator is not None
                    else None
                ),
            )
            for s in self.node_sets
        ]
        out.problems.extend(tally.problems)

        declared = self.declared_provisioned_power
        if declared is not None:
            # The declared figure governs, so the component block beneath it is only a
            # cross-check: its defaults do not tag the result, and its gaps and
            # misquoted defaults do not reject it (E.1, E.6.2).
            out.declared = True
            out.warnings.extend(tally.misquoted)
            watts = declared.watts
            if watts is None:
                out.problems.append(fragment("power-descriptor", "declared-provisioned-power-give"))
            else:
                out.provisioned_power_kw = round(watts / _W_PER_KW, 2)
                if declared.is_default:
                    # Appendix D has no system-level figure to default to.
                    out.problems.append(
                        fragment(
                            "power-descriptor",
                            "mlc-default-no-subsection",
                            label="declared_provisioned_power",
                        )
                    )
        else:
            out.problems.extend(tally.gaps)
            out.problems.extend(tally.misquoted)
            out.warnings.extend(tally.unconfirmed)
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
    ) -> tuple[float, float]:
        """One set's ``(major, published)`` watts. A set that cannot be costed adds 0."""
        label = f"node_sets[{node_set.node_set_id}]"
        y = node_set.nodes_provisioned
        if node_set.power_method == "component_sum":
            if node_set.components is None:
                out.problems.append(
                    fragment("power-descriptor", "component-sum-requires-components", label=label)
                )
                return 0.0, 0.0
            per_node = self._per_node_w(
                node_set.components, cores.get(node_set.system_node_ensemble_id), label, tally
            )
            return (0.0 if per_node is None else y * per_node), 0.0

        if node_set.published_power is None:
            out.problems.append(
                fragment(
                    "power-descriptor",
                    "requires-published-power",
                    label=label,
                    power_method=node_set.power_method,
                )
            )
            return 0.0, 0.0
        published = tally.watts(f"{label}.published_power", node_set.published_power)
        if published is None:
            return 0.0, 0.0
        if node_set.power_method == "published_system":
            return 0.0, y * published
        n = node_set.nodes_in_published_rack
        if n is None or n <= y:
            out.problems.append(
                fragment("power-descriptor", "node-scaling-requires-nodes", label=label, y=y, n=n)
            )
            return 0.0, 0.0
        return 0.0, published * (y / n)

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
                    fragment("power-descriptor", "combined-cpu-accelerator-mutually", label=label)
                )
            parts.append(
                tally.watts(
                    f"{label}.combined_cpu_accelerator", components.combined_cpu_accelerator
                )
            )
        else:
            if cpu is None or acc is None:
                tally.problems.append(
                    fragment("power-descriptor", "needs-cpu-accelerator-combined", label=label)
                )
                return None
            cpu_fallback = cpu_default(cpu.model, cores)
            cpu_w = tally.watts(
                f"{label}.cpu.tdp_per_unit",
                cpu.tdp_per_unit,
                cpu_fallback,
                kind="cpu.tdp_per_unit",
                defaults=cpu_default_candidates(cpu.model, cores),
                determined=cpu_fallback is not None,
                subject=(cpu.model or "an unnamed CPU")
                + (f" with {cores} cores" if cores is not None else ""),
            )
            acc_fallback = accelerator_default(acc.model)
            acc_w = tally.watts(
                f"{label}.accelerator.tdp_per_unit",
                acc.tdp_per_unit,
                acc_fallback,
                kind="accelerator.tdp_per_unit",
                defaults=() if acc_fallback is None else (acc_fallback,),
                subject=acc.model or "an unnamed accelerator",
            )
            parts.append(None if cpu_w is None else cpu.count_per_node * cpu_w)
            parts.append(None if acc_w is None else acc.count_per_node * acc_w)

        up = components.scale_up_network
        if up is None:
            tally.problems.append(
                fragment("power-descriptor", "scale-up-network-required", label=label)
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
                tally.problems.append(
                    fragment("power-descriptor", "declared-tdp-requires-switch", label=label)
                )
                return None
            kind = "scale_up_network.tdp_per_switch"
            # D.1's Ethernet point scales with the switch's bandwidth (C.7), which the
            # descriptor does not carry, so only a default's subsection is checked.
            per_switch = tally.watts(
                f"{label}.tdp_per_switch", up.tdp_per_switch, kind=kind, defaults=None
            )
            sv = up.tdp_per_switch
            if (
                per_switch is not None
                and sv is not None
                and sv.is_default
                and sv.source == SUBSECTIONS[kind]
            ):
                tally.unconfirmed.append(
                    fragment(
                        "power-descriptor",
                        "mlc-default-ethernet-scaled",
                        label=f"{label}.tdp_per_switch",
                        value=per_switch,
                        reference_w=ETHERNET_SWITCH_W,
                        reference_tbps=ETHERNET_SWITCH_TBPS_PER_DIRECTION,
                    )
                )
            return None if per_switch is None else up.switch_count * per_switch
        # bandwidth_estimate. The worked examples (C.8, C.9, E.6.2) read the aggregate
        # as terabytes per second — 14.4 TB/s × 8 bit/B × 5 pJ/bit = 576 W — despite
        # the field's `_tbps` suffix, so that is how it is read here.
        if up.aggregate_bandwidth_tbps is None:
            tally.problems.append(
                fragment("power-descriptor", "bandwidth-estimate-requires-aggregate", label=label)
            )
            return None
        energy = up.energy_per_bit_pj
        if energy is None:
            tally.gaps.append(fragment("power-descriptor", "energy-per-bit-pj", label=label))
            return None
        if energy.value_pj is None:
            tally.problems.append(fragment("power-descriptor", "energy-per-bit-pj-2", label=label))
            return None
        if energy.is_default:
            tally.estimated.append(f"{label}.energy_per_bit_pj ({energy.source} default)")
            tally.appendix_d(
                f"{label}.energy_per_bit_pj",
                energy,
                energy.value_pj,
                kind="scale_up_network.energy_per_bit_pj",
                expected=(NVLINK_PJ_PER_BIT,),
                subject="the NVLink reference, D.1's only energy-per-bit figure",
                unit="pJ/bit",
            )
        return up.aggregate_bandwidth_tbps * 8.0 * energy.value_pj

    def _carries_nics(self, node_set: NodeSet) -> bool:
        """Whether counted scale-out NICs are added to *node_set*'s power.

        §4.5.2: a formula-built node has no NIC term, so its NICs MUST be added; a
        published node figure is assumed to include them and they MUST NOT be added
        again. ``nics.counted`` is one flag for the whole system, so on a system mixing
        the two paths it decides only whether the formula-built sets get them. The
        published sets get them too only where ``excluded_from_published_power``
        evidences that their figure left the adapters out.
        """
        nics = self.scale_out.nics
        if not self.scale_out.present or nics is None or not nics.counted:
            return False
        return node_set.power_method == "component_sum" or bool(nics.excluded_from_published_power)

    def _check_fabric_declared(self, out: PowerComputation) -> None:
        """E.4: ``present`` follows from whether the nodes need a scale-out fabric.

        E.4 sets ``present: false`` "for a single-node submission, or where the nodes
        are joined only by a fabric already counted in ``scale_up_network``". So a
        multi-node system may legitimately declare none — C.2's rack — but only if
        something in the descriptor joins its nodes:

        - every set built from components with ``scale_up_network.method: none`` has
          nothing joining its nodes, so ``present: false`` leaves the fabric out;
        - a published node figure, or a scale-up network, may or may not span nodes,
          which the descriptor cannot say, so that is a warning;
        - ``present: true`` on a single node contradicts E.4 but overstates rather
          than flatters, so that is a warning too.
        """
        nodes = sum(s.nodes_provisioned for s in self.node_sets)
        if self.scale_out.present:
            if nodes == 1:
                out.warnings.append(fragment("power-descriptor", "scale-out-present-true-2"))
            return
        if nodes == 1:
            return
        no_scale_up = all(
            s.power_method == "component_sum"
            and s.components is not None
            and s.components.scale_up_network is not None
            and s.components.scale_up_network.method == "none"
            for s in self.node_sets
        )
        if no_scale_up:
            out.problems.append(
                fragment("power-descriptor", "scale-out-present-false", nodes=nodes)
            )
        else:
            out.warnings.append(
                fragment("power-descriptor", "scale-out-present-false-2", nodes=nodes)
            )

    def _add_scale_out(self, out: PowerComputation, tally: _Tally) -> float:
        """Add the switches to *out*; return the counted NIC watts **per node**.

        The NICs are major components, so the caller adds them to the major figure of
        each set that carries them (:meth:`_carries_nics`), where they take the
        overhead fraction like anything else in the node.
        """
        fabric = self.scale_out
        nic_per_node_w = 0.0
        self._check_fabric_declared(out)
        if not fabric.present:
            return nic_per_node_w
        missing = [
            name
            for name in ("cabling", "required_bandwidth_tbps", "nics", "switches")
            if getattr(fabric, name) is None
        ]
        if missing:
            out.problems.append(
                fragment("power-descriptor", "scale-out-present-true", missing=", ".join(missing))
            )
            return nic_per_node_w
        assert fabric.nics is not None and fabric.switches is not None
        assert fabric.required_bandwidth_tbps is not None

        nodes = sum(s.nodes_provisioned for s in self.node_sets)
        nics = fabric.nics
        by_formula = any(s.power_method == "component_sum" for s in self.node_sets)
        if nics.counted:
            if not by_formula and not nics.excluded_from_published_power:
                out.problems.append(fragment("power-descriptor", "scale-out-nics-counted"))
            per_nic = tally.watts(
                "scale_out.nics.tdp_per_nic",
                nics.tdp_per_nic,
                nic_default(),
                kind="scale_out.nics.tdp_per_nic",
                defaults=(nic_default(),),
                subject="a scale-out NIC",
            )
            if per_nic is not None:
                nic_per_node_w = nics.count_per_node * per_nic
        elif by_formula:
            out.problems.append(fragment("power-descriptor", "scale-out-nics-counted-2"))

        # E.4 defines the requirement as the NICs' sum, so a smaller declared figure —
        # which would let fewer switches through, and so less switch power — is held
        # to the derived one. A larger one only asks for more switches.
        from_nics = nodes * nics.count_per_node * nics.bandwidth_per_nic_gbps / 1000.0
        required = max(fabric.required_bandwidth_tbps, from_nics)
        if fabric.required_bandwidth_tbps < from_nics - 1e-6:
            out.problems.append(
                fragment(
                    "power-descriptor",
                    "scale-out-required-bandwidth",
                    required_bandwidth_tbps=fabric.required_bandwidth_tbps,
                    from_nics=from_nics,
                    nodes=nodes,
                    count_per_node=nics.count_per_node,
                    bandwidth_per_nic_gbps=nics.bandwidth_per_nic_gbps,
                )
            )
        elif fabric.required_bandwidth_tbps > from_nics + 1e-6:
            out.warnings.append(
                fragment(
                    "power-descriptor",
                    "scale-out-required-bandwidth-2",
                    required_bandwidth_tbps=fabric.required_bandwidth_tbps,
                    from_nics=from_nics,
                    nodes=nodes,
                    count_per_node=nics.count_per_node,
                    bandwidth_per_nic_gbps=nics.bandwidth_per_nic_gbps,
                )
            )

        offered = 0.0
        for index, switch in enumerate(fabric.switches):
            label = f"scale_out.switches[{index}].power_per_switch"
            reference = switch_default(switch.model, fabric.cabling)
            watts = tally.watts(
                label,
                switch.power_per_switch,
                reference,
                kind="scale_out.switches.power_per_switch",
                defaults=() if reference is None else (reference,),
                subject=f"{switch.model} with {fabric.cabling} cabling",
            )
            if watts is not None:
                out.scale_out_switch_power_w += switch.count * watts
            offered += switch.count * switch.bandwidth_tbps
        if offered < required - 1e-6:
            out.problems.append(
                fragment(
                    "power-descriptor",
                    "scale-out-switches-offer",
                    offered=offered,
                    required=required,
                )
            )
        return nic_per_node_w

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
                    problems.append(
                        fragment(
                            "power-descriptor",
                            "computed-recomputed",
                            name=name,
                            theirs=theirs,
                            value=value,
                        )
                    )
            if (
                stated.overhead_fraction is not None
                and abs(stated.overhead_fraction - out.overhead_fraction) > 1e-9
            ):
                problems.append(
                    fragment(
                        "power-descriptor",
                        "computed-overhead-fraction-cooling",
                        overhead_fraction=stated.overhead_fraction,
                        cooling=self.cooling,
                        overhead_fraction_2=out.overhead_fraction,
                    )
                )
        kw = out.provisioned_power_kw
        stated_kw = self.provisioned_power_kw
        if stated_kw is not None and kw is not None and abs(stated_kw - kw) > _KW_TOL:
            problems.append(
                fragment(
                    "power-descriptor",
                    "provisioned-power-kw-recomputed",
                    stated_kw=stated_kw,
                    kw=kw,
                )
            )
        return problems
