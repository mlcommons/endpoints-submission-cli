# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Provisioned power — §4.5.2's `system_power.json` descriptor.

§4.5 normalises throughput by the system's **provisioned** power: what the system is
built to draw, not what a given measurement point actually drew. §4.5.3 is explicit
that the denominator is therefore constant across a submission's whole Pareto curve,
and that two otherwise identical systems with different provisioned power are
different systems.

§4.5.2's power model:

.. code-block:: text

    System Power     = Major_components + Other_components
    Major_components = CPU_power + Accelerator_power + Network_scale_up_power
    Other_components = overhead_fraction × Major_components
    overhead_fraction = 0.30 liquid-cooled, 0.50 air-cooled

Note what ``Major_components`` leaves out. §4.5.2 defines ``Other_components`` as
"**scale-out networking**, storage, power-supply overhead, and cooling" — so a
scale-out fabric is already inside the overhead fraction and must not also be summed
into the majors. §4.5.2's field-group table nonetheless gives Scale-out its own row
with ``num_switches`` / ``tdp_per_switch``, which the power model has no slot for;
the formula is the normative statement, so a declared scale-out group is read and
reported but does not enter the total.

§4.5.2 publishes field *names* but no JSON schema, so the group keys here
(``cpu``, ``accelerator``, …) and the fields the table does not name
(``provisioned_power_w``, ``cooling``, the rack-scaling trio) are this checker's
naming. The names §4.5.2 *does* give — ``num_cpu``, ``tdp_per_cpu``,
``num_accelerator``, ``tdp_per_accelerator``, ``num_switches``, ``tdp_per_switch`` —
are read exactly as written.

Every field is optional: §4.5.2 says a submitter who omits a value leaves it to be
auto-populated, "which triggers the estimated-power tag". The checker reports what was
supplied rather than rejecting gaps — but it does not silently substitute zero for a
missing overhead fraction, because that understates the denominator and inflates
``system_tps_per_kw``.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

__all__ = [
    "AIR_COOLED_OVERHEAD",
    "LIQUID_COOLED_OVERHEAD",
    "ComponentGroup",
    "SystemPower",
    "overhead_for_cooling",
]

#: Watts per kilowatt — §4.5.3's normalisation is expressed in kW.
_W_PER_KW = 1000.0

#: §4.5.2's overhead fractions, reaffirmed in the [POWER-NORM] resolved table.
LIQUID_COOLED_OVERHEAD = 0.30
AIR_COOLED_OVERHEAD = 0.50


def overhead_for_cooling(cooling: str | None) -> float | None:
    """§4.5.2's overhead fraction for a §8.2 ``cooling`` description, if it names one.

    §8.2's ``cooling`` field "describes if the node uses any liquid cooling, only
    air-cooling, or only passive cooling" as free text, so this matches on the words
    rather than an enum. Liquid is checked first: "liquid-cooled with air-cooled PSUs"
    is a liquid-cooled system.

    Passive cooling names neither fraction. §4.5.2 gives only two, so rather than
    inventing a third this returns ``None`` and the caller reports that the fraction
    could not be established.
    """
    if not cooling:
        return None
    text = cooling.lower()
    if "liquid" in text or "water" in text or "immersion" in text:
        return LIQUID_COOLED_OVERHEAD
    if "air" in text:
        return AIR_COOLED_OVERHEAD
    return None


class ComponentGroup(BaseModel):
    """One §4.5.2 component group: a count, a per-unit TDP, and its evidence.

    The aliases are §4.5.2's own field names. They are grouped per component in the
    spec's table (``num_cpu`` under CPU, ``num_switches`` under both network rows), and
    accepted on any group here — a mis-keyed group is a submitter error worth catching,
    but rejecting it in the schema would fail the file rather than report it, and
    §4.5.2's whole posture is to report gaps rather than reject.

    Attributes:
        count: Units of this component the system is provisioned with.
        tdp_per_unit: Rated power per unit, in watts.
        link: Public specification supporting the rating. §4.5.2 requires every power
            value to be publicly verifiable.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    count: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "count",
            "num_cpu",
            "num_accelerator",
            "num_switches",
            "num_compute",
        ),
    )
    tdp_per_unit: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "tdp_per_unit",
            "tdp_per_cpu",
            "tdp_per_accelerator",
            "tdp_per_switch",
            "tdp_per_compute",
        ),
    )
    link: str | None = Field(
        default=None,
        validation_alias=AliasChoices("link", "public_specification"),
    )

    @property
    def total_w(self) -> float | None:
        """This group's contribution in watts, or None when either factor is absent."""
        if self.count is None or self.tdp_per_unit is None:
            return None
        return self.count * self.tdp_per_unit


class SystemPower(BaseModel):
    """Parsed contents of a system's ``system_power.json`` (§4.5.2).

    Attributes:
        provisioned_power_w: A directly declared total, in watts. §4.5.2: "If the
            estimated total system power is higher than a submitter believes their
            system is rated at, they may instead provide a provisioned power number
            directly." Where a published spec states a range, §4.5.2 takes the upper
            bound — a submitter obligation this checker cannot verify.
        cpu: CPU group — ``num_cpu`` × ``tdp_per_cpu``.
        accelerator: Accelerator group — ``num_accelerator`` × ``tdp_per_accelerator``.
        compute: Optional combined CPU + accelerator group. §4.5.2: "In some systems
            CPU and accelerator power are published as a single combined value. That
            is a valid alternative formulation." When present it replaces both.
        scale_up_network: Intra-node and rack-level switching. A major component.
        scale_out_network: Optional. Read and reported, but **not** summed into the
            majors — §4.5.2 places scale-out inside ``Other_components``.
        overhead_fraction: §4.5.2's ``Other_components`` multiplier. Normally derived
            from the cooling method rather than declared.
        cooling: §8.2's cooling description, which fixes the overhead fraction.
            Declarable here, and otherwise injected by the checker from the system
            description.
        rack_power_w: §4.5.2.1 rack-level node scaling — published power of the full
            rack, ``P_rack``.
        rack_nodes: ``N``, the number of nodes in that published rack.
        submitted_nodes: ``Y``, the number of nodes actually submitted.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    provisioned_power_w: float | None = Field(
        default=None,
        validation_alias=AliasChoices("provisioned_power_w", "provisioned_power_watts"),
    )
    cpu: ComponentGroup = Field(default_factory=ComponentGroup)
    accelerator: ComponentGroup = Field(default_factory=ComponentGroup)
    compute: ComponentGroup = Field(default_factory=ComponentGroup)
    scale_up_network: ComponentGroup = Field(default_factory=ComponentGroup)
    scale_out_network: ComponentGroup = Field(default_factory=ComponentGroup)
    overhead_fraction: float | None = None
    cooling: str | None = None

    rack_power_w: float | None = None
    rack_nodes: float | None = None
    submitted_nodes: float | None = None

    @property
    def resolved_overhead_fraction(self) -> float | None:
        """§4.5.2's overhead fraction, declared or derived from the cooling method.

        An explicit declaration wins, so a submitter can record a fraction the WG has
        agreed for a case §4.5.2's two-way split does not cover. Otherwise it comes
        from ``cooling``. ``None`` means it could not be established — deliberately not
        zero, which would drop ``Other_components`` from the total and inflate
        ``system_tps_per_kw`` by 30–50 %.
        """
        if self.overhead_fraction is not None:
            return self.overhead_fraction
        return overhead_for_cooling(self.cooling)

    @property
    def major_components_w(self) -> float | None:
        """§4.5.2's ``Major_components``: CPU + accelerator + scale-up.

        Scale-out is excluded by §4.5.2's definition of ``Other_components``. A
        combined ``compute`` group replaces CPU and accelerator when supplied.
        """
        compute = self.compute.total_w
        if compute is not None:
            parts = [compute, self.scale_up_network.total_w]
        else:
            parts = [self.cpu.total_w, self.accelerator.total_w, self.scale_up_network.total_w]
        stated = [p for p in parts if p is not None]
        return sum(stated) if stated else None

    @property
    def rack_scaled_power_w(self) -> float | None:
        """§4.5.2.1's ``P_rack × (Y / N)``, when all three values are present.

        Applies at node granularity only; §4.5.2.1 forbids it within a node, which is
        a property of the configuration rather than of these numbers.
        """
        if self.rack_power_w is None or self.rack_nodes is None or self.submitted_nodes is None:
            return None
        if self.rack_nodes <= 0:
            return None
        return self.rack_power_w * (self.submitted_nodes / self.rack_nodes)

    @property
    def derived_power_w(self) -> float | None:
        """Total provisioned power in watts, by whichever §4.5.2 path the file states.

        Precedence follows §4.5.2's own ordering for partially provisioned systems:
        a directly declared figure, then rack-level node scaling, then the component
        formula. Returns ``None`` when the components are stated but the overhead
        fraction is not — the sum of the majors alone is not a §4.5.2 total.
        """
        if self.provisioned_power_w is not None:
            return self.provisioned_power_w
        scaled = self.rack_scaled_power_w
        if scaled is not None:
            return scaled
        major = self.major_components_w
        overhead = self.resolved_overhead_fraction
        if major is None or overhead is None:
            return None
        return major * (1.0 + overhead)

    @property
    def provisioned_power_kw(self) -> float | None:
        """Provisioned power in kilowatts — §4.5.3's normalisation denominator."""
        watts = self.derived_power_w
        return None if watts is None else watts / _W_PER_KW

    @property
    def missing_groups(self) -> list[str]:
        """What the submitter left for MLCommons to auto-populate.

        §4.5.2: omitting a value "triggers the estimated-power tag". Nothing is missing
        when the total comes from a declared figure or from §4.5.2.1's rack scaling,
        since neither path uses the components. Scale-out is never listed — §4.5.2
        marks it optional and does not count it among the majors.
        """
        if self.provisioned_power_w is not None or self.rack_scaled_power_w is not None:
            return []
        groups: list[tuple[str, ComponentGroup]] = (
            [("compute", self.compute)]
            if self.compute.total_w is not None
            else [("cpu", self.cpu), ("accelerator", self.accelerator)]
        )
        groups.append(("scale_up_network", self.scale_up_network))
        missing = [name for name, group in groups if group.total_w is None]
        if self.resolved_overhead_fraction is None:
            missing.append("overhead_fraction (no declared value and cooling method unknown)")
        return missing
