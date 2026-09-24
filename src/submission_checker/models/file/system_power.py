# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Provisioned power — §4.5.2's `system_power.json` descriptor.

§4.5 normalises throughput by the system's **provisioned** power: what the system is
built to draw, not what a given measurement point actually drew. §4.5.3 is explicit
that the denominator is therefore constant across a submission's whole Pareto curve,
and that two otherwise identical systems with different provisioned power are
different systems.

§4.5.2 does not publish a literal JSON schema — it lists field *groups* and says
MLCommons auto-populates anything the submitter omits, which "triggers the
estimated-power tag". This model follows that: every field is optional, and the
checker reports which of them the submitter supplied rather than rejecting gaps.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

__all__ = ["ComponentGroup", "SystemPower"]

#: Watts per kilowatt — §4.5.3's normalisation is expressed in kW.
_W_PER_KW = 1000.0


class ComponentGroup(BaseModel):
    """One §4.5.2 component group: a count, a per-unit TDP, and its evidence.

    Attributes:
        count: Units of this component the system is provisioned with.
        tdp_per_unit: Rated power per unit, in watts.
        link: Public specification supporting the rating. §4.5.2 requires every power
            value to be publicly verifiable.
    """

    model_config = ConfigDict(extra="allow")

    count: float | None = Field(
        default=None,
        validation_alias=AliasChoices("count", "num_cpu", "num_accelerator", "num_switches"),
    )
    tdp_per_unit: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "tdp_per_unit",
            "tdp_per_cpu",
            "tdp_per_cpu_watts",
            "tdp_per_accelerator",
            "tdp_per_accelerator_watts",
            "tdp_per_switch",
            "tdp_per_switch_watts",
        ),
    )
    link: str | None = Field(
        default=None, validation_alias=AliasChoices("link", "public_specification")
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
        provisioned_power_w: A directly declared total, in watts. §4.5.2 lets a
            submitter supply this instead of the component build-up when the estimate
            comes out higher than the system is rated at.
        cpu: CPU group — ``num_cpu`` × ``tdp_per_cpu``.
        accelerator: Accelerator group.
        scale_up_network: Intra-node and rack-level switching.
        scale_out_network: Optional; multi-node submissions using a scale-out fabric.
        overhead_fraction: §4.5.2's "other components and cooling" multiplier —
            0.30 liquid-cooled, 0.50 air-cooled.
    """

    model_config = ConfigDict(extra="allow")

    provisioned_power_w: float | None = Field(
        default=None,
        validation_alias=AliasChoices("provisioned_power_w", "provisioned_power_watts"),
    )
    cpu: ComponentGroup = Field(default_factory=ComponentGroup)
    accelerator: ComponentGroup = Field(default_factory=ComponentGroup)
    scale_up_network: ComponentGroup = Field(default_factory=ComponentGroup)
    scale_out_network: ComponentGroup = Field(default_factory=ComponentGroup)
    overhead_fraction: float | None = None

    @property
    def major_components_w(self) -> float | None:
        """CPU + accelerator + scale-up (+ scale-out), or None if nothing is stated.

        §4.5.2 makes scale-out optional, so its absence is not a gap; a group with no
        numbers simply contributes nothing.
        """
        parts = [
            g.total_w
            for g in (self.cpu, self.accelerator, self.scale_up_network, self.scale_out_network)
        ]
        stated = [p for p in parts if p is not None]
        return sum(stated) if stated else None

    @property
    def derived_power_w(self) -> float | None:
        """Total provisioned power in watts, declared directly or built up (§4.5.2).

        A directly declared figure wins: §4.5.2 offers it precisely so a submitter can
        override an estimate they consider too high.
        """
        if self.provisioned_power_w is not None:
            return self.provisioned_power_w
        major = self.major_components_w
        if major is None:
            return None
        return major * (1.0 + (self.overhead_fraction or 0.0))

    @property
    def provisioned_power_kw(self) -> float | None:
        """Provisioned power in kilowatts — §4.5.3's normalisation denominator."""
        watts = self.derived_power_w
        return None if watts is None else watts / _W_PER_KW

    @property
    def missing_groups(self) -> list[str]:
        """Component groups the submitter left for MLCommons to auto-populate.

        §4.5.2: omitting a value "triggers the estimated-power tag". Scale-out is
        excluded — §4.5.2 marks it optional, used only by multi-node submissions.
        """
        if self.provisioned_power_w is not None:
            return []
        return [
            name
            for name, group in (
                ("cpu", self.cpu),
                ("accelerator", self.accelerator),
                ("scale_up_network", self.scale_up_network),
            )
            if group.total_w is None
        ]
