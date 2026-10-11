# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for provisioned power (§4.5), its Appendix E descriptor, and its normalized metric.

The expected figures come from the rules, not from this implementation: E.6's worked
descriptors and C.11's multi-node example each state their total, and the arithmetic
here has to land on it.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from submission_checker.checker import SubmissionChecker
from submission_checker.models import Severity, SystemPower
from submission_checker.power_defaults import (
    accelerator_default,
    cpu_default,
    nic_default,
    switch_default,
)

from .conftest import TEST_SUBMISSIONS


def _sv(value: float, unit: str = "value_w", source_type: str = "vendor_spec") -> dict[str, Any]:
    source = "D.4" if source_type == "mlc_default" else "https://example.com/spec"
    return {unit: value, "source_type": source_type, "source": source}


#: E.6.1 — the AMD MI455X Helios rack, every component on an Appendix D default.
HELIOS: dict[str, Any] = {
    "system_desc_id": "helios_mi455x",
    "cooling": "liquid",
    "node_sets": [
        {
            "node_set_id": 0,
            "system_node_ensemble_id": 0,
            "nodes_provisioned": 1,
            "power_method": "component_sum",
            "components": {
                "cpu": {
                    "model": "AMD Venice 256-core",
                    "count_per_node": 18,
                    "tdp_per_unit": {"value_w": 500, "source_type": "mlc_default", "source": "D.2"},
                },
                "accelerator": {
                    "model": "AMD Instinct MI455X",
                    "count_per_node": 72,
                    "tdp_per_unit": {
                        "value_w": 2500,
                        "source_type": "mlc_default",
                        "source": "D.3",
                    },
                },
                "scale_up_network": {
                    "method": "declared_tdp",
                    "switch_count": 6,
                    "tdp_per_switch": _sv(7000, source_type="publication"),
                },
            },
        }
    ],
    "scale_out": {"present": False},
    "computed": {
        "major_components_w": 231000,
        "overhead_fraction": 0.30,
        "other_components_w": 69300,
        "published_node_power_w": 0,
        "scale_out_switch_power_w": 0,
        "total_system_power_w": 300300,
    },
    "provisioned_power_kw": 300.30,
    "mlc_estimated_power": True,
}

#: E.6.2 — one DGX B300 node: a 15.11 kW component sum under a published 14.5 kW.
DGX_B300: dict[str, Any] = {
    "system_desc_id": "dgx_b300_1node",
    "cooling": "air",
    "node_sets": [
        {
            "node_set_id": 0,
            "system_node_ensemble_id": 0,
            "nodes_provisioned": 1,
            "power_method": "component_sum",
            "components": {
                "cpu": {
                    "model": "Intel Xeon 6776P",
                    "count_per_node": 2,
                    "tdp_per_unit": _sv(350),
                },
                "accelerator": {
                    "model": "NVIDIA B300",
                    "count_per_node": 8,
                    "tdp_per_unit": _sv(1100),
                },
                "scale_up_network": {
                    "method": "bandwidth_estimate",
                    "switch_count": 2,
                    "aggregate_bandwidth_tbps": 14.4,
                    "energy_per_bit_pj": {
                        "value_pj": 5,
                        "source_type": "mlc_default",
                        "source": "D.1",
                    },
                },
            },
        }
    ],
    "scale_out": {"present": False},
    "declared_provisioned_power": _sv(14.50, "value_kw"),
    "computed": {
        "major_components_w": 10076,
        "overhead_fraction": 0.50,
        "other_components_w": 5038,
        "published_node_power_w": 0,
        "scale_out_switch_power_w": 0,
        "total_system_power_w": 15114,
    },
    "provisioned_power_kw": 14.50,
    "mlc_estimated_power": False,
}

#: E.6.3 — ten DGX B300 nodes on their published figure, joined over Ethernet.
DGX_CLUSTER: dict[str, Any] = {
    "system_desc_id": "dgx_b300_10node_eth",
    "cooling": "air",
    "node_sets": [
        {
            "node_set_id": 0,
            "system_node_ensemble_id": 0,
            "nodes_provisioned": 10,
            "power_method": "published_system",
            "published_power": _sv(14.50, "value_kw"),
        }
    ],
    "scale_out": {
        "present": True,
        "cabling": "passive",
        "required_bandwidth_tbps": 64.0,
        "nics": {"count_per_node": 8, "bandwidth_per_nic_gbps": 800, "counted": False},
        "switches": [
            {
                "model": "NVIDIA Spectrum SN5610",
                "count": 2,
                "bandwidth_tbps": 51.2,
                "power_per_switch": _sv(900, source_type="mlc_default"),
            }
        ],
    },
    "computed": {
        "major_components_w": 0,
        "overhead_fraction": 0.50,
        "other_components_w": 0,
        "published_node_power_w": 145000,
        "scale_out_switch_power_w": 1800,
        "total_system_power_w": 146800,
    },
    "provisioned_power_kw": 146.80,
    "mlc_estimated_power": True,
}


def _compute(data: dict[str, Any], **kwargs: Any):
    return SystemPower.model_validate(data).compute(**kwargs)


def _edit(data: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(data)


@pytest.mark.unit
class TestWorkedDescriptors:
    """E.6 and C.11 each state a total; the arithmetic must reproduce it."""

    def test_component_sum_on_defaults(self) -> None:
        """E.6.1: 231,000 W majors x 1.30 = 300.30 kW, tagged as estimated."""
        out = _compute(HELIOS)
        assert out.provisioned_power_kw == 300.30
        assert not out.problems
        assert len(out.estimated) == 2  # the D.2 CPU and the D.3 accelerator

    def test_declared_figure_governs_and_its_defaults_do_not_tag(self) -> None:
        """E.6.2: the D.1 default sits under a declared figure, so it tags nothing."""
        out = _compute(DGX_B300)
        assert out.provisioned_power_kw == 14.50
        assert out.total_system_power_w == pytest.approx(15114.0)
        assert not out.problems
        assert not out.estimated

    def test_published_nodes_and_switches_sit_outside_the_overhead(self) -> None:
        """E.6.3: 10 x 14.5 kW + 2 x 900 W = 146.80 kW, with no 1.50 applied to either."""
        out = _compute(DGX_CLUSTER)
        assert out.provisioned_power_kw == 146.80
        assert out.other_components_w == 0
        assert not out.problems
        assert out.estimated  # the D.4 switch

    def test_active_optical_cabling_selects_the_other_figure(self) -> None:
        """E.6.3's note: active optical gives 149.16 kW. Left absent, D.4 supplies it."""
        data = _edit(DGX_CLUSTER)
        data["scale_out"]["cabling"] = "active_optical"
        del data["scale_out"]["switches"][0]["power_per_switch"]
        del data["computed"], data["provisioned_power_kw"]
        out = _compute(data)
        assert out.provisioned_power_kw == 149.16
        assert any("auto-populated from D.4" in e for e in out.estimated)

    def test_node_scaling_per_homogeneous_set(self) -> None:
        """E.6.4: 120 x 5/8 + 96 x 3/8 = 111.00 kW."""
        data = {
            "system_desc_id": "mixed_rack",
            "cooling": "liquid",
            "scale_out": {"present": False},
            "node_sets": [
                {
                    "node_set_id": 0,
                    "system_node_ensemble_id": 0,
                    "nodes_provisioned": 5,
                    "nodes_in_published_rack": 8,
                    "power_method": "node_scaling",
                    "published_power": _sv(120.0, "value_kw"),
                },
                {
                    "node_set_id": 1,
                    "system_node_ensemble_id": 1,
                    "nodes_provisioned": 3,
                    "nodes_in_published_rack": 8,
                    "power_method": "node_scaling",
                    "published_power": _sv(96.0, "value_kw"),
                },
            ],
        }
        assert _compute(data).provisioned_power_kw == 111.00

    def test_formula_nodes_count_their_nics_as_majors(self) -> None:
        """C.11: (700 + 8,800 + 576 + 8 x 75) x 1.50 per node, x 10, + 1.80 kW = 161.94."""
        data = _edit(DGX_B300)
        data["node_sets"][0]["nodes_provisioned"] = 10
        for key in ("declared_provisioned_power", "computed", "provisioned_power_kw"):
            del data[key]
        data["scale_out"] = _edit(DGX_CLUSTER["scale_out"])
        data["scale_out"]["nics"]["counted"] = True
        out = _compute(data)
        assert out.provisioned_power_kw == 161.94
        assert not out.problems


@pytest.mark.unit
class TestScaleOutRules:
    def test_formula_path_must_count_nics(self) -> None:
        """§4.5.2: the formula "has no NIC term", so the NICs MUST be included."""
        data = _edit(DGX_B300)
        del data["declared_provisioned_power"], data["computed"], data["provisioned_power_kw"]
        data["node_sets"][0]["nodes_provisioned"] = 10
        data["scale_out"] = _edit(DGX_CLUSTER["scale_out"])
        out = _compute(data)
        assert any("MUST be included" in p for p in out.problems)

    def test_published_path_must_not_count_nics_twice(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["scale_out"]["nics"]["counted"] = True
        out = _compute(data)
        assert any("excluded_from_published_power" in p for p in out.problems)

    def test_evidenced_exclusion_permits_counting(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["scale_out"]["nics"]["counted"] = True
        data["scale_out"]["nics"]["excluded_from_published_power"] = "https://example.com/x"
        del data["computed"], data["provisioned_power_kw"]
        out = _compute(data)
        assert not out.problems
        # 80 NICs x 75 W (D.4), as majors, x 1.50.
        assert out.major_components_w == pytest.approx(6000.0)

    def test_switches_must_carry_the_required_bandwidth(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["scale_out"]["switches"][0]["count"] = 1
        out = _compute(data)
        assert any("below the required" in p for p in out.problems)

    def test_cabling_is_mandatory_when_present(self) -> None:
        data = _edit(DGX_CLUSTER)
        del data["scale_out"]["cabling"]
        out = _compute(data)
        assert any("cabling" in p for p in out.problems)
        assert out.provisioned_power_kw is None

    def test_understated_required_bandwidth_is_rejected(self) -> None:
        """E.4 defines the requirement as the NICs' sum; 10 Tb/s would admit one SN4700."""
        data = _edit(DGX_CLUSTER)
        data["scale_out"]["required_bandwidth_tbps"] = 10.0
        data["scale_out"]["switches"] = [
            {
                "model": "NVIDIA Spectrum SN4700",
                "count": 1,
                "bandwidth_tbps": 12.8,
                "power_per_switch": _sv(630, source_type="mlc_default"),
            }
        ]
        out = _compute(data)
        assert out.provisioned_power_kw is None
        assert any("below the 64 Tb/s of NIC bandwidth" in p for p in out.problems)
        # The switches are held to the derived requirement, not the declared one.
        assert any("below the required 64 Tb/s" in p for p in out.problems)

    def test_overstated_required_bandwidth_warns(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["scale_out"]["required_bandwidth_tbps"] = 110.0
        out = _compute(data)
        assert any("above the 64 Tb/s" in w for w in out.warnings)
        # A declared figure above the NICs' sum still binds: 2 x 51.2 Tb/s falls short.
        assert any("102.4 Tb/s, below the required 110 Tb/s" in p for p in out.problems)

    def test_mixed_sets_count_nics_on_formula_nodes_only(self) -> None:
        """§4.5.2: published node figures already include their NICs."""
        data = _edit(DGX_CLUSTER)
        data["node_sets"][0]["nodes_provisioned"] = 5
        formula = _edit(DGX_B300["node_sets"][0])
        formula.update(node_set_id=1, system_node_ensemble_id=1, nodes_provisioned=5)
        data["node_sets"].append(formula)
        data["scale_out"]["nics"]["counted"] = True
        del data["computed"], data["provisioned_power_kw"]
        out = _compute(data)
        assert not out.problems
        shares = {s.node_set_id: s.power_w for s in out.sets}
        assert shares[0] == pytest.approx(5 * 14500)
        # (700 + 8,800 + 576 + 8 x 75) x 1.50 per node, as in C.11's formula path.
        assert shares[1] == pytest.approx(5 * 16014)

    def test_evidenced_exclusion_counts_nics_on_published_nodes_too(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["node_sets"][0]["nodes_provisioned"] = 5
        formula = _edit(DGX_B300["node_sets"][0])
        formula.update(node_set_id=1, system_node_ensemble_id=1, nodes_provisioned=5)
        data["node_sets"].append(formula)
        data["scale_out"]["nics"].update(
            counted=True, excluded_from_published_power="https://example.com/x"
        )
        del data["computed"], data["provisioned_power_kw"]
        shares = {s.node_set_id: s.power_w for s in _compute(data).sets}
        assert shares[0] == pytest.approx(5 * (14500 + 8 * 75 * 1.5))


@pytest.mark.unit
class TestFabricDeclared:
    """E.4: ``present`` is false for a single node, or nodes joined by scale-up."""

    def _formula_nodes(self, nodes: int, scale_up: str) -> dict[str, Any]:
        data = _edit(DGX_B300)
        del data["declared_provisioned_power"], data["computed"], data["provisioned_power_kw"]
        data["node_sets"][0]["nodes_provisioned"] = nodes
        if scale_up == "none":
            data["node_sets"][0]["components"]["scale_up_network"] = {"method": "none"}
        return data

    def test_unjoined_nodes_without_a_fabric_are_rejected(self) -> None:
        out = _compute(self._formula_nodes(4, "none"))
        assert out.provisioned_power_kw is None
        assert any("no scale-up network joins them" in p for p in out.problems)

    def test_nodes_with_a_scale_up_network_warn(self) -> None:
        """C.2's rack is legitimate, but the descriptor cannot show the fabric spans nodes."""
        out = _compute(self._formula_nodes(4, "bandwidth_estimate"))
        assert not out.problems
        assert any("cannot show" in w for w in out.warnings)

    def test_published_nodes_without_a_fabric_warn(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["scale_out"] = {"present": False}
        del data["computed"], data["provisioned_power_kw"]
        out = _compute(data)
        assert not out.problems
        assert any("cannot show" in w for w in out.warnings)

    def test_single_node_needs_no_fabric(self) -> None:
        out = _compute(self._formula_nodes(1, "none"))
        assert not out.problems and not out.warnings

    def test_single_node_with_a_fabric_warns(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["node_sets"][0]["nodes_provisioned"] = 1
        data["scale_out"]["required_bandwidth_tbps"] = 6.4
        del data["computed"], data["provisioned_power_kw"]
        out = _compute(data)
        assert not out.problems
        assert any("single-node" in w for w in out.warnings)


@pytest.mark.unit
class TestNodeSetRules:
    @pytest.mark.parametrize("n", [None, 5, 4])
    def test_node_scaling_needs_a_larger_rack(self, n: int | None) -> None:
        """E.7: N absent, or not greater than Y, is a rejection."""
        data = {
            "system_desc_id": "x",
            "cooling": "air",
            "scale_out": {"present": False},
            "node_sets": [
                {
                    "node_set_id": 0,
                    "system_node_ensemble_id": 0,
                    "nodes_provisioned": 5,
                    "nodes_in_published_rack": n,
                    "power_method": "node_scaling",
                    "published_power": _sv(100.0, "value_kw"),
                }
            ],
        }
        assert _compute(data).problems

    def test_published_path_needs_a_published_figure(self) -> None:
        data = _edit(DGX_CLUSTER)
        del data["node_sets"][0]["published_power"]
        assert any("requires published_power" in p for p in _compute(data).problems)

    def test_node_set_ids_are_unique(self) -> None:
        data = _edit(HELIOS)
        data["node_sets"].append(_edit(data["node_sets"][0]))
        assert any("unique" in p for p in _compute(data).problems)

    def test_combined_figure_excludes_the_separate_tdps(self) -> None:
        data = _edit(HELIOS)
        data["node_sets"][0]["components"]["combined_cpu_accelerator"] = _sv(200000)
        assert any("mutually exclusive" in p for p in _compute(data).problems)

    def test_combined_figure_replaces_cpu_and_accelerator(self) -> None:
        data = _edit(HELIOS)
        components = data["node_sets"][0]["components"]
        del components["cpu"], components["accelerator"]
        components["combined_cpu_accelerator"] = _sv(189000)
        del data["computed"], data["provisioned_power_kw"]
        # (189,000 + 42,000) x 1.30 — the same 300.30 kW as the separate blocks.
        assert _compute(data).provisioned_power_kw == 300.30


@pytest.mark.unit
class TestDefaultsAndGaps:
    def test_absent_value_is_auto_populated_and_tagged(self) -> None:
        data = _edit(HELIOS)
        del data["node_sets"][0]["components"]["accelerator"]["tdp_per_unit"]
        out = _compute(data)
        assert out.provisioned_power_kw == 300.30
        assert any("auto-populated from D.3" in e for e in out.estimated)

    def test_cpu_default_uses_the_system_description_core_count(self) -> None:
        data = _edit(DGX_B300)
        data["node_sets"][0]["components"]["cpu"]["model"] = "Intel Xeon 6776P"
        del data["node_sets"][0]["components"]["cpu"]["tdp_per_unit"]
        assert _compute(data, cores_by_ensemble={0: 64}).provisioned_power_kw == 14.50
        # Without a core count there is nothing to choose a D.2 row by — but the
        # declared figure governs, so the gap does not reject.
        assert not _compute(data).problems

    def test_gap_with_no_default_rejects_when_it_reaches_the_total(self) -> None:
        data = _edit(HELIOS)
        accelerator = data["node_sets"][0]["components"]["accelerator"]
        accelerator["model"] = "Some Future Accelerator"
        del accelerator["tdp_per_unit"]
        out = _compute(data)
        assert any("no default" in p for p in out.problems)
        assert out.provisioned_power_kw is None

    def test_submitter_estimated_flag_is_ignored(self) -> None:
        data = _edit(HELIOS)
        data["mlc_estimated_power"] = False
        assert _compute(data).estimated


def _default(value: float, source: str, unit: str = "value_w") -> dict[str, Any]:
    return {unit: value, "source_type": "mlc_default", "source": source}


def _formula_b300() -> dict[str, Any]:
    """E.6.2's DGX B300 on its component sum alone, so every value reaches the total."""
    data = _edit(DGX_B300)
    for key in ("declared_provisioned_power", "computed", "provisioned_power_kw"):
        del data[key]
    return data


def _counted_nics() -> dict[str, Any]:
    """C.11's formula path: ten DGX B300 nodes counting their NICs, D.4 switches."""
    data = _formula_b300()
    data["node_sets"][0]["nodes_provisioned"] = 10
    data["scale_out"] = _edit(DGX_CLUSTER["scale_out"])
    data["scale_out"]["nics"]["counted"] = True
    return data


def _no_computed(data: dict[str, Any]) -> dict[str, Any]:
    data.pop("computed", None)
    data.pop("provisioned_power_kw", None)
    return data


@pytest.mark.unit
class TestAppendixDDefaultsArePinned:
    """E.1: an mlc_default is "its own reference" — the Appendix D figure, from its subsection."""

    # D.3 — accelerators, keyed by model.
    def test_accelerator_default_matching_d3_passes(self) -> None:
        out = _compute(HELIOS)
        assert not out.problems and not out.warnings

    def test_accelerator_default_with_the_wrong_value_fails(self) -> None:
        """The gap this closes: a B300 at 500 W "from D.3", where D.3 gives 1,100 W."""
        data = _formula_b300()
        data["node_sets"][0]["components"]["accelerator"]["tdp_per_unit"] = _default(500, "D.3")
        out = _compute(data)
        assert out.provisioned_power_kw is None
        assert any("D.3 gives 1100 W for NVIDIA B300" in p for p in out.problems)

    def test_accelerator_default_from_the_wrong_subsection_fails(self) -> None:
        data = _no_computed(_edit(HELIOS))
        data["node_sets"][0]["components"]["accelerator"]["tdp_per_unit"]["source"] = "D.2"
        out = _compute(data)
        assert out.provisioned_power_kw is None
        assert any("cites D.2" in p and "in D.3" in p for p in out.problems)

    def test_unlisted_accelerator_has_no_default_to_cite(self) -> None:
        data = _formula_b300()
        accelerator = data["node_sets"][0]["components"]["accelerator"]
        accelerator["model"] = "NVIDIA H100"
        accelerator["tdp_per_unit"] = _default(700, "D.3")
        out = _compute(data)
        assert any("D.3 has no default for NVIDIA H100" in p for p in out.problems)

    # D.2 — processors, keyed by architecture and core count.
    def test_cpu_default_matching_d2_passes(self) -> None:
        data = _formula_b300()
        data["node_sets"][0]["components"]["cpu"]["tdp_per_unit"] = _default(350, "D.2")
        out = _compute(data, cores_by_ensemble={0: 64})
        assert not out.problems and not out.warnings

    def test_cpu_default_with_the_wrong_value_fails(self) -> None:
        """A 256-core x86 part is D.2's > 64-core row: 500 W, not 350 W."""
        data = _no_computed(_edit(HELIOS))
        data["node_sets"][0]["components"]["cpu"]["tdp_per_unit"]["value_w"] = 350
        out = _compute(data)
        assert any("D.2 gives 500 W for AMD Venice 256-core" in p for p in out.problems)

    def test_cpu_default_from_the_wrong_subsection_fails(self) -> None:
        data = _edit(HELIOS)
        data["node_sets"][0]["components"]["cpu"]["tdp_per_unit"]["source"] = "D.3"
        assert any("in D.2" in p for p in _compute(data).problems)

    def test_arm_above_128_cores_has_no_default_to_cite(self) -> None:
        """D.2's open item: no ARM default above 128 cores."""
        data = _formula_b300()
        cpu = data["node_sets"][0]["components"]["cpu"]
        cpu["model"] = "NVIDIA Grace"
        cpu["tdp_per_unit"] = _default(300, "D.2")
        out = _compute(data, cores_by_ensemble={0: 144})
        assert any("D.2 has no default for NVIDIA Grace with 144 cores" in p for p in out.problems)

    def test_cpu_default_whose_row_cannot_be_established_warns(self) -> None:
        """No core count: 350 W is a D.2 figure, but which row applies is unknown."""
        data = _formula_b300()
        data["node_sets"][0]["components"]["cpu"]["tdp_per_unit"] = _default(350, "D.2")
        out = _compute(data)
        assert not out.problems
        assert any("could not be established" in w for w in out.warnings)

    def test_cpu_default_that_is_no_d2_figure_fails_even_unconfirmed(self) -> None:
        data = _formula_b300()
        data["node_sets"][0]["components"]["cpu"]["tdp_per_unit"] = _default(400, "D.2")
        out = _compute(data)
        assert any("D.2 gives 350 W or 500 W" in p for p in out.problems)

    # D.1 — scale-up network.
    def test_energy_per_bit_default_matching_d1_passes(self) -> None:
        out = _compute(_formula_b300())
        assert not out.problems and not out.warnings

    def test_energy_per_bit_default_with_the_wrong_value_fails(self) -> None:
        data = _formula_b300()
        up = data["node_sets"][0]["components"]["scale_up_network"]
        up["energy_per_bit_pj"]["value_pj"] = 3
        out = _compute(data)
        assert any("D.1 gives 5 pJ/bit" in p for p in out.problems)

    def test_energy_per_bit_default_from_the_wrong_subsection_fails(self) -> None:
        data = _formula_b300()
        up = data["node_sets"][0]["components"]["scale_up_network"]
        up["energy_per_bit_pj"]["source"] = "D.4"
        assert any("in D.1" in p for p in _compute(data).problems)

    def test_switch_tdp_from_d1_is_checked_by_subsection_and_warned(self) -> None:
        """D.1's Ethernet point scales with bandwidth the descriptor does not carry."""
        data = _no_computed(_edit(HELIOS))
        up = data["node_sets"][0]["components"]["scale_up_network"]
        up["tdp_per_switch"] = _default(1750, "D.1")
        out = _compute(data)
        assert not out.problems
        assert any("could not be checked" in w for w in out.warnings)
        up["tdp_per_switch"]["source"] = "D.4"
        assert any("in D.1" in p for p in _compute(data).problems)

    # D.4 — reference scale-out switches, keyed by model and cabling, and NICs.
    def test_switch_default_matching_d4_passes(self) -> None:
        out = _compute(DGX_CLUSTER)
        assert not out.problems

    def test_switch_default_with_the_wrong_value_fails(self) -> None:
        data = _no_computed(_edit(DGX_CLUSTER))
        data["scale_out"]["cabling"] = "active_optical"  # D.4: 2.08 kW, not 900 W
        out = _compute(data)
        assert any(
            "D.4 gives 2080 W for NVIDIA Spectrum SN5610 with active_optical cabling" in p
            for p in out.problems
        )

    def test_switch_default_from_the_wrong_subsection_fails(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["scale_out"]["switches"][0]["power_per_switch"]["source"] = "D.1"
        assert any("in D.4" in p for p in _compute(data).problems)

    def test_switch_without_a_figure_for_its_cabling_has_no_default(self) -> None:
        """D.4: the SN4700 publishes no active-optical figure."""
        data = _no_computed(_edit(DGX_CLUSTER))
        data["scale_out"]["cabling"] = "active_optical"
        data["scale_out"]["switches"] = [
            {
                "model": "NVIDIA Spectrum SN4700",
                "count": 5,
                "bandwidth_tbps": 12.8,
                "power_per_switch": _default(630, "D.4"),
            }
        ]
        assert any("D.4 has no default" in p for p in _compute(data).problems)

    def test_nic_default_matching_d4_passes(self) -> None:
        data = _counted_nics()
        data["scale_out"]["nics"]["tdp_per_nic"] = _default(75, "D.4")
        out = _compute(data)
        assert not out.problems
        assert out.provisioned_power_kw == 161.94

    def test_nic_default_with_the_wrong_value_fails(self) -> None:
        data = _counted_nics()
        data["scale_out"]["nics"]["tdp_per_nic"] = _default(25, "D.4")
        assert any("D.4 gives 75 W for a scale-out NIC" in p for p in _compute(data).problems)

    def test_nic_default_from_the_wrong_subsection_fails(self) -> None:
        data = _counted_nics()
        data["scale_out"]["nics"]["tdp_per_nic"] = _default(75, "D.2")
        assert any("in D.4" in p for p in _compute(data).problems)

    # Fields Appendix D gives no default for.
    def test_published_power_has_no_appendix_d_default(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["node_sets"][0]["published_power"] = _default(14.5, "D.4", "value_kw")
        out = _compute(data)
        assert any("defines no default" in p for p in out.problems)

    def test_declared_provisioned_power_has_no_appendix_d_default(self) -> None:
        data = _edit(DGX_B300)
        data["declared_provisioned_power"] = _default(14.5, "D.3", "value_kw")
        out = _compute(data)
        assert out.provisioned_power_kw is None
        assert any("declared_provisioned_power" in p for p in out.problems)

    # E.6.2: under a governing declared figure the component block is a cross-check.
    def test_misquoted_default_under_a_declared_figure_only_warns(self) -> None:
        data = _edit(DGX_B300)
        data["node_sets"][0]["components"]["accelerator"]["tdp_per_unit"] = _default(500, "D.3")
        del data["computed"]
        out = _compute(data)
        assert not out.problems
        assert out.provisioned_power_kw == 14.50
        assert any("D.3 gives 1100 W" in w for w in out.warnings)


@pytest.mark.unit
class TestComputedBlock:
    def test_disagreement_is_a_rejection(self) -> None:
        """E.7: the last case "is a rejection rather than a silent correction"."""
        data = _edit(HELIOS)
        data["computed"]["major_components_w"] = 200000
        assert any("computed.major_components_w" in p for p in _compute(data).problems)

    def test_wrong_overhead_is_a_rejection(self) -> None:
        data = _edit(HELIOS)
        data["computed"]["overhead_fraction"] = 0.5
        assert any("overhead_fraction" in p for p in _compute(data).problems)

    def test_wrong_provisioned_power_is_a_rejection(self) -> None:
        data = _edit(HELIOS)
        data["provisioned_power_kw"] = 250.0
        assert any("provisioned_power_kw" in p for p in _compute(data).problems)

    def test_rounding_is_to_two_places(self) -> None:
        data = _edit(HELIOS)
        data["node_sets"][0]["components"]["cpu"]["count_per_node"] = 17
        del data["computed"], data["provisioned_power_kw"]
        # (230,500) x 1.30 = 299,650 W
        assert _compute(data).provisioned_power_kw == 299.65


@pytest.mark.unit
class TestStructure:
    """What fails to load at all — E.7's required fields and E.1's sourced values."""

    @pytest.mark.parametrize("field", ["system_desc_id", "cooling", "node_sets", "scale_out"])
    def test_required_top_level_fields(self, field: str) -> None:
        data = _edit(HELIOS)
        del data[field]
        with pytest.raises(ValidationError):
            SystemPower.model_validate(data)

    def test_self_declaration_is_not_a_source_type(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["node_sets"][0]["published_power"]["source_type"] = "self_declared"
        with pytest.raises(ValidationError):
            SystemPower.model_validate(data)

    def test_verifiable_sources_are_urls(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["node_sets"][0]["published_power"]["source"] = "our datasheet"
        with pytest.raises(ValidationError, match="resolvable URL"):
            SystemPower.model_validate(data)

    def test_defaults_name_their_appendix_d_subsection(self) -> None:
        data = _edit(HELIOS)
        data["node_sets"][0]["components"]["cpu"]["tdp_per_unit"]["source"] = "D.9"
        with pytest.raises(ValidationError, match="Appendix D"):
            SystemPower.model_validate(data)

    def test_a_sourced_value_carries_exactly_one_value(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["node_sets"][0]["published_power"]["value_w"] = 14500
        with pytest.raises(ValidationError, match="exactly one"):
            SystemPower.model_validate(data)

    @pytest.mark.parametrize("nodes", [0, -1, 4.5])
    def test_nodes_provisioned_is_a_positive_whole_number(self, nodes: float) -> None:
        """§4.5.2.1: 4.5 nodes are declared as 5."""
        data = _edit(HELIOS)
        data["node_sets"][0]["nodes_provisioned"] = nodes
        with pytest.raises(ValidationError):
            SystemPower.model_validate(data)

    def test_cooling_is_liquid_or_air(self) -> None:
        data = _edit(HELIOS)
        data["cooling"] = "passive"
        with pytest.raises(ValidationError):
            SystemPower.model_validate(data)


@pytest.mark.unit
class TestAppendixD:
    @pytest.mark.parametrize(
        ("model", "cores", "watts"),
        [
            ("Intel Xeon 6776P", 64, 350.0),
            ("AMD EPYC 9654", 96, 500.0),
            ("AMD Venice 256-core", None, 500.0),
            ("NVIDIA Grace CPU (Armv9)", 72, 300.0),
        ],
    )
    def test_cpu(self, model: str, cores: int | None, watts: float) -> None:
        found = cpu_default(model, cores)
        assert found is not None and found.watts == watts

    @pytest.mark.parametrize(
        ("model", "cores"),
        [("NVIDIA Grace", 144), ("Mystery CPU", 32), ("Intel Xeon", None)],
    )
    def test_cpu_without_a_default(self, model: str, cores: int | None) -> None:
        """ARM above 128 cores is D.2's open item; the others give no row to pick."""
        assert cpu_default(model, cores) is None

    @pytest.mark.parametrize(
        ("model", "watts"),
        [("NVIDIA GB300", 1400.0), ("NVIDIA B300", 1100.0), ("AMD Instinct MI355X", 1400.0)],
    )
    def test_accelerator(self, model: str, watts: float) -> None:
        found = accelerator_default(model)
        assert found is not None and found.watts == watts

    def test_unlisted_accelerator(self) -> None:
        assert accelerator_default("NVIDIA H100") is None

    def test_switch_by_cabling(self) -> None:
        assert switch_default("SN5610", "active_optical").watts == 2080.0  # type: ignore[union-attr]
        # D.4: no published active-optical figure, so no reference for that case.
        assert switch_default("SN4700", "active_optical") is None

    def test_nic(self) -> None:
        assert nic_default().watts == 75.0


def _copy(tmp_path: Path) -> Path:
    dest = tmp_path / "sub"
    shutil.copytree(TEST_SUBMISSIONS / "valid_standardized", dest)
    return dest


def _power_files(root: Path) -> list[Path]:
    return list(root.rglob("system_power.json"))


def _rewrite(root: Path, edit) -> None:
    for path in _power_files(root):
        data = json.loads(path.read_text())
        edit(data)
        path.write_text(json.dumps(data))


def _hits(report, rule: str, severity: Severity | None = None):
    return [
        r for r in report.results if r.rule == rule and (severity is None or r.severity == severity)
    ]


@pytest.mark.unit
class TestPowerDescriptorRule:
    """§9.1 "Power descriptor": one per system, valid under E.7, or rejected."""

    def test_present_and_valid_passes(self, tmp_path: Path) -> None:
        report = SubmissionChecker(_copy(tmp_path)).run()
        hits = _hits(report, "power-descriptor")
        assert hits and all(r.severity == Severity.INFO for r in hits)
        # (2 x 350 + 8 x 700 + 3,500) x 1.50, air-cooled.
        assert any("14.70 kW" in r.message for r in hits)

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        for path in _power_files(root):
            path.unlink()
        assert _hits(SubmissionChecker(root).run(), "power-descriptor", Severity.ERROR)

    def test_unparseable_file_is_rejected(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        for path in _power_files(root):
            path.write_text("{not json")
        assert _hits(SubmissionChecker(root).run(), "power-descriptor", Severity.ERROR)

    def test_pre_appendix_e_form_is_rejected(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        for path in _power_files(root):
            path.write_text(
                json.dumps({"accelerator": {"num_accelerator": 8, "tdp_per_accelerator": 700}})
            )
        assert _hits(SubmissionChecker(root).run(), "power-descriptor", Severity.ERROR)

    def test_cooling_must_agree_with_the_system_description(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _rewrite(root, lambda d: d.update(cooling="liquid"))
        errors = _hits(SubmissionChecker(root).run(), "power-descriptor", Severity.ERROR)
        assert errors and "agree" in errors[0].message

    def test_d2_default_reads_cores_from_the_system_description(self, tmp_path: Path) -> None:
        """The fixture's AMD EPYC 9654 has 96 cores, so D.2 gives 500 W per CPU."""
        root = _copy(tmp_path)
        _rewrite(root, lambda d: d["node_sets"][0]["components"]["cpu"].pop("tdp_per_unit"))
        report = SubmissionChecker(root).run()
        assert not _hits(report, "power-descriptor", Severity.ERROR)
        assert _hits(report, "power-estimated", Severity.WARNING)
        # (2 x 500 + 5,600 + 3,500) x 1.50
        assert any("15.15 kW" in r.message for r in _hits(report, "power-descriptor"))

    def test_unknown_node_ensemble_warns(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _rewrite(root, lambda d: d["node_sets"][0].update(system_node_ensemble_id=7))
        warnings = _hits(SubmissionChecker(root).run(), "power-descriptor", Severity.WARNING)
        assert warnings and "[7]" in warnings[0].message


def _point_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("point.yaml"))


def _rewrite_desc(root: Path, edit, point: int | None = None) -> None:
    """Apply *edit* to every point's system_desc.json, or to point r<point> only."""
    for point_dir in _point_dirs(root):
        if point is not None and point_dir.name != f"r{point}":
            continue
        path = point_dir / "system_desc.json"
        data = json.loads(path.read_text())
        edit(data)
        path.write_text(json.dumps(data))


def _set_parallelism(root: Path, point: int | None = None, **values: Any) -> None:
    _rewrite_desc(root, lambda d: d["config_summary"].update(values), point)


def _rewrite_point(root: Path, point: int, edit) -> None:
    path = root / "results" / "acme_h100x8_001" / "llama3_1-8b" / f"r{point}" / "point.yaml"
    data = yaml.safe_load(path.read_text())
    edit(data)
    path.write_text(yaml.safe_dump(data))


def _three_nodes(root: Path) -> None:
    """Make the valid fixture three 8-accelerator nodes: 3 x 9,800 W x 1.50 = 44.10 kW.

    Every point runs one TP=16 replica — floor(24 / 16) = 1 — so it engages 16
    accelerators on two of the three nodes, the C.3 shape: a remainder too small for
    another replica.
    """
    _rewrite(root, lambda d: d["node_sets"][0].update(nodes_provisioned=3))

    def desc(d: dict[str, Any]) -> None:
        d["node_types"][0]["number_of_nodes"] = 3
        d["system_node_ensemble_total"] = 3
        d["config_summary"].update(tensor_parallel=16, data_parallel=1)

    _rewrite_desc(root, desc)


def _kw(result) -> str:
    return result.message.split("point_power_kw ")[1].split(",")[0]


@pytest.mark.unit
class TestPointPower:
    """§4.5.3's point_power_kw, against the rules' own worked figures."""

    def test_single_node_is_constant(self) -> None:
        out = _compute(DGX_B300)
        assert out.point_power_kw() == out.point_power_kw({0: 1}) == 14.50

    def test_rack_by_engaged_nodes(self) -> None:
        """C.3: the 152.10 kW C.2 rack, 16 of 18 nodes engaged, divides by 135.20 kW."""
        data = _edit(DGX_CLUSTER)
        data["cooling"] = "liquid"
        data["scale_out"] = {"present": False}
        data["node_sets"][0].update(nodes_provisioned=18, published_power=_sv(8450))
        for key in ("computed", "provisioned_power_kw"):
            del data[key]
        out = _compute(data)
        assert out.provisioned_power_kw == 152.10
        assert out.point_power_kw({0: 16}) == 135.20
        assert out.point_power_kw({0: 4}) == 33.80  # C.3's DP = 1 counterfactual

    def test_node_scaling_composes(self) -> None:
        """C.5: 140 kW x 6/18 = 46.67 kW provisioned; 2 of those 6 nodes give 15.56 kW."""
        data = {
            "system_desc_id": "partial_rack",
            "cooling": "liquid",
            "scale_out": {"present": False},
            "node_sets": [
                {
                    "node_set_id": 0,
                    "system_node_ensemble_id": 0,
                    "nodes_provisioned": 6,
                    "nodes_in_published_rack": 18,
                    "power_method": "node_scaling",
                    "published_power": _sv(140.0, "value_kw"),
                }
            ],
        }
        out = _compute(data)
        assert out.provisioned_power_kw == 46.67
        assert out.point_power_kw({0: 2}) == 15.56

    def test_switch_power_scales_with_the_nodes(self) -> None:
        """§4.5.3: S scales by Σ Y / Σ N — half the E.6.3 cluster is 72.50 + 0.90 kW."""
        out = _compute(DGX_CLUSTER)
        assert out.point_power_kw({0: 5}) == 73.40

    def test_heterogeneous_sets_scale_separately(self) -> None:
        """E.6.4's sets at 75.00 and 36.00 kW: all of the first, a third of the second."""
        data = {
            "system_desc_id": "mixed_rack",
            "cooling": "liquid",
            "scale_out": {"present": False},
            "node_sets": [
                {
                    "node_set_id": i,
                    "system_node_ensemble_id": i,
                    "nodes_provisioned": n,
                    "nodes_in_published_rack": 8,
                    "power_method": "node_scaling",
                    "published_power": _sv(kw, "value_kw"),
                }
                for i, n, kw in ((0, 5, 120.0), (1, 3, 96.0))
            ],
        }
        out = _compute(data)
        assert out.point_power_kw({1: 1}) == 87.00

    def test_set_shares_and_switches_sum_to_the_total(self) -> None:
        """P_s carries its overhead and counted NICs, so Σ P_s + S is C.11's 161.94 kW."""
        data = _edit(DGX_B300)
        data["node_sets"][0]["nodes_provisioned"] = 10
        for key in ("declared_provisioned_power", "computed", "provisioned_power_kw"):
            del data[key]
        data["scale_out"] = _edit(DGX_CLUSTER["scale_out"])
        data["scale_out"]["nics"]["counted"] = True
        out = _compute(data)
        shares = sum(s.power_w for s in out.sets) + out.scale_out_switch_power_w
        assert shares == pytest.approx(out.total_system_power_w)
        assert out.sets[0].accelerators_per_node == 8

    def test_declared_figure_scales_homogeneously(self) -> None:
        """§4.5.3's homogeneous form: provisioned_power_kw x (Y / N)."""
        data = _edit(DGX_CLUSTER)
        data["declared_provisioned_power"] = _sv(140.0, "value_kw")
        del data["computed"], data["provisioned_power_kw"]
        out = _compute(data)
        assert out.point_power_kw({0: 5}) == 70.00

    def test_declared_figure_over_several_sets_takes_the_largest_fraction(self) -> None:
        data = {
            "system_desc_id": "mixed_rack",
            "cooling": "liquid",
            "scale_out": {"present": False},
            "declared_provisioned_power": _sv(100.0, "value_kw"),
            "node_sets": [
                {
                    "node_set_id": i,
                    "system_node_ensemble_id": i,
                    "nodes_provisioned": n,
                    "power_method": "published_system",
                    "published_power": _sv(10.0, "value_kw"),
                }
                for i, n in ((0, 5), (1, 3))
            ],
        }
        out = _compute(data)
        assert out.point_power_kw({0: 4, 1: 1}) == 80.00
        assert out.point_power_kw({0: 4}) == 100.00  # set 1 undeclared: fully engaged


@pytest.mark.unit
class TestNormalizedMetric:
    """§9.1 "Per-point denominator": system_tps_per_kw = system_tps / point_power_kw."""

    def _set_stored(self, root: Path, value: float) -> None:
        for path in root.rglob("result_summary.json"):
            data = json.loads(path.read_text())
            data["system_tps_per_kw"] = value
            path.write_text(json.dumps(data))

    def test_derived_when_nothing_is_stored(self, tmp_path: Path) -> None:
        report = SubmissionChecker(_copy(tmp_path)).run()
        hits = _hits(report, "metric-consistency-tps-per-kw")
        assert hits and all(r.severity == Severity.INFO for r in hits)

    def test_stored_mismatch_errors(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        self._set_stored(root, 999999.0)
        report = SubmissionChecker(root).run()
        assert _hits(report, "metric-consistency-tps-per-kw", Severity.ERROR)

    def test_no_provisioned_power_means_no_normalisation(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _rewrite(root, lambda d: d["scale_out"].update(present=True))
        report = SubmissionChecker(root).run()
        assert not _hits(report, "metric-consistency-tps-per-kw")

    def test_single_node_denominator_is_constant(self, tmp_path: Path) -> None:
        """§4.5.3: with Y = N = 1 every point divides by the full node power."""
        report = SubmissionChecker(_copy(tmp_path)).run()
        kws = {_kw(r) for r in _hits(report, "metric-consistency-tps-per-kw", Severity.INFO)}
        assert kws == {"14.70"}

    def test_undeclared_point_uses_full_power(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _three_nodes(root)
        report = SubmissionChecker(root).run()
        assert not report.errors, [r.message for r in report.errors]
        kws = {_kw(r) for r in _hits(report, "metric-consistency-tps-per-kw", Severity.INFO)}
        assert kws == {"44.10"}

    def test_declared_nodes_scale_the_point(self, tmp_path: Path) -> None:
        """Two of three nodes engaged: 44.10 x 2/3 = 29.40 kW, at that point only."""
        root = _copy(tmp_path)
        _three_nodes(root)
        _rewrite_point(
            root, 16, lambda d: d.update(nodes_used=[{"system_node_ensemble_id": 0, "nodes": 2}])
        )
        report = SubmissionChecker(root).run()
        assert not report.errors, [r.message for r in report.errors]
        by_point = {
            r.path.parent.name: _kw(r)
            for r in _hits(report, "metric-consistency-tps-per-kw", Severity.INFO)
        }
        assert by_point.pop("r16") == "29.40"
        assert set(by_point.values()) == {"44.10"}

    def test_stored_value_checked_against_the_point_denominator(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _three_nodes(root)
        _rewrite_point(
            root, 16, lambda d: d.update(nodes_used=[{"system_node_ensemble_id": 0, "nodes": 2}])
        )
        summary_path = next(p for p in _point_dirs(root) if p.name == "r16") / "result_summary.json"
        summary = json.loads(summary_path.read_text())
        tps = summary["output_sequence_lengths"]["total"] / (summary["duration_ns"] / 1e9)
        summary["system_tps_per_kw"] = round(tps / 44.10, 1)  # the full figure: wrong
        summary_path.write_text(json.dumps(summary))
        errors = _hits(
            SubmissionChecker(root).run(), "metric-consistency-tps-per-kw", Severity.ERROR
        )
        assert errors and "29.40" in errors[0].message


@pytest.mark.unit
class TestNodesUsed:
    """§9.1 "Per-point node declaration"."""

    def _declare(self, root: Path, *entries: tuple[int, int]) -> None:
        _rewrite_point(
            root,
            16,
            lambda d: d.update(
                nodes_used=[{"system_node_ensemble_id": e, "nodes": n} for e, n in entries]
            ),
        )

    @pytest.mark.parametrize(
        ("entries", "fragment"),
        [
            (((0, 4),), "no greater than N_s = 3"),
            (((0, 0),), "positive whole number"),
            (((7, 2),), "matches 0 node sets"),
            (((0, 2), (0, 2)), "declared twice"),
            (((0, 1),), "hold 8 accelerators"),
        ],
    )
    def test_rejected(self, tmp_path: Path, entries, fragment: str) -> None:
        root = _copy(tmp_path)
        _three_nodes(root)
        self._declare(root, *entries)
        report = SubmissionChecker(root).run()
        errors = _hits(report, "nodes-used", Severity.ERROR)
        assert errors and fragment in errors[0].message
        # No denominator is checked against a declaration that did not validate.
        assert not [
            r for r in _hits(report, "metric-consistency-tps-per-kw") if r.path.parent.name == "r16"
        ]

    def test_more_nodes_than_engaged_warns(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _three_nodes(root)
        self._declare(root, (0, 3))
        report = SubmissionChecker(root).run()
        assert not _hits(report, "nodes-used", Severity.ERROR)
        warnings = _hits(report, "nodes-used", Severity.WARNING)
        assert warnings and "ceil(16 / 8) = 2 nodes" in warnings[0].message


@pytest.mark.unit
class TestMaximalEngagement:
    """§9.1 "Maximal engagement": DP = floor(A_provisioned / A_replica)."""

    def test_fully_engaged_passes(self, tmp_path: Path) -> None:
        report = SubmissionChecker(_copy(tmp_path)).run()
        hits = _hits(report, "maximal-engagement")
        assert hits and all(r.severity == Severity.INFO for r in hits)

    def test_shortfall_without_declaration_is_rejected(self, tmp_path: Path) -> None:
        """TP=4 on 8 accelerators fits two replicas; running one is not a valid point."""
        root = _copy(tmp_path)
        _set_parallelism(root, 16, tensor_parallel=4, data_parallel=1)
        errors = _hits(SubmissionChecker(root).run(), "maximal-engagement", Severity.ERROR)
        assert errors and "floor(8 / 4) = 2" in errors[0].message

    def test_declared_shortfall_is_left_to_peer_review(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _set_parallelism(root, 16, tensor_parallel=4, data_parallel=1)
        _rewrite_point(
            root,
            16,
            lambda d: d.update(
                dp_shortfall={"dp_actual": 1, "dp_formula": 2, "reason": "host memory binds"}
            ),
        )
        report = SubmissionChecker(root).run()
        assert not _hits(report, "maximal-engagement", Severity.ERROR)
        warnings = _hits(report, "maximal-engagement", Severity.WARNING)
        assert warnings and "host memory binds" in warnings[0].message

    def test_shortfall_must_match_the_point(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _set_parallelism(root, 16, tensor_parallel=2, data_parallel=1)
        _rewrite_point(
            root,
            16,
            lambda d: d.update(dp_shortfall={"dp_actual": 1, "dp_formula": 2, "reason": "x"}),
        )
        errors = _hits(SubmissionChecker(root).run(), "maximal-engagement", Severity.ERROR)
        assert errors and "runs DP 1 of floor(8 / 2) = 4" in errors[0].message

    def test_more_than_provisioned_is_rejected(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _set_parallelism(root, 16, tensor_parallel=4, data_parallel=3)
        errors = _hits(SubmissionChecker(root).run(), "maximal-engagement", Severity.ERROR)
        assert errors and "more than the 8 provisioned" in errors[0].message

    def test_parallelism_may_vary_between_points(self, tmp_path: Path) -> None:
        """§8.1 / C.3: each point's system_desc.json carries its own parallelism."""
        root = _copy(tmp_path)
        _set_parallelism(root, 16, tensor_parallel=4, data_parallel=2)
        report = SubmissionChecker(root).run()
        assert not report.errors, [r.message for r in report.errors]

    def test_disaggregated_is_left_to_peer_review(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _set_parallelism(root, 16, disaggregated=True, tensor_parallel=1)
        report = SubmissionChecker(root).run()
        assert not _hits(report, "maximal-engagement", Severity.ERROR)
        assert _hits(report, "maximal-engagement", Severity.WARNING)

    def test_undeclared_parallelism_warns(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _rewrite_desc(root, lambda d: d.update(config_summary="free-form summary"), 16)
        report = SubmissionChecker(root).run()
        warnings = _hits(report, "maximal-engagement", Severity.WARNING)
        assert warnings and "declares no" in warnings[0].message


@pytest.mark.unit
class TestDivisionScope:
    """§4.5: required for Standardized; RDI may, Serviced is deferred."""

    @pytest.mark.parametrize("division", ["RDI", "Serviced"])
    def test_optional_outside_standardized(self, tmp_path: Path, division: str) -> None:
        root = _copy(tmp_path)
        _rewrite_desc(root, lambda d: d.update(division=division))
        for path in _power_files(root):
            path.unlink()
        report = SubmissionChecker(root).run()
        assert not _hits(report, "power-descriptor", Severity.ERROR)
        assert _hits(report, "power-descriptor", Severity.INFO)

    def test_supplied_descriptor_is_still_validated(self, tmp_path: Path) -> None:
        root = _copy(tmp_path)
        _rewrite_desc(root, lambda d: d.update(division="RDI"))
        _rewrite(root, lambda d: d.update(cooling="liquid"))
        assert _hits(SubmissionChecker(root).run(), "power-descriptor", Severity.ERROR)
