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

    def test_required_bandwidth_inconsistent_with_nics_warns(self) -> None:
        data = _edit(DGX_CLUSTER)
        data["scale_out"]["required_bandwidth_tbps"] = 32.0
        out = _compute(data)
        assert out.warnings and "64 Tb/s" in out.warnings[0]


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


@pytest.mark.unit
class TestNormalizedMetric:
    """§4.5.3: system_tps_per_kw = system_tps / provisioned_power_kw."""

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

    def test_denominator_is_constant_across_the_curve(self, tmp_path: Path) -> None:
        """§4.5.3: a low-concurrency point is normalised by the *full* provisioned power."""
        report = SubmissionChecker(_copy(tmp_path)).run()
        kws = {
            r.message.split("/")[-1].strip().removesuffix(" kW)")
            for r in _hits(report, "metric-consistency-tps-per-kw", Severity.INFO)
        }
        assert len(kws) == 1, f"denominator varies across the curve: {kws}"
