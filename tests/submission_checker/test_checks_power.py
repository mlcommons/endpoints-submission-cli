# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for provisioned power (§4.5) and its normalized metric.

§4.5.2 publishes no JSON schema — it lists field groups and says MLCommons
auto-populates omissions — so most of what is asserted here is that the checker is
permissive about gaps and strict about arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from submission_checker.checker import SubmissionChecker
from submission_checker.models import Severity, SystemPower

from .conftest import TEST_SUBMISSIONS


@pytest.mark.unit
class TestProvisionedPower:
    def test_component_build_up(self) -> None:
        """major × (1 + overhead); §4.5.2's "other components and cooling"."""
        power = SystemPower.model_validate(
            {
                "cpu": {"count": 2, "tdp_per_unit": 350},  #   700 W
                "accelerator": {"count": 8, "tdp_per_unit": 700},  # 5,600 W
                "scale_up_network": {"count": 1, "tdp_per_unit": 3500},
                "overhead_fraction": 0.5,
            }
        )
        assert power.major_components_w == pytest.approx(9800.0)
        assert power.derived_power_w == pytest.approx(14700.0)
        assert power.provisioned_power_kw == pytest.approx(14.7)

    def test_declared_total_overrides_the_build_up(self) -> None:
        """§4.5.2 offers this so a submitter can override an over-high estimate."""
        power = SystemPower.model_validate(
            {
                "provisioned_power_w": 10000.0,
                "accelerator": {"count": 8, "tdp_per_unit": 700},
                "overhead_fraction": 0.5,
            }
        )
        assert power.provisioned_power_kw == pytest.approx(10.0)

    def test_scale_out_is_not_a_major_component(self) -> None:
        """§4.5.2 puts scale-out inside Other_components, so it must not also be summed
        into the majors — that would count it twice, once explicitly and once via the
        overhead fraction."""
        base = {"accelerator": {"count": 1, "tdp_per_unit": 1000}}
        without = SystemPower.model_validate(base)
        with_scale_out = SystemPower.model_validate(
            {**base, "scale_out_network": {"count": 2, "tdp_per_unit": 500}}
        )
        assert without.major_components_w == pytest.approx(1000.0)
        assert with_scale_out.major_components_w == pytest.approx(1000.0)

    def test_the_power_model_matches_the_spec_formula(self) -> None:
        """System Power = (CPU + Accelerator + Scale-up) x (1 + overhead_fraction)."""
        power = SystemPower.model_validate(
            {
                "cpu": {"num_cpu": 2, "tdp_per_cpu": 350},
                "accelerator": {"num_accelerator": 8, "tdp_per_accelerator": 700},
                "scale_up_network": {"num_switches": 4, "tdp_per_switch": 200},
                "scale_out_network": {"num_switches": 2, "tdp_per_switch": 500},
                "cooling": "Liquid cooled",
            }
        )
        majors = 2 * 350 + 8 * 700 + 4 * 200
        assert power.major_components_w == pytest.approx(majors)
        assert power.derived_power_w == pytest.approx(majors * 1.30)

    def test_no_numbers_yields_no_power(self) -> None:
        assert SystemPower().provisioned_power_kw is None

    def test_an_unresolvable_overhead_yields_no_power(self) -> None:
        """Components alone are not a §4.5.2 total.

        Substituting zero would drop Other_components and shrink the denominator by
        23-33%, inflating system_tps_per_kw in the submitter's favour — silently.
        """
        power = SystemPower.model_validate({"accelerator": {"count": 1, "tdp_per_unit": 1000}})
        assert power.resolved_overhead_fraction is None
        assert power.derived_power_w is None

    def test_missing_groups_drive_the_estimated_tag(self) -> None:
        """§4.5.2: omitting a value "triggers the estimated-power tag"."""
        power = SystemPower.model_validate(
            {"accelerator": {"count": 8, "tdp_per_unit": 700}, "cooling": "air-cooled"}
        )
        assert power.missing_groups == ["cpu", "scale_up_network"]

    def test_an_unresolvable_overhead_is_itself_a_gap(self) -> None:
        power = SystemPower.model_validate({"accelerator": {"count": 8, "tdp_per_unit": 700}})
        assert any(m.startswith("overhead_fraction") for m in power.missing_groups)

    def test_a_declared_total_leaves_nothing_estimated(self) -> None:
        assert SystemPower.model_validate({"provisioned_power_w": 1.0}).missing_groups == []

    def test_a_half_stated_group_counts_as_missing(self) -> None:
        """A count with no TDP contributes nothing, so it is a gap not a value."""
        power = SystemPower.model_validate({"cpu": {"count": 2}})
        assert power.cpu.total_w is None
        assert "cpu" in power.missing_groups


@pytest.mark.unit
class TestPowerDescriptorRule:
    """§9.1 "Power descriptor": one per system, or the submission is rejected."""

    def _check(self, path: Path):
        return SubmissionChecker(path).run()

    def _copy(self, tmp_path: Path) -> Path:
        import shutil

        dest = tmp_path / "sub"
        shutil.copytree(TEST_SUBMISSIONS / "valid_standardized", dest)
        return dest

    def test_present_and_valid_passes(self, tmp_path: Path) -> None:
        report = self._check(self._copy(tmp_path))
        hits = [r for r in report.results if r.rule == "power-descriptor"]
        assert hits and all(r.severity == Severity.INFO for r in hits)

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        root = self._copy(tmp_path)
        for path in root.rglob("system_power.json"):
            path.unlink()
        report = self._check(root)
        assert [
            r
            for r in report.results
            if r.rule == "power-descriptor" and r.severity == Severity.ERROR
        ]

    def test_unparseable_file_is_rejected(self, tmp_path: Path) -> None:
        root = self._copy(tmp_path)
        for path in root.rglob("system_power.json"):
            path.write_text("{not json")
        report = self._check(root)
        assert [
            r
            for r in report.results
            if r.rule == "power-descriptor" and r.severity == Severity.ERROR
        ]

    def test_no_derivable_power_is_rejected(self, tmp_path: Path) -> None:
        """A file that parses but states nothing cannot normalise anything."""
        root = self._copy(tmp_path)
        for path in root.rglob("system_power.json"):
            path.write_text(json.dumps({"note": "to be filled in"}))
        report = self._check(root)
        assert [
            r
            for r in report.results
            if r.rule == "power-descriptor" and r.severity == Severity.ERROR
        ]

    def test_omitted_group_is_flagged_not_rejected(self, tmp_path: Path) -> None:
        """MLCommons auto-populates gaps, so a gap is a tag rather than a failure."""
        root = self._copy(tmp_path)
        for path in root.rglob("system_power.json"):
            path.write_text(
                json.dumps(
                    {"accelerator": {"count": 8, "tdp_per_unit": 700}, "overhead_fraction": 0.5}
                )
            )
        report = self._check(root)
        assert not [
            r
            for r in report.results
            if r.rule == "power-descriptor" and r.severity == Severity.ERROR
        ]
        assert [
            r
            for r in report.results
            if r.rule == "power-estimated" and r.severity == Severity.WARNING
        ]


@pytest.mark.unit
class TestNormalizedMetric:
    """§4.5.3: system_tps_per_kw = system_tps / provisioned_power_kw."""

    def _copy(self, tmp_path: Path) -> Path:
        import shutil

        dest = tmp_path / "sub"
        shutil.copytree(TEST_SUBMISSIONS / "valid_standardized", dest)
        return dest

    def _set_stored(self, root: Path, value: float) -> None:
        for path in root.rglob("result_summary.json"):
            data = json.loads(path.read_text())
            data["system_tps_per_kw"] = value
            path.write_text(json.dumps(data))

    def test_derived_when_nothing_is_stored(self, tmp_path: Path) -> None:
        report = SubmissionChecker(self._copy(tmp_path)).run()
        hits = [r for r in report.results if r.rule == "metric-consistency-tps-per-kw"]
        assert hits and all(r.severity == Severity.INFO for r in hits)

    def test_stored_mismatch_errors(self, tmp_path: Path) -> None:
        root = self._copy(tmp_path)
        self._set_stored(root, 999999.0)
        report = SubmissionChecker(root).run()
        assert [
            r
            for r in report.results
            if r.rule == "metric-consistency-tps-per-kw" and r.severity == Severity.ERROR
        ]

    def test_denominator_is_constant_across_the_curve(self, tmp_path: Path) -> None:
        """§4.5.3: a low-concurrency point is normalised by the *full* provisioned power.

        The reported values must therefore be the throughput curve scaled by one
        constant — so every point's system_tps / system_tps_per_kw is the same kW.
        """
        report = SubmissionChecker(self._copy(tmp_path)).run()
        kws = {
            r.message.split("/")[-1].strip().removesuffix(" kW)")
            for r in report.results
            if r.rule == "metric-consistency-tps-per-kw" and r.severity == Severity.INFO
        }
        assert len(kws) == 1, f"denominator varies across the curve: {kws}"


@pytest.mark.unit
class TestSpecFieldNames:
    """§4.5.2's table names the fields; a conforming file must not read as empty.

    Before these aliases a fully compliant ``system_power.json`` parsed without
    complaint and produced no power at all, because ``extra="allow"`` swallowed every
    spec-named key — so the submitter got the estimated-power tag for a file that had
    stated everything.
    """

    @pytest.mark.parametrize(
        ("group", "count_key", "tdp_key"),
        [
            ("cpu", "num_cpu", "tdp_per_cpu"),
            ("accelerator", "num_accelerator", "tdp_per_accelerator"),
            ("scale_up_network", "num_switches", "tdp_per_switch"),
            ("scale_out_network", "num_switches", "tdp_per_switch"),
        ],
    )
    def test_spec_names_are_read(self, group: str, count_key: str, tdp_key: str) -> None:
        power = SystemPower.model_validate({group: {count_key: 4, tdp_key: 250}})
        assert getattr(power, group).total_w == pytest.approx(1000.0)

    def test_generic_names_still_work(self) -> None:
        """The pre-existing spelling stays valid — §4.5.2 publishes no schema, so
        neither spelling can be declared the wrong one."""
        power = SystemPower.model_validate({"cpu": {"count": 4, "tdp_per_unit": 250}})
        assert power.cpu.total_w == pytest.approx(1000.0)

    def test_public_specification_is_read_as_the_link(self) -> None:
        power = SystemPower.model_validate(
            {"cpu": {"num_cpu": 1, "tdp_per_cpu": 1, "public_specification": "https://x/spec"}}
        )
        assert power.cpu.link == "https://x/spec"

    def test_provisioned_power_watts_spelling(self) -> None:
        assert SystemPower.model_validate(
            {"provisioned_power_watts": 5000.0}
        ).provisioned_power_kw == pytest.approx(5.0)


@pytest.mark.unit
class TestCoolingDerivedOverhead:
    """§4.5.2 fixes the fraction by cooling method: 0.30 liquid, 0.50 air."""

    @pytest.mark.parametrize(
        ("cooling", "expected"),
        [
            ("Liquid cooling", 0.30),
            ("direct-to-chip liquid", 0.30),
            ("Immersion cooled", 0.30),
            ("Air-cooled only", 0.50),
            ("air", 0.50),
            # Liquid wins: a liquid-cooled system with air-cooled PSUs is liquid-cooled.
            ("liquid cooled with air-cooled PSUs", 0.30),
        ],
    )
    def test_derived(self, cooling: str, expected: float) -> None:
        power = SystemPower.model_validate({"cooling": cooling})
        assert power.resolved_overhead_fraction == pytest.approx(expected)

    @pytest.mark.parametrize("cooling", ["passive cooling only", "", None, "unspecified"])
    def test_unrecognised_cooling_resolves_to_nothing(self, cooling: str | None) -> None:
        """§4.5.2 gives two fractions; inventing a third for passive cooling would be
        a guess, so the gap is reported instead."""
        power = SystemPower.model_validate({"cooling": cooling})
        assert power.resolved_overhead_fraction is None

    def test_an_explicit_fraction_wins(self) -> None:
        power = SystemPower.model_validate({"cooling": "air-cooled", "overhead_fraction": 0.35})
        assert power.resolved_overhead_fraction == pytest.approx(0.35)


@pytest.mark.unit
class TestAlternativeFormulations:
    def test_combined_compute_replaces_cpu_and_accelerator(self) -> None:
        """§4.5.2: "In some systems CPU and accelerator power are published as a single
        combined value. That is a valid alternative formulation.\""""
        power = SystemPower.model_validate(
            {
                "compute": {"num_compute": 1, "tdp_per_compute": 6000},
                "scale_up_network": {"num_switches": 2, "tdp_per_switch": 500},
                "cooling": "liquid",
            }
        )
        assert power.major_components_w == pytest.approx(7000.0)
        assert power.missing_groups == []

    def test_rack_level_node_scaling(self) -> None:
        """§4.5.2.1: provisioned_power(Y nodes) = P_rack x (Y / N)."""
        power = SystemPower.model_validate(
            {"rack_power_w": 140_000, "rack_nodes": 18, "submitted_nodes": 9}
        )
        assert power.provisioned_power_kw == pytest.approx(70.0)
        assert power.missing_groups == []

    def test_a_declared_total_beats_rack_scaling(self) -> None:
        """§4.5.2 lists the direct declaration first among the three paths."""
        power = SystemPower.model_validate(
            {
                "provisioned_power_w": 100_000,
                "rack_power_w": 140_000,
                "rack_nodes": 18,
                "submitted_nodes": 9,
            }
        )
        assert power.provisioned_power_kw == pytest.approx(100.0)

    def test_partial_rack_values_are_not_used(self) -> None:
        power = SystemPower.model_validate({"rack_power_w": 140_000, "rack_nodes": 18})
        assert power.rack_scaled_power_w is None


@pytest.mark.unit
class TestCoolingComesFromTheSystemDescription:
    """§8.2 already carries `cooling`, so `system_power.json` should not restate it."""

    def _copy(self, tmp_path: Path) -> Path:
        import shutil

        dest = tmp_path / "sub"
        shutil.copytree(TEST_SUBMISSIONS / "valid_standardized", dest)
        return dest

    def _drop_overhead(self, root: Path) -> None:
        for path in root.rglob("system_power.json"):
            data = json.loads(path.read_text())
            data.pop("overhead_fraction", None)
            path.write_text(json.dumps(data))

    def _set_cooling(self, root: Path, value: str, *, nodes: str | None = None) -> None:
        """Set the system-level cooling, and each node type's unless *nodes* differs.

        §8.2's table puts ``cooling`` on the system while §8.2.1's template nests it
        under ``node_types``, so a fixture carries both and both must be set.
        """
        for path in root.rglob("system_desc.json"):
            data = json.loads(path.read_text())
            data["cooling"] = value
            for node in data.get("node_types") or []:
                node["cooling"] = value if nodes is None else nodes
            path.write_text(json.dumps(data))

    def test_declared_cooling_supplies_the_fraction(self, tmp_path: Path) -> None:
        root = self._copy(tmp_path)
        self._drop_overhead(root)
        self._set_cooling(root, "Direct-to-chip liquid cooling")
        report = SubmissionChecker(root).run()
        hits = [r for r in report.results if r.rule == "power-descriptor"]
        assert hits and all(r.severity == Severity.INFO for r in hits)
        # majors (700 + 5600 + 3500 = 9800 W) x 1.30, not the 9.800 kW of majors alone.
        assert any("12.740 kW" in r.message for r in hits), [r.message for r in hits]

    def test_air_cooled_costs_more_than_liquid(self, tmp_path: Path) -> None:
        kw = {}
        for label, cooling in (("liquid", "liquid cooled"), ("air", "air cooled")):
            root = self._copy(tmp_path / label)
            self._drop_overhead(root)
            self._set_cooling(root, cooling)
            report = SubmissionChecker(root).run()
            kw[label] = [r.message for r in report.results if r.rule == "power-descriptor"]
        assert kw["liquid"] != kw["air"]

    def test_no_cooling_anywhere_is_reported_not_assumed(self, tmp_path: Path) -> None:
        """The failure mode this replaces: a silent zero overhead, which shrinks the
        denominator by a third and inflates system_tps_per_kw."""
        root = self._copy(tmp_path)
        self._drop_overhead(root)
        self._set_cooling(root, "")
        report = SubmissionChecker(root).run()
        errors = [
            r
            for r in report.results
            if r.rule == "power-descriptor" and r.severity == Severity.ERROR
        ]
        assert errors and "cooling method" in errors[0].message

    def test_mixed_node_cooling_takes_the_conservative_fraction(self, tmp_path: Path) -> None:
        """§4.5.2: "Estimation is done conservatively." Air is the larger overhead, so
        a system with both takes the bigger denominator rather than the flattering one."""
        root = self._copy(tmp_path)
        self._drop_overhead(root)
        self._set_cooling(root, "liquid cooled", nodes="Air-cooled")
        report = SubmissionChecker(root).run()
        hits = [r for r in report.results if r.rule == "power-descriptor"]
        assert any("14.700 kW" in r.message for r in hits), [r.message for r in hits]
