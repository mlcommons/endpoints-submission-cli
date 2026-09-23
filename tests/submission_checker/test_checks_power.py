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

    def test_scale_out_is_optional_and_additive(self) -> None:
        """§4.5.2 marks it optional — multi-node submissions only."""
        base = {"accelerator": {"count": 1, "tdp_per_unit": 1000}}
        without = SystemPower.model_validate(base)
        with_scale_out = SystemPower.model_validate(
            {**base, "scale_out_network": {"count": 2, "tdp_per_unit": 500}}
        )
        assert without.major_components_w == pytest.approx(1000.0)
        assert with_scale_out.major_components_w == pytest.approx(2000.0)

    def test_no_numbers_yields_no_power(self) -> None:
        assert SystemPower().provisioned_power_kw is None

    def test_overhead_defaults_to_none_meaning_zero(self) -> None:
        """A descriptor stating components but no overhead is still usable."""
        power = SystemPower.model_validate({"accelerator": {"count": 1, "tdp_per_unit": 1000}})
        assert power.derived_power_w == pytest.approx(1000.0)

    def test_missing_groups_drive_the_estimated_tag(self) -> None:
        """§4.5.2: omitting a value "triggers the estimated-power tag"."""
        power = SystemPower.model_validate({"accelerator": {"count": 8, "tdp_per_unit": 700}})
        assert power.missing_groups == ["cpu", "scale_up_network"]

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
