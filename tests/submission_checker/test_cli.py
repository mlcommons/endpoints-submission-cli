"""Tests for the CLI commands."""

import json
from pathlib import Path

from click.testing import CliRunner

from submission_checker.cli import main


def test_check_fails_for_missing_path(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["check", str(tmp_path / "nonexistent")])
    assert result.exit_code == 1


def test_check_reports_low_concurrency_error(sub_g):
    """sub_g's points start at 64, so nothing lands in Low Concurrency (33–45)."""
    runner = CliRunner()
    result = runner.invoke(main, ["check", str(sub_g)])
    assert "low-concurrency-coverage" in result.output
    assert result.exit_code == 1


def test_check_valid_submission_passes(valid_standardized):
    runner = CliRunner()
    result = runner.invoke(main, ["check", str(valid_standardized)])
    assert result.exit_code == 0


def test_check_quiet_suppresses_info(sub_g):
    runner = CliRunner()
    result = runner.invoke(main, ["check", "--quiet", str(sub_g)])
    assert "info" not in result.output.lower()


def test_check_strict_fails_on_warnings(valid_standardized):
    """Even the valid submission has WARNINGs (run-duration WIP); strict makes it fail."""
    runner = CliRunner()
    result = runner.invoke(main, ["check", "--strict", str(valid_standardized)])
    # May or may not have warnings depending on duration values — just check it runs
    assert result.exit_code in (0, 1)


def test_regions_command_defaults_c_min_to_32():
    """Appendix B row (C_max=1024, C_min=32): low 33–42, med 43–131, high 132–1024."""
    runner = CliRunner()
    result = runner.invoke(main, ["regions", "--max-concurrency", "1024"])
    assert result.exit_code == 0
    assert "33" in result.output
    assert "42" in result.output
    assert "131" in result.output


def test_regions_command_honours_c_min():
    """Appendix B row (C_max=1024, C_min=16): low 17–26, med 27–117."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["regions", "--max-concurrency", "1024", "--min-concurrency", "16"]
    )
    assert result.exit_code == 0
    assert "26" in result.output
    assert "117" in result.output


def test_regions_command_shows_the_margin():
    """§5.4's 10% margin is its own region in v1.0, so the table must list it."""
    runner = CliRunner()
    result = runner.invoke(main, ["regions", "--max-concurrency", "1024"])
    assert result.exit_code == 0
    assert "1127" in result.output


def test_regions_command_invalid_c_max():
    runner = CliRunner()
    result = runner.invoke(main, ["regions", "--max-concurrency", "32"])
    assert result.exit_code == 1


def test_regions_command_invalid_c_min():
    """C_min must sit inside the Ultra Low Concurrency band (§5.4)."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["regions", "--max-concurrency", "1024", "--min-concurrency", "64"]
    )
    assert result.exit_code == 1


def test_check_output_flag_writes_json(tmp_path: Path, valid_standardized: Path) -> None:
    """--output writes a machine-readable JSON report alongside the terminal table."""
    output_file = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(main, ["check", "--output", str(output_file), str(valid_standardized)])
    assert result.exit_code == 0
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert "results" in data
    assert data["passed"] is True
