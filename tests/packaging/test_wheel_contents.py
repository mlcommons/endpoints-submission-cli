# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""The built wheel carries everything the installed tools need.

``[tool.setuptools] packages`` is an explicit list and package data is opted into by
glob, so a new module or data file can be missing from the wheel while every other
test, which runs from the source tree, still passes.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> zipfile.ZipFile:
    if shutil.which("uv") is None:
        pytest.skip("uv is needed to build the wheel")
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out), str(_ROOT)],
        check=True,
        capture_output=True,
    )
    (path,) = out.glob("*.whl")
    return zipfile.ZipFile(path)


def _metadata(wheel: zipfile.ZipFile, name: str) -> str:
    (member,) = [n for n in wheel.namelist() if n.endswith(f".dist-info/{name}")]
    return wheel.read(member).decode()


@pytest.mark.parametrize(
    "member",
    [
        "endpoints_submission_cli/__main__.py",
        "endpoints_submission_cli/mcp_server.py",
        "endpoints_submission_cli/commands/install_skill.py",
        "endpoints_submission_cli/skills/mlperf-submissions/SKILL.md",
        "submission_checker/data/seed_sets.yaml",
        "submission_checker/data/approved_drafters.yaml",
    ],
)
def test_wheel_contains(wheel: zipfile.ZipFile, member: str) -> None:
    assert member in wheel.namelist()


def test_both_scripts_are_installed(wheel: zipfile.ZipFile) -> None:
    entry_points = _metadata(wheel, "entry_points.txt")
    assert "endpoints-submission-cli = endpoints_submission_cli.main:main" in entry_points
    assert "endpoints-submission-mcp = endpoints_submission_cli.mcp_server:main" in entry_points


def test_mcp_is_an_extra_not_a_dependency(wheel: zipfile.ZipFile) -> None:
    metadata = _metadata(wheel, "METADATA")
    assert "Provides-Extra: mcp" in metadata
    requires = [line for line in metadata.splitlines() if line.startswith("Requires-Dist: mcp")]
    assert requires and all("extra ==" in line for line in requires)
