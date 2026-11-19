# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""``submitter-org-dir``: a submission sits under the directory its org name gives.

§8.1 lays a submission out as ``<submitting_organization>/<submission_id>/``. The
builder names that directory from ``submitter_org_names`` (required by §8.2) with
:func:`layout.org_dir_name`, and the checker holds a submission to the same name.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from submission_checker import layout
from submission_checker.checker import SubmissionChecker
from submission_checker.models import CheckResult, Report, Severity

from .conftest import copy_submission

_RULE = "submitter-org-dir"


def _check(path: Path) -> Report:
    return SubmissionChecker(path).run()


def _org_results(report: Report) -> list[CheckResult]:
    return [r for r in report.results if r.rule == _RULE]


def _system_descs(root: Path) -> list[Path]:
    return sorted((root / layout.RESULTS_DIR).rglob(layout.SYSTEM_DESC_JSON))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ACME Corp", "ACME_Corp"),
        ("Submitter-A", "Submitter-A"),
        ("  Acme, Inc.  ", "Acme_Inc"),
        ("...", "unknown"),
    ],
)
def test_org_dir_name(name: str, expected: str) -> None:
    assert layout.org_dir_name(name) == expected


def test_matching_org_dir_passes(valid_standardized: Path) -> None:
    """``ACME Corp`` is filed under ``ACME_Corp/``, the builder's directory form."""
    results = _org_results(_check(valid_standardized))
    assert results and all(r.severity == Severity.INFO for r in results)
    assert "ACME_Corp/" in results[0].message


def test_org_root_passes(valid_standardized: Path) -> None:
    """Pointing the checker at the org directory resolves the same parent."""
    results = _org_results(_check(valid_standardized.parent))
    assert results and all(r.severity == Severity.INFO for r in results)


def test_relative_submission_path(valid_standardized: Path, monkeypatch) -> None:
    """``.`` has no name of its own, so the parent must come from the resolved path."""
    monkeypatch.chdir(valid_standardized)
    results = _org_results(_check(Path(".")))
    assert results and all(r.severity == Severity.INFO for r in results)


def test_wrong_org_dir_is_an_error(valid_standardized: Path, tmp_path: Path) -> None:
    root = tmp_path / "Someone_Else" / "sub"
    shutil.copytree(valid_standardized, root)

    report = _check(root)
    errors = [r for r in _org_results(report) if r.severity == Severity.ERROR]
    assert errors and not report.passed
    assert "ACME_Corp/" in errors[0].message
    assert "Someone_Else/" in errors[0].message


def test_raw_org_name_as_dir_is_an_error(valid_standardized: Path, tmp_path: Path) -> None:
    """The directory form is required: ``ACME Corp/`` is not ``ACME_Corp/``."""
    root = tmp_path / "ACME Corp" / "sub"
    shutil.copytree(valid_standardized, root)
    assert any(r.severity == Severity.ERROR for r in _org_results(_check(root)))


def _set_org_name(root: Path, value: object, *, drop: bool = False) -> None:
    for path in _system_descs(root):
        data = json.loads(path.read_text())
        if drop:
            del data["submitter_org_names"]
        else:
            data["submitter_org_names"] = value
        path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    ("value", "drop"),
    [(None, True), ("   ", False), (None, False)],
    ids=["absent", "blank", "null"],
)
def test_missing_org_name_is_invalid(
    valid_standardized: Path, tmp_path: Path, value: object, drop: bool
) -> None:
    """§8.2 requires the field; its absence is the schema's to report, not this rule's."""
    root = copy_submission(valid_standardized, tmp_path)
    _set_org_name(root, value, drop=drop)

    report = _check(root)
    assert {e.rule for e in report.errors} == {"system-description-valid"}
    assert any("submitter_org_names" in e.message for e in report.errors)
    assert not _org_results(report)


def test_org_dir_still_checked_when_description_invalid(
    valid_standardized: Path, tmp_path: Path
) -> None:
    """An unrelated invalid field must not hide the directory check."""
    root = tmp_path / "Someone_Else" / "sub"
    shutil.copytree(valid_standardized, root)
    for path in _system_descs(root):
        data = json.loads(path.read_text())
        data["division"] = "Standardised"
        path.write_text(json.dumps(data))

    errors = {e.rule for e in _check(root).errors}
    assert errors == {"system-description-valid", _RULE}
