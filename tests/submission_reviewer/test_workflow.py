# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""The reusable ``Review hints`` workflow keeps the properties the review repo relies on."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from submission_reviewer.render import MARKER

_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "review-submission.yml"


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    data = yaml.safe_load(_WORKFLOW.read_text())
    assert isinstance(data, dict)
    return data


def _steps(workflow: dict[Any, Any]) -> list[dict[str, Any]]:
    steps = workflow["jobs"]["review"]["steps"]
    assert isinstance(steps, list)
    return steps


def test_is_reusable(workflow) -> None:
    # PyYAML reads the bare key `on` as True.
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_call"}


def test_names_do_not_collide_with_the_compliance_gate(workflow) -> None:
    # lifecycle_manager matches these exact names to hand a submission over to review.
    names = {workflow["name"], *(job.get("name") for job in workflow["jobs"].values())}
    assert not names & {"Check submission", "Run submission checker"}


def test_least_privilege(workflow) -> None:
    assert workflow["permissions"] == {}
    assert workflow["jobs"]["review"]["permissions"] == {
        "contents": "read",
        "pull-requests": "write",
    }
    (checkout,) = [s for s in _steps(workflow) if s.get("uses", "").startswith("actions/checkout")]
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["fetch-depth"] == 0


def test_installs_from_pypi_never_from_the_checkout(workflow) -> None:
    (install,) = [s for s in _steps(workflow) if s.get("name") == "Install the reviewer"]
    assert "endpoints-submission-cli${CLI_VERSION:+==$CLI_VERSION}" in install["run"]
    assert "pip install ." not in _WORKFLOW.read_text()


def test_inputs_reach_scripts_only_through_env(workflow) -> None:
    for step in _steps(workflow):
        assert "${{" not in step.get("run", ""), step.get("name")


def test_comment_marker_matches_the_renderer(workflow) -> None:
    (post,) = [s for s in _steps(workflow) if s.get("name") == "Post or update the PR comment"]
    assert f"marker='{MARKER}'" in post["run"]
    assert 'select(.user.login == \\"github-actions[bot]\\")' in post["run"]


def test_review_uses_the_pr_base(workflow) -> None:
    (step,) = [s for s in _steps(workflow) if s.get("id") == "review"]
    assert step["env"]["PR_BASE"] == "${{ github.event.pull_request.base.sha }}"
    assert '--pr-base "$PR_BASE"' in step["run"]
