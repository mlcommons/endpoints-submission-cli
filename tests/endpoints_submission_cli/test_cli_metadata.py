# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-bundle cli_metadata.json marker.

The marker is generated entirely from what the command already holds: the API's
submission record plus the version of the CLI running now. Nothing is read back
out of the previous bundle, and no new API field is involved.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from endpoints_submission_cli.commands.common import (
    CLI_METADATA_FILENAME,
    _write_cli_metadata,
)

_CREATED = "2026-08-20T17:08:02.936070Z"


def _running(version: str):
    return patch("endpoints_submission_cli.commands.common._cli_version", return_value=version)


@pytest.mark.unit
class TestCreate:
    """On create the running CLI *is* the creating one, so both versions match."""

    def test_both_versions_are_the_running_cli(self, tmp_path: Path) -> None:
        record = {"cli_version": "1.0.0.0", "created_at": _CREATED}
        with _running("1.0.0.0"):
            meta = _write_cli_metadata(tmp_path, "create", record)
        assert meta["cli_version"] == "1.0.0.0"
        assert meta["most_recent_cli_used"] == "1.0.0.0"

    def test_created_at_comes_from_the_record(self, tmp_path: Path) -> None:
        record = {"cli_version": "1.0.0.0", "created_at": _CREATED}
        with _running("1.0.0.0"):
            meta = _write_cli_metadata(tmp_path, "create", record)
        assert meta["created_at"] == _CREATED


@pytest.mark.unit
class TestLaterCommands:
    """Every other command reports the DB's creating version plus its own."""

    _RECORD = {"cli_version": "1.0.0.0", "created_at": _CREATED}

    @pytest.mark.parametrize("command", ["add-run", "remove-run", "update", "create-local"])
    def test_splits_creating_and_current_version(self, tmp_path: Path, command: str) -> None:
        with _running("1.2.0.0"):
            meta = _write_cli_metadata(tmp_path, command, self._RECORD)
        assert meta["command"] == command
        assert meta["cli_version"] == "1.0.0.0"
        assert meta["most_recent_cli_used"] == "1.2.0.0"

    def test_created_at_tracks_the_submission_not_the_rebuild(self, tmp_path: Path) -> None:
        """A rebuild must not restamp creation time — it comes from the DB record."""
        with _running("1.2.0.0"):
            meta = _write_cli_metadata(tmp_path, "update", self._RECORD)
        assert meta["created_at"] == _CREATED

    def test_same_version_rebuild_matches(self, tmp_path: Path) -> None:
        with _running("1.0.0.0"):
            meta = _write_cli_metadata(tmp_path, "add-run", self._RECORD)
        assert meta["cli_version"] == meta["most_recent_cli_used"] == "1.0.0.0"


@pytest.mark.unit
class TestFallbacks:
    def test_no_record_falls_back_to_running_version(self, tmp_path: Path) -> None:
        with _running("1.5.0.0"):
            meta = _write_cli_metadata(tmp_path, "create")
        assert meta["cli_version"] == "1.5.0.0"
        assert meta["most_recent_cli_used"] == "1.5.0.0"

    def test_record_without_cli_version_falls_back(self, tmp_path: Path) -> None:
        """An API too old to report cli_version must not blank the field."""
        with _running("1.5.0.0"):
            meta = _write_cli_metadata(tmp_path, "update", {"created_at": _CREATED})
        assert meta["cli_version"] == "1.5.0.0"

    def test_missing_created_at_falls_back_to_now(self, tmp_path: Path) -> None:
        with _running("1.5.0.0"):
            meta = _write_cli_metadata(tmp_path, "update", {"cli_version": "1.0.0.0"})
        parsed = datetime.fromisoformat(meta["created_at"])
        assert parsed.tzinfo is not None


@pytest.mark.unit
class TestOnDisk:
    def test_written_into_the_given_directory(self, tmp_path: Path) -> None:
        with _running("1.0.0.0"):
            _write_cli_metadata(tmp_path, "create", {"cli_version": "1.0.0.0"})
        on_disk = json.loads((tmp_path / CLI_METADATA_FILENAME).read_text())
        assert set(on_disk) == {"command", "cli_version", "most_recent_cli_used", "created_at"}
