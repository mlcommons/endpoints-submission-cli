# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the submissions command group."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from endpoints_submission_cli.exceptions import APIError, SubmissionCheckError
from endpoints_submission_cli.main import app
from endpoints_submission_cli.submissions.builder import PENDING_SUBMISSION_ID
from tests.endpoints_submission_cli.conftest import (
    RUN_ID,
    SUBMISSION_ID,
    SUBMISSION_OUT,
)

TOKEN = "mlc_test"
_TOKEN_ARGS = ["--token", TOKEN]

_PR_URL = "https://github.com/mlcommons/submissions/pull/42"
_PR_NUMBER = 42

_runner = CliRunner()


def _flat(output: str) -> str:
    """Output with whitespace collapsed, so Rich's line wrapping doesn't split a phrase."""
    return " ".join(output.split())


def _run_app(*args: str) -> None:
    """Invoke the app and assert exit code 0."""
    result = _runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output


@pytest.mark.unit
class TestSubmissionsList:
    def test_list_success(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.list_submissions",
            return_value=[SUBMISSION_OUT],
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app("submissions", "list", *_TOKEN_ARGS)

    def test_list_json_flag(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.list_submissions",
            return_value=[SUBMISSION_OUT],
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["submissions", "list", "-j", *_TOKEN_ARGS])
        assert result.exit_code == 0
        assert SUBMISSION_ID in result.output

    def test_list_api_error_exits_1(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.list_submissions",
            side_effect=APIError("fail"),
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(app, ["submissions", "list", *_TOKEN_ARGS])
        assert result.exit_code == 1

    def test_list_no_token_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRISM_USER_API_TOKEN", raising=False)
        result = _runner.invoke(app, ["submissions", "list"])
        assert result.exit_code == 1


@pytest.mark.unit
class TestSubmissionsGet:
    def test_get_success(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.get_submission", return_value=SUBMISSION_OUT
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app("submissions", "get", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)

    def test_get_json_flag(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.get_submission", return_value=SUBMISSION_OUT
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app,
                    ["submissions", "get", "--submission-id", SUBMISSION_ID, "-j", *_TOKEN_ARGS],
                )
        assert result.exit_code == 0
        assert SUBMISSION_ID in result.output

    def test_get_api_error_exits_1(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.get_submission", side_effect=APIError("404")
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app,
                    ["submissions", "get", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS],
                )
        assert result.exit_code == 1

    def test_get_download_to_saves_archive(self, tmp_path: Path) -> None:
        archive = tmp_path / f"{SUBMISSION_ID}.tar.gz"
        with patch(
            "endpoints_submission_cli.submissions.api.get_submission", return_value=SUBMISSION_OUT
        ):
            with patch(
                "endpoints_submission_cli.submissions.api.download_submission_archive",
                return_value=archive,
            ) as mock_dl:
                with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                    result = _runner.invoke(
                        app,
                        [
                            "submissions",
                            "get",
                            "--submission-id",
                            SUBMISSION_ID,
                            "--download-to",
                            str(tmp_path),
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(TOKEN, SUBMISSION_ID, tmp_path)
        assert "Archive saved to" in result.output

    def test_get_download_api_error_exits_1(self, tmp_path: Path) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.get_submission", return_value=SUBMISSION_OUT
        ):
            with patch(
                "endpoints_submission_cli.submissions.api.download_submission_archive",
                side_effect=APIError("download failed"),
            ):
                with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                    result = _runner.invoke(
                        app,
                        [
                            "submissions",
                            "get",
                            "--submission-id",
                            SUBMISSION_ID,
                            "--download-to",
                            str(tmp_path),
                            *_TOKEN_ARGS,
                        ],
                    )
        assert result.exit_code == 1


@pytest.mark.unit
class TestSubmissionsUpdate:
    _OLD_RUN_ID = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"

    def test_update_run_ids(self, tmp_path: Path) -> None:
        """--run-ids triggers full rebuild pipeline; update_submission called with new run list."""
        # Removing _OLD_RUN_ID and keeping RUN_ID: §8 permits withdrawing a point,
        # not adding one, so the desired list must be a subset of the current one.
        current_sub = {
            **SUBMISSION_OUT,
            "run_ids": [self._OLD_RUN_ID, RUN_ID],
            "pr_number": _PR_NUMBER,
        }
        updated_sub = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        # These commands pass submission_id to the builder, so the rebuilt tree
        # carries the real id rather than the placeholder.
        (fake_sub_dir / SUBMISSION_ID).mkdir(exist_ok=True)
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.get_submission", return_value=current_sub
            ):
                with patch(
                    "endpoints_submission_cli.submissions.api.update_submission",
                    return_value=updated_sub,
                ) as mock_patch:
                    with patch(
                        "endpoints_submission_cli.runs.api.download_run_archive",
                        return_value=fake_archive,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.update.build_submission_folder",
                            return_value=fake_sub_dir,
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.update._run_submission_checker"
                            ):
                                with patch(
                                    "endpoints_submission_cli.commands.submissions.update.create_bundle_archive",
                                    return_value=fake_bundle,
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.api.upload_submission_archive"
                                    ):
                                        _run_app(
                                            "submissions",
                                            "update",
                                            "--submission-id",
                                            SUBMISSION_ID,
                                            "--run-ids",
                                            RUN_ID,
                                            *_TOKEN_ARGS,
                                        )
        mock_patch.assert_called_once_with(TOKEN, SUBMISSION_ID, {"run_ids": [RUN_ID]})

    def test_update_run_ids_merge_failure_rolls_back(self, tmp_path: Path) -> None:
        """A rebuild failure rolls back the DB PATCH with the original run IDs."""
        # Removing _OLD_RUN_ID and keeping RUN_ID: §8 permits withdrawing a point,
        # not adding one, so the desired list must be a subset of the current one.
        current_sub = {
            **SUBMISSION_OUT,
            "run_ids": [self._OLD_RUN_ID, RUN_ID],
            "pr_number": _PR_NUMBER,
        }
        updated_sub = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        # These commands pass submission_id to the builder, so the rebuilt tree
        # carries the real id rather than the placeholder.
        (fake_sub_dir / SUBMISSION_ID).mkdir(exist_ok=True)

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.get_submission", return_value=current_sub
            ):
                with patch(
                    "endpoints_submission_cli.submissions.api.update_submission",
                    return_value=updated_sub,
                ) as mock_patch:
                    with patch(
                        "endpoints_submission_cli.runs.api.download_run_archive",
                        return_value=fake_archive,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.update.build_submission_folder",
                            return_value=fake_sub_dir,
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.update._run_submission_checker"
                            ):
                                result = _runner.invoke(
                                    app,
                                    [
                                        "submissions",
                                        "update",
                                        "--submission-id",
                                        SUBMISSION_ID,
                                        "--run-ids",
                                        RUN_ID,
                                        *_TOKEN_ARGS,
                                    ],
                                )
        assert result.exit_code == 1
        assert mock_patch.call_count == 2
        mock_patch.assert_any_call(TOKEN, SUBMISSION_ID, {"run_ids": [RUN_ID]})
        mock_patch.assert_any_call(TOKEN, SUBMISSION_ID, {"run_ids": [self._OLD_RUN_ID, RUN_ID]})

    def test_update_run_ids_download_failure_rolls_back(self, tmp_path: Path) -> None:
        """Download failure rolls back the DB PATCH with original run IDs."""
        current_sub = {**SUBMISSION_OUT, "run_ids": [self._OLD_RUN_ID, RUN_ID]}
        updated_sub = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.get_submission", return_value=current_sub
            ):
                with patch(
                    "endpoints_submission_cli.submissions.api.update_submission",
                    return_value=updated_sub,
                ) as mock_patch:
                    with patch(
                        "endpoints_submission_cli.runs.api.download_run_archive",
                        side_effect=APIError("not found"),
                    ):
                        result = _runner.invoke(
                            app,
                            [
                                "submissions",
                                "update",
                                "--submission-id",
                                SUBMISSION_ID,
                                "--run-ids",
                                RUN_ID,
                                *_TOKEN_ARGS,
                            ],
                        )
        assert result.exit_code == 1
        # First call: PATCH forward; second call: rollback with original run IDs
        assert mock_patch.call_count == 2
        mock_patch.assert_any_call(TOKEN, SUBMISSION_ID, {"run_ids": [RUN_ID]})
        mock_patch.assert_any_call(TOKEN, SUBMISSION_ID, {"run_ids": [self._OLD_RUN_ID, RUN_ID]})

    def test_update_run_ids_rejects_additions(self) -> None:
        """§8 no longer provides a post-submission window for adding points.

        The guard fires before anything is fetched or rebuilt, so a mistaken --run-ids
        list cannot leave the submission half-updated.
        """
        current_sub = {**SUBMISSION_OUT, "run_ids": [self._OLD_RUN_ID]}
        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.get_submission", return_value=current_sub
            ):
                with patch(
                    "endpoints_submission_cli.submissions.api.update_submission"
                ) as mock_patch:
                    with patch("endpoints_submission_cli.runs.api.download_run_archive") as mock_dl:
                        result = _runner.invoke(
                            app,
                            [
                                "submissions",
                                "update",
                                "--submission-id",
                                SUBMISSION_ID,
                                "--run-ids",
                                RUN_ID,
                                *_TOKEN_ARGS,
                            ],
                        )
        assert result.exit_code == 1
        assert "Cannot add runs" in result.output
        mock_patch.assert_not_called()
        mock_dl.assert_not_called()

    def test_update_run_ids_allows_pure_removal(self, tmp_path: Path) -> None:
        """Withdrawing a point is still permitted (§8.1), so a shrinking list works."""
        current_sub = {**SUBMISSION_OUT, "run_ids": [self._OLD_RUN_ID, RUN_ID]}
        updated_sub = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        (fake_sub_dir / SUBMISSION_ID).mkdir(exist_ok=True)
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.get_submission", return_value=current_sub
            ):
                with patch(
                    "endpoints_submission_cli.submissions.api.update_submission",
                    return_value=updated_sub,
                ) as mock_patch:
                    with patch(
                        "endpoints_submission_cli.runs.api.download_run_archive",
                        return_value=fake_archive,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.update.build_submission_folder",
                            return_value=fake_sub_dir,
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.update._run_submission_checker"
                            ):
                                with patch(
                                    "endpoints_submission_cli.commands.submissions.update.create_bundle_archive",
                                    return_value=fake_bundle,
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.api.upload_submission_archive"
                                    ):
                                        _run_app(
                                            "submissions",
                                            "update",
                                            "--submission-id",
                                            SUBMISSION_ID,
                                            "--run-ids",
                                            RUN_ID,
                                            *_TOKEN_ARGS,
                                        )
        mock_patch.assert_called_once_with(TOKEN, SUBMISSION_ID, {"run_ids": [RUN_ID]})

    def test_update_run_ids_no_change_applies_date_only(self) -> None:
        """When desired run list equals current, skip rebuild and apply date patch only."""
        current_sub = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        updated_sub = {**SUBMISSION_OUT, "target_availability_date": "2026-06-01"}

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.get_submission", return_value=current_sub
            ):
                with patch(
                    "endpoints_submission_cli.submissions.api.update_submission",
                    return_value=updated_sub,
                ) as mock_patch:
                    _run_app(
                        "submissions",
                        "update",
                        "--submission-id",
                        SUBMISSION_ID,
                        "--run-ids",
                        RUN_ID,
                        "--target-availability-date",
                        "2026-06-01",
                        *_TOKEN_ARGS,
                    )
        mock_patch.assert_called_once_with(
            TOKEN, SUBMISSION_ID, {"target_availability_date": "2026-06-01"}
        )

    def test_update_target_availability_date(self) -> None:
        """--target-availability-date only triggers a DB-only PATCH (no rebuild)."""
        updated = {**SUBMISSION_OUT, "target_availability_date": "2026-06-01"}
        with patch(
            "endpoints_submission_cli.submissions.api.update_submission", return_value=updated
        ) as mock_patch:
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app(
                    "submissions",
                    "update",
                    "--submission-id",
                    SUBMISSION_ID,
                    "--target-availability-date",
                    "2026-06-01",
                    *_TOKEN_ARGS,
                )
        mock_patch.assert_called_once_with(
            TOKEN, SUBMISSION_ID, {"target_availability_date": "2026-06-01"}
        )

    def test_update_no_fields_is_noop(self) -> None:
        with patch("endpoints_submission_cli.submissions.api.update_submission") as mock_patch:
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                _run_app("submissions", "update", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS)
        mock_patch.assert_not_called()

    def test_update_api_error_exits_1(self) -> None:
        """Date-only PATCH API error exits with code 1."""
        with patch(
            "endpoints_submission_cli.submissions.api.update_submission",
            side_effect=APIError("500"),
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app,
                    [
                        "submissions",
                        "update",
                        "--submission-id",
                        SUBMISSION_ID,
                        "--target-availability-date",
                        "2026-06-01",
                        *_TOKEN_ARGS,
                    ],
                )
        assert result.exit_code == 1


@pytest.mark.unit
class TestSubmissionsWithdraw:
    def test_withdraw_success(self) -> None:
        withdrawn = {**SUBMISSION_OUT, "status": "WITHDRAWN", "pr_number": _PR_NUMBER}
        with patch(
            "endpoints_submission_cli.submissions.api.withdraw_submission", return_value=withdrawn
        ):
            with patch("endpoints_submission_cli.submissions.api.delete_submission_archive"):
                with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                    _run_app(
                        "submissions",
                        "withdraw",
                        "--submission-id",
                        SUBMISSION_ID,
                        *_TOKEN_ARGS,
                    )

    def test_withdraw_archive_failure_is_warning(self) -> None:
        withdrawn = {**SUBMISSION_OUT, "status": "WITHDRAWN", "pr_number": None}
        with patch(
            "endpoints_submission_cli.submissions.api.withdraw_submission", return_value=withdrawn
        ):
            with patch(
                "endpoints_submission_cli.submissions.api.delete_submission_archive",
                side_effect=APIError("blob error"),
            ):
                with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                    _run_app(
                        "submissions",
                        "withdraw",
                        "--submission-id",
                        SUBMISSION_ID,
                        *_TOKEN_ARGS,
                    )

    def test_withdraw_api_error_exits_1(self) -> None:
        with patch(
            "endpoints_submission_cli.submissions.api.withdraw_submission",
            side_effect=APIError("locked"),
        ):
            with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
                result = _runner.invoke(
                    app,
                    ["submissions", "withdraw", "--submission-id", SUBMISSION_ID, *_TOKEN_ARGS],
                )
        assert result.exit_code == 1


def _make_fake_archive(tmp_path: Path) -> Path:
    """Return a tiny tar.gz for use as a fake downloaded archive."""
    import tarfile

    folder = tmp_path / "fake_run"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "system_info.json").write_text("{}")
    archive = tmp_path / "fake.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)
    return archive


@pytest.mark.unit
class TestSubmissionsCreate:
    def test_create_success(self, tmp_path: Path) -> None:
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.runs.api.download_run_archive", return_value=fake_archive
            ):
                with patch(
                    "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
                    return_value=fake_sub_dir,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.create._run_submission_checker"
                    ):
                        with patch(
                            "endpoints_submission_cli.submissions.api.create_submission",
                            return_value=SUBMISSION_OUT,
                        ) as mock_create:
                            with patch(
                                "endpoints_submission_cli.commands.submissions.create.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch(
                                    "endpoints_submission_cli.submissions.api.upload_submission_archive"
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.api.update_submission"
                                    ) as mock_update:
                                        _run_app(
                                            "submissions",
                                            "create",
                                            "--division",
                                            "standardized",
                                            "--scenario",
                                            "cop",
                                            "--availability",
                                            "available",
                                            "--run-ids",
                                            RUN_ID,
                                            *_TOKEN_ARGS,
                                        )
        mock_create.assert_called_once()
        # Status is left COMPLIANCE_CHECKING; the lifecycle manager hands it to review.
        mock_update.assert_not_called()
        # Without --test, the submission is created as a non-test entry.
        assert mock_create.call_args.args[1]["is_test"] is False
        # The marker lives inside <submission_id>/, not at the organisation level:
        # the org directory is shared across submissions.
        meta_path = fake_sub_dir / SUBMISSION_ID / "cli_metadata.json"
        assert meta_path.is_file()
        assert not (fake_sub_dir / "cli_metadata.json").exists()
        meta = json.loads(meta_path.read_text())
        assert meta["command"] == "create"
        assert "cli_version" in meta and "created_at" in meta

    def test_shared_tree_flags_reach_the_builder(self, tmp_path: Path) -> None:
        """``--shared-src`` / ``--shared-docs`` are passed through as given, repeatably."""
        import contextlib

        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")
        impl_a = tmp_path / "vllm"
        impl_b = tmp_path / "sglang"
        docs = tmp_path / "shared_docs"
        for d in (impl_a, impl_b, docs):
            d.mkdir()

        with contextlib.ExitStack() as stack:
            patches = [
                patch("endpoints_submission_cli._http.get_token", return_value=TOKEN),
                patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ),
                patch(
                    "endpoints_submission_cli.commands.submissions.create._run_submission_checker"
                ),
                patch(
                    "endpoints_submission_cli.submissions.api.create_submission",
                    return_value=SUBMISSION_OUT,
                ),
                patch(
                    "endpoints_submission_cli.commands.submissions.create.create_bundle_archive",
                    return_value=fake_bundle,
                ),
                patch("endpoints_submission_cli.submissions.api.upload_submission_archive"),
                patch("endpoints_submission_cli.submissions.api.update_submission"),
            ]
            for pt in patches:
                stack.enter_context(pt)
            mock_build = stack.enter_context(
                patch(
                    "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
                    return_value=fake_sub_dir,
                )
            )
            _run_app(
                "submissions",
                "create",
                "--division",
                "standardized",
                "--scenario",
                "cop",
                "--availability",
                "available",
                "--run-ids",
                RUN_ID,
                "--shared-src",
                str(impl_a),
                "--shared-src",
                str(impl_b),
                "--shared-docs",
                str(docs),
                *_TOKEN_ARGS,
            )

        kwargs = mock_build.call_args.kwargs
        assert list(kwargs["shared_src_dirs"]) == [impl_a, impl_b]
        assert list(kwargs["shared_docs_dirs"]) == [docs]

    def test_shared_tree_flags_default_to_empty(self, tmp_path: Path) -> None:
        """Omitting them must not change how the builder is called."""
        import contextlib

        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with contextlib.ExitStack() as stack:
            for pt in [
                patch("endpoints_submission_cli._http.get_token", return_value=TOKEN),
                patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ),
                patch(
                    "endpoints_submission_cli.commands.submissions.create._run_submission_checker"
                ),
                patch(
                    "endpoints_submission_cli.submissions.api.create_submission",
                    return_value=SUBMISSION_OUT,
                ),
                patch(
                    "endpoints_submission_cli.commands.submissions.create.create_bundle_archive",
                    return_value=fake_bundle,
                ),
                patch("endpoints_submission_cli.submissions.api.upload_submission_archive"),
                patch("endpoints_submission_cli.submissions.api.update_submission"),
            ]:
                stack.enter_context(pt)
            mock_build = stack.enter_context(
                patch(
                    "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
                    return_value=fake_sub_dir,
                )
            )
            _run_app(
                "submissions",
                "create",
                "--division",
                "standardized",
                "--scenario",
                "cop",
                "--availability",
                "available",
                "--run-ids",
                RUN_ID,
                *_TOKEN_ARGS,
            )

        kwargs = mock_build.call_args.kwargs
        assert tuple(kwargs["shared_src_dirs"]) == ()
        assert tuple(kwargs["shared_docs_dirs"]) == ()

    def test_create_test_flag_sets_is_test(self, tmp_path: Path) -> None:
        import contextlib

        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")
        C = "endpoints_submission_cli"

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(f"{C}._http.get_token", return_value=TOKEN))
            stack.enter_context(
                patch(f"{C}.runs.api.download_run_archive", return_value=fake_archive)
            )
            stack.enter_context(
                patch(
                    f"{C}.commands.submissions.create.build_submission_folder",
                    return_value=fake_sub_dir,
                )
            )
            stack.enter_context(patch(f"{C}.commands.submissions.create._run_submission_checker"))
            mock_create = stack.enter_context(
                patch(f"{C}.submissions.api.create_submission", return_value=SUBMISSION_OUT)
            )
            stack.enter_context(
                patch(
                    f"{C}.commands.submissions.create.create_bundle_archive",
                    return_value=fake_bundle,
                )
            )
            stack.enter_context(patch(f"{C}.submissions.api.upload_submission_archive"))
            stack.enter_context(patch(f"{C}.submissions.api.update_submission"))
            _run_app(
                "submissions",
                "create",
                "--division",
                "standardized",
                "--scenario",
                "cop",
                "--availability",
                "available",
                "--run-ids",
                RUN_ID,
                "--test",
                *_TOKEN_ARGS,
            )

        mock_create.assert_called_once()
        assert mock_create.call_args.args[1]["is_test"] is True

    def test_create_download_failure_exits_1(self) -> None:
        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.runs.api.download_run_archive",
                side_effect=APIError("not found"),
            ):
                result = _runner.invoke(
                    app,
                    [
                        "submissions",
                        "create",
                        "--division",
                        "standardized",
                        "--scenario",
                        "cop",
                        "--availability",
                        "available",
                        "--run-ids",
                        RUN_ID,
                        *_TOKEN_ARGS,
                    ],
                )
        assert result.exit_code == 1

    def test_create_checker_failure_exits_1(self, tmp_path: Path) -> None:

        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.runs.api.download_run_archive", return_value=fake_archive
            ):
                with patch(
                    "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
                    return_value=fake_sub_dir,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.create._run_submission_checker",
                        side_effect=SubmissionCheckError("1 error"),
                    ):
                        result = _runner.invoke(
                            app,
                            [
                                "submissions",
                                "create",
                                "--division",
                                "standardized",
                                "--scenario",
                                "cop",
                                "--availability",
                                "available",
                                "--run-ids",
                                RUN_ID,
                                *_TOKEN_ARGS,
                            ],
                        )
        assert result.exit_code == 1

    def test_create_api_error_exits_1(self, tmp_path: Path) -> None:
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.runs.api.download_run_archive", return_value=fake_archive
            ):
                with patch(
                    "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
                    return_value=fake_sub_dir,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.create._run_submission_checker"
                    ):
                        with patch(
                            "endpoints_submission_cli.submissions.api.create_submission",
                            side_effect=APIError("500"),
                        ):
                            result = _runner.invoke(
                                app,
                                [
                                    "submissions",
                                    "create",
                                    "--division",
                                    "standardized",
                                    "--scenario",
                                    "cop",
                                    "--availability",
                                    "available",
                                    "--run-ids",
                                    RUN_ID,
                                    *_TOKEN_ARGS,
                                ],
                            )
        assert result.exit_code == 1

    def test_create_upload_failure_rolls_back(self, tmp_path: Path) -> None:
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.runs.api.download_run_archive", return_value=fake_archive
            ):
                with patch(
                    "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
                    return_value=fake_sub_dir,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.create._run_submission_checker"
                    ):
                        with patch(
                            "endpoints_submission_cli.submissions.api.create_submission",
                            return_value=SUBMISSION_OUT,
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.create.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch(
                                    "endpoints_submission_cli.submissions.api.upload_submission_archive",
                                    side_effect=APIError("upload failed"),
                                ):
                                    with patch(
                                        "endpoints_submission_cli.submissions.api.withdraw_submission"
                                    ) as mock_withdraw:
                                        result = _runner.invoke(
                                            app,
                                            [
                                                "submissions",
                                                "create",
                                                "--division",
                                                "standardized",
                                                "--scenario",
                                                "cop",
                                                "--availability",
                                                "available",
                                                "--run-ids",
                                                RUN_ID,
                                                *_TOKEN_ARGS,
                                            ],
                                        )
        assert result.exit_code == 1
        mock_withdraw.assert_called_once_with(TOKEN, SUBMISSION_ID)


@pytest.mark.unit
class TestSubmissionsRemoveRun:
    _REMOVED_RUN_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"

    def test_remove_run_success(self, tmp_path: Path) -> None:
        sub_out = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}
        fake_archive = _make_fake_archive(tmp_path)
        fake_sub_dir = tmp_path / "sub"
        fake_sub_dir.mkdir()
        # build_submission_folder returns the org dir; the submission level lives below it.
        (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir()
        # These commands pass submission_id to the builder, so the rebuilt tree
        # carries the real id rather than the placeholder.
        (fake_sub_dir / SUBMISSION_ID).mkdir(exist_ok=True)
        fake_bundle = tmp_path / "bundle.tar.gz"
        fake_bundle.write_bytes(b"bundle")

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.remove_run_from_submission",
                return_value=sub_out,
            ):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    return_value=fake_archive,
                ):
                    with patch(
                        "endpoints_submission_cli.commands.submissions.remove_run.build_submission_folder",
                        return_value=fake_sub_dir,
                    ):
                        with patch(
                            "endpoints_submission_cli.commands.submissions.remove_run._run_submission_checker"
                        ):
                            with patch(
                                "endpoints_submission_cli.commands.submissions.remove_run.create_bundle_archive",
                                return_value=fake_bundle,
                            ):
                                with patch(
                                    "endpoints_submission_cli.submissions.api.upload_submission_archive"
                                ):
                                    _run_app(
                                        "submissions",
                                        "remove-run",
                                        "--submission-id",
                                        SUBMISSION_ID,
                                        "--run-id",
                                        self._REMOVED_RUN_ID,
                                        *_TOKEN_ARGS,
                                    )

    def test_remove_run_no_runs_left(self) -> None:
        sub_out = {**SUBMISSION_OUT, "run_ids": []}

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.remove_run_from_submission",
                return_value=sub_out,
            ):
                _run_app(
                    "submissions",
                    "remove-run",
                    "--submission-id",
                    SUBMISSION_ID,
                    "--run-id",
                    self._REMOVED_RUN_ID,
                    *_TOKEN_ARGS,
                )

    def test_remove_run_api_error_exits_1(self) -> None:
        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.remove_run_from_submission",
                side_effect=APIError("not found"),
            ):
                result = _runner.invoke(
                    app,
                    [
                        "submissions",
                        "remove-run",
                        "--submission-id",
                        SUBMISSION_ID,
                        "--run-id",
                        self._REMOVED_RUN_ID,
                        *_TOKEN_ARGS,
                    ],
                )
        assert result.exit_code == 1

    def test_remove_run_download_failure_rolls_back(self, tmp_path: Path) -> None:
        sub_out = {**SUBMISSION_OUT, "run_ids": [RUN_ID]}

        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.remove_run_from_submission",
                return_value=sub_out,
            ):
                with patch(
                    "endpoints_submission_cli.runs.api.download_run_archive",
                    side_effect=APIError("not found"),
                ):
                    with patch(
                        "endpoints_submission_cli.submissions.api.add_run_to_submission"
                    ) as mock_rollback:
                        result = _runner.invoke(
                            app,
                            [
                                "submissions",
                                "remove-run",
                                "--submission-id",
                                SUBMISSION_ID,
                                "--run-id",
                                self._REMOVED_RUN_ID,
                                *_TOKEN_ARGS,
                            ],
                        )
        assert result.exit_code == 1
        mock_rollback.assert_called_once_with(TOKEN, SUBMISSION_ID, self._REMOVED_RUN_ID)


_PROMPT_TEXT = "Would you like to continue?"
_PROVISIONAL = ["--publication-mode", "provisional"]
_EMBARGOED = ["--publication-mode", "confidential_embargoed"]
_WARNING_TEXT = "publicly viewable on the visualizer"


@contextlib.contextmanager
def _patched_create(tmp_path: Path):
    """Patch out every side effect of `submissions create` past the confirmation prompt."""
    fake_sub_dir = tmp_path / "sub"
    fake_sub_dir.mkdir(exist_ok=True)
    # build_submission_folder returns the org dir; the submission level lives below it.
    (fake_sub_dir / PENDING_SUBMISSION_ID).mkdir(exist_ok=True)
    fake_bundle = tmp_path / "bundle.tar.gz"
    fake_bundle.write_bytes(b"bundle")
    targets = [
        patch("endpoints_submission_cli._http.get_token", return_value=TOKEN),
        patch(
            "endpoints_submission_cli.runs.api.download_run_archive",
            return_value=_make_fake_archive(tmp_path),
        ),
        patch(
            "endpoints_submission_cli.commands.submissions.create.build_submission_folder",
            return_value=fake_sub_dir,
        ),
        patch("endpoints_submission_cli.commands.submissions.create._run_submission_checker"),
        patch(
            "endpoints_submission_cli.commands.submissions.create.create_bundle_archive",
            return_value=fake_bundle,
        ),
        patch("endpoints_submission_cli.submissions.api.upload_submission_archive"),
        patch("endpoints_submission_cli.submissions.api.update_submission"),
    ]
    with contextlib.ExitStack() as stack:
        for target in targets:
            stack.enter_context(target)
        yield stack.enter_context(
            patch(
                "endpoints_submission_cli.submissions.api.create_submission",
                return_value=SUBMISSION_OUT,
            )
        )


@pytest.mark.unit
class TestProvisionalConfirmation:
    """`--publication-mode provisional` must be confirmed before anything is submitted."""

    _CREATE_ARGS = [
        "submissions",
        "create",
        "--division",
        "standardized",
        "--scenario",
        "cop",
        "--availability",
        "available",
        "--run-ids",
        RUN_ID,
        *_TOKEN_ARGS,
    ]

    def test_no_prompt_without_provisional(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, self._CREATE_ARGS)
        assert result.exit_code == 0, result.output
        assert _PROMPT_TEXT not in result.output
        assert mock_create.call_args[0][1]["publication_mode"] == "confidential"
        assert "early_publish" not in mock_create.call_args[0][1]

    def test_declining_prompt_exits_1_without_downloading(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, *_PROVISIONAL], input="n\n")
        assert result.exit_code == 1
        assert _WARNING_TEXT in result.output
        assert "peer review pending" in result.output
        mock_create.assert_not_called()

    def test_accepting_prompt_sets_flag(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, *_PROVISIONAL], input="y\n")
        assert result.exit_code == 0, result.output
        assert _PROMPT_TEXT in result.output
        assert "as soon as the automated compliance checks pass" in _flat(result.output)
        assert mock_create.call_args[0][1]["publication_mode"] == "provisional"

    def test_empty_answer_declines(self, tmp_path: Path) -> None:
        """The prompt defaults to "no" — a bare Enter must not publish."""
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, *_PROVISIONAL], input="\n")
        assert result.exit_code == 1
        mock_create.assert_not_called()

    def test_yes_flag_skips_prompt(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, *_PROVISIONAL, "--yes"])
        assert result.exit_code == 0, result.output
        assert _PROMPT_TEXT not in result.output
        assert _WARNING_TEXT in result.output
        assert mock_create.call_args[0][1]["publication_mode"] == "provisional"

    def test_dry_run_skips_prompt(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, *_PROVISIONAL, "--dry-run"])
        assert result.exit_code == 0, result.output
        assert _PROMPT_TEXT not in result.output
        mock_create.assert_not_called()


@pytest.mark.unit
class TestPublicationMode:
    """--publication-mode declares one of the three modes of rules §6.2; the embargo must fit it."""

    _CREATE_ARGS = TestProvisionalConfirmation._CREATE_ARGS
    _FUTURE = "2099-01-06T16:00:00Z"

    def test_embargo_alone_is_confidential_embargoed(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(
                app, [*self._CREATE_ARGS, *_EMBARGOED, "--embargo-date", self._FUTURE]
            )
        assert result.exit_code == 0, result.output
        assert _PROMPT_TEXT not in result.output
        assert "not published before 2099-01-06T16:00:00+00:00" in _flat(result.output)
        payload = mock_create.call_args[0][1]
        assert payload["publication_mode"] == "confidential_embargoed"
        assert payload["embargo_date"] == "2099-01-06T16:00:00+00:00"

    def test_provisional_with_embargo(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(
                app,
                [*self._CREATE_ARGS, *_PROVISIONAL, "--embargo-date", self._FUTURE],
                input="y\n",
            )
        assert result.exit_code == 0, result.output
        assert "from the embargo date, 2099-01-06T16:00:00+00:00" in _flat(result.output)
        payload = mock_create.call_args[0][1]
        assert payload["publication_mode"] == "provisional"

    def test_embargo_without_offset_is_sent_as_utc(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(
                app, [*self._CREATE_ARGS, *_EMBARGOED, "--embargo-date", "2099-01-06T16:00:00"]
            )
        assert result.exit_code == 0, result.output
        assert "taking it as UTC" in _flat(result.output)
        assert mock_create.call_args[0][1]["embargo_date"] == "2099-01-06T16:00:00+00:00"

    def test_embargo_offset_is_kept(self, tmp_path: Path) -> None:
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(
                app,
                [*self._CREATE_ARGS, *_EMBARGOED, "--embargo-date", "2099-01-06T08:00:00-08:00"],
            )
        assert result.exit_code == 0, result.output
        assert mock_create.call_args[0][1]["embargo_date"] == "2099-01-06T08:00:00-08:00"

    @pytest.mark.parametrize(
        ("value", "message"),
        [("next tuesday", "Invalid --embargo-date"), ("2020-01-01T00:00:00Z", "not in the future")],
    )
    def test_bad_embargo_exits_before_downloading(
        self, tmp_path: Path, value: str, message: str
    ) -> None:
        with _patched_create(tmp_path) as mock_create:
            with patch("endpoints_submission_cli.runs.api.download_run_archive") as mock_download:
                result = _runner.invoke(
                    app, [*self._CREATE_ARGS, *_EMBARGOED, "--embargo-date", value]
                )
        assert result.exit_code == 1
        assert message in _flat(result.output)
        mock_download.assert_not_called()
        mock_create.assert_not_called()

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--embargo-date", "2099-01-06T16:00:00Z"], "--embargo-date needs"),
            (_EMBARGOED, "requires --embargo-date"),
        ],
        ids=["confidential-with-embargo", "embargoed-without-embargo"],
    )
    def test_embargo_must_fit_mode(self, tmp_path: Path, args: list[str], message: str) -> None:
        with _patched_create(tmp_path) as mock_create:
            with patch("endpoints_submission_cli.runs.api.download_run_archive") as mock_download:
                result = _runner.invoke(app, [*self._CREATE_ARGS, *args])
        assert result.exit_code == 1
        assert message in _flat(result.output)
        mock_download.assert_not_called()
        mock_create.assert_not_called()

    def test_provisional_flag_is_gone(self, tmp_path: Path) -> None:
        """Replaced by --publication-mode provisional; no alias is kept."""
        with _patched_create(tmp_path) as mock_create:
            result = _runner.invoke(app, [*self._CREATE_ARGS, "--provisional"])
        assert result.exit_code == 2
        mock_create.assert_not_called()

    def test_update_normalizes_embargo_and_allows_past(self) -> None:
        """Moving the embargo up, even into the past, is how a submitter lifts it."""
        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch(
                "endpoints_submission_cli.submissions.api.update_submission",
                return_value=SUBMISSION_OUT,
            ) as mock_patch:
                _run_app(
                    "submissions",
                    "update",
                    "--submission-id",
                    SUBMISSION_ID,
                    "--embargo-date",
                    "2020-01-01T00:00:00",
                    *_TOKEN_ARGS,
                )
        assert mock_patch.call_args[0][2] == {"embargo_date": "2020-01-01T00:00:00+00:00"}

    def test_update_rejects_unparseable_embargo(self) -> None:
        with patch("endpoints_submission_cli._http.get_token", return_value=TOKEN):
            with patch("endpoints_submission_cli.submissions.api.update_submission") as mock_patch:
                result = _runner.invoke(
                    app,
                    [
                        "submissions",
                        "update",
                        "--submission-id",
                        SUBMISSION_ID,
                        "--embargo-date",
                        "soon",
                        *_TOKEN_ARGS,
                    ],
                )
        assert result.exit_code == 1
        mock_patch.assert_not_called()


class TestNoPublicationCycleFlag:
    """Nobody picks a cycle at submit time any more; the lifecycle manager
    publishes from the finalization time and records the cycle it used."""

    @pytest.mark.parametrize(
        "args",
        [
            ["submissions", "create", "--publication-cycle", "2026-11-C0"],
            [
                "submissions",
                "update",
                "--submission-id",
                "x",
                "--publication-cycle",
                "2026-11-C0",
            ],
        ],
    )
    def test_flag_is_rejected(self, args: list[str]) -> None:
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 2
        assert "No such option: --publication-cycle" in result.output


@pytest.mark.unit
class TestCreateLocalRemoved:
    """`submissions create-local` was deprecated in 1.1 and is gone."""

    def test_command_no_longer_exists(self) -> None:
        result = CliRunner().invoke(app, ["submissions", "create-local", "--help"])
        assert result.exit_code == 2
        assert "No such command 'create-local'" in result.output

    def test_not_listed_in_help(self) -> None:
        result = CliRunner().invoke(app, ["submissions", "--help"])
        assert result.exit_code == 0
        assert "create-local" not in result.output
