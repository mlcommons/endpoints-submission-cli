# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Custom exceptions for endpoints-submission-cli."""

__all__ = [
    "APIError",
    "AuthError",
    "ArchiveError",
    "GitHubError",
    "RunFolderError",
    "SubmissionBuildError",
    "SubmissionCheckError",
]


class APIError(Exception):
    """Raised when a Submission API call fails."""


class AuthError(APIError):
    """Raised when the API key is missing or rejected."""


class ArchiveError(Exception):
    """Raised when archive upload, download, or extraction fails."""


class GitHubError(Exception):
    """Raised when a gh CLI operation fails."""


class RunFolderError(Exception):
    """Raised when the run folder is missing required files or has invalid content."""


class SubmissionBuildError(Exception):
    """Raised when assembling the submission folder structure fails."""


class SubmissionCheckError(Exception):
    """Raised when the Submission Checker reports validation errors."""
