# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Shared helper for trimming the verbose ``responses`` list in results.json.

The raw ``responses`` array captured during a run can be very large (one entry
per sample). It is only kept for spot-checking, so both the run-upload archiver
and the submission builder cap it to keep payloads small and uploads reliable.
"""

from __future__ import annotations

import json

from .exceptions import TruncationError

__all__ = ["RESPONSES_LIMIT", "truncate_responses"]

RESPONSES_LIMIT = 10 * 1024  # 10 KB


def truncate_responses(content: bytes) -> bytes:
    """Truncate the ``responses`` collection in a results.json payload to stay under 10 KB.

    ``responses`` may be either a list (``[entry, ...]``) or a dict keyed by sample
    id (``{uuid: output, ...}``); both are produced by different run modes. Entries
    are kept in iteration order until adding the next would exceed the limit. Returns
    *content* unchanged when it is not JSON or has no non-empty ``responses``. Only the
    ``responses`` key is affected; all other fields (accuracy scores, config, ...) are
    preserved.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return content
    responses = data.get("responses")
    if responses is None or (isinstance(responses, (list, dict)) and not responses):
        return content  # nothing to truncate
    if isinstance(responses, list):
        # Each item contributes its own bytes plus 2 for the ", " separator after the first.
        total = 2  # "[]"
        idx = 0
        for i, r in enumerate(responses):
            total += len(json.dumps(r).encode()) + (2 if i > 0 else 0)
            if total > RESPONSES_LIMIT:
                break
            idx = i + 1
        data["responses"] = responses[:idx]
    elif isinstance(responses, dict):
        # Each entry contributes ``"key": value`` plus 2 for the ", " separator after
        # the first. Approximate the key/value cost conservatively to stay under the cap.
        total = 2  # "{}"
        kept: dict[str, object] = {}
        for i, (k, v) in enumerate(responses.items()):
            entry = len(json.dumps(k).encode()) + len(json.dumps(v).encode()) + 2  # ': '
            if i > 0:
                entry += 2  # ", "
            if total + entry > RESPONSES_LIMIT:
                break
            total += entry
            kept[k] = v
        data["responses"] = kept
    else:
        # Unknown shape — refuse to ship a payload we cannot bound. A silent pass-through
        # here is what let a multi-GB results.json reach a submission bundle.
        raise TruncationError(
            f"Cannot truncate 'responses' of type {type(responses).__name__}; "
            "expected a list or dict."
        )
    # Defense-in-depth: never return a payload whose responses still exceed the cap.
    if len(json.dumps(data["responses"]).encode()) > RESPONSES_LIMIT:
        raise TruncationError("responses still exceed the size limit after truncation")
    return json.dumps(data, indent=2).encode()
