# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Shared helper for trimming the verbose ``responses`` list in results.json.

The raw ``responses`` array captured during a run can be very large (one entry
per sample). It is only kept for spot-checking, so both the run-upload archiver
and the submission builder cap it to keep payloads small and uploads reliable.
"""

from __future__ import annotations

import json

__all__ = ["RESPONSES_LIMIT", "truncate_responses"]

RESPONSES_LIMIT = 10 * 1024  # 10 KB


def truncate_responses(content: bytes) -> bytes:
    """Truncate the ``responses`` list in a results.json payload to stay under 10 KB.

    Returns *content* unchanged when it is not JSON, has no ``responses`` list, or
    already fits within the limit. Only the ``responses`` key is affected; all
    other fields (accuracy scores, config, results, ...) are preserved.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return content
    responses = data.get("responses")
    if not isinstance(responses, list) or not responses:
        return content
    # Walk items and stop as soon as adding the next one would exceed the limit.
    # Each item contributes its own bytes plus 2 for the ", " separator after the first.
    total = 2  # "[]"
    idx = 0
    for i, r in enumerate(responses):
        total += len(json.dumps(r).encode()) + (2 if i > 0 else 0)
        if total > RESPONSES_LIMIT:
            break
        idx = i + 1
    data["responses"] = responses[:idx]
    return json.dumps(data, indent=2).encode()
