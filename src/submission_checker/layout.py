# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""The submission directory layout, in one place.

Both the builder (which writes a submission) and the checker (which reads one) need
to agree on every file and directory name in §8.1's tree. Before this module those
strings were duplicated across six modules and the point-directory regex was
hand-rolled in three of them, so a format change meant finding every copy.

Layout as of `mlcommons/endpoints_policies` PR #119 (`feat/remove_runtime_meta.json`)::

    <submitting_organization>/
      <submission_id>/
        src/<implementation>/README.md
        docs/
        results/<system>/<model_name>/r<N>/
            point.yaml
            result_summary.json
            accuracy_results.json
            system_desc.json
            server_configs/

Only symbols used by more than one module belong here; anything narrower stays with
its caller so this does not become a dumping ground.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

__all__ = [
    "ACCURACY_RESULTS_JSON",
    "ACCURACY_SUBDIR",
    "CONFIG_YAML",
    "DOCS_DIR",
    "DOCUMENTATION_SUBDIR",
    "POINT_DIR_RE",
    "POINT_YAML",
    "README_MD",
    "REQUIRED_RUN_FILES",
    "RESULTS_DIR",
    "RESULTS_JSON",
    "RESULT_SUMMARY_JSON",
    "SERVER_CONFIGS_DIR",
    "SRC_DIR",
    "SYSTEM_DESC_JSON",
    "iter_curves",
    "iter_point_dirs",
    "parse_point_dir",
    "point_dir_name",
    "resolve_shared_path",
]

# ── File names inside a Pareto-point directory ────────────────────────────────

#: §8.3 measurement-point disclosure.
POINT_YAML = "point.yaml"
#: §8.2 system description. Per point since PR #119; there is no per-system copy.
SYSTEM_DESC_JSON = "system_desc.json"
#: Aggregate metrics for the point.
RESULT_SUMMARY_JSON = "result_summary.json"
#: §6.6 accuracy results.
ACCURACY_RESULTS_JSON = "accuracy_results.json"
#: Raw per-request log. Not in §8.1's tree, but shipped and read for accuracy scores.
RESULTS_JSON = "results.json"
#: Benchmark configuration. Optional as of v1.0 — shipped when the run supplies it.
CONFIG_YAML = "config.yaml"
README_MD = "README.md"

# ── Directory names ───────────────────────────────────────────────────────────

RESULTS_DIR = "results"
DOCS_DIR = "docs"
SRC_DIR = "src"
#: Optional, point-specific, submitter-defined layout.
SERVER_CONFIGS_DIR = "server_configs"

# Subdirectory names as they appear inside a *run* folder, before assembly.
ACCURACY_SUBDIR = "accuracy"
DOCUMENTATION_SUBDIR = "documentation"

#: Files a run folder must contain for `runs create` to accept it.
REQUIRED_RUN_FILES = (SYSTEM_DESC_JSON, POINT_YAML, RESULT_SUMMARY_JSON)

# ── Pareto-point directories ──────────────────────────────────────────────────

#: A Pareto-point directory: "r" followed by the concurrency level (r1, r32, r256).
POINT_DIR_RE = re.compile(r"r(\d+)")


def point_dir_name(concurrency: int) -> str:
    """Return the directory name for a Pareto point at *concurrency*."""
    return f"r{concurrency}"


def parse_point_dir(name: str) -> int | None:
    """Return the concurrency encoded in a point-directory name, or None.

    Args:
        name: A directory name, e.g. ``"r32"``.

    Returns:
        The concurrency level, or ``None`` when *name* is not a point directory.
    """
    match = POINT_DIR_RE.fullmatch(name)
    return int(match.group(1)) if match else None


def iter_point_dirs(model_dir: Path) -> list[Path]:
    """Return *model_dir*'s Pareto-point directories, ordered by concurrency.

    Sorted numerically rather than lexically, so ``r256`` follows ``r32``.
    """
    if not model_dir.is_dir():
        return []
    found = [(parse_point_dir(d.name), d) for d in model_dir.iterdir() if d.is_dir()]
    return [d for concurrency, d in sorted((c, d) for c, d in found if c is not None)]


def iter_curves(results_dir: Path) -> Iterator[tuple[Path, Path]]:
    """Yield every ``(system_dir, model_dir)`` pair under *results_dir*.

    One pair is one Pareto curve — §8.5 defines a result as "one system, one
    benchmark model, one dataset". Region boundaries and TPS-utilisation
    normalisation are both per curve, so both callers walk the tree this way.
    """
    if not results_dir.is_dir():
        return
    for system_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in system_dir.iterdir() if p.is_dir()):
            yield system_dir, model_dir


# ── Shared-content pointers (§8.1, §9.1 "Shared path resolution") ─────────────


def resolve_shared_path(submission_root: Path, value: str) -> Path | None:
    """Resolve a ``shared_src`` / ``shared_docs`` pointer against the submission root.

    §9.1 requires each pointer to "resolve to an existing directory under the
    submission root". Anything absolute or containing a ``..`` component is rejected
    rather than resolved: a pointer that escapes the bundle cannot be reviewed from
    the bundle, and permitting traversal would let a submission reference content it
    does not ship.

    Args:
        submission_root: The ``<submission_id>/`` directory.
        value: The pointer as written in ``point.yaml``.

    Returns:
        The resolved directory, or ``None`` when the value is empty, unsafe, or does
        not name an existing directory.
    """
    if not value or not value.strip():
        return None
    raw = value.strip()
    if raw.startswith("/") or raw.startswith("~"):
        return None
    parts = _path_parts(raw)
    if not parts or any(part == ".." for part in parts):
        return None
    candidate = submission_root.joinpath(*parts)
    return candidate if candidate.is_dir() else None


def _path_parts(value: str) -> list[str]:
    """Split a POSIX-style relative path into its meaningful components."""
    return [part for part in value.replace("\\", "/").split("/") if part and part != "."]
