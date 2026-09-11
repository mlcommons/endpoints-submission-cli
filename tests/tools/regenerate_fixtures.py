#!/usr/bin/env python3
"""Regenerate ``test_submissions/`` from v0.7 layout to MLPerf Endpoints v1.0.

Committed and idempotent: every step reads the current state and writes the desired
one, so a second run produces no diff. That matters because the corpus is edited by
hand between rounds — a one-shot migration script would be unrunnable the moment
someone touched a fixture, and the next format change would start from scratch.

What it does, per submission tree:

1. **Per-point ``system_desc.json``** (policies PR #119). The per-system
   ``results/<system>/system_desc_id.json`` is folded into every ``r<N>/`` together
   with the §8.2 fields absorbed from that point's ``run_metadata.json``; both source
   files are then deleted.
2. **§8.2 renames.** ``system_availability_status`` → ``publication_status``; the flat
   node-level ``accelerator_*`` fields → ``accelerator_info[]``.
3. **``tps_utilization`` recomputed** per Pareto curve as ``system_tps / max(system_tps)``
   with ``system_tps`` derived from ``result_summary.json``, so the stored value agrees
   with the measurement the checker recomputes it from.
4. **§8.3 disclosure added to every ``point.yaml``** — the ten §8.3 fields plus
   ``shared_src``, ``shared_docs``, ``seed_set`` and ``target_cohort``, and the §4.6 RNG
   seeds under their v1.0 names.
5. **``result_summary.json`` gains ``tpot``** (migrated from ``run_metadata``'s
   ``measured_latency_tpot_*``, in nanoseconds), a TTFT P90, and ``system_tps`` /
   ``tps_per_user`` consistent with the v1.0 formulas.
6. **Stale artifacts removed** — ``results_summary.json`` and ``point_<N>.yaml`` from
   the pre-``r<N>`` naming.
7. **``valid_standardized`` point repair** — see :data:`POINT_RENAMES`.

Run with ``uv run python tests/tools/regenerate_fixtures.py``; ``--check`` exits 1 if
anything would change, which is what the idempotence test asserts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from submission_checker import layout  # noqa: E402
from submission_checker.seed_sets import load_seed_sets  # noqa: E402

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "test_submissions"

#: The seed set every fixture binds, and the cohort it targets.
SEED_SET_ID = "A"
TARGET_COHORT = "2026-09-C0"

#: Pareto points that must move for the fixture to remain valid under v1.0.
#:
#: ``valid_standardized`` is the corpus's must-pass tree, and it stopped passing:
#: ``C_max=1000`` with a derived ``C_min=16`` puts Low Concurrency at 17–26, and its
#: points were 16/38/88/256/512/768/1000 — nothing landed in that window. v0.7's fixed
#: 33–42 window caught 38, which is why the fixture was built that way. Moving the
#: point to 20 is the smallest change that restores coverage.
POINT_RENAMES: dict[str, dict[int, int]] = {
    "valid_standardized": {38: 20},
}

#: §8.2 fields absorbed from ``run_metadata.json`` when policies PR #119 removed it.
_ABSORBED_FIELDS = (
    "node_config",
    "config_summary",
    "config_summary_notes",
    "disaggregated",
    "expert_parallel",
    "tensor_parallel",
    "pipeline_parallel",
    "data_parallel",
    "batch",
    "link_config",
)

#: Node fields §8.2.1 moved into ``accelerator_info[]``.
_ACCELERATOR_FIELDS = (
    "accelerator_model_name",
    "accelerators_per_node",
    "accelerator_memory_capacity",
    "accelerator_memory_type",
    "accelerator_interconnect",
    "accelerator_host_interconnect",
)

#: Stale filenames from the pre-``r<N>`` layout, deleted wherever they appear.
_STALE_GLOBS = ("results_summary.json", "point_*.yaml", "run_metadata.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would change and exit 1 if anything would, without writing.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=FIXTURE_ROOT,
        help=f"Fixture corpus to regenerate (default: {FIXTURE_ROOT}).",
    )
    args = parser.parse_args(argv)

    seed_set = load_seed_sets()[SEED_SET_ID]
    changes: list[str] = []
    for submission_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        changes += regenerate_submission(submission_dir, seed_set, dry_run=args.check)

    for change in changes:
        print(change)
    if args.check:
        print(f"{len(changes)} change(s) pending" if changes else "up to date")
        return 1 if changes else 0
    print(f"{len(changes)} change(s) applied" if changes else "already up to date")
    return 0


def regenerate_submission(submission_dir: Path, seed_set: Any, *, dry_run: bool) -> list[str]:
    """Bring one ``<submission>/`` tree up to the v1.0 layout."""
    changes: list[str] = []
    results_dir = submission_dir / layout.RESULTS_DIR
    if not results_dir.is_dir():
        return changes

    changes += _apply_point_renames(submission_dir, dry_run=dry_run)

    for system_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        legacy_desc_path = system_dir / "system_desc_id.json"
        base_desc = _system_description_for(system_dir, legacy_desc_path)

        for model_dir in sorted(p for p in system_dir.iterdir() if p.is_dir()):
            changes += _regenerate_curve(model_dir, base_desc, seed_set, dry_run=dry_run)

        if legacy_desc_path.exists():
            changes.append(f"delete {_rel(legacy_desc_path)}")
            if not dry_run:
                legacy_desc_path.unlink()

    # After the curves: _implementation_name reads the migrated system_desc.json.
    changes += _ensure_shared_trees(submission_dir, dry_run=dry_run)
    changes += _remove_stale_files(submission_dir, dry_run=dry_run)
    return changes


def _regenerate_curve(
    model_dir: Path,
    base_desc: dict[str, Any],
    seed_set: Any,
    *,
    dry_run: bool,
) -> list[str]:
    """Regenerate every point of one Pareto curve (§8.5: one system, one model)."""
    changes: list[str] = []
    point_dirs = layout.iter_point_dirs(model_dir)
    if not point_dirs:
        return changes

    # tps_utilization normalises against the peak of this curve, so the whole curve
    # has to be read before any point can be written.
    summaries = {d: _read_json(d / layout.RESULT_SUMMARY_JSON) or {} for d in point_dirs}
    tps = {d: _derive_system_tps(s) for d, s in summaries.items()}
    max_tps = max((v for v in tps.values() if v is not None), default=None)

    for point_dir in point_dirs:
        legacy_meta = _read_json(point_dir / "run_metadata.json") or {}
        concurrency = layout.parse_point_dir(point_dir.name) or 0

        desc = dict(base_desc)
        desc.update({k: legacy_meta[k] for k in _ABSORBED_FIELDS if legacy_meta.get(k) is not None})
        point_tps, util = tps[point_dir], None
        if max_tps and max_tps > 0 and point_tps is not None:
            util = round(point_tps / max_tps, 6)
        if util is not None:
            desc["tps_utilization"] = util
        changes += _write_json(point_dir / layout.SYSTEM_DESC_JSON, desc, dry_run=dry_run)

        summary = _modernise_summary(summaries[point_dir], legacy_meta)
        changes += _write_json(point_dir / layout.RESULT_SUMMARY_JSON, summary, dry_run=dry_run)

        changes += _write_point_yaml(point_dir, concurrency, base_desc, seed_set, dry_run=dry_run)
    return changes


# ---------------------------------------------------------------------------
# system_desc.json
# ---------------------------------------------------------------------------


def _system_description_for(system_dir: Path, legacy_path: Path) -> dict[str, Any]:
    """The system's description, from the legacy per-system file or an already-migrated point.

    Reading back from a point is what makes a second run a no-op: the first run deletes
    ``system_desc_id.json``, so without this the regenerator would overwrite every
    ``system_desc.json`` with an empty one.
    """
    legacy = _read_json(legacy_path)
    if legacy is not None:
        return _modernise_system_desc(legacy)
    for _system, model_dir in layout.iter_curves(system_dir.parent):
        if model_dir.parent != system_dir:
            continue
        for point_dir in layout.iter_point_dirs(model_dir):
            migrated = _read_json(point_dir / layout.SYSTEM_DESC_JSON)
            if migrated is not None:
                # tps_utilization is per point; the curve loop sets it again.
                return {k: v for k, v in migrated.items() if k != "tps_utilization"}
    return {}


def _modernise_system_desc(desc: dict[str, Any]) -> dict[str, Any]:
    """Apply the §8.2 renames to a system description."""
    out = dict(desc)
    for legacy in ("system_availability_status", "availability_status"):
        value = out.pop(legacy, None)
        if value is not None and "publication_status" not in out:
            out["publication_status"] = value
    out["node_types"] = [_modernise_node(n) for n in out.get("node_types") or []]
    return out


def _modernise_node(node: dict[str, Any]) -> dict[str, Any]:
    """Fold a node's flat ``accelerator_*`` fields into ``accelerator_info[]`` (§8.2.1)."""
    out = dict(node)
    if out.get("accelerator_info"):
        return out
    flat = {name: out.pop(name) for name in _ACCELERATOR_FIELDS if out.get(name) is not None}
    for name in _ACCELERATOR_FIELDS:
        out.pop(name, None)
    if flat:
        out["accelerator_info"] = [flat]
    return out


# ---------------------------------------------------------------------------
# result_summary.json
# ---------------------------------------------------------------------------


def _modernise_summary(summary: dict[str, Any], legacy_meta: dict[str, Any]) -> dict[str, Any]:
    """Add the v1.0 metric fields, migrating TPOT out of ``run_metadata.json``.

    ``run_metadata`` recorded latencies in milliseconds; ``result_summary`` serialises
    them in nanoseconds like ``ttft``, so every migrated value is scaled.
    """
    out = dict(summary)

    tpot_ns = {
        percentile: float(legacy_meta[f"measured_latency_tpot_p{percentile}"]) * 1e6
        for percentile in ("50", "90", "95", "99")
        if legacy_meta.get(f"measured_latency_tpot_p{percentile}") is not None
    }
    if tpot_ns:
        existing = out.get("tpot") if isinstance(out.get("tpot"), dict) else {}
        average_ms = legacy_meta.get("measured_latency_tpot_average")
        out["tpot"] = {
            **existing,
            "total": float(average_ms) * 1e6 * int(out.get("n_samples_completed") or 0)
            if average_ms is not None
            else existing.get("total", 0.0),
            "percentiles": tpot_ns,
        }

    # §4.1 reports TTFT at P90; the corpus was built with 50/95 only. Interpolated
    # from the summary's own percentiles rather than migrated from run_metadata: the
    # two files were never reconciled in this corpus, and run_metadata's TTFT numbers
    # would land above the summary's own P95, producing a non-monotonic distribution.
    ttft = out.get("ttft")
    if isinstance(ttft, dict) and isinstance(ttft.get("percentiles"), dict):
        percentiles = dict(ttft["percentiles"])
        if "90" not in percentiles and {"50", "95"} <= set(percentiles):
            p50, p95 = percentiles["50"], percentiles["95"]
            percentiles["90"] = p50 + (p95 - p50) * 0.8
        out["ttft"] = {**ttft, "percentiles": percentiles}

    # Stored metrics must agree with the formulas the checker recomputes.
    system_tps = _derive_system_tps(out)
    if system_tps is not None:
        out["system_tps"] = round(system_tps, 4)
    tpot_p90_ms = tpot_ns.get("90", 0.0) / 1e6
    if tpot_p90_ms > 0:
        out["tps_per_user"] = round(1000.0 / tpot_p90_ms, 4)
    return out


def _dataset_type(value: object) -> str:
    """Normalise a dataset_type to one of §8.3's three spellings."""
    text = str(value or "Performance").strip().lower()
    return {
        "accuracy": "Accuracy",
        "performance": "Performance",
        "accuracy + performance": "Accuracy + Performance",
    }.get(text, "Performance")


def _derive_system_tps(summary: dict[str, Any]) -> float | None:
    """``total output tokens / elapsed seconds``, matching ``PointSummary.system_tps``."""
    try:
        duration_ns = float(summary.get("duration_ns") or 0.0)
        lengths = summary.get("output_sequence_lengths") or {}
        total = float(lengths.get("total") or 0.0)
    except (TypeError, ValueError):
        return None
    return total / (duration_ns / 1e9) if duration_ns > 0 else None


# ---------------------------------------------------------------------------
# point.yaml
# ---------------------------------------------------------------------------


def _write_point_yaml(
    point_dir: Path,
    concurrency: int,
    desc: dict[str, Any],
    seed_set: Any,
    *,
    dry_run: bool,
) -> list[str]:
    """Add the §8.3 disclosure and §4.6 seeds to a point's ``point.yaml``."""
    path = point_dir / layout.POINT_YAML
    current = yaml.safe_load(path.read_text()) if path.exists() else None
    point: dict[str, Any] = dict(current) if isinstance(current, dict) else {}
    point["concurrency"] = concurrency

    point.setdefault("division", desc.get("division") or "Standardized")
    if desc.get("max_supported_concurrency") is not None:
        point["max_supported_concurrency"] = desc["max_supported_concurrency"]
    point.setdefault("model_name", desc.get("model_name") or desc.get("model_id") or "")
    point.setdefault("model_precision", desc.get("model_precision") or "FP16")
    point.setdefault("link_to_model", desc.get("link_to_model") or "https://example.com/model")
    point.setdefault("link_to_model_transformation", "https://example.com/quantization")
    point.setdefault("model_notes", "")
    point.setdefault("dataset_name", desc.get("dataset_name") or "CNN/DailyMail")
    # §8.3 spells these "Accuracy", "Performance", "Accuracy + Performance".
    point.setdefault("dataset_type", _dataset_type(desc.get("dataset_type")))
    point.setdefault("dataset_link", desc.get("dataset_link") or "https://example.com/dataset")

    # §8.1 fixes both locations, so the pointers are the directory names themselves;
    # a fixture missing one fails required-dir / src-dir, which is the right report.
    point["shared_src"] = layout.SRC_DIR
    point["shared_docs"] = layout.DOCS_DIR
    point["seed_set"] = seed_set.id
    point["target_cohort"] = TARGET_COHORT

    runtime_settings = dict(point.get("runtime_settings") or {})
    runtime = dict(runtime_settings.get("runtime") or {})
    runtime.pop("scheduler_random_seed", None)
    runtime.pop("dataloader_random_seed", None)
    runtime.update(seed_set.seeds)
    runtime_settings["runtime"] = runtime
    point["runtime_settings"] = runtime_settings

    warmup = point.get("warmup")
    if isinstance(warmup, dict):
        warmup = dict(warmup)
        warmup.setdefault("logs_retained", True)
        warmup.setdefault("link_logs", "https://example.com/warmup-logs")
        # A renamed point leaves its warmup block describing the old concurrency, so
        # rescale it: warmup runs at the point's own concurrency by construction here.
        old = warmup.get("concurrency")
        if isinstance(old, int) and old > 0 and old != concurrency and concurrency > 0:
            ratio = concurrency / old
            warmup["concurrency"] = concurrency
            for key in ("requests_issued", "requests_completed"):
                value = warmup.get(key)
                if isinstance(value, int):
                    warmup[key] = max(1, round(value * ratio))
        point["warmup"] = warmup

    return _write_yaml(path, point, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Renames, deletions, and idempotent writes
# ---------------------------------------------------------------------------


def _ensure_shared_trees(submission_dir: Path, *, dry_run: bool) -> list[str]:
    """Give the fixture the ``src/<impl>/README.md`` §2.2.1 requires of every division.

    The corpus predates that rule, so every ``sub_*`` tree failed ``src-dir`` — an error
    no test asserted and every test had to look past. Now that each point's
    ``shared_src`` must also resolve, a missing tree would report the same gap eleven
    more times per fixture. A fixture should fail for the reason it documents and
    nothing else, so the tree is created here rather than tolerated.
    """
    changes: list[str] = []
    impl = _implementation_name(submission_dir)
    readme = submission_dir / layout.SRC_DIR / impl / layout.README_MD
    if not readme.is_file():
        changes.append(f"write {_rel(readme)}")
        if not dry_run:
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                f"# {impl}\n\nAnonymised fixture implementation directory."
                " Real submissions document how to build the SUT and reproduce a point here.\n"
            )
    docs_dir = submission_dir / layout.DOCS_DIR
    if not docs_dir.is_dir():
        changes.append(f"mkdir {_rel(docs_dir)}")
        if not dry_run:
            docs_dir.mkdir(parents=True, exist_ok=True)
            (docs_dir / ".gitkeep").write_text("")
    return changes


def _implementation_name(submission_dir: Path) -> str:
    """Name the implementation directory after whatever the fixture says it served with."""
    for _system, model_dir in layout.iter_curves(submission_dir / layout.RESULTS_DIR):
        for point_dir in layout.iter_point_dirs(model_dir):
            desc = _read_json(point_dir / layout.SYSTEM_DESC_JSON) or {}
            framework = str(desc.get("serving_framework") or "").strip()
            if framework:
                return framework.split()[0].lower().replace("/", "_")
    return "reference"


def _apply_point_renames(submission_dir: Path, *, dry_run: bool) -> list[str]:
    """Move the Pareto points listed in :data:`POINT_RENAMES` to their new concurrency."""
    renames = POINT_RENAMES.get(submission_dir.name)
    if not renames:
        return []
    changes: list[str] = []
    for _system_dir, model_dir in layout.iter_curves(submission_dir / layout.RESULTS_DIR):
        for old, new in renames.items():
            src = model_dir / layout.point_dir_name(old)
            dst = model_dir / layout.point_dir_name(new)
            if not src.is_dir() or dst.exists():
                continue
            changes.append(f"rename {_rel(src)} -> {dst.name}/")
            if not dry_run:
                shutil.move(str(src), str(dst))
    return changes


def _remove_stale_files(submission_dir: Path, *, dry_run: bool) -> list[str]:
    """Delete artifacts left over from layouts the spec no longer defines."""
    changes: list[str] = []
    for pattern in _STALE_GLOBS:
        for path in sorted(submission_dir.rglob(pattern)):
            if not path.is_file():
                continue
            changes.append(f"delete {_rel(path)}")
            if not dry_run:
                path.unlink()
    return changes


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any], *, dry_run: bool) -> list[str]:
    """Write *data* only when it differs from what is on disk, so reruns are no-ops."""
    text = json.dumps(data, indent=2) + "\n"
    if path.is_file() and path.read_text() == text:
        return []
    if not dry_run:
        path.write_text(text)
    return [f"write {_rel(path)}"]


def _write_yaml(path: Path, data: dict[str, Any], *, dry_run: bool) -> list[str]:
    text = yaml.safe_dump(data, sort_keys=True, default_flow_style=False)
    if path.is_file() and path.read_text() == text:
        return []
    if not dry_run:
        path.write_text(text)
    return [f"write {_rel(path)}"]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(FIXTURE_ROOT.parent.parent.parent))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
