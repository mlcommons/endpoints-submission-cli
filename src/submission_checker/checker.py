"""Submission checker — orchestrates §9.1 automated compliance checks.

Loading and structural validation live here; all rule logic lives in Pydantic
model validators on PointConfig, PointResult, and ModelContext.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import yaml

__all__ = ["SubmissionChecker"]

from . import layout
from .drafters import ApprovedDrafter, DrafterListError, load_approved_drafters
from .messages import Invalid
from .models import (
    AccuracyResult,
    CheckResult,
    DrafterBinding,
    ModelContext,
    ModelDir,
    Parallelism,
    PointConfig,
    PointPower,
    PointResult,
    PointSummary,
    PowerComputation,
    RegionPlacement,
    Regions,
    Report,
    SeedBinding,
    Severity,
    SrcDir,
    SubmissionDir,
    SystemDescription,
    compute_regions,
)
from .models import err as _err
from .models import ok as _ok
from .models import warn as _warn
from .models.file.system import PIPELINE_ASSIGNED_FIELDS
from .models.file.system_power import AIR_COOLED_OVERHEAD, overhead_for_cooling
from .models.loader import (
    load_accuracy_result,
    load_accuracy_scores,
    load_point_config,
    load_result_summary,
    load_system_description,
    load_system_power,
)
from .models.regions import ULTRA_LOW_CONCURRENCY_MAX
from .seed_sets import SeedSet, SeedSetError, load_seed_sets

if TYPE_CHECKING:
    from pathlib import Path

# Absolute tolerance for the tps_utilization consistency check.
_TPS_UTILIZATION_ABS_TOL = 0.1


# §3.2 — the benchmark models accepted this submission round. The name is read from
# each point's `point.yaml` (§8.3): policies PR #130 removed `model_name` from §8.2's
# `system_desc.json` table and template, and §8.5's Result ID now says `model_id`
# "Must match `model_name` in `point.yaml` (§8.3)".
#
# Every entry is in `layout.canonical_model_name` form, and a declared name must match
# one exactly: the checker never rewrites what a submitter wrote. The canonical form is
# also the §8.1 directory name, so `llama3_1-8b` rather than `llama3.1-8b`.
#
# The agentic three come from the reference implementation's Agentic Inference
# example, which §3.2 makes the authority: "The set of supported benchmark models is
# defined per submission round and maintained in the MLPerf Endpoints reference
# repository."
#
# That sentence also says this list does not belong in a release: §3.2 publishes it
# "at least 6 weeks before the submission round opens", so a new round should not need
# a new checker. `data/seed_sets.yaml` and `data/approved_drafters.yaml` are the
# pattern to follow when that is worth doing.
#
# `deepseek-v4_1-flash` is the README's "DeepSeek-V4.1-Flash", which its example config
# serves as `deepseek-ai/DeepSeek-V4.1-Flash`. It replaced DeepSeek-V4-Pro
# (`deepseek-v4-pro`), which is no longer an accepted benchmark model.
_ALLOWED_MODEL_NAMES = (
    "llama3_1-8b",
    "gpt-oss-120b",
    "deepseek-r1",
    "kimi-k3",
    "qwen3_6-35b-a3b",
    "deepseek-v4_1-flash",
)


def _results_has_accuracy_scores(path: Path) -> bool:
    """True if a results.json carries a non-empty accuracy mapping or native list."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get("accuracy_scores"), (dict, list))
        and bool(data["accuracy_scores"])
    )


#: Fields of ``system_desc.json`` allowed to differ between points of one curve.
#: ``tps_utilization`` is a per-point quantity that §8.2 nonetheless places in the
#: system description — worth raising with the WG, but not a submission defect.
#:
#: The parallelism fields vary by design: §8.1 calls each point's copy the
#: "framework/parallelism/precision for this point", and §4.5.3's worked example (C.3)
#: changes the replica size between points. ``config_summary`` concatenates them, and
#: ``batch`` is tuned per concurrency like any other server setting.
_PER_POINT_SYSTEM_DESC_FIELDS = frozenset(
    {
        "tps_utilization",
        "tensor_parallel",
        "pipeline_parallel",
        "expert_parallel",
        "data_parallel",
        "disaggregated",
        "batch",
        "config_summary",
        "config_summary_notes",
    }
)

#: Keys of ``system_desc.json`` that are not §8.2 fields and are never compared.
#: The pipeline assigns the submission ID and dates (§8.2); a copy left in a
#: submitter's file from a v0.7 template says nothing about the system, so differing
#: copies are not flagged.
_IGNORED_SYSTEM_DESC_FIELDS = frozenset(PIPELINE_ASSIGNED_FIELDS)

#: §4.5 scope: "Power normalization applies to all Standardized division submissions
#: … RDI submissions MAY report normalized throughput but are not required to", and
#: Serviced normalisation "will be introduced in a later version".
_POWER_OPTIONAL_DIVISIONS = frozenset({"rdi", "serviced"})

#: §8.2's parallelism fields, read per point for §4.5.3.
_PARALLELISM_FIELDS = ("tensor_parallel", "pipeline_parallel", "expert_parallel", "data_parallel")


@dataclass
class _LoadedPoint:
    """One Pareto point whose ``point.yaml`` parsed, before regions are known."""

    point_dir: Path
    yaml_path: Path
    config: PointConfig


@dataclass
class _UnparsedPoint:
    """A Pareto point whose ``point.yaml`` is missing or did not validate.

    It is still a point: it counts toward ``C_min`` and coverage, and its other files
    are still checked, so one bad field does not hide the rest of the point or shift
    the curve's region boundaries. ``concurrency`` comes from the raw YAML when that
    field is readable, else from the ``r<N>`` directory name, which
    ``point-dirname-concurrency`` already requires to agree with it.
    """

    point_dir: Path
    yaml_path: Path
    concurrency: int | None


@dataclass
class _SystemFacts:
    """What the power descriptor is checked against from §8.2's system description.

    Attributes:
        overhead: The overhead fraction §8.2's ``cooling`` implies, or ``None`` where it
            names neither liquid nor air.
        cores: ``host_processor_core_count`` per ``system_node_ensemble_id``, for D.2.
        ensembles: Every ``system_node_ensemble_id`` the description declares.
        accelerators: Accelerators per node per ``system_node_ensemble_id`` — §4.5.3's
            divisor where a node set does not state ``accelerator.count_per_node``.
        division: The declared division, lower-cased, or ``None`` if unreadable.
    """

    overhead: float | None
    cores: dict[int, int]
    ensembles: set[int]
    accelerators: dict[int, int]
    division: str | None


def _system_desc_identity(data: dict[str, object]) -> dict[str, object]:
    """Strip the fields that legitimately vary between points of the same curve."""
    return {
        k: v
        for k, v in data.items()
        if k not in _PER_POINT_SYSTEM_DESC_FIELDS and k not in _IGNORED_SYSTEM_DESC_FIELDS
    }


def _read_json(path: Path) -> dict[str, object] | None:
    """Parse *path* as a JSON object, returning None on any read or decode failure."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _positive_int(value: object) -> int | None:
    """*value* if it is a positive integer (and not a bool), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _fallback_concurrency(yaml_path: Path, point_dir: Path) -> int | None:
    """A point's concurrency without a valid ``point.yaml``: raw field, else ``r<N>``."""
    try:
        data = yaml.safe_load(yaml_path.read_text())
    except (OSError, yaml.YAMLError):
        data = None
    raw = _positive_int(data.get("concurrency")) if isinstance(data, dict) else None
    return raw if raw is not None else layout.parse_point_dir(point_dir.name)


def _raw_max_concurrency(sd_path: Path | None) -> int | None:
    """``max_supported_concurrency`` from a ``system_desc.json`` that failed validation."""
    data = _read_json(sd_path) if sd_path is not None else None
    return _positive_int(data.get("max_supported_concurrency")) if data else None


def _as_float(value: object) -> float | None:
    """Coerce a JSON number to float, rejecting bools and everything non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _derived_system_tps(summary_path: Path) -> float | None:
    """Compute ``total output tokens / elapsed seconds`` from a result summary.

    Mirrors :attr:`~submission_checker.models.PointSummary.system_tps` but reads the
    raw JSON, so a summary that fails full validation still contributes its
    throughput to the curve's peak rather than silently shrinking the denominator.
    """
    data = _read_json(summary_path)
    if data is None:
        return None
    duration_ns = _as_float(data.get("duration_ns"))
    lengths = data.get("output_sequence_lengths")
    total = _as_float(lengths.get("total")) if isinstance(lengths, dict) else None
    if duration_ns is None or total is None or duration_ns <= 0:
        return None
    return total / (duration_ns / 1e9)


def _accelerators_per_node(node: dict[str, object]) -> int | None:
    """Accelerators in one §8.2.1 node type, across its ``accelerator_info`` entries.

    v0.7's flat ``accelerators_per_node`` on the node is read where the list is absent,
    as :class:`~submission_checker.models.NodeType` does.
    """
    info = node.get("accelerator_info")
    entries = info if isinstance(info, list) and info else [node]
    counts = [e.get("accelerators_per_node") for e in entries if isinstance(e, dict)]
    ints = [c for c in counts if isinstance(c, int) and not isinstance(c, bool)]
    return sum(ints) if ints else None


def _declared_tps_utilization(system_desc_path: Path) -> float | None:
    """Read ``tps_utilization`` from a point's ``system_desc.json``, or None."""
    data = _read_json(system_desc_path)
    return None if data is None else _as_float(data.get("tps_utilization"))


def _point_parallelism(system_desc_path: Path) -> Parallelism | None:
    """A point's parallelism from its own ``system_desc.json``, or None if it declares none.

    §8.2 places the fields at the top level; the structured ``config_summary`` object
    is read as a fallback. A field left out is 1 — §8.2 defines "TP=1 means no
    partitioning" and likewise for the others — but a description with none of them
    says nothing about its deployment, so it yields ``None`` rather than a guess.
    """
    data = _read_json(system_desc_path)
    if data is None:
        return None
    summary = data.get("config_summary")
    summary = summary if isinstance(summary, dict) else {}
    values: dict[str, int] = {}
    for name in _PARALLELISM_FIELDS:
        value = data.get(name, summary.get(name))
        if isinstance(value, int) and not isinstance(value, bool):
            values[name] = value
    if not values:
        return None
    # §8.2: "Indicates whether the system is disaggregated (disaggregated > 1)", while
    # the model reads it as a bool; either spelling of "yes" counts.
    flag = data.get("disaggregated", summary.get("disaggregated"))
    disaggregated = flag is True or (
        isinstance(flag, int) and not isinstance(flag, bool) and flag > 1
    )
    return Parallelism(**values, disaggregated=disaggregated)


def _resolve_submission_root(path: Path) -> Path:
    """Return the ``<submission_id>/`` level, descending from an org root if needed.

    Submissions are laid out as ``<organisation>/<submission_id>/``, so a path may
    point at either level. A directory that already holds ``results/`` is the
    submission root; otherwise a single child holding ``results/`` is used. An
    ambiguous org root (several submissions) is returned unchanged so the caller
    gets the normal "missing results/" error rather than an arbitrary pick.
    """
    if not path.is_dir() or (path / "results").is_dir():
        return path
    candidates = [d for d in sorted(path.iterdir()) if d.is_dir() and (d / "results").is_dir()]
    return candidates[0] if len(candidates) == 1 else path


class SubmissionChecker:
    """Validates an MLPerf Endpoints submission directory against §9.1 rules.

    The *submission_path* should be a ``<submission_id>/`` directory containing
    ``results/`` and ``docs/`` as specified in §8.1. The submitting organisation's
    directory is also accepted — submissions are laid out as
    ``<organisation>/<submission_id>/`` and the single submission level below the
    org root is resolved automatically.

    Args:
        submission_path: Directory of the submission to validate.

    Example::

        checker = SubmissionChecker(Path("/submissions/acme_corp/sub-123"))
        report = checker.run()
        for err in report.errors:
            print(err.rule, err.message)
    """

    def __init__(
        self,
        submission_path: Path,
        seed_sets_path: Path | None = None,
        approved_drafters_path: Path | None = None,
    ) -> None:
        self.submission_path = _resolve_submission_root(submission_path)
        self.seed_sets_path = seed_sets_path
        self.approved_drafters_path = approved_drafters_path
        self._seed_sets: dict[str, SeedSet] = {}
        self._seed_sets_error: str | None = None
        self._drafters: dict[str, list[ApprovedDrafter]] = {}
        self._drafters_error: str | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Report:
        """Run all §9.1 automated checks and return an aggregated report.

        Returns:
            A :class:`~submission_checker.models.Report` with every
            :class:`~submission_checker.models.CheckResult` produced.
        """
        report = Report(submission_path=self.submission_path)

        try:
            self._seed_sets = load_seed_sets(self.seed_sets_path)
        except SeedSetError as exc:
            # Without the registry every seed rule would report a defect the submitter
            # cannot fix, so say once that the checker is misconfigured and skip them.
            self._seed_sets = {}
            self._seed_sets_error = str(exc)
            report.results.append(_warn("seed-set-registry", "unavailable", None, detail=str(exc)))

        try:
            self._drafters = load_approved_drafters(self.approved_drafters_path)
        except DrafterListError as exc:
            self._drafters = {}
            self._drafters_error = str(exc)
            report.results.append(
                _warn("drafter-list-registry", "unavailable", None, detail=str(exc))
            )

        if not self.submission_path.exists():
            report.results.append(
                _err(
                    "path-exists",
                    "fail",
                    self.submission_path,
                    submission_path=self.submission_path,
                )
            )
            return report
        report.results.append(_ok("path-exists", "pass", self.submission_path))

        submission_dir = SubmissionDir(root=self.submission_path)
        report.results.extend(submission_dir._check_results)

        # src/ is shared by the whole submission (§2.2.1), so it is checked once here
        # rather than per curve, and whatever else is missing.
        report.results.extend(SrcDir(root=self.submission_path)._check_results)

        # Only a missing results/ leaves nothing further to check. A missing docs/
        # is reported above and must not hide every result check behind it.
        results_dir = submission_dir.results_dir
        if not results_dir.is_dir():
            return report

        # Since policies PR #119 there is no per-system file: a system is simply a
        # directory under results/, and its description lives in every point's
        # system_desc.json.
        system_dirs = [d for d in sorted(results_dir.iterdir()) if d.is_dir()]
        if not system_dirs:
            report.results.append(_err("system-results-dir", "fail", results_dir))
            return report

        for system_dir in system_dirs:
            report.results.extend(self._check_system(system_dir))

        # Per-curve: tps_utilization must match system_tps / max(system_tps)
        # within each <system>/<benchmark_model> pareto curve.
        report.results.extend(self._check_tps_utilization(results_dir))

        # §15: at least one model must carry accuracy results — either as accuracy_scores
        # embedded in a results.json, or as a standalone accuracy/results.json.
        has_full_accuracy = any(True for _ in results_dir.rglob("accuracy_results.json")) or any(
            _results_has_accuracy_scores(p) for p in results_dir.rglob("results.json")
        )

        if has_full_accuracy:
            report.results.append(_ok("accuracy-present", "pass", results_dir))
        else:
            report.results.append(_err("accuracy-present", "fail", results_dir))

        return report

    # ------------------------------------------------------------------
    # Submission-wide checks
    # ------------------------------------------------------------------

    def _check_tps_utilization(self, results_dir: Path) -> list[CheckResult]:
        """Verify each point's ``tps_utilization`` equals ``system_tps / max(system_tps)``.

        ``tps_utilization`` normalises a point to the peak ``system_tps`` of the
        system+model curve it belongs to — NOT across the whole submission.
        Normalising submission-wide is wrong when a submission contains more than
        one system: e.g. an ``MI355X_1x`` and an ``MI355X_8x`` config sharing a
        folder would force every 1x point to be divided by the 8x peak, so the
        smaller system can never match its stored (per-curve) values.

        The stored value lives in the point's ``system_desc.json`` (§8.2 absorbed it
        when policies PR #119 removed ``run_metadata.json``), while ``system_tps`` is
        derived from ``result_summary.json`` — the measurement itself — so a submitter
        cannot declare a throughput that disagrees with what they measured.

        Stored values are compared to the recomputed expectation within an absolute
        tolerance of ``_TPS_UTILIZATION_ABS_TOL``. Files that are missing or
        structurally invalid are left to the per-file presence and validity checks.
        """
        results: list[CheckResult] = []
        for system_dir, model_dir in layout.iter_curves(results_dir):
            entries: list[tuple[Path, float, float]] = []
            for point_dir in layout.iter_point_dirs(model_dir):
                tps = _derived_system_tps(point_dir / layout.RESULT_SUMMARY_JSON)
                sd_path = point_dir / layout.SYSTEM_DESC_JSON
                util = _declared_tps_utilization(sd_path)
                if tps is not None and util is not None:
                    entries.append((sd_path, tps, util))

            if not entries:
                continue
            max_tps = max(tps for _, tps, _ in entries)
            if max_tps <= 0:
                continue
            curve = f"{system_dir.name}/{model_dir.name}"
            for sd_path, tps, util in entries:
                expected = tps / max_tps
                if abs(util - expected) <= _TPS_UTILIZATION_ABS_TOL:
                    results.append(
                        _ok("tps-utilization", "pass", sd_path, util=util, expected=expected)
                    )
                else:
                    results.append(
                        _err(
                            "tps-utilization",
                            "fail",
                            sd_path,
                            util=util,
                            expected=expected,
                            tps=tps,
                            max_tps=max_tps,
                            curve=curve,
                            tps_utilization_abs_tol=_TPS_UTILIZATION_ABS_TOL,
                        )
                    )
        return results

    # ------------------------------------------------------------------
    # Per-system orchestration
    # ------------------------------------------------------------------

    def _check_system(self, system_dir: Path) -> list[CheckResult]:
        """Run every check scoped to one ``results/<system>/`` directory.

        The system description is no longer a per-system file, so a system contributes
        no checks of its own beyond holding at least one benchmark-model directory;
        everything else is per curve.
        """
        results: list[CheckResult] = []
        system_id = system_dir.name

        facts = self._system_facts(system_dir)
        power, power_results = self._load_system_power(system_dir, facts)
        results.extend(power_results)

        model_dirs = [d for d in sorted(system_dir.iterdir()) if d.is_dir()]
        if not model_dirs:
            results.append(_err("benchmark-model-dir", "fail", system_dir, system_id=system_id))
            return results

        for model_dir in model_dirs:
            results.extend(self._check_model(system_id, model_dir, power))

        return results

    def _system_facts(self, system_dir: Path) -> _SystemFacts:
        """What Appendix E needs from §8.2's description of this system.

        The system description is per point since policies PR #119, but §8.2 describes
        one system, and ``system-description-consistency`` already reports points of a
        curve that disagree. So the first readable one answers the question.

        §8.2's table lists ``cooling`` as a system field while §8.2.1's template nests
        it under ``node_types``, so both placements are read. Node types cooled
        differently resolve to the *air-cooled* fraction: §4.5.2 says estimation "is
        done conservatively", and air is the larger overhead.
        """
        for desc_path in sorted(system_dir.glob(f"*/r*/{layout.SYSTEM_DESC_JSON}")):
            try:
                data = json.loads(desc_path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            declared: list[str] = []
            top = data.get("cooling")
            if isinstance(top, str) and top.strip():
                declared.append(top)
            cores: dict[int, int] = {}
            ensembles: set[int] = set()
            accelerators: dict[int, int] = {}
            for node in data.get("node_types") or []:
                if not isinstance(node, dict):
                    continue
                value = node.get("cooling")
                if isinstance(value, str) and value.strip():
                    declared.append(value)
                ensemble = node.get("system_node_ensemble_id")
                if isinstance(ensemble, str) and ensemble.strip().isdigit():
                    ensemble = int(ensemble)  # as NodeType coerces it
                if isinstance(ensemble, int):
                    ensembles.add(ensemble)
                    count = node.get("host_processor_core_count")
                    if isinstance(count, int):
                        cores[ensemble] = count
                    per_node = _accelerators_per_node(node)
                    if per_node is not None:
                        accelerators[ensemble] = per_node
            fractions = {overhead_for_cooling(value) for value in declared}
            fractions.discard(None)
            if len(fractions) > 1:
                overhead: float | None = AIR_COOLED_OVERHEAD
            else:
                overhead = next(iter(fractions), None)
            division = data.get("division")
            return _SystemFacts(
                overhead=overhead,
                cores=cores,
                ensembles=ensembles,
                accelerators=accelerators,
                division=division.strip().lower() if isinstance(division, str) else None,
            )
        return _SystemFacts(
            overhead=None, cores={}, ensembles=set(), accelerators={}, division=None
        )

    def _load_system_power(
        self, system_dir: Path, facts: _SystemFacts
    ) -> tuple[PowerComputation | None, list[CheckResult]]:
        """§9.1 "Power descriptor": a `system_power.json` per system, valid under E.7.

        Per *system*, not per point: provisioned power, and each node set's ``P_s`` and
        ``N_s``, are properties of the system. Each point scales them by the nodes it
        engages (§4.5.3), which :class:`PointPower` checks.

        Required for Standardized. An RDI or Serviced system may omit it (§4.5's
        scope); one that supplies it is held to the same schema, since it still
        produces a published normalised figure.

        The structure is checked when the file loads; this adds the E.7 rules that need
        the system description, and reports what :meth:`SystemPower.compute` found.
        Every E.7 finding is a rejection, so each is its own error rather than one
        summary line.
        """
        results: list[CheckResult] = []
        path = system_dir / layout.SYSTEM_POWER_JSON
        if not path.is_file() and facts.division in _POWER_OPTIONAL_DIVISIONS:
            results.append(_ok("power-descriptor", "pass", path, division=facts.division))
            return None, results
        if not path.is_file():
            results.append(
                _err(
                    "power-descriptor",
                    "fail",
                    path,
                    relative_to=system_dir.relative_to(self.submission_path),
                )
            )
            return None, results

        power, load_results = load_system_power(path)
        results.extend(load_results)
        if power is None:
            return None, results

        if facts.overhead is not None and facts.overhead != power.overhead_fraction:
            results.append(
                _err(
                    "power-descriptor",
                    "fail-2",
                    path,
                    cooling=power.cooling,
                    overhead=facts.overhead,
                )
            )
        unknown = sorted({s.system_node_ensemble_id for s in power.node_sets} - facts.ensembles)
        if facts.ensembles and unknown:
            results.append(_warn("power-descriptor", "warn", path, unknown=unknown))

        computation = power.compute(facts.cores)
        # The power computation words its own findings; they pass through as detail.
        for problem in computation.problems:
            results.append(_err("power-descriptor", "computation-problem", path, detail=problem))
        for warning in computation.warnings:
            results.append(_warn("power-descriptor", "computation-warning", path, detail=warning))
        kw = computation.provisioned_power_kw
        if kw is None:
            return None, results
        # §4.5.3 divides by accelerators per node; a set on a published path has no
        # components block, so §8.2's node type supplies it.
        computation.sets = [
            s
            if s.accelerators_per_node is not None
            else replace(s, accelerators_per_node=facts.accelerators.get(s.system_node_ensemble_id))
            for s in computation.sets
        ]

        if computation.estimated:
            results.append(
                _warn("power-estimated", "warn", path, estimates="; ".join(computation.estimated))
            )
        if not computation.problems:
            results.append(_ok("power-descriptor", "pass-2", path, kw=kw))
        return computation, results

    # ------------------------------------------------------------------
    # Per benchmark-model orchestration
    # ------------------------------------------------------------------

    def _check_model(
        self, system_id: str, model_dir: Path, power: PowerComputation | None = None
    ) -> list[CheckResult]:
        """Run every check scoped to one Pareto curve (§8.5: one system, one model).

        Two-phase, because v1.0 derives ``C_min`` from the submitted points (§5.4)
        rather than taking it as a declaration: phase one parses every ``point.yaml``
        with no notion of regions, phase two computes the boundaries from what parsed
        and runs the region-dependent rules against them.
        """
        results: list[CheckResult] = []
        benchmark_model = model_dir.name

        model_structure = ModelDir(
            root=model_dir, system_id=system_id, benchmark_model=benchmark_model
        )
        results.extend(model_structure._check_results)
        if any(r.severity == Severity.ERROR for r in model_structure._check_results):
            return results

        point_dirs = model_structure.point_dirs

        # ── Phase 1: parse every point.yaml, with no region context ───────────
        loaded, unparsed, load_results = self._load_point_configs(point_dirs)
        results.extend(load_results)
        if not loaded and not point_dirs:
            return results

        # An invalid system description is reported, not fatal: the curve's other
        # checks do not depend on it, and C_max can still come from the raw field.
        system_desc, sd_path, sd_results = self._load_curve_system_desc(point_dirs, model_dir)
        results.extend(sd_results)
        c_max = (
            system_desc.max_supported_concurrency
            if system_desc is not None
            else _raw_max_concurrency(sd_path)
        )

        results.extend(self._check_model_name(loaded))

        # ── Phase 1b: derive C_min, then the region boundaries ────────────────
        regions, region_results = self._derive_regions(
            c_max, loaded, unparsed, len(point_dirs), model_dir, sd_path
        )
        results.extend(region_results)

        # ── Phase 2: the region-dependent, per-point rules ────────────────────
        valid_points: list[tuple[Path, PointConfig]] = [(p.yaml_path, p.config) for p in loaded]
        loaded_points: list[tuple[PointConfig, PointSummary]] = []

        for point in loaded:
            results.extend(self._check_point(point, regions, loaded_points, power))
        for unparsed_point in unparsed:
            results.extend(self._check_unparsed_point(unparsed_point))

        results.extend(self._check_shared_paths(loaded))

        if self._seed_sets_error is None:
            seed_binding = SeedBinding(
                points=valid_points, registry=self._seed_sets, model_dir=model_dir
            )
            results.extend(seed_binding._check_results)

        if self._drafters_error is None:
            # §8.3's disclosure names the benchmark; §8.2 no longer carries the field.
            # The directory is the fallback — §8.1 names it after the same value, so it
            # is the same answer wherever a point failed to declare one.
            benchmark = next(
                (p.config.model_name for p in loaded if p.config.model_name), model_dir.name
            )
            drafter_binding = DrafterBinding(
                points=valid_points,
                approved=self._drafters.get(benchmark, []),
                benchmark=benchmark,
                model_dir=model_dir,
            )
            results.extend(drafter_binding._check_results)

        present = [(p.point_dir, p.config.concurrency) for p in loaded] + [
            (u.point_dir, u.concurrency) for u in unparsed if u.concurrency is not None
        ]
        accuracy_by_point, accuracy_dir, accuracy_results = self._load_curve_accuracy(present)
        results.extend(accuracy_results)

        # ModelContext validates point-count, coverage, config-consistency, accuracy-gate
        model_ctx = ModelContext(
            system_id=system_id,
            system_desc=system_desc,
            raw_c_max=c_max if system_desc is None else None,
            model_dir=model_dir,
            regions=regions,
            points_dir=model_dir,
            accuracy_dir=accuracy_dir,
            all_point_count=len(point_dirs),
            valid_points=valid_points,
            loaded_points=loaded_points,
            unparsed_concurrencies=[u.concurrency for u in unparsed if u.concurrency is not None],
            accuracy_by_point=accuracy_by_point,
        )
        results.extend(model_ctx._check_results)

        return results

    # ------------------------------------------------------------------
    # Phase 1 — parsing, before regions exist
    # ------------------------------------------------------------------

    def _load_point_configs(
        self, point_dirs: list[Path]
    ) -> tuple[list[_LoadedPoint], list[_UnparsedPoint], list[CheckResult]]:
        """Parse each point directory's ``point.yaml``; region rules run later.

        A point whose ``point.yaml`` is missing or invalid is returned as an
        :class:`_UnparsedPoint` rather than dropped, so it still counts.
        """
        results: list[CheckResult] = []
        loaded: list[_LoadedPoint] = []
        unparsed: list[_UnparsedPoint] = []

        for point_dir in point_dirs:
            yaml_path = point_dir / layout.POINT_YAML
            if not yaml_path.exists():
                results.append(
                    _err(
                        "measurement-points-present",
                        "fail-2",
                        yaml_path,
                        relative_to=point_dir.relative_to(self.submission_path),
                    )
                )
                unparsed.append(
                    _UnparsedPoint(point_dir, yaml_path, layout.parse_point_dir(point_dir.name))
                )
                continue

            config, config_results = load_point_config(yaml_path, context={"yaml_path": yaml_path})
            results.extend(config_results)
            if config is None:
                concurrency = _fallback_concurrency(yaml_path, point_dir)
                unparsed.append(_UnparsedPoint(point_dir, yaml_path, concurrency))
                continue

            # ModelDir.point_dirs only yields names matching r<digits>, so this parses.
            dir_concurrency = layout.parse_point_dir(point_dir.name)
            if dir_concurrency is not None and dir_concurrency != config.concurrency:
                results.append(
                    _warn(
                        "point-dirname-concurrency",
                        "warn",
                        yaml_path,
                        point_dir_name=point_dir.name,
                        dir_concurrency=dir_concurrency,
                        concurrency=config.concurrency,
                    )
                )

            loaded.append(_LoadedPoint(point_dir=point_dir, yaml_path=yaml_path, config=config))

        if not loaded and point_dirs:
            results.append(
                _err(
                    "measurement-points-present",
                    "fail",
                    point_dirs[0].parent,
                    relative_to=point_dirs[0].parent.relative_to(self.submission_path),
                )
            )
        return loaded, unparsed, results

    def _load_curve_system_desc(
        self, point_dirs: list[Path], model_dir: Path
    ) -> tuple[SystemDescription | None, Path | None, list[CheckResult]]:
        """Load the curve's system description from its points' ``system_desc.json``.

        Every point carries its own copy since policies PR #119, so the curve's
        description is whichever they agree on — and disagreeing on anything but
        :data:`_PER_POINT_SYSTEM_DESC_FIELDS` means the points do not describe one
        system, which §8.5 requires. Only the first copy is schema-validated: once the
        rest are byte-equivalent, validating them again would say nothing new.
        """
        results: list[CheckResult] = []
        rel = model_dir.relative_to(self.submission_path)
        present: list[tuple[Path, dict[str, object]]] = []

        for point_dir in point_dirs:
            sd_path = point_dir / layout.SYSTEM_DESC_JSON
            if not sd_path.exists():
                results.append(
                    _err(
                        "system-description-present",
                        "fail-2",
                        sd_path,
                        rel=rel,
                        point_dir_name=point_dir.name,
                    )
                )
                continue
            data = _read_json(sd_path)
            if data is None:
                results.append(_err("system-description-valid", "fail", sd_path))
                continue
            present.append((sd_path, data))

        if not present:
            results.append(_err("system-description-present", "fail", model_dir, rel=rel))
            return None, None, results

        ref_path, ref_data = present[0]
        results.extend(self._check_system_desc_consistency(present, model_dir))

        system_desc, load_results = load_system_description(ref_path)
        results.extend(load_results)
        if system_desc is None:
            return None, ref_path, results
        results.append(_ok("system-description-valid", "pass", ref_path, rel=rel))
        return system_desc, ref_path, results

    def _check_system_desc_consistency(
        self, present: list[tuple[Path, dict[str, object]]], model_dir: Path
    ) -> list[CheckResult]:
        """§8.5: every point of a curve must describe the same system."""
        ref_path, ref_data = present[0]
        ref_identity = _system_desc_identity(ref_data)
        for sd_path, data in present[1:]:
            identity = _system_desc_identity(data)
            if identity == ref_identity:
                continue
            differing = sorted(
                key
                for key in set(identity) | set(ref_identity)
                if identity.get(key) != ref_identity.get(key)
            )
            return [
                _err(
                    "system-description-consistency",
                    "fail",
                    sd_path,
                    ref_path_parent_name=ref_path.parent.name,
                    differing=", ".join(differing),
                )
            ]
        return [
            _ok("system-description-consistency", "pass", model_dir, present_count=len(present))
        ]

    def _derive_regions(
        self,
        c_max: int | None,
        loaded: list[_LoadedPoint],
        unparsed: list[_UnparsedPoint],
        point_dir_count: int,
        model_dir: Path,
        sd_path: Path | None,
    ) -> tuple[Regions | None, list[CheckResult]]:
        """Compute the curve's region boundaries from ``C_max`` and a derived ``C_min``.

        Every point contributes to ``C_min``, including one whose ``point.yaml`` did
        not validate (its concurrency comes from the raw field or ``r<N>``). Leaving
        such a point out moves the floor, and with it every boundary, so a single bad
        field elsewhere in the file used to report regions as uncovered that are not.
        """
        results: list[CheckResult] = []
        if c_max is None:
            results.append(_warn("region-basis", "warn", sd_path or model_dir))
            return None, results
        results.append(_ok("max-concurrency-declared", "pass", sd_path or model_dir, c_max=c_max))

        borrowed = [u.concurrency for u in unparsed if u.concurrency is not None]
        concurrencies = [point.config.concurrency for point in loaded] + borrowed
        if not concurrencies:
            results.append(_err("region-basis", "fail", model_dir))
            return None, results

        lowest = min(concurrencies)
        # A curve with no Ultra Low Concurrency point still gets usable boundaries;
        # ModelContext reports the missing coverage as its own error.
        c_min = min(lowest, ULTRA_LOW_CONCURRENCY_MAX)
        clamped = "" if c_min == lowest else f" (clamped from {lowest})"
        note = (
            f"; {len(borrowed)} from a {layout.POINT_YAML} that did not validate,"
            " read from its raw concurrency or r<N> name"
            if borrowed
            else ""
        )

        if len(concurrencies) < point_dir_count:
            results.append(
                _warn(
                    "region-basis",
                    "warn-2",
                    model_dir,
                    c_min=c_min,
                    clamped=clamped,
                    concurrencies_count=len(concurrencies),
                    point_dir_count=point_dir_count,
                    note=note,
                )
            )
        else:
            results.append(
                _ok(
                    "region-basis",
                    "pass",
                    model_dir,
                    c_min=c_min,
                    clamped=clamped,
                    point_dir_count=point_dir_count,
                    note=note,
                )
            )

        try:
            regions = compute_regions(c_max, c_min)
        except Invalid as exc:
            results.append(_err(exc.rule, exc.key, sd_path or model_dir, **exc.params))
            return None, results
        return regions, results

    # ------------------------------------------------------------------
    # Phase 2 — the region-dependent, per-point rules
    # ------------------------------------------------------------------

    def _check_point(
        self,
        point: _LoadedPoint,
        regions: Regions | None,
        loaded_points: list[tuple[PointConfig, PointSummary]],
        power: PowerComputation | None = None,
    ) -> list[CheckResult]:
        """Run the per-point rules that need the curve's regions or its result summary.

        Appends to *loaded_points* as a side effect so :class:`ModelContext` sees every
        point whose config and summary both loaded.
        """
        results: list[CheckResult] = []

        if regions is not None:
            placement = RegionPlacement(
                config=point.config, regions=regions, yaml_path=point.yaml_path
            )
            results.extend(placement._check_results)

        summary_path = point.point_dir / layout.RESULT_SUMMARY_JSON
        if not summary_path.exists():
            results.append(
                _err(
                    "result-summary-present",
                    "fail",
                    summary_path,
                    concurrency=point.config.concurrency,
                    relative_to=summary_path.relative_to(self.submission_path),
                )
            )
            return results

        summary, load_results = load_result_summary(summary_path)
        results.extend(load_results)
        if summary is None:
            return results

        # PointResult validates point-duration and metric-consistency
        point_result = PointResult.model_validate(
            {"config": point.config, "summary": summary, "yaml_path": point.yaml_path},
            context={"regions": regions, "summary_path": summary_path},
        )
        results.extend(point_result._check_results)
        loaded_points.append((point.config, summary))
        if power is not None:
            point_power = PointPower(
                config=point.config,
                power=power,
                parallelism=_point_parallelism(point.point_dir / layout.SYSTEM_DESC_JSON),
                system_tps=summary.system_tps,
                stored_tps_per_kw=_as_float((summary.model_extra or {}).get("system_tps_per_kw")),
                yaml_path=point.yaml_path,
                summary_path=summary_path,
            )
            results.extend(point_power._check_results)
        return results

    def _check_unparsed_point(self, point: _UnparsedPoint) -> list[CheckResult]:
        """Check what can be checked for a point whose ``point.yaml`` is unusable.

        Its result summary does not depend on ``point.yaml``, so it is loaded and
        validated as for any point. The rules that need the parsed config (region
        placement, metric consistency, power) cannot run, and the warning says so,
        so the submitter knows fixing the file may surface more.
        """
        results: list[CheckResult] = []
        summary_path = point.point_dir / layout.RESULT_SUMMARY_JSON
        if not summary_path.exists():
            results.append(
                _err(
                    "result-summary-present",
                    "fail-2",
                    summary_path,
                    point_dir_name=point.point_dir.name,
                    relative_to=summary_path.relative_to(self.submission_path),
                )
            )
        else:
            _summary, load_results = load_result_summary(summary_path)
            results.extend(load_results)
        results.append(
            _warn(
                "point-rules-skipped", "warn", point.yaml_path, point_dir_name=point.point_dir.name
            )
        )
        return results

    def _load_curve_accuracy(
        self, points: list[tuple[Path, int]]
    ) -> tuple[dict[int, AccuracyResult], Path | None, list[CheckResult]]:
        """Load accuracy results for **every** point that carries them.

        ``accuracy_results.json`` beside ``point.yaml`` wins; otherwise
        ``accuracy_scores`` embedded in the point's ``results.json`` is used.

        §5.3 moved accuracy from one run per submission to ``N`` points — the four
        mandatory concurrency points plus the Offline point — so the first usable
        result no longer stands in for the curve and each point is loaded on its own.

        Args:
            points: ``(point directory, concurrency)`` for every point of the curve,
                including those whose ``point.yaml`` did not validate.

        Returns:
            ``(results keyed by concurrency, a directory for messages, check results)``.
        """
        results: list[CheckResult] = []
        by_concurrency: dict[int, AccuracyResult] = {}
        first_dir: Path | None = None

        for point_dir, concurrency in points:
            acc: AccuracyResult | None = None

            accuracy_json = point_dir / layout.ACCURACY_RESULTS_JSON
            if accuracy_json.exists():
                acc, acc_results = load_accuracy_result(accuracy_json)
                results.extend(acc_results)
                if any(r.severity == Severity.ERROR for r in acc_results):
                    acc = None
            else:
                results_json = point_dir / layout.RESULTS_JSON
                if results_json.exists():
                    acc, acc_results, present = load_accuracy_scores(results_json)
                    if present:
                        results.extend(acc_results)
                        if any(r.severity == Severity.ERROR for r in acc_results):
                            acc = None
                    else:
                        acc = None

            if acc is not None:
                by_concurrency[concurrency] = acc
                if first_dir is None:
                    first_dir = point_dir

        return by_concurrency, first_dir, results

    # ------------------------------------------------------------------
    # Per-curve system-description rules
    # ------------------------------------------------------------------

    def _check_model_name(self, loaded: list[_LoadedPoint]) -> list[CheckResult]:
        """§3.2: ``model_name`` must be one of the accepted benchmark models, exactly.

        Read from ``point.yaml``, not ``system_desc.json``. §8.2 no longer defines
        ``model_name`` at all — policies PR #130 removed it from both the table and the
        template — and §8.5 makes the point's disclosure authoritative: a result ID's
        ``model_id`` "Must match ``model_name`` in ``point.yaml`` (§8.3)".

        The name must already be canonical (:func:`layout.canonical_model_name`). A
        name that only matches once canonicalised is still an error, and the message
        says which spelling to use instead.

        Reported per distinct name rather than per point, so a curve of 32 points does
        not produce 32 identical lines. Whether the points agree is
        ``config-consistency-model``'s question; absence is
        ``point-disclosure-complete``'s, so a point with no name is skipped here rather
        than reported twice.
        """
        seen: dict[str, Path] = {}
        for point in loaded:
            name = point.config.model_name
            if name:
                seen.setdefault(name, point.yaml_path)
        if not seen:
            return []
        results: list[CheckResult] = []
        for name, path in seen.items():
            if name in _ALLOWED_MODEL_NAMES:
                results.append(_ok("model-name-valid", "pass", path, name=name))
            else:
                allowed = ", ".join(_ALLOWED_MODEL_NAMES)
                canonical = layout.canonical_model_name(name)
                if canonical in _ALLOWED_MODEL_NAMES:
                    results.append(
                        _err(
                            "model-name-valid",
                            "noncanonical",
                            path,
                            name=name,
                            canonical=canonical,
                            allowed=allowed,
                        )
                    )
                else:
                    results.append(
                        _err("model-name-valid", "fail", path, name=name, allowed=allowed)
                    )
        return results

    def _check_shared_paths(self, loaded: list[_LoadedPoint]) -> list[CheckResult]:
        """§9.1: each point's ``shared_src`` / ``shared_docs`` must resolve under the root.

        The pointers are how a point names the shared trees §8.1 places outside
        ``results/``. A value that does not resolve leaves the point's implementation
        and documentation unreviewable, which §9.1 answers with "Reject submission".

        Reported per distinct value rather than per point, like ``model-name-valid``:
        every point normally names the same tree, so a missing ``docs/`` would
        otherwise be one cause reported once per point.
        """
        groups: dict[tuple[str, str], list[_LoadedPoint]] = {}
        for point in loaded:
            for field_name in ("shared_src", "shared_docs"):
                value = getattr(point.config, field_name)
                if value is None:
                    continue  # absence is reported by point-disclosure-complete
                groups.setdefault((field_name, value), []).append(point)

        results: list[CheckResult] = []
        for (field_name, value), points in groups.items():
            names = ", ".join(p.point_dir.name for p in points)
            where = f"{len(points)} point(s): {names}"
            resolved = layout.resolve_shared_path(self.submission_path, value)
            if resolved is None:
                results.append(
                    _err(
                        "shared-path-resolution",
                        "fail",
                        points[0].yaml_path,
                        field_name=field_name,
                        value=value,
                        where=where,
                    )
                )
            else:
                results.append(
                    _ok(
                        "shared-path-resolution",
                        "pass",
                        points[0].yaml_path,
                        field_name=field_name,
                        value=value,
                        relative_to=resolved.relative_to(self.submission_path),
                        where=where,
                    )
                )
        return results
