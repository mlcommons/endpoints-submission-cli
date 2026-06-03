# MLCommons Endpoints Submission Tools

This repository contains two tools:

- **`submission-checker`** — validates a submission folder against the §9.1 automated compliance rules (see below).
- **`endpoints-submission-cli`** — CLI for registering benchmark runs and creating rolling submissions via the PRISM API. See [docs/endpoints-submission-cli.md](docs/endpoints-submission-cli.md) for full usage.

---

# submission-checker

CLI tool for validating MLPerf Endpoints submissions against the §9.1 automated compliance checks.

## Installation

```bash
uv sync --extra dev
```

Or with pip:

```bash
pip install -e ".[dev]"
```

## Usage

### Check a submission

```bash
submission-checker check /path/to/submission
```

The tool expects the submission root to contain `systems/` and `pareto/` subdirectories as specified in §8.1.

**Options:**

| Flag | Description |
|------|-------------|
| `--strict` | Treat warnings as errors (exit 1 on any warning) |
| `--quiet` / `-q` | Suppress INFO-level passing checks |
| `--output FILE` / `-o FILE` | Write full results as JSON to *FILE* |

**Exit codes:** `0` = all checks passed, `1` = one or more errors (or warnings with `--strict`).

### Show region boundaries

```bash
submission-checker regions --max-concurrency 1024
```

Prints the concurrency ranges for each region given a declared Maximum Supported Concurrency *M* (§5.5).

## Required Files in submission structure

```
<org>/
├── systems/
│   └── <system_desc_id>.json         # §8.2 — hardware + software description
└── pareto/
    └── <system_desc_id>/
        └── <benchmark_model>/
            ├── points/
            │   └── point_<N>.yaml    # §8.3 — one config per measurement point
            ├── results/
            │   └── point_<N>/
            │       ├── mlperf_endpoints_log_summary.json
            │       └── mlperf_endpoints_log_detail.json
            └── accuracy/
                ├── accuracy.txt
                └── accuracy_result.json
```

## What gets checked

| Rule | Spec | Description |
|------|------|-------------|
| `path-exists` | §1 | Submission root directory exists |
| `required-dir` | §1 | `systems/` and `pareto/` present |
| `system-description-present` | §1 | At least one `*.json` file found in `systems/` |
| `system-description-valid` | §1 | `systems/*.json` parses against schema |
| `src-dir` | §1 | `src/` present for Standardized submissions |
| `pareto-dir-exists` | §1 | `pareto/<system_id>/` directory exists |
| `benchmark-model-dir` | §1 | At least one benchmark-model directory in `pareto/<system_id>/` |
| `pareto-subdir` | §1 | `points/`, `results/`, `accuracy/` present |
| `measurement-points-present` | §1 | At least one `point_*.yaml` found |
| `point-config-valid` | §1 | YAML parses against `PointConfig` schema |
| `point-filename-concurrency` | §1 | Filename concurrency matches declared value |
| `result-file-present` | §1 | Result summary log exists for each point config |
| `result-detail-present` | §1 | Result detail log exists for each point config |
| `result-file-valid` | §1 | Result summary log parses against `PointSummary` schema |
| `point-count` | §2, §8 | 7–32 measurement points |
| `point-cap` | §2, §8 | Point count does not exceed 32 |
| `low-latency-coverage` | §3 | At least one point in Low Latency region |
| `low-throughput-coverage` | §4 | At least one point in Low Throughput region |
| `med-throughput-coverage` | §5 | At least one point in Medium Throughput region |
| `high-throughput-coverage` | §6 | At least one point in High Throughput region |
| `max-concurrency-declared` | §7 | `max_supported_concurrency` field present |
| `region-computation` | §7 | *M* > 32 (required for region formula) |
| `concurrency-in-range` | §9 | Concurrency within region bounds (incl. 10% margin) |
| `load-pattern` | §10 | `load_pattern` is `concurrency` with a positive concurrency level |
| `point-duration` | §11 | Point meets per-region minimum duration |
| `min-query-count` | §12 | `n_samples_completed` meets dataset-specific minimum (§6.4) |
| `streaming-config` | §13 | `stream_all_chunks` is `True` |
| `metric-consistency-duration` | §14 | `duration_ns` > 0 |
| `metric-consistency-accounting` | §14 | `completed + failed == issued` |
| `metric-consistency-output-tokens` | §14 | `total_output_tokens` ≥ 0 |
| `metric-consistency-system-tps` | §9.1 | Stored `system_tps` consistent with derived value |
| `metric-consistency-tps-per-user` | §9.1 | Stored `tps_per_user` consistent with `system_tps / concurrency` |
| `accuracy-file` | §15 | `accuracy.txt` and `accuracy_result.json` present |
| `accuracy-valid` | §15 | `accuracy_result.json` parses correctly |
| `accuracy-consistency` | §15 | `passed` flag consistent with `score >= quality_target` |
| `accuracy-gate` | §15 | Score ≥ quality target |
| `config-consistency-dataset` | §16 | All points use the same dataset |
| `config-consistency-model` | §16 | Directory name matches `benchmark_model` |
| `region-declared` | §8.3 | Declared `region` field (if present) is valid and matches computed region |

## Programmatic API

```python
from submission_checker import SubmissionChecker, Report

checker = SubmissionChecker(Path("/submissions/acme_corp"))
report = checker.run()

if report.passed:
    print("All checks passed")
else:
    for result in report.errors:
        print(f"[{result.rule}] {result.message}")
```

The `Report` object also exposes `report.warnings` and serialises cleanly via `report.model_dump_json()`.

## Development

```bash
uv run pytest                                          # run tests (189 tests, 100% coverage)
uv run pytest --no-cov -x                             # fast fail on first error
uv run ruff check src/ tests/                         # lint
uv run ruff format src/ tests/                        # auto-format
uv run sphinx-build -W docs docs/_build/html          # build docs
```

## Architecture

```
cli.py          Entry point — Click commands, Rich table output
checker.py      SubmissionChecker — orchestrates loading and validation
loader.py       File I/O — JSON/YAML loading, returns (model | None, list[CheckResult])
structure.py    Directory structure validators (§8.1)
models/
  results.py         CheckResult, Severity, ok/warn/err helpers
  regions.py         Region boundary computation (§5.5 reference algorithm)
  file/              Per-artifact models — each validates a single file
    system.py          SystemDescription (systems/*.json)
    point_config.py    PointConfig + RuntimeSettings (points/point_<N>.yaml)
    point_summary.py   PointSummary + PercentileStats (mlperf_endpoints_log_summary.json)
    accuracy.py        AccuracyResult (accuracy/accuracy_result.json)
  aggregate/         Cross-artifact models — validate across multiple files
    point_result.py    PointResult — pairs one PointConfig with its PointSummary
    context.py         ModelContext — validates point count, coverage, consistency, accuracy
```

Validation logic is co-located with the data models: each Pydantic model runs its own
`@model_validator` methods and accumulates results in a private `_check_results` list.
`SubmissionChecker.run()` orchestrates loading, instantiates models, and collects results
into a `Report`. All loaders return `(model | None, list[CheckResult])` — failure surfaces
every Pydantic validation error, not just the first.
