# MLCommons Endpoints Submission Tools

A Python package with two tools for managing MLPerf Endpoints benchmark submissions:

- **`endpoints-submission-cli`** — registers benchmark runs, assembles submission packages, runs compliance checks, and opens GitHub pull requests via the PRISM API.
- **`submission-checker`** — validates a submission folder against the §9.1 automated compliance rules before or after upload.

---

## Installation

**With pip:**

```bash
pip install endpoints-submission-cli
```

**From source (editable):**

```bash
pip install -e ".[dev]"
```

**With [uv](https://github.com/astral-sh/uv):**

```bash
uv sync --extra dev
```

---

# endpoints-submission-cli

## Requirements

- Python 3.10 or later
- [`gh` CLI](https://cli.github.com/) — required for creating, updating, and withdrawing submissions

## Authentication

Every command requires a PRISM API token in `mlc_…` format. Supply it as an env var or pass `--token` per command:

```bash
# Persistent (add to shell profile)
export PRISM_USER_API_TOKEN=mlc_your_token_here

# Per-command override
endpoints-submission-cli runs list --token mlc_your_token_here
```

Submission commands that create or update GitHub pull requests also require the `gh` CLI:

```bash
gh auth login
```

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `PRISM_USER_API_TOKEN` | — | API key. Required unless `--token` is passed. |
| `MLPERF_SUBMISSION_REPO` | `MLCommons-Systems/test-endpoints-submission-repo` | Target GitHub repository for submission PRs (`owner/repo`). |

Add to your shell profile for a persistent setup:

```bash
export PRISM_USER_API_TOKEN=mlc_your_token_here
export MLPERF_SUBMISSION_REPO=MLCommons-Systems/endpoints-submission-repo
```

## Quick start

```bash
# 1. Verify connectivity
endpoints-submission-cli runs list

# 2. Register a benchmark run from a local result folder
endpoints-submission-cli runs create --path /results/llama3_h100_c4
# → Run created: d5d9873e-5eca-4f8d-a487-4be1cb8b440c
RUN_ID=d5d9873e-5eca-4f8d-a487-4be1cb8b440c

# 3. Create a submission (assembles, checks, uploads, opens PR)
endpoints-submission-cli submissions create \
  --division standardized \
  --availability available \
  --run-ids $RUN_ID
# → Submission created: a1b2c3d4-…
# → PR: https://github.com/MLCommons-Systems/…/pull/42
SUB_ID=a1b2c3d4-e5f6-7890-abcd-ef1234567890

# 4. Add another run later
endpoints-submission-cli submissions add-run \
  --submission-id $SUB_ID \
  --run-id <new-run-id>

# 5. Withdraw if needed
endpoints-submission-cli submissions withdraw --submission-id $SUB_ID
```

## Command reference

```
endpoints-submission-cli
├── runs
│   ├── list        List all runs
│   ├── create      Register a run from a local folder
│   ├── get         Fetch run details
│   ├── delete      Delete a run and its archive
│   ├── pin         Pin a run (prevent expiry)
│   └── unpin       Restore normal expiry
└── submissions
    ├── list        List all submissions
    ├── create      Create a submission from runs (full pipeline)
    ├── get         Fetch submission details
    ├── update      Update run list or metadata
    ├── withdraw    Withdraw a submission
    ├── add-run     Add a run to an existing submission
    └── remove-run  Remove a run from a submission
```

Use `--help` on any command for full flag details:

```bash
endpoints-submission-cli submissions create --help
```

### Terminal Pareto preview

`submissions create-local` renders a text-based Pareto-frontier chart of your
measurement points before anything is uploaded — the same throughput-vs-interactivity
trade-off shown by the web visualizer, but drawn with Unicode so it works over SSH on
a headless results box (no browser or local server needed). Pair it with `--dry-run`
to preview and run the Submission Checker without creating a submission:

```bash
endpoints-submission-cli submissions create-local \
    --path ./my_submission --division standardized \
    --scenario cop --availability available --dry-run
```

Each `pareto/<system>/<model>` curve is plotted from its runs' `run_metadata.json`
(system throughput vs. tokens/s per user). Systems are distinguished by both colour
**and** marker glyph (so curves stay readable in monochrome terminals, piped logs, and
for colour-blind readers), the efficient frontier is drawn across systems, and a stats
table plus a plain-language trade-off summary highlight which system wins on what.

---

# submission-checker

CLI tool for validating MLPerf Endpoints submissions against the §9.1 automated compliance checks.

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

---

## Development

```bash
uv run pytest                          # run all tests
uv run pytest --no-cov -x             # fast fail on first error
uv run ruff check src/ tests/          # lint
uv run ruff format src/ tests/         # auto-format
```
