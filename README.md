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

---

# submission-checker

CLI tool for validating MLPerf Endpoints submissions against the §9.1 automated compliance checks.

## Usage

### Check a submission

```bash
submission-checker check /path/to/submission
```

The path may be the submitting organisation's directory or a `<submission_id>/`
directory below it; a submission root is the level holding `results/` and `docs/` (§8.1).

**Options:**

| Flag | Description |
|------|-------------|
| `--strict` | Treat warnings as errors (exit 1 on any warning) |
| `--quiet` / `-q` | Suppress INFO-level passing checks |
| `--output FILE` / `-o FILE` | Write full results as JSON to *FILE* |
| `--seed-sets FILE` | Published seed sets to check against (§4.6). Defaults to the bundled set; also settable via `$MLPERF_ENDPOINTS_SEED_SETS`. |

**Exit codes:** `0` = all checks passed, `1` = one or more errors (or warnings with `--strict`).

### Show region boundaries

```bash
submission-checker regions --max-concurrency 1024 --min-concurrency 16
```

Prints the concurrency range for each region given a `(C_max, C_min)` pair, using the
§5.5 reference algorithm. `--min-concurrency` defaults to 32; in a real submission
`C_min` is derived from the lowest measurement point rather than declared (§5.4).

## Required Files in submission structure

Layout as of `mlcommons/endpoints_policies` PR #119: there is no per-system file — every
Pareto point carries its own `system_desc.json`.

```
<submitting_organization>/
└── <submission_id>/
    ├── src/
    │   └── <implementation>/         # §2.2.1 — README.md + endpoint interface code
    │       └── README.md
    ├── docs/                         # calibration, software disclosure, …
    └── results/
        └── <system>/
            └── <model_name>/
                └── r<N>/             # one directory per concurrency level
                    ├── point.yaml            # §8.3 measurement-point disclosure
                    ├── system_desc.json      # §8.2 — per point since PR #119
                    ├── result_summary.json   # aggregate metrics
                    ├── accuracy_results.json # §6.6 accuracy results
                    ├── config.yaml           # OPTIONAL as of v1.0
                    └── server_configs/       # OPTIONAL, submitter-defined
```

`src/` and `docs/` are shared across the whole submission; each `point.yaml` names them
via `shared_src` and `shared_docs`, which must resolve to a directory under the
submission root (§9.1).

## What gets checked

### Structure

| Rule | Spec | Description |
|------|------|-------------|
| `path-exists` | §1 | Submission root directory exists |
| `required-dir` | §1 | `results/` and `docs/` present |
| `src-dir` | §2.2.1 | `src/` present with at least one implementation directory |
| `src-readme` | §2.2.1 | Each `src/<implementation>/` has a `README.md` |
| `system-results-dir` | §1 | At least one `results/<system>/` directory exists |
| `benchmark-model-dir` | §1 | At least one benchmark-model directory per system |
| `point-dirs` | §1 | At least one `r<N>/` Pareto-point directory per model |
| `measurement-points-present` | §1 | Every `r<N>/` carries a `point.yaml` |
| `point-dirname-concurrency` | §1 | `r<N>/` name matches the declared concurrency (warn) |
| `result-summary-present` | §1 | `result_summary.json` exists for each point |
| `shared-path-resolution` | §9.1 | `shared_src` / `shared_docs` resolve under the submission root |

### System description (§8.2)

| Rule | Spec | Description |
|------|------|-------------|
| `system-description-present` | §8.2 | Every point has a `system_desc.json` |
| `system-description-valid` | §8.2 | It parses against the `SystemDescription` schema |
| `system-description-consistency` | §8.5 | Every point of a curve describes the same system |
| `model-name-valid` | §2 | `model_name` is one of the round's supported models |
| `model-name-consistency` | §16 | It matches the results directory name |
| `max-concurrency-declared` | §7 | `max_supported_concurrency` (C_max) present and > 32 |
| `tps-utilization` | §8.2 | Equals `system_tps / max(system_tps)` over the point's own curve |

### Regions (§5)

`C_min` is **derived** from the submission's own points in v1.0, not declared, so the
boundaries differ per curve. The 10 % margin above `C_max` is its own region and does
not satisfy High Concurrency coverage.

| Rule | Spec | Description |
|------|------|-------------|
| `region-basis` | §5.4 | Reports the derived `C_min` and how many points it came from |
| `region-computation` | §5.5 | `(C_max, C_min)` is a valid input to the reference algorithm |
| `concurrency-in-range` | §9.1 | Each concurrency falls in a valid region, margin included |
| `region-declared` | §8.3 | Declared `region` is one of the spec's values |
| `region-placement` | §8.3 | Declared region matches the computed one (warn) |
| `ultra-low-concurrency-coverage` | §5.4 | At least one point at concurrency ≤ 32 |
| `low-concurrency-coverage` | §9.1 | At least one point in the Low Concurrency region |
| `med-concurrency-coverage` | §9.1 | At least one point in the Medium Concurrency region |
| `high-concurrency-coverage` | §9.1 | At least one point in the High Concurrency region |
| `point-count` | §2, §8 | 7–32 measurement points |
| `point-cap` | §2, §8 | Point count does not exceed 32 |

### Measurement points (§8.3, §6)

| Rule | Spec | Description |
|------|------|-------------|
| `point-config-valid` | §8.3 | `point.yaml` parses against the `PointConfig` schema |
| `point-disclosure-complete` | §8.3 | Every required §8.3 disclosure field is present |
| `load-pattern` | §6.1 | `load_pattern` is `concurrency` with a positive level |
| `streaming-config` | §6.5 | `stream_all_chunks` is `True` |
| `point-duration` | §6.2 | Point meets its region's minimum duration (warn) |
| `min-query-count` | §6.4 | `n_samples_completed` meets the dataset minimum |
| `warmup-present` | §6.3.3 | Warmup declaration present |
| `warmup-logs-retained` | §6.3.2 | Warmup log retention declared (warn) |
| `warmup-salt` | §6.3.3 | Warns when the warmup salt is enabled |
| `config-consistency-dataset` | §16 | All points use the same dataset |

### Seed binding (§4.6)

| Rule | Spec | Description |
|------|------|-------------|
| `seed-set-consistency` | §9.1 | Every point records the same seed set |
| `seed-set-membership` | §9.1 | The bound set is one MLCommons published |
| `seed-runtime-match` | §2.1.1 | The RNG seeds equal the bound set's values |
| `target-cohort` | §4.6 | `target_cohort` matches `YYYY-MM-C0` / `YYYY-MM-C1` |
| `seed-set-adoption` | §4.6 | Set published for the target cohort or the three before it |
| `seed-config-legacy` | §4.6 | v0.7 fallback: seeds == 42 when no `seed_set` is declared |
| `seed-set-registry` | §4.6 | Warns when the seed-set file itself cannot be read |

The published sets ship as data (`src/submission_checker/data/seed_sets.yaml`). Point
`--seed-sets FILE` or `$MLPERF_ENDPOINTS_SEED_SETS` at a newer file to check against a
set published after this release.

### Metrics (§9.1, §14)

| Rule | Spec | Description |
|------|------|-------------|
| `result-file-valid` | §8.3 | `result_summary.json` parses against `PointSummary` |
| `metric-consistency-duration` | §14 | `duration_ns` > 0 |
| `metric-consistency-accounting` | §14 | `completed + failed == issued` |
| `metric-consistency-output-tokens` | §14 | `total_output_tokens` ≥ 0 |
| `metric-consistency-system-tps` | §9.1 | Stored `system_tps` matches the derived value |
| `metric-consistency-tpot-p90` | §9.1 | Reported TPOT P90 present, finite, strictly positive |
| `metric-consistency-tps-per-user` | §9.1 | Stored `tps_per_user` matches `1000 / tpot_p90_ms` |

### Accuracy (§15)

| Rule | Spec | Description |
|------|------|-------------|
| `accuracy-present` | §15 | At least one model in the submission carries accuracy results |
| `accuracy-valid` | §15 | `accuracy_results.json` parses correctly |
| `accuracy-sample-count` | §15 | Issued sample count meets the model's minimum |
| `accuracy-gate` | §15 | Score meets the benchmark quality target |

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
