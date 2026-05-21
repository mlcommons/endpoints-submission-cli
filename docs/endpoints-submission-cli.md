# endpoints-submission-cli

Command-line tool for managing MLPerf Endpoints benchmark runs and rolling
submissions against the PRISM Submission API.

---

## Table of contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [End-to-end workflow](#end-to-end-workflow)
4. [Run commands](#run-commands)
   - [runs list](#runs-list)
   - [runs create](#runs-create)
   - [runs get](#runs-get)
   - [runs delete](#runs-delete)
   - [runs pin / runs unpin](#runs-pin--runs-unpin)
5. [Submission commands](#submission-commands)
   - [submissions list](#submissions-list)
   - [submissions create](#submissions-create)
   - [submissions get](#submissions-get)
   - [submissions update](#submissions-update)
   - [submissions withdraw](#submissions-withdraw)
   - [submissions add-run](#submissions-add-run)
   - [submissions remove-run](#submissions-remove-run)
6. [Environment variable reference](#environment-variable-reference)
7. [Exit codes](#exit-codes)
8. [Run folder layout](#run-folder-layout)
9. [Development](#development)

---

## Installation

**Requires Python 3.10+.**

Install into a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv sync --extra dev
```

Verify the installation:

```bash
endpoints-submission-cli --help
```

---

## Configuration

All commands require an API token. You can supply it in two ways (the flag
takes precedence):

| Method | Example |
|--------|---------|
| Environment variable | `export PRISM_USER_API_TOKEN=mlc_your_token_here` |
| Per-command flag | `--token mlc_your_token_here` |

Two additional environment variables control where the CLI talks to:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLPERF_API_BASE_URL(for testing purpose)` | `http://localhost:8080` | Base URL of the PRISM Submission API |
| `MLPERF_SUBMISSION_REPO` | `mlcommons/inference_results_rolling` | GitHub repository for submission PRs |

Add these to your shell profile for persistent configuration:

```bash
export PRISM_USER_API_TOKEN=mlc_your_token_here
export MLPERF_API_BASE_URL=https://api.mlcommons.org
export MLPERF_SUBMISSION_REPO=mlcommons/inference_results_rolling
```

The `submissions create`, `submissions withdraw`, `submissions add-run`, and
`submissions remove-run` commands use the GitHub CLI (`gh`). Make sure `gh` is
installed and authenticated:

```bash
gh auth login
```

---

## End-to-end workflow

### Register a run and create a submission

```bash
# 1. Register a benchmark run from a local result folder
endpoints-submission-cli runs create --path /results/my_run_2025_04

# Output:  Run created: d5d9873e-5eca-4f8d-a487-4be1cb8b440c
RUN_ID=d5d9873e-5eca-4f8d-a487-4be1cb8b440c

# 2. (Optional) Pin the run to prevent automatic expiry
endpoints-submission-cli runs pin --run-id $RUN_ID

# 3. Create a submission from one or more runs
endpoints-submission-cli submissions create \
  --division standardized \
  --availability available \
  --run-ids $RUN_ID

# Output:
#   Submission created: a1b2c3d4-e5f6-7890-abcd-ef1234567890
#   PR: https://github.com/mlcommons/inference_results_rolling/pull/42
SUB_ID=a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### Add more runs to an existing submission

```bash
endpoints-submission-cli runs create --path /results/my_run_concurrency_16
# NEW_RUN_ID=...

endpoints-submission-cli submissions add-run \
  --submission-id $SUB_ID \
  --run-id $NEW_RUN_ID
```

### Withdraw a submission

```bash
endpoints-submission-cli submissions withdraw --submission-id $SUB_ID
# Closes the GitHub PR and deletes the stored bundle
```

---

## Run commands

### runs list

List all runs registered under your account.

```bash
endpoints-submission-cli runs list [--token TOKEN] [-j]
```

| Flag | Description |
|------|-------------|
| `--token TOKEN` | API token (falls back to `PRISM_USER_API_TOKEN`) |
| `-j` / `--json` | Print raw JSON instead of the rich table |

**Example:**

```bash
endpoints-submission-cli runs list
endpoints-submission-cli runs list -j | jq '.[].id'
```

---

### runs create

Parse a local run folder, register a run record with the API, and upload the
folder as a compressed archive.

```bash
endpoints-submission-cli runs create --path PATH [--token TOKEN]
```

| Flag | Description |
|------|-------------|
| `--path PATH` | Path to the local run folder (required) |
| `--token TOKEN` | API token |

The folder must contain exactly these three files (see [Run folder layout](#run-folder-layout)):

```
system_info.json
config.yaml
result_summary.json
```

If the archive upload fails the run record is automatically deleted (rollback
to clean state). The run ID is printed on success.

**Example:**

```bash
endpoints-submission-cli runs create --path ./results/llama3_h100_c4
```

---

### runs get

Fetch full details of a single run.

```bash
endpoints-submission-cli runs get --run-id RUN_ID [--token TOKEN] [-j]
```

| Flag | Description |
|------|-------------|
| `--run-id RUN_ID` | Run UUID (required) |
| `--token TOKEN` | API token |
| `-j` / `--json` | Print raw JSON |

---

### runs delete

Delete a run record and its stored archive.

```bash
endpoints-submission-cli runs delete --run-id RUN_ID [--token TOKEN]
```

> **Note:** Runs that belong to an active submission cannot be deleted. Withdraw
> the submission first (`submissions withdraw`).

Archive deletion after a successful DB delete is best-effort. If it fails the
orphaned URI is logged as a warning but the command still exits 0.

---

### runs pin / runs unpin

Pin a run to prevent automatic expiry, or unpin it to restore normal expiry.

```bash
endpoints-submission-cli runs pin   --run-id RUN_ID [--token TOKEN]
endpoints-submission-cli runs unpin --run-id RUN_ID [--token TOKEN]
```

Pinned runs have `expires_at = null`. Unpin them when you no longer need to
keep them indefinitely.

---

## Submission commands

### submissions list

List all submissions for your account.

```bash
endpoints-submission-cli submissions list [--token TOKEN] [-j]
```

---

### submissions create

Assemble and submit a new MLPerf rolling submission from one or more registered
runs. This command runs the full automated workflow:

1. Download run archives from the API.
2. Assemble the required [submission folder structure](#submission-folder-structure).
3. Run the Submission Checker — aborts if compliance errors are found.
4. Register the submission with the API (`POST /submissions`).
5. Upload the submission bundle.
6. Create a GitHub PR in the target repository.
7. Store the PR URL and number on the submission record.

```bash
endpoints-submission-cli submissions create \
  --division    DIVISION \
  --availability AVAILABILITY \
  --run-ids     RUN_ID_1 \
  --run-ids     RUN_ID_2 \
  [--token TOKEN] \
  [--early-publish] \
  [--publication-cycle CYCLE] \
  [--target-availability-date DATE]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--division` | yes | `standardized`, `serviced`, or `rdi` |
| `--availability` | yes | `available`, `preview`, or `rdi` |
| `--run-ids RUN_ID` | yes (repeatable) | Run UUID(s) to include; pass the flag once per run |
| `--token TOKEN` | no | API token |
| `--early-publish` | no | Request early publication (default: false) |
| `--publication-cycle CYCLE` | no | Target cycle, e.g. `2025-04-C1` |
| `--target-availability-date DATE` | no | `YYYY-MM-DD`; required when availability is `preview` |

**If the GitHub PR step fails** the submission record and uploaded bundle still
exist. Retry the PR step manually:

```bash
endpoints-submission-cli submissions update \
  --submission-id $SUB_ID \
  --pr-url https://github.com/org/repo/pull/N \
  --pr-number N
```

---

### submissions get

Fetch full details of a single submission, including embedded run records.

```bash
endpoints-submission-cli submissions get \
  --submission-id SUB_ID [--token TOKEN] [-j]
```

---

### submissions update

Patch one or more fields on an existing submission. Only the flags you provide
are changed.

```bash
endpoints-submission-cli submissions update \
  --submission-id SUB_ID \
  [--token TOKEN] \
  [--status STATUS] \
  [--pr-url URL] \
  [--pr-number N] \
  [--publication-cycle CYCLE] \
  [--target-availability-date DATE] \
  [--archive-uri URI] \
  [--availability-qualified-at DATETIME] \
  [--compliance-passed-at DATETIME] \
  [--first-published-at DATETIME] \
  [--peer-review-started-at DATETIME] \
  [--objection-resolution-started-at DATETIME] \
  [--finalized-at DATETIME]
```

| Flag | Description |
|------|-------------|
| `--submission-id` | Submission UUID (required) |
| `--status` | Override lifecycle status |
| `--pr-url` | GitHub PR URL |
| `--pr-number` | GitHub PR number |
| `--publication-cycle` | Target publication cycle |
| `--target-availability-date` | Target availability date (YYYY-MM-DD) |
| All `*-at` flags | ISO-8601 timestamp for the corresponding lifecycle event |

Providing no optional flags prints a warning and makes no API call.

---

### submissions withdraw

Withdraw a submission: marks it `WITHDRAWN`, closes its GitHub PR, and deletes
the stored bundle.

```bash
endpoints-submission-cli submissions withdraw \
  --submission-id SUB_ID [--token TOKEN]
```

The operations are ordered so that the DB state is updated first. PR closure
and archive deletion are best-effort — failures are reported as warnings but
do not affect the exit code.

---

### submissions add-run

Add a run to an existing submission, rebuild the submission bundle, re-run
compliance checking, and push a new commit to the PR branch.

```bash
endpoints-submission-cli submissions add-run \
  --submission-id SUB_ID \
  --run-id RUN_ID \
  [--token TOKEN]
```

**Rollback:** if any step after the API registration fails (download, build,
checker, upload) the run is automatically removed from the submission record
before exiting.

**GitHub:** if the push fails the bundle is already updated in blob storage and
the DB is consistent. Re-push manually with `git push` on the PR branch.

---

### submissions remove-run

Remove a run from an existing submission. If runs still remain, the bundle is
rebuilt, compliance-checked, re-uploaded, and pushed to the PR branch.

```bash
endpoints-submission-cli submissions remove-run \
  --submission-id SUB_ID \
  --run-id RUN_ID \
  [--token TOKEN]
```

If the submission has no runs left after removal the rebuild step is skipped
and a warning is printed.

**Rollback:** if a step fails after removal, the run is automatically re-added
to the submission record.

---

## Environment variable reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PRISM_USER_API_TOKEN` | yes* | — | API key for the PRISM Submission API (`mlc_…`). Can be passed per-command with `--token` instead. |
| `MLPERF_API_BASE_URL` | no | `http://localhost:8080` | Base URL of the PRISM Submission API. |
| `MLPERF_SUBMISSION_REPO` | no | `mlcommons/inference_results_rolling` | Target GitHub repository for submission PRs (`org/repo` format). |

\* Required unless `--token` is passed.

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Command completed successfully |
| `1` | Error — authentication failure, API error, invalid input, or compliance check failure |

---

## Run folder layout

`runs create` expects a directory containing these three files:

```
<run-folder>/
├── system_info.json      # Hardware and software description
├── config.yaml           # Benchmark configuration (model, concurrency, endpoints, …)
└── result_summary.json   # Aggregated performance metrics
```

**`system_info.json`** — hardware description:

```json
{
  "system_name": "H100x8-vLLM",
  "system_category": "datacenter",
  "accelerator_model_name": "NVIDIA H100 SXM5 80GB",
  "accelerators_per_node": 8,
  "framework": "vllm==0.4.2",
  "submitter_org_names": "Acme Corp",
  ...
}
```

**`config.yaml`** — benchmark run configuration:

```yaml
model_params:
  name: meta-llama/Llama-3.1-8B-Instruct
settings:
  load_pattern:
    type: concurrency
    target_concurrency: 4
endpoint_config:
  endpoints:
    - http://my-server:8000
  api_type: openai
```

**`result_summary.json`** — performance metrics from the run:

```json
{
  "duration_ns": 475700000000,
  "n_samples_completed": 2000,
  "ttft": { "avg": 80900000, "percentiles": { "50": 80500000, "99": 200000000 } },
  "output_sequence_lengths": { "avg": 50.0 }
}
```

---

## Submission folder structure

`submissions create` assembles this structure automatically from run archives.
The [Submission Checker](../README.md) validates it before the submission is
registered:

```
<org>/
├── systems/
│   └── <system_id>.json              # hardware + software description
└── pareto/
    └── <system_id>/
        └── <model>/
            ├── points/
            │   └── point_<N>.yaml    # one config per concurrency level
            ├── results/
            │   └── point_<N>/
            │       ├── mlperf_endpoints_log_summary.json
            │       └── mlperf_endpoints_log_detail.json
            └── accuracy/
                ├── accuracy.txt
                └── accuracy_result.json
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/endpoints_submission_cli/ -m unit

# Run with coverage
pytest tests/endpoints_submission_cli/ -m unit \
  --cov=src/endpoints_submission_cli --cov-report=term-missing

# Lint
ruff check src/endpoints_submission_cli/ tests/endpoints_submission_cli/

# Integration tests (requires a running API at http://localhost:8080)
pytest tests/endpoints_submission_cli/ -m integration
```
