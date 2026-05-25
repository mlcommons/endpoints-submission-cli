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
| `MLPERF_SUBMISSION_REPO` | `MLCommons-Systems/test-endpoints-submission-repo` | GitHub repository for submission PRs |

Add these to your shell profile for persistent configuration:

```bash
export PRISM_USER_API_TOKEN=mlc_your_token_here
export MLPERF_API_BASE_URL=https://api.mlcommons.org
export MLPERF_SUBMISSION_REPO=MLCommons-Systems/test-endpoints-submission-repo
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
endpoints-submission-cli runs create --path PATH [--token TOKEN] [--expires-at DATETIME] [--pinned] [--dry-run]
```

| Flag | Description |
|------|-------------|
| `--path PATH` | Path to the local run folder (required) |
| `--token TOKEN` | API token |
| `--expires-at DATETIME` | Expiry datetime in ISO 8601 format (e.g. `2026-01-01T00:00:00`). Defaults to server policy. |
| `--pinned` | Pin the run immediately to prevent automatic expiry. |
| `--dry-run` | Print the parsed API payload as JSON and exit without calling the API. |

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
7. Store the PR URL, PR number, and set status to `REVIEW_PENDING` on the submission record.

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
| `--embargo-date DATETIME` | no | Embargo datetime in ISO 8601 format (e.g. `2025-12-01T00:00:00`) |
| `--dry-run` | no | Assemble folder, run checker, print layout — exit without submitting |

**If the GitHub PR step fails** the submission record and uploaded bundle still
exist. Retry the PR step manually with `gh pr create` on the submission branch,
then contact MLCommons to update the PR linkage.

---

### submissions get

Fetch full details of a single submission, including embedded run records.

```bash
endpoints-submission-cli submissions get \
  --submission-id SUB_ID [--token TOKEN] [-j]
```

---

### submissions update

Update the run list or target availability date on an existing submission.

```bash
endpoints-submission-cli submissions update \
  --submission-id SUB_ID \
  [--token TOKEN] \
  [--run-ids RUN_ID] \
  [--target-availability-date DATE]
```

| Flag | Description |
|------|-------------|
| `--submission-id` | Submission UUID (required) |
| `--token TOKEN` | API token |
| `--run-ids RUN_ID` | Set the complete run UUID list. Repeatable — pass once per run. Runs not listed are removed. |
| `--target-availability-date DATE` | Target availability date (`YYYY-MM-DD`) |
| `--publication-cycle CYCLE` | Publication cycle (e.g. `2025-04-C1`) |
| `--embargo-date DATETIME` | Embargo datetime in ISO 8601 format (e.g. `2025-12-01T00:00:00`) |

**When `--run-ids` is provided** the command runs a full rebuild:

1. Check GitHub prerequisites (`gh` installed and authenticated).
2. GET current submission to determine the division, PR number, and existing run list.
3. Log added / removed runs.
4. PATCH the DB with the new run list (and any metadata fields in the same call).
5. Download all desired run archives (with progress bar).
6. Assemble the submission folder and run the Submission Checker — rollback and abort on errors.
7. Clone the submission repository, check out the existing PR branch, and apply the surgical merge — rollback and abort on errors.
8. Upload the merged bundle to blob storage (`POST /submissions/{id}/archive`) — rollback and abort on errors.
9. Push the merged branch to the GitHub PR.

**Rollback:** if any step 5–8 fails after the DB PATCH, the run list is automatically restored to
its original value. The GitHub push (step 9) is non-fatal — if it fails, blob storage and the DB
are already consistent; re-run `submissions update` to retry.

**GitHub PR branch file update strategy:** The CLI clones the submission repository and checks out
the existing PR branch. It then compares the fresh build against what is already on the branch —
only generated content is overwritten; files that may have been manually edited by reviewers are
preserved. `points/` and `accuracy/` are replaced entirely; log files are replaced per-point;
`system_desc.json` is preserved from the PR branch (seeded for new points); `systems/`, `src/`,
and `documentation/` are preserved from the PR branch. Commit message format:
`update: add <ids>; remove <ids> (<N> runs total)`.

> **Blob storage and GitHub PR branch content:** Both destinations receive the merged result — the
> fresh build with reviewer-edited files (`system_desc.json`, `systems/`) preserved from the PR
> branch. Blob storage and the GitHub PR branch always contain identical content.

**When only metadata flags are provided** (no `--run-ids`) the command is a DB-only PATCH —
no download, rebuild, archive upload, or GitHub push occurs.

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

**Pipeline:**

1. Check GitHub prerequisites (`gh` installed and authenticated).
2. `POST /submissions/{id}/runs/{run_id}` — register the addition.
3. Download all run archives (including the newly added run).
4. Rebuild the submission folder.
5. Run the Submission Checker — rollback and abort on errors.
6. Clone the submission repository, check out the existing PR branch, and apply the surgical merge — rollback and abort on errors.
7. Upload the merged bundle to blob storage (`POST /submissions/{id}/archive`) — rollback and abort on errors.
8. Push the merged branch to the GitHub PR.

**Rollback:** if any step 3–7 fails after registration, the run is automatically removed from the
submission record before exiting. The GitHub push (step 8) is non-fatal — if it fails, blob
storage and the DB are already consistent; re-run `submissions add-run` to retry.

**GitHub PR branch file update strategy:** The CLI clones the submission repository and checks
out the existing PR branch. It then compares the fresh build against what is already on the
branch — only generated content is overwritten; files that may have been manually edited by
reviewers are preserved. `points/` and `accuracy/` are replaced entirely; log files
(`mlperf_endpoints_log_*.json`) are replaced per-point; `system_desc.json` is preserved from
the PR branch (seeded for new points); `systems/`, `src/`, and `documentation/` are preserved
from the PR branch; `systems/` is seeded from the fresh build only if absent.

> **Blob storage and GitHub PR branch content:** Both destinations receive the merged result — the
> fresh build with reviewer-edited files (`system_desc.json`, `systems/`) preserved from the PR
> branch. Blob storage and the GitHub PR branch always contain identical content.

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

**Pipeline:**

1. Check GitHub prerequisites (`gh` installed and authenticated).
2. `DELETE /submissions/{id}/runs/{run_id}` — register the removal.
3. Download remaining run archives — skipped if no runs remain.
4. Rebuild the submission folder — skipped if no runs remain.
5. Run the Submission Checker — rollback and abort on errors; skipped if no runs remain.
6. Clone the submission repository, check out the existing PR branch, and apply the surgical merge — rollback and abort on errors; skipped if no runs remain.
7. Upload the merged bundle to blob storage (`POST /submissions/{id}/archive`) — rollback and abort on errors; skipped if no runs remain.
8. Push the merged branch to the GitHub PR — skipped if no runs remain.

If no runs remain after removal, steps 3–8 are skipped and a warning is printed.

**Rollback:** if any step 3–7 fails after removal, the run is automatically re-added to the
submission record. The GitHub push (step 8) is non-fatal — if it fails, blob storage and the DB
are already consistent; re-run `submissions remove-run` to retry.

**GitHub PR branch file update strategy:** The CLI clones the submission repository and checks
out the existing PR branch. It then compares the fresh build against what is already on the
branch — only generated content is overwritten; files that may have been manually edited by
reviewers are preserved. `points/` and `accuracy/` are replaced entirely; log files are
replaced per-point; `system_desc.json` is preserved from the PR branch; point dirs for the
removed run are deleted; `systems/`, `src/`, and `documentation/` are preserved from the PR
branch.

> **Blob storage and GitHub PR branch content:** Both destinations receive the merged result — the
> fresh build with reviewer-edited files (`system_desc.json`, `systems/`) preserved from the PR
> branch. Blob storage and the GitHub PR branch always contain identical content.

---

## Environment variable reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PRISM_USER_API_TOKEN` | yes* | — | API key for the PRISM Submission API (`mlc_…`). Can be passed per-command with `--token` instead. |
| `MLPERF_API_BASE_URL` | no | `http://localhost:8080` | Base URL of the PRISM Submission API. |
| `MLPERF_SUBMISSION_REPO` | no | `MLCommons-Systems/test-endpoints-submission-repo` | Target GitHub repository for submission PRs (`org/repo` format). |

\* Required unless `--token` is passed.

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Command completed successfully |
| `1` | Error — authentication failure, API error, invalid input, or compliance check failure |

---

## Run folder layout

`runs create` expects a directory containing these files:

```
<run-folder>/
├── system_info.json        # Hardware and software description (required)
├── config.yaml             # Benchmark configuration (model, concurrency, endpoints, …) (required)
├── result_summary.json     # Aggregated performance metrics (required)
└── runtime_settings.json   # Inference server runtime settings (optional)
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

**`runtime_settings.json`** *(optional)* — inference server runtime settings used
as the base for the `runtime_settings` block in each `point_<N>.yaml`:

```json
{
  "max_num_seqs": 256,
  "tensor_parallel_size": 8
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
