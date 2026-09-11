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

Add these to your shell profile for persistent configuration:

```bash
export PRISM_USER_API_TOKEN=mlc_your_token_here
export MLPERF_API_BASE_URL=https://api.mlcommons.org
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
SUB_ID=a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### Withdraw a submission

```bash
endpoints-submission-cli submissions withdraw --submission-id $SUB_ID
# Deletes the stored bundle
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
endpoints-submission-cli runs create --path PATH [--token TOKEN] [--expires-at DATETIME] [--pinned] [--test] [--dry-run]
```

| Flag | Description |
|------|-------------|
| `--path PATH` | Path to the local run folder (required) |
| `--token TOKEN` | API token |
| `--expires-at DATETIME` | Expiry datetime in ISO 8601 format (e.g. `2026-01-01T00:00:00`). Defaults to server policy. |
| `--pinned` | Pin the run immediately to prevent automatic expiry. |
| `--test` | Mark the run as a test run (excluded from published reporting; fixed at creation). |
| `--dry-run` | Print the parsed API payload as JSON and exit without calling the API. |

The folder must contain these three files (see [Run folder layout](#run-folder-layout)):

```
system_desc.json
point.yaml
performance/result_summary.json
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
6. Set status to `REVIEW_PENDING` on the submission record.

```bash
endpoints-submission-cli submissions create \
  --division    DIVISION \
  --availability AVAILABILITY \
  --run-ids     RUN_ID_1 \
  --run-ids     RUN_ID_2 \
  [--token TOKEN] \
  [--provisional] \
  [--yes] \
  [--publication-cycle CYCLE] \
  [--target-availability-date DATE]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--division` | yes | `standardized`, `serviced`, or `rdi` |
| `--availability` | yes | `available`, `preview`, or `rdi` |
| `--run-ids RUN_ID` | yes (repeatable) | Run UUID(s) to include; pass the flag once per run |
| `--token TOKEN` | no | API token |
| `--provisional` | no | Request provisional publication (default: false). Results become publicly viewable on the visualizer during the next cohort with a `peer review pending` disclaimer. Prompts for confirmation before submitting |
| `--yes`, `-y` | no | Skip the `--provisional` confirmation prompt (for non-interactive use) |
| `--publication-cycle CYCLE` | no | Target cycle, e.g. `2025-04-C1` |
| `--target-availability-date DATE` | no | `YYYY-MM-DD`; required when availability is `preview` |
| `--embargo-date DATETIME` | no | Embargo datetime in ISO 8601 format (e.g. `2025-12-01T00:00:00`) |
| `--dry-run` | no | Assemble folder, run checker, print layout — exit without submitting |

**If the final status PATCH fails** the submission record and uploaded bundle still
exist; the failure is a warning, not a fatal error. The CLI does not open the review
pull request — `pr_url` and `pr_number` stay on the record for whatever does.

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

1. GET the current submission to determine its division and existing run list.
2. **Reject the update if it would add a run** (see below); log removed runs.
3. PATCH the DB with the new run list (and any metadata fields in the same call).
4. Download the remaining run archives (with progress bar).
5. Assemble the submission folder and run the Submission Checker — rollback and abort on errors.
6. Upload the bundle to blob storage (`POST /submissions/{id}/archive`) — rollback and abort on errors.

**Rollback:** if any step 4–6 fails after the DB PATCH, the run list is automatically restored to
its original value.

> **`--run-ids` may only shrink the list.** MLPerf Endpoints Submission Rules §8 no longer
> provide a post-submission window for adding measurement points, so a list containing a run
> the submission does not already have is rejected before anything is fetched or patched.
> Removals are still permitted.

**When only metadata flags are provided** (no `--run-ids`) the command is a DB-only PATCH —
no download, rebuild, archive upload, or GitHub push occurs.

Providing no optional flags prints a warning and makes no API call.

---

### submissions withdraw

Withdraw a submission: marks it `WITHDRAWN` and deletes
the stored bundle.

```bash
endpoints-submission-cli submissions withdraw \
  --submission-id SUB_ID [--token TOKEN]
```

The operations are ordered so that the DB state is updated first. Archive deletion
is best-effort — a failure is reported as a warning but does not affect the exit
code. The CLI does not close the review pull request; it no longer manages one.

### submissions remove-run

Remove a run from an existing submission. If runs still remain, the bundle is
rebuilt, compliance-checked, and re-uploaded.

```bash
endpoints-submission-cli submissions remove-run \
  --submission-id SUB_ID \
  --run-id RUN_ID \
  [--token TOKEN]
```

**Pipeline:**

1. `DELETE /submissions/{id}/runs/{run_id}` — register the removal.
2. Download remaining run archives — skipped if no runs remain.
3. Rebuild the submission folder — skipped if no runs remain.
4. Run the Submission Checker — rollback and abort on errors; skipped if no runs remain.
5. Upload the bundle to blob storage (`POST /submissions/{id}/archive`) — rollback and abort
   on errors; skipped if no runs remain.

If no runs remain after removal, steps 2–5 are skipped and a warning is printed.

**Rollback:** if any step 2–5 fails after removal, the run is automatically re-added to the
submission record.

> **A withdrawn point cannot be replaced.** Submission Rules §8.1 lets a submitter withdraw a
> faulty measurement point during peer review, which is what this command does — but §8 no
> longer provides a window for adding points, so a submission that drops below the 7-point
> minimum cannot be repaired by adding another. The Submission Checker reports the shortfall.

---

## Environment variable reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PRISM_USER_API_TOKEN` | yes* | — | API key for the PRISM Submission API (`mlc_…`). Can be passed per-command with `--token` instead. |
| `MLPERF_API_BASE_URL` | no | `http://localhost:8080` | Base URL of the PRISM Submission API. |

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
├── system_desc.json                  # §8.2 hardware/software description (required)
├── point.yaml                        # §8.3 Pareto-point disclosure (required)
│                                     #   ^ both authored by the submitter, not by endpoints
├── performance/result_summary.json   # Performance metrics (required)
├── accuracy/accuracy_results.json    # Per-dataset accuracy scores
├── config.yaml                       # Resolved benchmark configuration (optional as of v1.0)
├── events.jsonl                      # Per-event log (largest file by far)
├── report.txt                        # Human-readable report
├── sample_idx_map.json               # Sample index mapping
├── metrics/final_snapshot.json       # Metrics snapshot
├── src/<implementation>/             # Merged into the bundle's shared src/ (README.md required)
└── documentation/                    # Merged into the bundle's shared docs/
```

This is the layout `mlcommons/endpoints` writes to its `report_dir` — `performance/`
exists when the performance phase ran, `accuracy/` when the accuracy phase ran, and
`--mode both` produces both. See
[run-folder layout](endpoints-cli/reference/run-folder-layout.md) for a captured example. Flat
layouts that put `result_summary.json` at the top level are not accepted by
`runs create`.

**`system_desc.json`** — §8.2 system description:

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

**`point.yaml`** — the §8.3 Pareto-point disclosure for this run, validated by the
Submission Checker and copied into the bundle **verbatim**:

```yaml
concurrency: 4
region: low_latency
dataset: cnn_dailymail

# §8.3 disclosure
division: Standardized
max_supported_concurrency: 1024
model_name: llama3.1-8b
model_precision: FP16
link_to_model: https://example.com/model
link_to_model_transformation: https://example.com/quantization
dataset_name: CNN/DailyMail
dataset_type: Performance
dataset_link: https://example.com/dataset

# §4.6 seed binding — the set is named here and its values used below
seed_set: A
target_cohort: 2026-09-C0

# §8.1 bundle-internal pointers; injected by the builder when absent
shared_src: src/trtllm
shared_docs: docs

runtime_settings:
  load_pattern: concurrency
  stream_all_chunks: true
  min_duration_ms: 1200000
  min_sample_count: 2000
  runtime:
    scheduler_rng_seed: 10487924139932647040
    sample_index_rng_seed: 586478644936801402
    model_seed: 9315206023656308754
warmup:
  duration_s: 60.0
  requests_issued: 40
  requests_completed: 40
  data_source: cnn_dailymail validation split
  concurrency: 4
  initialization_steps: [model loaded, kv-cache warmed]
  logs_retained: true
```

The CLI does **not** derive this from `config.yaml`. It used to, which silently produced
nulls for any disclosure field a harness did not happen to put in its config — and the
checker then rejected the bundle. `point.yaml` is now required and passed through
untouched, so the harness owns the disclosure. `config.yaml` became **optional** in
v1.0: it records what the harness was told to do and carries no disclosure of its own.

The one exception is `shared_src` / `shared_docs`. The builder fills these in when a
run does not declare them, because their referent — `src/<implementation>/`, the union
of every run's `src/` folder — does not exist until the builder assembles the bundle, so
no single run archive can name it. A value the run *does* declare is validated and never
overwritten; one that does not resolve fails the build.

v1.0 also rotates the RNG seeds (§4.6). v0.7 fixed every seed at 42; a v1.0 point names
a published seed set and seeds `scheduler_rng_seed`, `sample_index_rng_seed` and
`model_seed` from it.

**`runtime_settings.json`** *(optional)* — inference server runtime settings recorded
alongside the run:

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
<submitting_organization>/
└── <submission_id>/                       # assigned by MLC; one per submission
    ├── src/                               # SHARED across the whole submission
    │   └── <implementation>/              # e.g. trtllm/, vllm/, sglang/
    │       ├── README.md                  # how to build/launch the SUT, reproduce a point
    │       └── <endpoint interface code, infra/cluster setup, client harness>
    │
    ├── docs/                              # SHARED across the whole submission
    │   ├── calibration.adoc               # if weight transformations applied (§3.3)
    │   ├── software_disclosure.md         # §8.4
    │   └── <additional documentation>
    │
    └── results/
        └── <system>/                      # e.g. H200-SXM-141GBx8_TRT/
            └── <benchmark_model>/         # e.g. deepseek-r1/, gpt-oss-120b/
                └── r<N>/                  # one PARETO POINT per concurrency (r1, r32, …)
                    ├── point.yaml             # §8.3
                    ├── system_desc.json       # §8.2 — per point since policies PR #119
                    ├── result_summary.json    # aggregate metrics (QPS, TPS, TTFT, TPOT)
                    ├── accuracy_results.json  # §6.6
                    ├── config.yaml            # OPTIONAL as of v1.0
                    └── server_configs/        # OPTIONAL, point-specific backend configs
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
