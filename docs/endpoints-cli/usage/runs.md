# Run commands

Runs represent individual benchmark executions. Each run is registered from a local result folder, uploaded as a compressed archive, and given a UUID that can later be referenced in a submission.

---

## runs list

List all runs registered under the authenticated account.

```bash
endpoints-submission-cli runs list [--token TOKEN] [-j]
```

| Flag | Description |
|---|---|
| `--token TOKEN` | API key. Falls back to `PRISM_USER_API_TOKEN`. |
| `-j` / `--json` | Print raw JSON instead of the Rich table. |

**Example:**

```bash
endpoints-submission-cli runs list

endpoints-submission-cli runs list -j
```

**Output (table):**

```
 ID                                    Model                             Concurrency  Started At
 d5d9873e-5eca-4f8d-a487-4be1cb8b440c  meta-llama/Llama-3.1-8B-Instruct  4            2025-04-10T09:00:00
```

---

## runs create

Parse a local run folder, register a run record with the API, and upload the folder as a compressed archive (`.tar.gz`).

```bash
endpoints-submission-cli runs create \
  --path PATH \
  [--token TOKEN] \
  [--expires-at DATETIME] \
  [--pinned] \
  [--dry-run]
```

| Flag | Required | Description |
|---|---|---|
| `--path PATH` | yes | Path to the local run folder. |
| `--token TOKEN` | no | API key. |
| `--expires-at DATETIME` | no | Expiry datetime in ISO 8601 format (e.g. `2026-01-01T00:00:00`). Defaults to server policy when omitted. |
| `--pinned` | no | Pin the run immediately to prevent automatic expiry. |
| `--dry-run` | no | Print the parsed API payload as JSON and exit without calling the API. |

**Run folder layout** — the folder must contain:

```
<run-folder>/
├── system_info.json        # Hardware and software description (required)
├── config.yaml             # Benchmark configuration (required)
├── result_summary.json     # Aggregated performance metrics (required)
└── runtime_settings.json   # Inference server settings (optional)
```

**Rollback behaviour:** if the archive upload fails after the run record has been created, the CLI automatically deletes the run record to leave a clean state. If that delete also fails, the orphaned run ID is printed so it can be cleaned up manually.

**Example:**

```bash
# Register and upload
endpoints-submission-cli runs create --path ./results/llama3_h100_c4

# Preview the payload without calling the API
endpoints-submission-cli runs create --path ./results/llama3_h100_c4 --dry-run

# Register with a custom expiry
endpoints-submission-cli runs create \
  --path ./results/llama3_h100_c4 \
  --expires-at 2026-06-01T00:00:00

# Register and pin immediately
endpoints-submission-cli runs create --path ./results/llama3_h100_c4 --pinned
```

**Output:**

```
Run created: d5d9873e-5eca-4f8d-a487-4be1cb8b440c
Archive: gs://mlperf-runs/d5d9873e-…/d5d9873e-….tar.gz
```

---

## runs get

Fetch full details of a single run and optionally download its archive.

```bash
endpoints-submission-cli runs get \
  --run-id RUN_ID \
  [--download-to DIR] \
  [--token TOKEN]
```

| Flag | Required | Description |
|---|---|---|
| `--run-id RUN_ID` | yes | Run UUID. |
| `--download-to DIR` | no | Directory to download the run archive (`.tar.gz`) into. Created automatically if it does not exist. |
| `--token TOKEN` | no | API key. |

Run details are always printed as JSON (syntax-highlighted in a terminal, plain when piped). When `--download-to` is provided, the archive is saved as `<run-id>.tar.gz` inside the specified directory and the saved path is printed on a separate line after the JSON.

**Example:**

```bash
# Fetch run details only
endpoints-submission-cli runs get --run-id d5d9873e-5eca-4f8d-a487-4be1cb8b440c

# Fetch details and download the archive to ./downloads/
endpoints-submission-cli runs get \
  --run-id d5d9873e-5eca-4f8d-a487-4be1cb8b440c \
  --download-to ./downloads
```

**Output (with `--download-to`):**

```
{ ... }  ← run details JSON
Archive saved to ./downloads/d5d9873e-5eca-4f8d-a487-4be1cb8b440c.tar.gz
```

---

## runs delete

Delete a run record and its stored archive.

```bash
endpoints-submission-cli runs delete \
  --run-id RUN_ID \
  [--token TOKEN]
```

| Flag | Required | Description |
|---|---|---|
| `--run-id RUN_ID` | yes | Run UUID. |
| `--token TOKEN` | no | API key. |

> **Note:** Runs that belong to an active submission cannot be deleted. Withdraw the submission first (`submissions withdraw`), then delete the run.

**Order of operations:** archive is deleted from storage first (best-effort; a 404 is silently ignored), then the run record is deleted from the database.

**Example:**

```bash
endpoints-submission-cli runs delete --run-id d5d9873e-5eca-4f8d-a487-4be1cb8b440c
```

---

## runs pin

Pin a run to prevent automatic expiry (`expires_at` is set to `null`).

```bash
endpoints-submission-cli runs pin \
  --run-id RUN_ID \
  [--token TOKEN]
```

| Flag | Required | Description |
|---|---|---|
| `--run-id RUN_ID` | yes | Run UUID. |
| `--token TOKEN` | no | API key. |

**Example:**

```bash
endpoints-submission-cli runs pin --run-id d5d9873e-5eca-4f8d-a487-4be1cb8b440c
# → Run pinned: d5d9873e-5eca-4f8d-a487-4be1cb8b440c
```

---

## runs unpin

Restore normal expiry behaviour on a pinned run.

```bash
endpoints-submission-cli runs unpin \
  --run-id RUN_ID \
  [--token TOKEN]
```

| Flag | Required | Description |
|---|---|---|
| `--run-id RUN_ID` | yes | Run UUID. |
| `--token TOKEN` | no | API key. |

**Example:**

```bash
endpoints-submission-cli runs unpin --run-id d5d9873e-5eca-4f8d-a487-4be1cb8b440c
# → Run unpinned: d5d9873e-5eca-4f8d-a487-4be1cb8b440c
```