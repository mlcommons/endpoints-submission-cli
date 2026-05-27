# CLI → API mapping

This table maps every CLI command to the API endpoints it calls, in the order they are called.

---

## Run commands

| CLI Command | Step | HTTP Method | Endpoint | Notes |
|---|---|---|---|---|
| `runs list` | 1 | GET | `/runs` | Returns list of run objects. |
| `runs create` | 1 | POST | `/runs` | Registers the run record. |
| | 2 | POST | `/runs/{run_id}/archive` | Uploads the `.tar.gz` archive. |
| | (rollback) | DELETE | `/runs/{run_id}/archive` | Best-effort; only on upload failure. |
| | (rollback) | DELETE | `/runs/{run_id}` | Only on upload failure, after archive delete attempt. |
| `runs get` | 1 | GET | `/runs/{run_id}` | |
| `runs delete` | 1 | DELETE | `/runs/{run_id}/archive` | Best-effort; 404 silently ignored. |
| | 2 | DELETE | `/runs/{run_id}` | |
| `runs pin` | 1 | PATCH | `/runs/{run_id}/pin` | Sets `expires_at = null`. |
| `runs unpin` | 1 | PATCH | `/runs/{run_id}/unpin` | Restores server-policy expiry. |

---

## Submission commands

| CLI Command | Step | HTTP Method | Endpoint | Notes |
|---|---|---|---|---|
| `submissions list` | 1 | GET | `/submissions` | Returns list of submission objects. |
| `submissions create` | 1 | GET | `/runs/{run_id}/archive` | Repeated for each `--run-ids` value. |
| | 2 | POST | `/submissions` | Registers the submission record. |
| | 3 | POST | `/submissions/{id}/archive` | Uploads the bundle `.tar.gz`. |
| | (rollback) | DELETE | `/submissions/{id}` | Only on GitHub PR failure, after upload. |
| | 4 | PATCH | `/submissions/{id}` | Sets `pr_url`, `pr_number`, `status=REVIEW_PENDING`. |
| `submissions get` | 1 | GET | `/submissions/{id}` | Query param `include_runs=true` by default. |
| `submissions update` (run-ids) | 1 | GET | `/submissions/{id}` | Fetch current run list and division. |
| | 2 | PATCH | `/submissions/{id}` | Updates `run_ids` (and any metadata fields). |
| | 3 | GET | `/runs/{run_id}/archive` | Repeated for each desired run. |
| | 4 | POST | `/submissions/{id}/archive` | Uploads updated bundle. |
| | (rollback) | PATCH | `/submissions/{id}` | Restores `run_ids` to original on failure. |
| `submissions update` (metadata only) | 1 | PATCH | `/submissions/{id}` | Single call with only metadata fields. |
| `submissions withdraw` | 1 | DELETE | `/submissions/{id}` | Marks `WITHDRAWN`; returns PR number. |
| | 2 | DELETE | `/submissions/{id}/archive` | Best-effort; failure is a warning. |
| `submissions add-run` | 1 | POST | `/submissions/{id}/runs/{run_id}` | Registers the addition. Returns updated run list. |
| | 2 | GET | `/runs/{run_id}/archive` | Repeated for all runs (including new). |
| | 3 | POST | `/submissions/{id}/archive` | Uploads updated bundle. |
| | (rollback) | DELETE | `/submissions/{id}/runs/{run_id}` | Only on failure after step 1. |
| `submissions remove-run` | 1 | DELETE | `/submissions/{id}/runs/{run_id}` | Returns updated run list. |
| | 2 | GET | `/runs/{run_id}/archive` | Repeated for remaining runs (skipped if none). |
| | 3 | POST | `/submissions/{id}/archive` | Uploads updated bundle (skipped if no runs remain). |
| | (rollback) | POST | `/submissions/{id}/runs/{run_id}` | Only on failure after step 1. |

---

## Authentication

All API calls send the token as the `X-API-Key` request header. The token is resolved in this order:

1. `--token` flag on the command line.
2. `PRISM_USER_API_TOKEN` environment variable.

A missing token raises `AuthError` and the CLI exits with code 1 before any API call is made.

---

## Timeout profile

| Profile | Connect | Read | Write | Used for |
|---|---|---|---|---|
| `_API_TIMEOUT` | 10 s | 30 s | 30 s | All JSON API calls |
| `_UPLOAD_TIMEOUT` | 10 s | 120 s | 300 s | Archive upload (POST …/archive) |
| `_DOWNLOAD_TIMEOUT` | 10 s | 300 s | 30 s | Archive download (GET …/archive) |

All values are **idle timeouts** (time without a byte transferred), not wall-clock totals. A slow-but-steady transfer never times out; only a stalled connection does.
