# Submission commands

A submission groups one or more benchmark runs into a package that is compliance-checked, uploaded, and submitted as a GitHub pull request to the MLCommons review repository.

---

## Submission status lifecycle

| Status | Set by | Meaning |
|---|---|---|
| `REVIEW_PENDING` | `submissions create` (step 9) | Submission created, PR open, awaiting review. |
| `WITHDRAWN` | `submissions withdraw` | Submission retracted; PR closed, archive deleted. |

Additional statuses set by the review workflow (server-side, not by the CLI):

| Status | Meaning |
|---|---|
| `FINALIZED` | Review complete; submission accepted. |
| `PUBLISHED` | Results published in the MLPerf leaderboard. |

---

## submissions list

List all submissions for the authenticated account.

```bash
endpoints-submission-cli submissions list [--token TOKEN] [-j]
```

| Flag | Description |
|---|---|
| `--token TOKEN` | API key. Falls back to `PRISM_USER_API_TOKEN`. |
| `-j` / `--json` | Print raw JSON instead of the Rich table. |

**Example:**

```bash
endpoints-submission-cli submissions list

endpoints-submission-cli submissions list -j
```

---

## submissions create

Create a new submission from one or more registered runs. This command runs the full automated pipeline:

1. Check GitHub prerequisites (`gh` installed and authenticated).
2. Download all run archives from the API (with progress bar).
3. Assemble the submission folder structure.
4. Run the Submission Checker — aborts with exit code 1 on compliance errors.
5. `POST /submissions` to register the submission.
6. Upload the submission bundle (`POST /submissions/{id}/archive`).
7. Clone the submission repository, create a branch, commit, push.
8. Create a GitHub PR (`gh pr create`).
9. Call `update_submission` internally (`PATCH /submissions/{id}`) to store `pr_url`, `pr_number`, and set `status=REVIEW_PENDING`.

```bash
endpoints-submission-cli submissions create \
  --division DIVISION \
  --scenario SCENARIO \
  --availability AVAILABILITY \
  --run-ids RUN_ID \
  [--run-ids RUN_ID ...] \
  [--token TOKEN] \
  [--provisional] \
  [--yes] \
  [--publication-cycle CYCLE] \
  [--target-availability-date DATE] \
  [--embargo-date DATETIME] \
  [--dry-run]
```

| Flag | Required | Description |
|---|---|---|
| `--division` | yes | `standardized`, `serviced`, or `rdi`. |
| `--scenario` | yes | `cop` (Client-on-Premises) or `con` (Client-over-Network). |
| `--availability` | yes | `available`, `preview`, or `rdi`. |
| `--run-ids RUN_ID` | yes (repeatable) | Run UUID to include. Pass the flag once per run. |
| `--token TOKEN` | no | API key. |
| `--provisional` | no | Request provisional publication (default: false). Results become publicly viewable on the visualizer during the next cohort with a `peer review pending` disclaimer. Prompts for confirmation before submitting. |
| `--yes`, `-y` | no | Skip the `--provisional` confirmation prompt (for non-interactive use). |
| `--publication-cycle CYCLE` | no | Target publication cycle, e.g. `2025-04-C1`. |
| `--target-availability-date DATE` | no | Target availability date (`YYYY-MM-DD`). Required when `--availability preview`. |
| `--embargo-date DATETIME` | no | Embargo datetime in ISO 8601 format, e.g. `2025-12-01T00:00:00`. |
| `--dry-run` | no | Assemble folder and run checker, then print the folder layout and exit without creating the submission or PR. |

**Rollback behaviour:** if the GitHub PR step fails, the submission is automatically withdrawn (`DELETE /submissions/{id}`) to leave a clean state. If that rollback also fails, the orphaned submission ID is printed.

**If only the PATCH step (step 9) fails** — the submission and PR both exist; the failure is a warning, not a fatal error. The PR URL can be linked manually.

**Example:**

```bash
# Basic submission with one run
endpoints-submission-cli submissions create \
  --division standardized \
  --scenario cop \
  --availability available \
  --run-ids d5d9873e-5eca-4f8d-a487-4be1cb8b440c

# Multiple runs
endpoints-submission-cli submissions create \
  --division standardized \
  --scenario cop \
  --availability available \
  --run-ids d5d9873e-5eca-4f8d-a487-4be1cb8b440c \
  --run-ids f7e6d5c4-b3a2-1098-7654-321fedcba098

# Preview availability with required date
endpoints-submission-cli submissions create \
  --division standardized \
  --scenario con \
  --availability preview \
  --run-ids d5d9873e-5eca-4f8d-a487-4be1cb8b440c \
  --target-availability-date 2025-09-01

# Dry run — check compliance without submitting
endpoints-submission-cli submissions create \
  --division standardized \
  --scenario cop \
  --availability available \
  --run-ids d5d9873e-5eca-4f8d-a487-4be1cb8b440c \
  --dry-run
```

**Output:**

```
Checking GitHub prerequisites…
Downloading run archives…
Assembling submission folder…
Running Submission Checker…
Checker report written to submission_checker_20250410_090123.log
Uploading submission bundle…
Creating GitHub PR…
Submission created: a1b2c3d4-e5f6-7890-abcd-ef1234567890
PR: https://github.com/MLCommons-Systems/test-endpoints-submission-repo/pull/42
```

**Submission folder structure** assembled by the CLI:

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
            │       ├── mlperf_endpoints_log_detail.json
            │       └── system_desc.json
            └── accuracy/
                ├── accuracy.txt
                └── accuracy_result.json
```

---

## submissions get

Fetch full details of a single submission, including embedded run records.

```bash
endpoints-submission-cli submissions get \
  --submission-id SUB_ID \
  [--token TOKEN] \
  [-j]
```

| Flag | Required | Description |
|---|---|---|
| `--submission-id SUB_ID` | yes | Submission UUID. |
| `--token TOKEN` | no | API key. |
| `-j` / `--json` | no | Print raw JSON. |

**Example:**

```bash
endpoints-submission-cli submissions get \
  --submission-id a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

## submissions update

Update the run list or metadata fields on an existing submission.

```bash
endpoints-submission-cli submissions update \
  --submission-id SUB_ID \
  [--token TOKEN] \
  [--run-ids RUN_ID ...] \
  [--target-availability-date DATE] \
  [--publication-cycle CYCLE] \
  [--embargo-date DATETIME]
```

| Flag | Required | Description |
|---|---|---|
| `--submission-id SUB_ID` | yes | Submission UUID. |
| `--token TOKEN` | no | API key. |
| `--run-ids RUN_ID` | no (repeatable) | Replace the complete run list. Pass once per run. Runs not listed are removed. |
| `--target-availability-date DATE` | no | Target availability date (`YYYY-MM-DD`). |
| `--publication-cycle CYCLE` | no | Publication cycle (e.g. `2025-04-C1`). |
| `--embargo-date DATETIME` | no | Embargo datetime in ISO 8601 format. |

Providing no flags prints a warning and makes no API call.

### When `--run-ids` is provided (full rebuild)

The command runs the full rebuild pipeline:

1. Check GitHub prerequisites (`gh` installed and authenticated).
2. `GET /submissions/{id}` — fetch current run list, division, and PR number.
3. Log added/removed runs.
4. `PATCH /submissions/{id}` with new `run_ids` (and any metadata fields) in a single call.
5. Download all desired run archives (with progress bar).
6. Assemble submission folder and run the Submission Checker — rollback on errors.
7. Clone the submission repository, check out the existing PR branch, and apply the surgical merge — rollback on errors.
8. Upload the merged bundle to blob storage (`POST /submissions/{id}/archive`) — rollback on errors.
9. Push the merged branch to the GitHub PR.

**Rollback:** if any step 5–8 fails after the DB PATCH, the run list is automatically restored to its original value (`PATCH /submissions/{id}` with `original_run_ids`). 

**GitHub PR branch file update strategy:** The CLI clones the submission repository and checks out the existing PR branch. It then compares the fresh build against what is already on the branch and makes per-directory decisions — only generated content is overwritten; files that may have been manually edited by reviewers are preserved:

| Path | Action |
|---|---|
| `points/` | Replaced entirely from the fresh build. |
| `accuracy/` | Replaced entirely from the fresh build. |
| `results/<point>/mlperf_endpoints_log_*.json` | Replaced from the fresh build. |
| `results/<point>/system_desc.json` | Preserved from the PR branch. Seeded from the fresh build only for points not yet on the branch. |
| Point dirs in `results/` removed from the fresh build | Deleted from the PR branch. |
| `systems/` | Preserved from the PR branch. Seeded from the fresh build only if the directory does not yet exist on the branch. |
| `src/`, `documentation/` | Preserved from the PR branch. |

> **Blob storage and GitHub PR branch content:** Both destinations receive the merged result — the fresh build with reviewer-edited files (`system_desc.json`, `systems/`) preserved from the PR branch. Blob storage and the GitHub PR branch always contain identical content.

### When only metadata flags are provided (DB-only PATCH)

No download or rebuild. A single `PATCH /submissions/{id}` is sent with only the specified fields.

**Example:**

```bash
# Replace run list (triggers full rebuild)
endpoints-submission-cli submissions update \
  --submission-id a1b2c3d4-… \
  --run-ids d5d9873e-… \
  --run-ids f7e6d5c4-…

# Update availability date only (no rebuild)
endpoints-submission-cli submissions update \
  --submission-id a1b2c3d4-… \
  --target-availability-date 2025-10-01

# Update publication cycle and embargo date
endpoints-submission-cli submissions update \
  --submission-id a1b2c3d4-… \
  --publication-cycle 2025-04-C1 \
  --embargo-date 2025-12-01T00:00:00

# Combine run list update with metadata update
endpoints-submission-cli submissions update \
  --submission-id a1b2c3d4-… \
  --run-ids d5d9873e-… \
  --run-ids f7e6d5c4-… \
  --target-availability-date 2025-10-01
```

---

## submissions withdraw

Withdraw a submission: marks it `WITHDRAWN`, closes its GitHub PR, and deletes the stored bundle.

```bash
endpoints-submission-cli submissions withdraw \
  --submission-id SUB_ID \
  [--token TOKEN]
```

**Order of operations:** DB update (`DELETE /submissions/{id}`) → close PR (`gh pr close`) → delete archive (`DELETE /submissions/{id}/archive`).

PR closure and archive deletion are best-effort — failures are reported as warnings but do not change the exit code. The submission is already `WITHDRAWN` in the database.

**Example:**

```bash
endpoints-submission-cli submissions withdraw \
  --submission-id a1b2c3d4-e5f6-7890-abcd-ef1234567890
# → Submission withdrawn: a1b2c3d4-…
```

If PR close fails the CLI prints the manual close command:

```
PR close failed (submission already WITHDRAWN): …
Close manually: gh pr close 42 --repo MLCommons-Systems/test-endpoints-submission-repo
```

---

## submissions add-run

Add a run to an existing submission, rebuild the bundle, re-run compliance checking, and push a new commit to the PR branch.

```bash
endpoints-submission-cli submissions add-run \
  --submission-id SUB_ID \
  --run-id RUN_ID \
  [--token TOKEN]
```

| Flag | Required | Description |
|---|---|---|
| `--submission-id SUB_ID` | yes | Submission UUID. |
| `--run-id RUN_ID` | yes | Run UUID to add. |
| `--token TOKEN` | no | API key. |

**Pipeline:**

1. Check GitHub prerequisites (`gh` installed and authenticated).
2. `POST /submissions/{id}/runs/{run_id}` — register the addition.
3. Download all run archives (including the newly added run).
4. Rebuild submission folder and run Submission Checker — rollback on errors.
5. Clone the submission repository, check out the existing PR branch, and apply the surgical merge — rollback on errors.
6. Upload the merged bundle to blob storage (`POST /submissions/{id}/archive`) — rollback on errors.
7. Push the merged branch to the GitHub PR.

**Rollback:** if any step 3–6 fails after registration, the run is automatically removed from the submission record (`DELETE /submissions/{id}/runs/{run_id}`).

**GitHub PR branch file update strategy:** The CLI clones the submission repository and checks out the existing PR branch. It then compares the fresh build against what is already on the branch and makes per-directory decisions:

| Path | Action |
|---|---|
| `points/` | Replaced entirely from the fresh build. |
| `accuracy/` | Replaced entirely from the fresh build. |
| `results/<point>/mlperf_endpoints_log_*.json` | Replaced from the fresh build. |
| `results/<point>/system_desc.json` | Preserved from the PR branch. Seeded from the fresh build only for points not yet on the branch. |
| Point dirs in `results/` removed from the fresh build | Deleted from the PR branch. |
| `systems/` | Preserved from the PR branch. Seeded from the fresh build only if the directory does not yet exist on the branch. |
| `src/`, `documentation/` | Preserved from the PR branch. |

> **Blob storage and GitHub PR branch content:** Both destinations receive the merged result — the fresh build with reviewer-edited files (`system_desc.json`, `systems/`) preserved from the PR branch. Blob storage and the GitHub PR branch always contain identical content.

**Example:**

```bash
endpoints-submission-cli submissions add-run \
  --submission-id a1b2c3d4-… \
  --run-id f7e6d5c4-b3a2-1098-7654-321fedcba098
# → Run f7e6d5c4 added to submission a1b2c3d4-…
```

---

## submissions remove-run

Remove a run from an existing submission. If runs still remain, the bundle is rebuilt, compliance-checked, re-uploaded, and pushed to the PR branch.

```bash
endpoints-submission-cli submissions remove-run \
  --submission-id SUB_ID \
  --run-id RUN_ID \
  [--token TOKEN]
```

| Flag | Required | Description |
|---|---|---|
| `--submission-id SUB_ID` | yes | Submission UUID. |
| `--run-id RUN_ID` | yes | Run UUID to remove. |
| `--token TOKEN` | no | API key. |

**Pipeline:**

1. Check GitHub prerequisites (`gh` installed and authenticated).
2. `DELETE /submissions/{id}/runs/{run_id}` — register the removal.
3. Download remaining run archives. Skipped if no runs remain.
4. Rebuild submission folder and run Submission Checker — rollback on errors. Skipped if no runs remain.
5. Clone the submission repository, check out the existing PR branch, and apply the surgical merge — rollback on errors. Skipped if no runs remain.
6. Upload the merged bundle to blob storage (`POST /submissions/{id}/archive`) — rollback on errors. Skipped if no runs remain.
7. Push the merged branch to the GitHub PR. Skipped if no runs remain.

If no runs remain after removal, steps 3–7 are skipped and a warning is printed.

**Rollback:** if any step 3–6 fails after removal, the run is automatically re-added (`POST /submissions/{id}/runs/{run_id}`).  `submissions remove-run` to retry.

**GitHub PR branch file update strategy:** The CLI clones the submission repository and checks out the existing PR branch. It then compares the fresh build against what is already on the branch and makes per-directory decisions:

| Path | Action |
|---|---|
| `points/` | Replaced entirely from the fresh build. |
| `accuracy/` | Replaced entirely from the fresh build. |
| `results/<point>/mlperf_endpoints_log_*.json` | Replaced from the fresh build. |
| `results/<point>/system_desc.json` | Preserved from the PR branch. Seeded from the fresh build only for points not yet on the branch. |
| Point dirs in `results/` for the removed run | Deleted from the PR branch. |
| `systems/` | Preserved from the PR branch. Seeded from the fresh build only if the directory does not yet exist on the branch. |
| `src/`, `documentation/` | Preserved from the PR branch. |

> **Blob storage and GitHub PR branch content:** Both destinations receive the merged result — the fresh build with reviewer-edited files (`system_desc.json`, `systems/`) preserved from the PR branch. Blob storage and the GitHub PR branch always contain identical content.

**Example:**

```bash
endpoints-submission-cli submissions remove-run \
  --submission-id a1b2c3d4-… \
  --run-id f7e6d5c4-b3a2-1098-7654-321fedcba098
# → Run f7e6d5c4 removed from submission a1b2c3d4-…
```
