# Submission commands

A submission groups one or more benchmark runs into a package that is compliance-checked and
uploaded to the MLCommons review pipeline. The CLI no longer opens the review pull request
itself — `pr_url` and `pr_number` appear in `submissions get` once whatever does has set them.

---

## Submission status lifecycle

| Status | Set by | Meaning |
|---|---|---|
| `COMPLIANCE_CHECKING` | `submissions create` | Submission created and bundle uploaded; waiting for the lifecycle manager to hand it to review. |
| `WITHDRAWN` | `submissions withdraw` | Submission retracted; PR closed, archive deleted. |

Additional statuses set by the review workflow (server-side, not by the CLI):

| Status | Meaning |
|---|---|
| `REVIEW_PENDING` | Handed to review by the lifecycle manager, which stamps `compliance_passed_at` at that moment. Review deadlines run from there. |
| `FINALIZED` | Review complete; submission accepted. |
| `PUBLISHED` | Results published in the MLPerf leaderboard. |

---

## submissions list

List all submissions for the authenticated account.

```bash
endpoints-submission-cli submissions list [--token TOKEN] [--status STATUS] [-j]
```

| Flag | Description |
|---|---|
| `--token TOKEN` | API key. Falls back to `PRISM_USER_API_TOKEN`. |
| `--status STATUS` | Filter by status, ignoring case. Repeat to include more statuses. `SUBMITTED` includes `COMPLIANCE_CHECKING` and `REVIEW_PENDING`. |
| `-j` / `--json` | Print raw JSON instead of the Rich table. |

**Example:**

```bash
endpoints-submission-cli submissions list

endpoints-submission-cli submissions list -j

endpoints-submission-cli submissions list --status SUBMITTED
```

---

## submissions create

Create a new submission from one or more registered runs. This command runs the full automated pipeline:

1. Download all run archives from the API (with progress bar).
2. Assemble the submission folder structure.
3. Run the Submission Checker — aborts with exit code 1 on compliance errors.
4. `POST /submissions` to register the submission.
5. Upload the submission bundle (`POST /submissions/{id}/archive`).

The submission is left in `COMPLIANCE_CHECKING`. The lifecycle manager moves it to
`REVIEW_PENDING` on its next run once it sees the uploaded bundle, and that is the time
review deadlines are measured from.

The CLI does not open the review pull request. `pr_url` and `pr_number` remain on the
submission record and are shown by `submissions get` once whatever opens it has set them.

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
| `--target-availability-date DATE` | no | Target availability date (`YYYY-MM-DD`). Required when `--availability preview`. |
| `--embargo-date DATETIME` | no | Embargo datetime, ISO 8601 with a time zone, e.g. `2026-12-01T16:00:00Z` (UTC) or `2026-12-01T08:00:00-08:00` (Pacific). A time without one is rejected. Must be in the future. Nothing is published before it. It can be moved later with `submissions update` but not added, so declare it here. |
| `--dry-run` | no | Assemble folder and run checker, then print the folder layout and exit without creating the submission or PR. |

To override the lower concurrency boundary, include `"c_min_override": 32` (for example)
in each performance point's `system_desc.json` before registering the runs. Use the same
integer from 1 through 32 for every point on a system/model frontier, and include a point
at that concurrency. The builder preserves the declaration; separate accuracy donors do
not replace a performance point's metadata. Omit the field everywhere to retain default
C_min derivation and Ultra Low Concurrency coverage of 1–32. With an override, that region
ends at C_min and any point within it may supply its required accuracy result.

The publication mode (Submission Rules §6.2) follows from `--provisional` and `--embargo-date`, and is fixed once the submission is created:

| `--provisional` | `--embargo-date` | Mode |
|---|---|---|
| no | no | Confidential review; published at the first publication date after finalization. |
| no | yes | Confidential review; the finalized result is held until the embargo date. |
| yes | either | Published during review with a `peer review pending` disclaimer, from the embargo date if one is given. |

There is no publication-cycle flag. Nobody picks a cycle at submit time: the lifecycle manager publishes a finalized submission on the first publication date after it is finalized, later if it is under embargo or carries non-response penalties, and records the cycle it published in.

**`cli_metadata.json`** is written inside the `<submission_id>/` directory of every bundle
(not at the organisation level, which is shared across submissions). It records which CLI
shaped the submission, generated from the submission record the command already received
from the API plus the version of the CLI running now:

```json
{
  "command": "remove-run",
  "cli_version": "1.0.0.0",
  "most_recent_cli_used": "1.2.0.0",
  "created_at": "2026-08-20T17:08:02.936070Z"
}
```

| Key | Meaning |
|---|---|
| `command` | the command that assembled this bundle |
| `cli_version` | the CLI that **created** the submission (from the API record) |
| `most_recent_cli_used` | the CLI that ran **this** command |
| `created_at` | when the submission was created (from the API record) |

On `create` the two versions are identical, because that command *is* the creating one.
`remove-run` and `update` report the creating version from the
API alongside their own, so a bundle built by 1.0.0.0 and later amended by 1.2.0.0 shows
both. `created_at` is the submission's creation time, not the rebuild time.

No extra API calls and no new API fields are involved — every value is already in hand by
the time the marker is written.

**Rollback behaviour:** if the bundle upload fails, the submission is automatically withdrawn (`DELETE /submissions/{id}`) to leave a clean state. If that rollback also fails, the orphaned submission ID is printed.

**If only the PATCH step (step 6) fails** — the submission and its bundle both exist; the failure is a warning, not a fatal error, and the status can be set manually.

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
Downloading run archives…
Assembling submission folder…
Running Submission Checker…
Checker report written to submission_checker_20250410_090123.log
Uploading submission bundle…
Submission created: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Submission folder structure** assembled by the CLI:

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
                    ├── config.yaml            # benchmark configuration as run
                    └── server_configs/        # OPTIONAL, point-specific backend configs
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

The default table renders every field the API returns for a submission — classification (division, scenario, availability), the `Test Submission` flag, `Published In Cycle` (blank until the lifecycle manager publishes the submission) and embargo date, `Reviewers Assigned` (a count; the reviewer identities are never exposed by the API), `Penalties Imposed` and `Business Days Waiting` (the non-response penalty level from rules §6.3 — 1 and 2 each delay publication by a cycle, 3 withdraws — and the longest any one finding has waited on the submitter; shown as `—` when the API does not report them), the checker/API/CLI versions, PR references, and the full set of lifecycle timestamps in chronological order. Embedded runs follow in their own table.

**Example:**

```bash
endpoints-submission-cli submissions get \
  --submission-id a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

In `submissions list`, a submission flagged `is_test` is marked with a yellow `*` before the status, with a `* = test submission` legend under the table. That view stays deliberately narrow — use `submissions get` for the full field set.

---

## submissions update

Update the run list or metadata fields on an existing submission.

```bash
endpoints-submission-cli submissions update \
  --submission-id SUB_ID \
  [--token TOKEN] \
  [--run-ids RUN_ID ...] \
  [--target-availability-date DATE] \
  [--embargo-date DATETIME]
```

| Flag | Required | Description |
|---|---|---|
| `--submission-id SUB_ID` | yes | Submission UUID. |
| `--token TOKEN` | no | API key. |
| `--run-ids RUN_ID` | no (repeatable) | Replace the complete run list. Pass once per run. Runs not listed are removed. |
| `--target-availability-date DATE` | no | Target availability date (`YYYY-MM-DD`). |
| `--embargo-date DATETIME` | no | Embargo datetime, ISO 8601 with a time zone, e.g. `2026-12-01T16:00:00Z` (UTC) or `2026-12-01T08:00:00-08:00` (Pacific). A time without one is rejected. Moves an embargo declared at submission; one cannot be added to a submission created without one. |

Providing no flags prints a warning and makes no API call.

### When `--run-ids` is provided (full rebuild)

The command runs the full rebuild pipeline:

1. `GET /submissions/{id}` — fetch the current run list and division.
2. **Reject the update if it would add a run** (see below); log removed runs.
3. `PATCH /submissions/{id}` with the new `run_ids` (and any metadata fields) in a single call.
4. Download the remaining run archives (with progress bar).
5. Assemble the submission folder and run the Submission Checker — rollback on errors.
6. Upload the bundle to blob storage (`POST /submissions/{id}/archive`) — rollback on errors.

**Rollback:** if any step 4–6 fails after the DB PATCH, the run list is automatically restored to its original value (`PATCH /submissions/{id}` with `original_run_ids`).

> [!IMPORTANT]
> **`--run-ids` may only shrink the list.** Submission Rules §8 no longer provide a
> post-submission window for adding measurement points, so a list containing a run the
> submission does not already have is rejected before anything is fetched or patched.
> Removals are still permitted — see [`submissions remove-run`](#submissions-remove-run).

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

# Update the embargo date
endpoints-submission-cli submissions update \
  --submission-id a1b2c3d4-… \
  --embargo-date 2026-12-01T16:00:00Z

# Combine run list update with metadata update
endpoints-submission-cli submissions update \
  --submission-id a1b2c3d4-… \
  --run-ids d5d9873e-… \
  --run-ids f7e6d5c4-… \
  --target-availability-date 2025-10-01
```

---

## submissions withdraw

Withdraw a submission: marks it `WITHDRAWN` and deletes the stored bundle.

```bash
endpoints-submission-cli submissions withdraw \
  --submission-id SUB_ID \
  [--token TOKEN]
```

**Order of operations:** DB update (`DELETE /submissions/{id}`) → delete archive (`DELETE /submissions/{id}/archive`).

Archive deletion is best-effort — a failure is reported as a warning but does not change the exit code. The submission is already `WITHDRAWN` in the database. The CLI does not close the review pull request; it no longer manages one.

**Example:**

```bash
endpoints-submission-cli submissions withdraw \
  --submission-id a1b2c3d4-e5f6-7890-abcd-ef1234567890
# → Submission withdrawn: a1b2c3d4-…
```

## submissions remove-run

Withdraw a run from an existing submission. If runs still remain, the bundle is rebuilt,
compliance-checked, and re-uploaded.

> [!IMPORTANT]
> **A submission's points are fixed at creation.** MLPerf Endpoints Submission Rules §8
> no longer provide a post-submission window for adding measurement points — the §8.1
> *Pareto Updates* section that allowed it was removed. There is no `add-run` command,
> and `submissions update --run-ids` rejects a list that would add one.
>
> §8.1 *Corrections* still lets a submitter withdraw a faulty point during peer review,
> which is what this command does. Note the consequence: withdrawn points do not count
> toward the 7-point minimum, and that shortfall **cannot be repaired by adding another
> point**. The Submission Checker will report it. If a submission needs a different set
> of runs, create a new one.

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

1. `DELETE /submissions/{id}/runs/{run_id}` — register the removal.
2. Download the remaining run archives. Skipped if no runs remain.
3. Rebuild the submission folder and run the Submission Checker — rollback on errors. Skipped if no runs remain.
4. Upload the bundle to blob storage (`POST /submissions/{id}/archive`) — rollback on errors. Skipped if no runs remain.

If no runs remain after removal, steps 2–4 are skipped and a warning is printed.

**Rollback:** if any step 2–4 fails after removal, the run is automatically re-added to the submission record.

**Example:**

```bash
endpoints-submission-cli submissions remove-run \
  --submission-id a1b2c3d4-… \
  --run-id f7e6d5c4-b3a2-1098-7654-321fedcba098
# → Run f7e6d5c4 removed from submission a1b2c3d4-…
```
