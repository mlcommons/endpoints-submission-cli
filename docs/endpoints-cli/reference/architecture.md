# Architecture

## Module map

```
src/endpoints_submission_cli/
├── main.py               Entry point — Click group, registers `runs` and `submissions`
├── api_client.py         HTTP client wrapping every PRISM API endpoint (httpx)
├── run_parser.py         Parses local run folder → API payload; builds .tar.gz archive
├── submission_builder.py Assembles submission folder from run archives; creates bundle
├── github_ops.py         Shells out to `gh` CLI for PR create/update/close
├── formatters.py         Rich table output and JSON mode for runs and submissions
├── exceptions.py         Custom exception hierarchy
└── commands/
    ├── runs.py           6 run commands (list, create, get, delete, pin, unpin)
    └── submissions.py    7 submission commands + 3 rollback helpers
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Registers command groups. Entry point for the `endpoints-submission-cli` script. |
| `api_client.py` | All HTTP calls to the PRISM API. Three timeout profiles. Token resolution. Error translation to `APIError`/`AuthError`. |
| `run_parser.py` | Reads `system_info.json`, `config.yaml`, `result_summary.json` from a local folder. Derives `benchmark_version` from `git_sha` or `"unknown"`. Builds `.tar.gz` archive. |
| `submission_builder.py` | Extracts run archives, groups by `(system_id, model)`, writes the submission folder tree (`systems/`, `pareto/`). Calls `submission_checker.models` for `compute_regions`. |
| `github_ops.py` | Clones the submission repo, creates/updates branches, commits, pushes, creates/closes PRs. All via `gh` CLI subprocess calls. |
| `formatters.py` | Rich table renderers for run and submission lists/details. JSON output with syntax highlighting. |
| `exceptions.py` | `APIError`, `AuthError`, `ArchiveError`, `GitHubError`, `RunFolderError`, `SubmissionBuildError`, `SubmissionCheckError`. |
| `commands/runs.py` | Click command implementations for all run operations. Delegates to `api_client` and `run_parser`. |
| `commands/submissions.py` | Click command implementations for all submission operations. Orchestrates the full pipeline and rollback helpers. |

---

## Module dependency diagram

```mermaid
graph TD
    main["main.py\n(entry point)"] --> runs_cmd["commands/runs.py"]
    main --> sub_cmd["commands/submissions.py"]

    runs_cmd --> api["api_client.py"]
    runs_cmd --> parser["run_parser.py"]
    runs_cmd --> fmt["formatters.py"]

    sub_cmd --> api
    sub_cmd --> github["github_ops.py"]
    sub_cmd --> builder["submission_builder.py"]
    sub_cmd --> fmt

    builder --> checker["submission_checker\n(external package)"]

    api --> exc["exceptions.py"]
    github --> exc
    parser --> exc
    builder --> exc
```

---

## Command flow diagrams

### `runs create`

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as runs.py
    participant P as run_parser
    participant A as api_client

    U->>CLI: runs create --path ./results/my_run
    CLI->>P: parse_run_folder(path)
    P-->>CLI: payload (system_info, config, result_summary, benchmark_version)
    CLI->>A: POST /runs  →  run_id
    CLI->>P: build_archive(path)  →  archive.tar.gz
    CLI->>A: POST /runs/{run_id}/archive
    alt upload fails
        CLI->>A: DELETE /runs/{run_id}/archive  (best-effort)
        CLI->>A: DELETE /runs/{run_id}  (rollback)
        CLI-->>U: error + exit 1
    end
    CLI-->>U: Run created: {run_id}
```

---

### `submissions create`

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as submissions.py
    participant A as api_client
    participant B as submission_builder
    participant Ch as SubmissionChecker
    participant GH as github_ops

    U->>CLI: submissions create --division ... --run-ids ...
    CLI->>GH: check_prerequisites(target_repo)
    loop for each run_id
        CLI->>A: GET /runs/{run_id}/archive  →  archive file
    end
    CLI->>B: build_submission_folder(archives, division)
    B-->>CLI: submission_dir
    CLI->>Ch: SubmissionChecker(submission_dir).run()
    Ch-->>CLI: report (errors / warnings)
    alt compliance errors found
        CLI-->>U: checker failed + exit 1
    end
    CLI->>A: POST /submissions  →  submission_id
    CLI->>B: create_bundle_archive(submission_dir)
    CLI->>A: POST /submissions/{id}/archive
    alt upload fails
        CLI->>A: DELETE /submissions/{id}  (rollback)
        CLI-->>U: error + exit 1
    end
    CLI->>GH: prepare_submission_branch(submission_dir, branch)
    CLI->>GH: create_pr(submission_id, branch)  →  pr_url, pr_number
    alt PR creation fails
        CLI->>A: DELETE /submissions/{id}  (rollback)
        CLI-->>U: error + exit 1
    end
    CLI->>A: PATCH /submissions/{id}  {pr_url, pr_number, status=REVIEW_PENDING}
    CLI-->>U: Submission created: {id}\nPR: {pr_url}
```

---

### `submissions update` (with `--run-ids`)

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as submissions.py
    participant A as api_client
    participant B as submission_builder
    participant Ch as SubmissionChecker
    participant GH as github_ops

    U->>CLI: submissions update --submission-id ... --run-ids ...
    CLI->>A: GET /submissions/{id}  →  current state
    CLI->>A: PATCH /submissions/{id}  {run_ids: desired}
    loop for each desired run_id
        CLI->>A: GET /runs/{run_id}/archive
    end
    CLI->>B: build_submission_folder(archives, division)
    CLI->>Ch: SubmissionChecker(submission_dir).run()
    alt checker / build error
        CLI->>A: PATCH /submissions/{id}  {run_ids: original}  (rollback)
        CLI-->>U: error + exit 1
    end
    CLI->>B: create_bundle_archive(submission_dir)
    CLI->>A: POST /submissions/{id}/archive
    alt upload fails
        CLI->>A: PATCH /submissions/{id}  {run_ids: original}  (rollback)
        CLI-->>U: error + exit 1
    end
    CLI->>GH: update_pr_branch(pr_number, submission_dir, ...)
    CLI-->>U: Submission {id} updated.
```

---

### PR branch update strategy (`update_pr_branch`)

Used by `submissions update`, `submissions add-run`, and `submissions remove-run`:

```
For each <system_id>/<model> in the fresh build:
│
├── points/      → replace entirely from fresh build
├── accuracy/    → replace entirely from fresh build
└── results/
    ├── Remove point dirs no longer in fresh build
    └── For each point dir in fresh build:
        ├── Copy log files (mlperf_endpoints_log_*.json)
        └── system_desc.json:
            ├── Preserve from PR branch (manual edits survive)
            └── Seed from fresh build only for new points

systems/   → preserve from PR branch (seed if absent)
src/       → preserve from PR branch
```

---

## Exception hierarchy

```
Exception
├── APIError          — any PRISM API call failure (4xx/5xx or network)
│   └── AuthError     — 401/403 or missing token
├── ArchiveError      — archive file open/upload/download/extract failure
├── GitHubError       — gh CLI subprocess failure
├── RunFolderError    — missing or malformed files in run folder
├── SubmissionBuildError — failure assembling submission folder tree
└── SubmissionCheckError — Submission Checker found compliance errors
```

`AuthError` extends `APIError` so callers that only catch `APIError` also handle auth failures.

---

## Deviations from the OpenAPI spec

| Area | Implementation | Spec |
|---|---|---|
| `benchmark_version` | Derived from `result_summary.git_sha`; falls back to `"unknown"`. Never reads the CLI package's own git hash. | Field in `RunCreate`. |
| `download_submission_archive` | Function exists in `api_client.py` but is not exported in `__all__` and not called by any command. | `GET /submissions/{id}/archive` defined in spec. |
| Submission status on create | Set to `REVIEW_PENDING` after PR creation (step 9 of `submissions create`), not at registration time. | `status` is a field on `SubmissionCreate`. |
| Rollback on PR failure | Uses `DELETE /submissions/{id}` (withdraw) as the rollback for a failed PR creation. This sets status to `WITHDRAWN`. | Spec defines `DELETE /submissions/{id}` as the withdraw endpoint. |
