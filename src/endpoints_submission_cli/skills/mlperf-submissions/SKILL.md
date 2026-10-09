---
name: mlperf-submissions
description: Validate, upload and manage MLPerf Endpoints submissions with endpoints-submission-cli. Use when the user wants to check a submission folder against the rules, upload a benchmark run, or list, inspect, create, update or withdraw runs and submissions on PRISM.
---

# MLPerf Endpoints submissions

`endpoints-submission-cli` validates submission folders against the MLPerf Endpoints
rules (§9.1 automated checks) and manages runs and submissions on the PRISM API.

## Conventions

- Pass `--json` wherever a command accepts it, and read the result from stdout. Status
  messages and the upgrade notice go to stderr.
- Authentication comes from the `PRISM_USER_API_TOKEN` environment variable. Never put
  the token on the command line, and never print it. If a command fails with "No API
  token provided", ask the user to set the variable.
- Run IDs and submission IDs are UUIDs, passed as flags (`--run-id`, `--submission-id`),
  never positionally.
- When unsure of a flag, run the command with `--help` rather than guessing.

## Read-only: run these freely

| Task | Command |
|---|---|
| Validate a submission folder | `endpoints-submission-cli check-submission <path> --json` |
| Same, treating warnings as errors | `endpoints-submission-cli check-submission <path> --strict --json` |
| List runs | `endpoints-submission-cli runs list --json` |
| Inspect a run | `endpoints-submission-cli runs get --run-id <uuid> --json` |
| List submissions | `endpoints-submission-cli submissions list --json` |
| Inspect a submission | `endpoints-submission-cli submissions get --submission-id <uuid> --json` |

`check-submission` exits 1 when the submission fails, and that is a result, not an
error. Its JSON has a `results` list of `{rule, message, severity, path, spec_ref}`.
Report the `error` results grouped by `rule`, each with its `spec_ref` (the rules
section it enforces) and the file it points at. Mention `warning` results briefly, and
leave out `info` results unless asked.

## Writes: confirm with the user first

Each of these changes state on PRISM. Show the user the exact command, and run it only
after they agree.

| Task | Command | First |
|---|---|---|
| Upload a run | `runs create --path <dir>` | Run it with `--dry-run` and show the parsed payload |
| Create a submission | `submissions create --division … --scenario cop\|con --availability … --run-ids <uuid> …` | Run it with `--dry-run`: it assembles the folder and runs the checker without submitting |
| Update a submission | `submissions update --submission-id <uuid> …` | |
| Remove a run from a submission | `submissions remove-run --submission-id <uuid> --run-id <uuid>` | |
| Withdraw a submission | `submissions withdraw --submission-id <uuid>` | Irreversible: say so |
| Delete a run | `runs delete --run-id <uuid>` | Irreversible: deletes the stored archive |
| Pin / unpin a run | `runs pin --run-id <uuid>` / `runs unpin --run-id <uuid>` | |

Never add `--yes` to `submissions create --provisional` on the user's behalf: it skips
the confirmation that provisional results become publicly viewable.

Use `--test` on `runs create` and `submissions create` when the user is trying things
out, so the entry is not counted as a real result.
