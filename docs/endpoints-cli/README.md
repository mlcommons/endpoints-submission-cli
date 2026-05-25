# endpoints-submission-cli

`endpoints-submission-cli` is the command-line tool for managing MLPerf Endpoints benchmark runs and rolling submissions against the PRISM Submission API. It handles the full lifecycle: registering benchmark runs, assembling submission packages, running compliance checks, uploading bundles, and creating GitHub pull requests — all in a single command.

---

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Authenticate
export PRISM_USER_API_TOKEN=mlc_your_token_here

# Register a benchmark run
endpoints-submission-cli runs create --path ./results/my_run

# Create a submission from that run
endpoints-submission-cli submissions create \
  --division standardized \
  --availability available \
  --run-ids <run-id>
```

---

## Documentation

| Document | Description |
|---|---|
| [Getting started](getting-started.md) | Install, configure, and authenticate |
| [Usage: runs](usage/runs.md) | All run commands with flags and examples |
| [Usage: submissions](usage/submissions.md) | All submission commands with flags and examples |
| [API mapping](reference/api-mapping.md) | CLI command → HTTP endpoint reference |
| [Architecture](reference/architecture.md) | Module map and command flow diagrams |

The legacy combined reference is also available at [../endpoints-submission-cli.md](../endpoints-submission-cli.md).

---

## Command overview

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
