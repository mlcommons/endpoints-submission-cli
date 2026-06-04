# Getting started

## Requirements

- Python 3.10 or later
- [`gh` CLI](https://cli.github.com/) — required for creating, updating and withdrawing submissions.
---

## Installation

**With pip (editable install from source):**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

**With [uv](https://github.com/astral-sh/uv):**

```bash
uv sync --extra dev
```

Verify:

```bash
endpoints-submission-cli --version
endpoints-submission-cli --help
```

---

## Authentication

### PRISM API token

Every command requires an API key obtained from `PRISM` in `mlc_…` format. Supply it once as an env var or pass `--token` per command:

```bash
# Persistent (recommended — add to shell profile)
export PRISM_USER_API_TOKEN=your_token_here

# Per-command override
endpoints-submission-cli runs list --token mlc_your_token_here
```

The env var and the `--token` flag are supported on every command. The flag takes precedence when both are set.

### GitHub CLI (`gh`)

Submission commands that create or update GitHub pull requests use the `gh` CLI. Install and authenticate before using them:

```bash
# Install: https://cli.github.com/
brew install gh                 # macOS
sudo apt install gh             # Ubuntu/Debian

# Authenticate
gh auth login
```

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `PRISM_USER_API_TOKEN` | — | API key. Required unless `--token` is passed. |
| `MLPERF_API_BASE_URL` | `https://api.mlcommons.org` | Base URL of the PRISM Submission API. Override only for dev/staging environments. |
| `MLPERF_SUBMISSION_REPO` | `MLCommons-Systems/test-endpoints-submission-repo` | Target GitHub repository for submission PRs (`owner/repo` format). |

Add to your shell profile for a persistent setup:

```bash
export PRISM_USER_API_TOKEN=mlc_your_token_here
export MLPERF_SUBMISSION_REPO=MLCommons-Systems/endpoints-submission-repo
```

---

## First command

List your runs to verify connectivity and authentication:

```bash
endpoints-submission-cli runs list
```

Expected output (Rich table):

```
 ID                                    Model                             Concurrency  Started At
 d5d9873e-5eca-4f8d-a487-4be1cb8b440c  meta-llama/Llama-3.1-8B-Instruct  4            2025-04-10T09:00:00
```

Use `-j` for machine-readable JSON:

```bash
endpoints-submission-cli runs list -j | jq '.[].id'
```

---

## End-to-end example

```bash
# 1. Register a benchmark run from a local result folder
endpoints-submission-cli runs create --path /results/llama3_h100_c4
# → Run created: d5d9873e-5eca-4f8d-a487-4be1cb8b440c
RUN_ID=d5d9873e-5eca-4f8d-a487-4be1cb8b440c

# 2. Create a submission
endpoints-submission-cli submissions create \
  --division standardized \
  --availability available \
  --run-ids $RUN_ID
# → Submission created: a1b2c3d4-…
# → PR: https://github.com/MLCommons-Systems/…/pull/42
SUB_ID=a1b2c3d4-e5f6-7890-abcd-ef1234567890

# 3. Add another run later
endpoints-submission-cli runs create --path /results/llama3_h100_c8
# → Run created: f7e6d5c4-…
endpoints-submission-cli submissions add-run \
  --submission-id $SUB_ID \
  --run-id f7e6d5c4-…

# 4. Withdraw if needed
endpoints-submission-cli submissions withdraw --submission-id $SUB_ID
```

See [usage/runs.md](usage/runs.md) and [usage/submissions.md](usage/submissions.md) for complete command references.
