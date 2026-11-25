# Submission reviewer (§9.2 review hints)

`endpoints-submission-reviewer` runs heuristics over an MLPerf Endpoints submission and
reports what looks out of the ordinary for the **§9.2 Manual Review Focus Areas** of the
[Endpoints Rules](https://github.com/mlcommons/endpoints_policies/blob/main/endpoints_rules.md#92-manual-review-focus-areas).
Its Markdown output is meant to be posted on the review PR as one comment, updated in
place on every push, so reviewers know where to start.

It complements the §9.1 checker (`check-submission`) and never replaces it:

| | `check-submission` (§9.1) | `endpoints-submission-reviewer` (§9.2) |
|---|---|---|
| Decides | Compliance; failures reject the submission | Nothing; every finding is a hint for a person |
| Exit status | Non-zero on errors | 0 whatever it finds |
| Review-repo workflow | `Check submission`, which gates the lifecycle | Advisory only, under another name |

The reviewer reads a submission exactly as the checker does (same loaders, same tree
walk) but leniently. A file that does not load becomes a note, and only the checks that
need it are affected. It does not repeat what §9.1 rejects.

## What it checks

Every §9.2 bullet appears in the comment under one of four headings. **Findings** lists
what was found. **Checked, nothing unusual** means the checks ran clean, and it says what
they cover. **Not evaluated** means an input was missing, and it says which. **Manual
review only** means no check covers the area. So the comment doubles as the reviewer's
checklist.

| §9.2 focus area | Check | Needs | What it looks for |
|---|---|---|---|
| Pareto curve plausibility | `pareto_shape` | `result_summary.json` | Throughput dips that later recover; throughput growing faster than concurrency; per-user rate high for the median TPOT (a hint: TPOT leaves out the first chunk); TTFT falling as load rises; `tps_per_user` rising with load; dominated points; Little's law (`latency.total / duration` vs concurrency) |
| Metric distributions | `metric_distributions` | `result_summary.json` | TTFT flat across a wide concurrency span; identical timing percentiles (TTFT, TPOT, latency) on different points; percentiles out of order or outside min/max; very narrow distributions; round throughput values |
| TTFT-triggering fragments | `ttft_fragments` | `events.jsonl` | Share of responses whose first streamed fragment is only whitespace, punctuation or control characters; a long pause after the first chunk |
| Duplicated or padded output | `output_padding` | `result_summary.json` | A point's mean output length out of line with its curve; most responses stopping at the token limit |
| | `output_padding_text` | `events.jsonl` | Output repeating the reasoning trace; mostly-repeated passages; trailing filler |
| Offline reordering | `offline_reordering` | `events.jsonl` | Completions of one dataset pass finishing among the previous pass's; the same sample from different passes completing back to back |
| Warmup data | `warmup` | `point.yaml` | Warmup source naming the performance dataset without salting; retained logs with no usable link; identical warmup on every point |
| | `warmup_prompts` | `events.jsonl` | Warmup prompts identical to performance prompts, salt stripped; declared warmup missing from the log, or a different count |
| System description | `system_description` | `system_desc.json` | Placeholders and links nobody can open; prose (`node_config`, notes) naming other accelerators or counts; `serving_framework` without matching `src/`; parallelism beyond the declared accelerators; `system_size` vs node types; `system_power.json` describing other hardware |
| Cross-submission consistency | — | — | Manual: needs other submissions' results |
| Division eligibility | `division` | `system_desc.json` | Division declared inconsistently; RDI division with a non-RDI status; Preview date outside the 180-day window, or no statement of what is not yet available; Serviced endpoint that is missing, private, not https or a preview tier, and missing pricing, rate-limit or ToS disclosure; placeholder `availability_url` |
| Post-submission updates | `updates` | git history | Hardware, division or `C_max` changed; software stack changed; per-point metrics improved (Submission Rules §6.3 requires a justification) or regressed; run configuration changed; points added or removed; power descriptor changed; implementation code changed |
| Speculative decoding | `specdec_info` | `point.yaml` | What each point declares, and a drafter that differs between points. The approved-list match is §9.1's check. |

Levels: 🔶 **attention**, unlikely in a genuine run; 🔹 **note**, unusual but often
explainable; ℹ️ **info**, context.

## Running it

```bash
endpoints-submission-reviewer PATH [--format markdown|json] [-o FILE]
    [--base-ref REV | --pr-base REV] [--logs-dir DIR] [--thresholds FILE]
```

- `PATH` is a submission (`<submission_id>/`), its organization directory, or a checkout
  holding `<org>/<submission_id>/`. Every submission found is reviewed.
- `--base-ref REV` compares the submission with its version at `REV` (the original
  submission).
- `--pr-base REV` finds the original in a review PR instead; see below.
- `--logs-dir DIR` points at a directory laid out like `results/`
  (`<system>/<model>/r<N>/events.jsonl` or `events.jsonl.gz`, with `sample_idx_map.json`
  beside it), holding the stored run logs. When it is given, logs are read from it
  only. A log committed in the submission is then not used, and a note says so when it
  differs from the stored one or has no stored counterpart. Without `--logs-dir`, the
  logs in the point directories are used.
- `--thresholds FILE` overrides any subset of the packaged tolerances in
  [`reviewer_thresholds.yaml`](../../../src/submission_reviewer/data/reviewer_thresholds.yaml).
- The Markdown is also appended to `$GITHUB_STEP_SUMMARY` when that is set
  (`--no-step-summary` to stop it).

The command exits 0 whatever it finds. It exits non-zero only when it cannot review at
all: no submission under `PATH`, or an invalid thresholds file.

### How the original submission is found

The lifecycle manager opens a review PR by committing the submission archive to
`submission/<id>`, a branch made from the review repo's default branch. It commits the
archive as one or more commits titled `Add submission <id>`; a large archive is split
into batches. Each later sync adds an `Update submission <id>` commit containing only the
changed files, and people may add commits of their own through the web UI. When the
archive was not ready, the first commit is `Placeholder for <id>`, and the archive arrives
with the first sync.

`--pr-base` takes the PR's base commit. It walks the PR's commits along first parents,
so commits that a merge of the base branch ("Update branch") brings in are not counted.
It also follows a checkout of GitHub's test merge commit back to the PR head.

It skips placeholders. Then it takes the leading run of commits that share the first
commit's title and author, each authored no earlier than the one before and at most ten
minutes after it. The original is the last commit of that run.

- The time limit keeps the next day's sync, which has the same title, out of the
  original.
- Author dates are used because a rebase keeps them but gives every commit the same
  committer date.
- The `<id>` in the title must begin the name of the submission directory under review;
  otherwise the original is reported as not found.
- If the commit titles are not the manager's, the PR's first commit is used.

Commit titles, authors and dates can be written by anyone who can push to the branch.
Those people can also change the workflow itself, so this is no defence against them.
A stronger source would be the lifecycle manager recording the original commit itself.

The original is read with `git ls-tree` and `git cat-file`, never `git archive`, so the
submission's own `.gitattributes` cannot hide or rewrite files. Only the small metadata
and result files are extracted.

Syncs never delete files on the branch, so a point withdrawn in storage still appears in
the PR tree. `updates` reports only the removals that reach the branch.

## Running it on review PRs

The command is meant to run on every review PR, from a workflow kept in the review
repository (`endpoints_submissions`, and the test repository). That workflow should:

- Run on `pull_request` (`opened`, `synchronize`, `reopened`), only for the lifecycle
  manager's branches: `if: startsWith(github.head_ref, 'submission/')`.
- Use names other than `Check submission` and `Run submission checker`. The lifecycle
  manager matches those exact names to hand a submission over to review, and a second
  workflow carrying them would confuse that gate.
- Check out the PR head with its full history (`fetch-depth: 0`), because the original
  submission is read from it, and with `persist-credentials: false`.
- Install a pinned `endpoints-submission-cli` from PyPI into a virtual environment
  outside the checkout, and run Python there with `-I`. `python -m venv` run from the
  checkout would import a `venv.py` the submission ships, in a job holding a write
  token.
- Run `endpoints-submission-reviewer . --pr-base "$PR_BASE" -o review.md`, with
  `PR_BASE` set from `github.event.pull_request.base.sha` through `env`.
- Post `review.md` as a PR comment, or update the earlier one. The comment starts with
  the hidden marker `<!-- submission-reviewer -->`; look for it only among comments
  written by `github-actions[bot]`, so a comment quoting it is never overwritten. This
  needs `pull-requests: write`, and `contents: read` for the checkout.
- Treat a non-zero exit as nothing to review (for example, a placeholder PR with no
  results yet): log a warning and post nothing.

### Supplying the per-request logs

The review PR holds what `submissions create` bundles, which does not include
`events.jsonl`. The run archives in MLCommons storage do. Until the lifecycle manager
ships the logs with the PR, the log checks are listed as *not evaluated*. To enable them,
download each point's `events.jsonl` (and `sample_idx_map.json`) into
`<system>/<model>/r<N>/` under one folder and pass it with `--logs-dir`. Downloading
needs storage credentials, so do it in a separate job that hands the folder over as an
artifact, and keep the credentials out of the reviewer job.

Logs committed in the review PR are not trusted over the stored ones: with
`--logs-dir`, only its logs are read, and a committed log that differs, or that has no
stored counterpart, is listed in the notes.

A log check counts only the points it reads a log for. When some logs are missing, or
unreadable (including a file with no sample events), the area is marked as partly
checked and the comment names those points. When none can be used, or no point applies
(for example, no dedicated Offline point for `offline_reordering`), the check is listed
as *not evaluated*.

### Security

The reviewer runs on content a submitter controls, and it only ever parses it:

- No code from the PR runs: the checks only read files. Install the CLI from PyPI, never
  from the checkout.
- Git is used through read-only plumbing commands, with external diff drivers and
  textconv disabled.
- A file that fails to load in any way, including deeply nested JSON, becomes a note;
  the rest of the review goes on.
- Logs are streamed, and file sizes are bounded where it matters: extraction skips blobs
  over 20 MB, docs reading is capped, and response text is truncated.
- Submitter text cannot add markup to the comment, which github-actions[bot] posts.
  Values from the submission are shown as code spans, capped at 200 characters. All
  other text is escaped so that it cannot form a link, an image, raw HTML, an autolink,
  an @-mention or an issue reference.
- A comment over GitHub's size limit is shortened by whole lines: first fewer findings
  per focus area, then whole submissions from the end. Every submission shown keeps its
  checklist, and a warning says what was left out.

Workflows triggered by `pull_request` run the workflow files from the PR's own branch.
Anyone who can push to a `submission/<id>` branch could therefore change them. That is
true of every workflow in the review repo, including `Check submission`, so give the
reviewer's job nothing beyond the `pull-requests: write` token.

## Thresholds and calibration

All tolerances live in `reviewer_thresholds.yaml`, with one comment per key explaining
what it measures. The defaults were set against the anonymised corpus in
`test_submissions/`:

- **Real-derived submissions (`sub_a` to `sub_j`).**
  - Curve and distribution findings: none, apart from two artifacts of anonymisation.
  - The first artifact is TPOT: every point carries the same synthetic TPOT block, so
    "identical tpot percentiles" fires on every submission.
  - The second is `node_config`: most systems copy "8x H100 test node", which the
    system-description check flags against their real accelerators.
  - Two findings are real anomalies in the data. `sub_j`'s TTFT p50 drops 92% between
    r4096 and r8192, where the completed-sample count also jumps from 8,776 to 417,284.
    `sub_f`'s `system_size` says "1 node" for a two-node system.
- **Synthetic submissions (`valid_standardized`, `invalid_submission`).** Their
  hand-written values trip the superlinear-scaling, round-value, output-length and
  identical-TTFT heuristics, as intended.
- **Every corpus submission** also gets notes for its `example.com` placeholder links and
  for warmup on a "validation split" of the performance dataset.

The log heuristics were exercised on synthetic logs in the reference client's exact
format. `tests/submission_reviewer/data/events_reference.jsonl` was written by the
client's own encoder. Their thresholds are first estimates. Recalibrate them on the first
real logs, especially `ttft_trivial_*`, because some models genuinely open with a
newline, and `duplicate_*`, because reasoning models often draft their answer in the
trace.

## Known limits

- Accuracy and performance use different datasets, so their output lengths are not
  compared. Output length is compared across the points of one curve instead.
- Warmup prompts are compared whole: a truncated or paraphrased performance sample,
  which §6.3.1 also bars, is not detected.
- Dataset passes are inferred from issue order, because the client does not log pass
  boundaries yet (§5.7, open item 5).
- Cross-submission consistency for the same hardware is not automated.
