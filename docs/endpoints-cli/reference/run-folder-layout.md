# Endpoints run folder layout

What `mlcommons/endpoints` writes to `report_dir` for a single benchmark run, and
how `endpoints-submission-cli runs create` consumes it.

Captured from an accuracy + performance run (`benchmark from-config --mode both`)
of `mlcommons/endpoints` @ `3614f36`.

## Layout

```
<report_dir>/
├── config.yaml                       # resolved BenchmarkConfig, as run          [written always]
├── report.txt                        # human-readable summary                    [written always]
├── events.jsonl                      # one JSON object per event (see size note) [written always]
├── sample_idx_map.json               # sample index -> dataset row mapping        [written always]
├── performance/
│   └── result_summary.json           # performance metrics                       [performance phase]
├── accuracy/
│   └── accuracy_results.json         # per-dataset accuracy scores               [accuracy phase]
└── metrics/
    ├── final_snapshot.json           # aggregator end-of-run snapshot            [written always]
    └── .ready                        # zero-byte completion sentinel             [written always]
```

Phase directories are created only for the phases that ran:

| Invocation | `performance/` | `accuracy/` |
|---|---|---|
| `--mode perf` (default) | yes | no |
| `--accuracy-only` | no | yes |
| `--mode both` | yes | yes |

## Sizes

From a 60 s, 16-concurrency run issuing 1,200 samples:

| File | Size | Scales with |
|---|---|---|
| `events.jsonl` | **46 MB** | samples x events per sample |
| `sample_idx_map.json` | 100 KB | samples issued |
| `metrics/final_snapshot.json` | 28 KB | fixed |
| `performance/result_summary.json` | 20 KB | fixed (histogram buckets) |
| `accuracy/accuracy_results.json` | 8 KB | datasets scored |
| `config.yaml`, `report.txt` | 4-8 KB | fixed |

`events.jsonl` dominates and grows with the run. A 600 s pareto point produces
several hundred MB, so `runs create` archives can be large.

## Key fields

`performance/result_summary.json`:

```
version, git_sha, test_started_at, n_samples_issued, n_samples_completed,
n_samples_failed, n_samples_succeeded, n_samples_dropped, duration_ns, state,
complete, ttft, tpot, latency, output_sequence_lengths, input_sequence_lengths,
legacy_loadgen_window_duration_ns, qps, tps, finish_reason_counts, run_config
```

`ttft` / `tpot` / `latency` / `*_sequence_lengths` are stat blocks of
`{total, min, max, median, avg, std_dev, percentiles, histogram}`. Percentile keys
are decimal strings — `"50.0"`, `"90.0"`, `"99.9"` — **not** `"50"` / `"90"`.

`accuracy/accuracy_results.json`:

```
{"osl_tokenization_s": float,
 "accuracy_scores": [ {dataset_name, extractor, ground_truth_column, score,
                       unit_samples, num_repeats, total_samples, duration_s,
                       complete, dataset_type, response_counts,
                       output_sequence_lengths, osl_tokenize_s}, ... ]}
```

`accuracy_scores` is a **list** of per-dataset entries; index it by `dataset_name`.

## What the CLI needs that endpoints does NOT write

`runs create` additionally requires **`system_desc.json`** in the run folder. This is
not an endpoints artifact — the submitter authors it (org, system, model, dataset
metadata, flat schema per rules §8.2) and drops it into `report_dir` before upload.

## Consumption by `runs create`

| Payload field | Source |
|---|---|
| `system_info` | `system_desc.json` (submitter-authored) |
| `config` | `config.yaml` |
| `result_summary` | `performance/result_summary.json` |
| `benchmark_version` | `result_summary.git_sha` |
| `started_at` / `finished_at` | `result_summary.test_started_at` + `duration_ns` |

`test_started_at` is **0** in current endpoints output, so the parser falls back to
`now() - duration` for the time window. Everything else in the folder — including
`accuracy/` and `events.jsonl` — reaches the server inside the uploaded archive
rather than the JSON payload.

This layout is the **only** one accepted. A flat folder with `result_summary.json` at
the top level is rejected:

```
Run folder error: ... is missing required file(s): performance/result_summary.json
```

Likewise, only `accuracy_results.json` has its `responses` list truncated before
archiving; a legacy `results.json` is archived verbatim.

## Known gaps

* **`run_metadata.json` is never written by endpoints.** `build_archive(..., run_date=…)`
  injects `run_date` into that file, so for an endpoints run folder the injection is a
  silent no-op. The field only takes effect if the submitter authors the file.
* **`test_started_at` is 0.** The real wall-clock start is not recorded in
  `result_summary.json`, so `started_at` / `finished_at` are reconstructed as
  `now() - duration_ns`. The window length is right; its absolute position is the
  upload time, not the run time.
* **`events.jsonl` dominates the archive.** 46 MB of a 48 MB run folder; it compresses
  to roughly 3 MB, but a 600 s point will be far larger.
