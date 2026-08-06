# `run_metadata.json` Authoring Guide

Each run corresponds to a point on the Pareto Curve; a given submission may have 7–30 runs.

**File location in run folder:** `<run_folder>/run_metadata.json`
**Location in assembled submission:** `pareto/<system_id>/<model>/results/point_<N>/run_metadata.json`

---

## Who fills in what

Fields fall into three categories:

- **Submitter-provided** — you must fill these in by hand.
- **Loadgen-derived** — produced by the Endpoints benchmark tool from `results_summary.json`; copy values directly from there.
- **CLI-injected** — `run_date` (written during `runs create`) and `tps_utilization` (written during `submissions create`). Leave both `null` in your source file; the CLI fills them in.

---

## Full JSON template

```json
{
  "run_date": null,
  "node_config": "32x Xeon 6503P + 48x GB200",
  "config_summary": {
    "disaggregated": null,
    "expert_parallel": null,
    "tensor_parallel": 4,
    "pipeline_parallel": 2,
    "data_parallel": null,
    "batch": null
  },
  "config_summary_notes": null,
  "concurrency": 128,
  "system_tps": 306.94,
  "tps_per_user": 2.40,
  "ttft": 312.5,
  "qps": 5.07,
  "tps_utilization": null,
  "measured_total_output_tokens": 60557,
  "measured_run_duration": 197.29,
  "measured_total_requests": 1000,
  "link_config": null,
  "link_logs": null,
  "measured_latency_ttft_min": 10.624,
  "measured_latency_ttft_average": 32.218,
  "measured_latency_ttft_p50": 28.4,
  "measured_latency_ttft_p90": 280.1,
  "measured_latency_ttft_p95": 300.7,
  "measured_latency_ttft_p99": 312.5,
  "measured_latency_ttft_p999": 316.572,
  "measured_latency_ttft_max": 316.675,
  "measured_latency_tpot_min": 3.076,
  "measured_latency_tpot_average": 4.049,
  "measured_latency_tpot_p50": 3.9,
  "measured_latency_tpot_p90": 5.4,
  "measured_latency_tpot_p95": 6.0,
  "measured_latency_tpot_p99": 6.5,
  "measured_latency_tpot_p999": 6.863,
  "measured_latency_tpot_max": 6.919,
  "measured_latency_request_min": 47.108,
  "measured_latency_request_average": 271.979,
  "measured_latency_request_p50": 250.3,
  "measured_latency_request_p90": 600.1,
  "measured_latency_request_p95": 620.4,
  "measured_latency_request_p99": 640.0,
  "measured_latency_request_p999": 643.234,
  "measured_latency_request_max": 650.2
}
```

Every measurement field (concurrency, the throughput/latency metrics, and all
`measured_latency_*` percentiles) must be present **and non-null** — the checker
rejects a null measurement. Only `config_summary_notes`, `link_config`, and
`link_logs` may be null, and even those keys must be present. `config_summary`
must be either a string of length ≥ 4 or the structured object shown above.

---

## Submitter-provided fields

These fields cannot be derived automatically and must be filled in by hand.

### `node_config`
Human-readable description of the nodes in the SUT for this run. Must contain enough detail to reproduce the submission.

Construct it from your system node ensemble data using this formula per node type:
```
[number_of_nodes × accelerators_per_node]x [accelerator_model_name]
```
If a node type has no accelerator, use processors instead:
```
[number_of_nodes × host_processors_per_node]x [processor_model_name]
```
Join multiple node types with `+`.

**Example:**
- Node type 1: 8 nodes, 4× Xeon 6503P each, no accelerator → `32x Xeon 6503P`
- Node type 2: 12 nodes, 4× GB200 each → `48x GB200`
- Result: `32x Xeon 6503P + 48x GB200`

### `config_summary` (object)

Parallelism configuration for this run. The display value shown in the UI is assembled from these sub-fields (omitting any that are 1 or null).

| Field | Type | Description |
|---|---|---|
| `disaggregated` | int \| null | Number of disaggregated stages. Set `null` if the system is not disaggregated. |
| `expert_parallel` | int \| null | EP degree. Only applies to Mixture-of-Experts models. EP=1 means no partitioning; use `null` or `1` for non-MoE models. |
| `tensor_parallel` | int | TP degree. The weight matrices of each layer are split across N processors. The number of attention heads must be divisible by N. TP=1 means no partitioning. |
| `pipeline_parallel` | int | PP degree. The layers of the model are split sequentially into N stages; each processor holds one stage. PP=1 means no partitioning. |
| `data_parallel` | int \| null | DP degree. The full model is replicated N times and requests are distributed across replicas. Set `null` or `1` if not replicated. |
| `batch` | int \| null | Maximum batch size. Set `null` if not applicable or unknown. |

### `config_summary_notes`
Free-form string for any configuration detail not captured by the fields above (e.g., quantization scheme, speculative decoding settings, KV cache configuration). Set `null` if nothing extra to add.

### `link_config`
URL to full configuration files for this run (e.g., a GitHub permalink to your server config). Set `null` if not available.

### `link_logs`
URL to full run logs. Set `null` if not available.

---

## Loadgen-derived fields

These values are produced by the Endpoints benchmark tool and should be read directly from your run output. The table below shows where each value comes from in `results_summary.json`.

### Top-level metrics

| Field | Type | Source in `results_summary.json` | Formula |
|---|---|---|---|
| `concurrency` | int | `config.yaml` → `settings.load_pattern.target_concurrency` | Direct copy. |
| `measured_total_output_tokens` | float | `output_sequence_lengths.total` | Direct copy. |
| `measured_run_duration` | float (seconds) | `duration_ns` | `duration_ns / 1_000_000_000` |
| `measured_total_requests` | int | `n_samples_completed` | Direct copy. |
| `system_tps` | float | Derived | `measured_total_output_tokens / measured_run_duration` |
| `tps_per_user` | float | Derived | `system_tps / concurrency` |
| `qps` | float | Derived | `measured_total_requests / measured_run_duration` |
| `ttft` | float (ms) | Derived | Same value as `measured_latency_ttft_p99` |

### TTFT latencies (milliseconds)

Source: `results_summary.json` → `ttft.percentiles`. Raw values are in nanoseconds; divide by 1,000,000 to get milliseconds.

| Field | `ttft.percentiles` key |
|---|---|
| `measured_latency_ttft_min` | `min` |
| `measured_latency_ttft_average` | `average` (or compute from `total / n_samples_completed`) |
| `measured_latency_ttft_p50` | `50` |
| `measured_latency_ttft_p90` | `90` |
| `measured_latency_ttft_p95` | `95` |
| `measured_latency_ttft_p99` | `99` ← also used as `ttft` above |
| `measured_latency_ttft_p999` | `99.9` |
| `measured_latency_ttft_max` | `max` |

### TPOT latencies (milliseconds)

Source: `results_summary.json` → `tpot.percentiles`. Same unit conversion (ns → ms).

| Field | `tpot.percentiles` key |
|---|---|
| `measured_latency_tpot_min` | `min` |
| `measured_latency_tpot_average` | `average` |
| `measured_latency_tpot_p50` | `50` |
| `measured_latency_tpot_p90` | `90` |
| `measured_latency_tpot_p95` | `95` |
| `measured_latency_tpot_p99` | `99` |
| `measured_latency_tpot_p999` | `99.9` |
| `measured_latency_tpot_max` | `max` |

### Request latencies (milliseconds)

End-to-end per-request latency. De-emphasized for UI/viz (WG decision, 1/23 meeting) but retained in the report. Source key varies by loadgen version — check your `results_summary.json` for a `request_latency` or `e2e_latency` field with the same `percentiles` structure.

| Field | percentile key |
|---|---|
| `measured_latency_request_min` | `min` |
| `measured_latency_request_average` | `average` |
| `measured_latency_request_p50` | `50` |
| `measured_latency_request_p90` | `90` |
| `measured_latency_request_p95` | `95` |
| `measured_latency_request_p99` | `99` |
| `measured_latency_request_p999` | `99.9` |
| `measured_latency_request_max` | `max` |

---

## CLI-injected fields

### `tps_utilization`
**Leave this `null`.** The submission CLI fills it in automatically during `submissions create`.

It is computed as `system_tps / max(system_tps across all runs in this submission)`, normalizing each run to the peak throughput of the curve. For example, if the highest `system_tps` across all your runs is 100 and this run achieved 37, the CLI will write `0.37`.

### `run_date`
**Leave this `null`.** The CLI fills it in automatically during `runs create`.

It is the date the run was executed, in `YYYY-MM-DD` format, derived from the run's `started_at` timestamp.
