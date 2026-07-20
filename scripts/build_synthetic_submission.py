#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Build a synthetic, checker-passing MLPerf endpoints submission for the Pareto demo.

Assembles ``test_submissions/synthetic_pareto/`` with three hardware systems all
running ``llama3.1-8b``. Each system contributes a full 8-point Pareto curve
covering the four concurrency regions, so the Submission Checker passes and the
terminal Pareto view has three curves to compare.

Metric values are anchored on v6.1-inference-samples but synthesized along a
saturating throughput model so the curves are internally consistent and show the
classic throughput-vs-interactivity trade-off:

    system_tps(c)   = T_max * c / (c + k)          # rises, then saturates
    tps_per_user(c) = system_tps(c) / c            # falls as load grows
    ttft(c)         = ttft0 + ttft_slope * c        # grows with load

Run:  python scripts/build_synthetic_submission.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "test_submissions" / "synthetic_pareto"

MODEL_DIR_NAME = "llama3_1-8b"  # normalizes to match model_id "llama3.1-8b"
DATASET = "llm-perf-dataset-v1"
OUTPUT_TOKEN_AVG = 128.0

# Concurrency ladder shared by every system — covers all four regions for M ≥ 1000:
#   low_latency 1-32 (16) · low_throughput 33-42 (40) · med_throughput 43-~140 (64,128)
#   high_throughput (256,512,768,1000)
CONCURRENCIES = [16, 40, 64, 128, 256, 512, 768, 1000]


# Each system: a distinct hardware profile producing a distinct trade-off curve.
SYSTEMS = [
    {
        "id": "acme_h100x8",
        "org": "ACME Corp",
        "contact": "mlperf@acme.example.com",
        "system_name": "ACME H100x8 Node",
        "max_concurrency": 1000,
        "accelerator": "NVIDIA H100 SXM5 80GB",
        "accelerators_per_node": 8,
        "framework": "vLLM 0.6.3",
        "t_max": 22000.0,
        "k": 250.0,
        "ttft0": 180.0,
        "ttft_slope": 0.62,
    },
    {
        "id": "hpc_system3_8x",
        "org": "Submitter3",
        "contact": "submitter3@hpc-system3.example.com",
        "system_name": "HPC-System3",
        "max_concurrency": 1024,
        "accelerator": "HPC-Accelerator-Gen3-80GB",
        "accelerators_per_node": 8,
        "framework": "HPC-Inference-Engine v2.8",
        "t_max": 26000.0,
        "k": 300.0,
        "ttft0": 150.0,
        "ttft_slope": 0.55,
    },
    {
        "id": "bluewave_mi300x",
        "org": "BlueWave Systems",
        "contact": "bench@bluewave.example.com",
        "system_name": "BlueWave MI300X-8",
        "max_concurrency": 1152,
        "accelerator": "AMD Instinct MI300X 192GB",
        "accelerators_per_node": 8,
        "framework": "vLLM-ROCm 0.6.2",
        "t_max": 18000.0,
        "k": 170.0,
        "ttft0": 210.0,
        "ttft_slope": 0.70,
    },
]


def system_desc(system: dict) -> dict:
    """Build a §8.2 SystemDescription for *system* (used for systems/ and per point)."""
    return {
        "submitter_org_names": system["org"],
        "submitter_contact": system["contact"],
        "system_name": system["system_name"],
        "system_category": "datacenter",
        "system_availability_status": "Available",
        "max_supported_concurrency": system["max_concurrency"],
        "system_size": f"1 node · {system['accelerators_per_node']}x {system['accelerator']}",
        "system_node_ensemble_count": 1,
        "system_node_ensemble_total": 1,
        "serving_framework": system["framework"],
        "node_types": [
            {
                "system_node_ensemble_id": 0,
                "number_of_nodes": 1,
                "host_processor_model_name": "AMD EPYC 9654",
                "host_processors_per_node": 2,
                "host_processor_core_count": 96,
                "host_memory_capacity": "1.5 TB",
                "host_memory_configuration": "24x 64GB DDR5-4800",
                "accelerator_model_name": system["accelerator"],
                "accelerators_per_node": system["accelerators_per_node"],
                "accelerator_memory_capacity": "80 GB",
                "accelerator_memory_type": "HBM3",
                "accelerator_interconnect": "High-speed fabric",
                "accelerator_host_interconnect": "PCIe Gen5",
                "host_network_card_count": "2x 400GbE",
                "host_networking": "Ethernet",
                "host_storage_capacity": "30 TB NVMe",
                "host_storage_type": "NVMe SSD",
                "cooling": "Air-cooled",
                "driver": "1.0",
                "operating_system": "Ubuntu 22.04",
                "filesystem": "ext4",
            }
        ],
        "division": "Standardized",
        "model_id": "llama3.1-8b",
        "model_name": "llama3.1-8b",
        "model_precision": "FP16",
        "link_to_model": "https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b",
        "dataset_id": DATASET,
        "dataset_name": "LLM Perf Dataset v1",
        "input_token_average": 800.0,
        "output_token_average": OUTPUT_TOKEN_AVG,
        "dataset_type": "text",
        "dataset_link": "https://example.com/dataset",
        "measured_accuracy_score": "45.12",
        "model_notes": "Synthetic demo submission for terminal Pareto view.",
    }


def curve_metrics(system: dict) -> list[dict]:
    """Return per-concurrency metric dicts for *system*, with tps_utilization filled."""
    rows = []
    for c in CONCURRENCIES:
        system_tps = system["t_max"] * c / (c + system["k"])
        rows.append(
            {
                "concurrency": c,
                "system_tps": round(system_tps, 2),
                "tps_per_user": round(system_tps / c, 4),
                "ttft": round(system["ttft0"] + system["ttft_slope"] * c, 3),
                "qps": round(system_tps / OUTPUT_TOKEN_AVG, 2),
            }
        )
    max_tps = max(r["system_tps"] for r in rows)
    for r in rows:
        r["tps_utilization"] = round(r["system_tps"] / max_tps, 4)
    return rows


def point_yaml(c: int) -> str:
    """Return a §8.3 points/point_<c>.yaml document."""
    return (
        f"concurrency: {c}\n"
        f"dataset: {DATASET}\n"
        "runtime_settings:\n"
        "  load_pattern: concurrency\n"
        "  min_duration_ms: 600000\n"
        "  runtime:\n"
        "    dataloader_random_seed: 42\n"
        "    scheduler_random_seed: 42\n"
        "  stream_all_chunks: true\n"
        "warmup:\n"
        f"  concurrency: {c}\n"
        f"  data_source: {DATASET} validation split\n"
        "  duration_s: 60.0\n"
        "  initialization_steps:\n"
        "  - model loaded\n"
        "  - kv-cache warmed\n"
        f"  requests_completed: {c * 10}\n"
        f"  requests_issued: {c * 10}\n"
    )


def results_summary(metrics: dict) -> dict:
    """Return a §8.3 results_summary.json consistent with *metrics*."""
    duration_ns = 600_000_000_000  # 600 s steady-state window (meets §6.2 minimum)
    total_output_tokens = int(metrics["system_tps"] * 600)
    n_samples = max(13600, total_output_tokens // int(OUTPUT_TOKEN_AVG))
    ttft_ns = metrics["ttft"] * 1_000_000
    return {
        "n_samples_issued": n_samples,
        "n_samples_completed": n_samples,
        "n_samples_failed": 0,
        "duration_ns": duration_ns,
        "ttft": {
            "total": ttft_ns * n_samples,
            "percentiles": {"50": ttft_ns * 0.55, "95": ttft_ns * 0.95},
        },
        "output_sequence_lengths": {"total": float(total_output_tokens), "percentiles": {}},
    }


def run_metadata(system: dict, metrics: dict) -> dict:
    """Return a §8.3 run_metadata.json for one point (source of truth for the Pareto view)."""
    c = metrics["concurrency"]
    ttft = metrics["ttft"]
    return {
        "run_date": "2026-07-04",
        "node_config": f"{system['accelerators_per_node']}x {system['accelerator']}",
        "config_summary": {
            "disaggregated": None,
            "expert_parallel": 1,
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "data_parallel": system["accelerators_per_node"],
            "batch": c,
        },
        "config_summary_notes": None,
        "concurrency": c,
        "system_tps": metrics["system_tps"],
        "tps_per_user": metrics["tps_per_user"],
        "ttft": ttft,
        "qps": metrics["qps"],
        "tps_utilization": metrics["tps_utilization"],
        "measured_total_output_tokens": int(metrics["system_tps"] * 600),
        "measured_run_duration": 600.0,
        "measured_total_requests": c * 10,
        "link_config": "config.yaml",
        "link_logs": "events.jsonl",
        "measured_latency_ttft_min": round(ttft * 0.15, 3),
        "measured_latency_ttft_average": round(ttft * 0.5, 3),
        "measured_latency_ttft_p50": round(ttft * 0.55, 3),
        "measured_latency_ttft_p90": round(ttft * 0.85, 3),
        "measured_latency_ttft_p95": round(ttft * 0.92, 3),
        "measured_latency_ttft_p99": round(ttft, 3),
        "measured_latency_ttft_p999": round(ttft * 1.05, 3),
        "measured_latency_ttft_max": round(ttft * 1.1, 3),
        "measured_latency_tpot_min": 8.0,
        "measured_latency_tpot_average": 12.0,
        "measured_latency_tpot_p50": 11.5,
        "measured_latency_tpot_p90": 15.0,
        "measured_latency_tpot_p95": 16.5,
        "measured_latency_tpot_p99": 18.0,
        "measured_latency_tpot_p999": 20.0,
        "measured_latency_tpot_max": 22.0,
        "measured_latency_request_min": round(ttft + 200, 3),
        "measured_latency_request_average": round(ttft + 900, 3),
        "measured_latency_request_p50": round(ttft + 800, 3),
        "measured_latency_request_p90": round(ttft + 1400, 3),
        "measured_latency_request_p95": round(ttft + 1600, 3),
        "measured_latency_request_p99": round(ttft + 1900, 3),
        "measured_latency_request_p999": round(ttft + 2200, 3),
        "measured_latency_request_max": round(ttft + 2500, 3),
    }


def accuracy_results() -> dict:
    """Return an accuracy/results.json that clears the llama3.1-8b gate (§15)."""
    return {
        DATASET: {
            "dataset_name": DATASET,
            "num_samples": 13368,
            "extractor": "RougeExtractor",
            "score": {"rouge1": "45.12", "rouge2": "22.01", "rougeL": "30.45"},
            "n_repeats": 1,
        }
    }


def write_json(path: Path, data: dict) -> None:
    """Write *data* as pretty JSON to *path*, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Assemble the synthetic submission directory tree."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    # Top-level required dirs.
    (OUT_DIR / "systems").mkdir()
    (OUT_DIR / "documentation").mkdir()
    (OUT_DIR / "src" / MODEL_DIR_NAME).mkdir(parents=True)
    (OUT_DIR / "src" / MODEL_DIR_NAME / ".gitkeep").write_text("", encoding="utf-8")
    (OUT_DIR / "documentation" / "README.md").write_text(
        "# Synthetic demo submission\n\nGenerated for the terminal Pareto view demo.\n",
        encoding="utf-8",
    )

    for system in SYSTEMS:
        sd = system_desc(system)
        write_json(OUT_DIR / "systems" / f"{system['id']}.json", sd)

        model_root = OUT_DIR / "pareto" / system["id"] / MODEL_DIR_NAME
        points_dir = model_root / "points"
        results_dir = model_root / "results"
        points_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)

        for metrics in curve_metrics(system):
            c = metrics["concurrency"]
            (points_dir / f"point_{c}.yaml").write_text(point_yaml(c), encoding="utf-8")

            point_dir = results_dir / f"point_{c}"
            point_dir.mkdir(parents=True)
            (point_dir / "config.yaml").write_text(f"concurrency: {c}\n", encoding="utf-8")
            write_json(point_dir / "system_desc.json", sd)
            write_json(point_dir / "results_summary.json", results_summary(metrics))
            write_json(point_dir / "run_metadata.json", run_metadata(system, metrics))
            write_json(point_dir / "accuracy" / "results.json", accuracy_results())

    print(f"Wrote synthetic submission to {OUT_DIR}")
    print(f"Systems: {len(SYSTEMS)} · points/curve: {len(CONCURRENCIES)}")


if __name__ == "__main__":
    main()
