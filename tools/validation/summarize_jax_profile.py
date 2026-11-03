"""Summarize JAX Perfetto traces without double-counting overlapping GPU streams."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(end - start for start, end in merge_intervals(intervals)) / 1e6


def intersection(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def category(event):
    name = event["name"].lower()
    if "nccl" in name:
        if "allgather" in name:
            return "all_gather"
        if "bf16" in name:
            return "bf16_collective"
        return "other_collective_including_wait"
    if "gemm" in name or "cublas" in name:
        return "gemm"
    if "scatter" in name:
        return "scatter"
    if "memcpy" in name or "memset" in name:
        return "copy_or_memset"
    return "other_compute"


def summarize(path, step_name="train_core"):
    with gzip.open(path, "rt") as source:
        events = json.load(source)["traceEvents"]
    gpu_pids = {
        event["pid"]
        for event in events
        if event.get("ph") == "M"
        and event["name"] == "process_name"
        and "GPU" in event.get("args", {}).get("name", "")
    }
    steps = [
        event
        for event in events
        if event.get("ph") == "X" and event["name"] == step_name
    ]
    if not steps:
        raise ValueError(f"no annotated {step_name} steps in {path}")
    if not gpu_pids:
        raise ValueError(
            f"no GPU streams found in {path}; check device profiling support"
        )
    if len(gpu_pids) != 1:
        raise ValueError(
            "summarize one GPU per trace to avoid mixing device utilization"
        )
    windows = [(event["ts"], event["ts"] + event["dur"]) for event in steps]
    active = [
        event
        for event in events
        if event.get("ph") == "X"
        and any(start <= event["ts"] < end for start, end in windows)
    ]
    gpu = [event for event in active if event.get("pid") in gpu_pids]
    allocations = [
        event for event in active if event["name"] == "Build buffer allocations"
    ]

    def ranges(selected):
        return [(event["ts"], event["ts"] + event["dur"]) for event in selected]

    gpu_ranges = ranges(gpu)
    alloc_ranges = ranges(allocations)
    categories, modules, kernels = (
        defaultdict(float),
        defaultdict(float),
        defaultdict(lambda: [0.0, 0]),
    )
    for event in gpu:
        seconds = event["dur"] / 1e6
        categories[category(event)] += seconds
        modules[event.get("args", {}).get("hlo_module", "unknown")] += seconds
        kernels[event["name"]][0] += seconds
        kernels[event["name"]][1] += 1
    step_details = []
    for start, end in windows:
        window = [(start, end)]
        busy = duration(intersection(gpu_ranges, window))
        allocation = duration(intersection(alloc_ranges, window))
        allocation_gpu = duration(
            intersection(intersection(alloc_ranges, window), gpu_ranges)
        )
        step_details.append(
            {
                "wall_seconds": (end - start) / 1e6,
                "gpu_busy_union_seconds": busy,
                "gpu_idle_seconds": (end - start) / 1e6 - busy,
                "host_buffer_allocation_seconds": allocation,
                "gpu_idle_during_host_allocation_seconds": allocation - allocation_gpu,
            }
        )
    return {
        "trace": str(path),
        "steps": step_details,
        "gpu_kernel_seconds_by_category": dict(categories),
        "gpu_kernel_seconds_by_module": dict(modules),
        "gpu_kernel_count": len(gpu),
        "top_kernels": [
            {"name": name, "seconds": seconds, "count": count}
            for name, (seconds, count) in sorted(
                kernels.items(), key=lambda pair: -pair[1][0]
            )[:30]
        ],
        "caveat": "Kernel sums include overlapping streams and collective waiting. Wall/busy times use interval unions.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-name", default="train_core")
    args = parser.parse_args()
    summaries = [
        summarize(path, args.step_name)
        for path in sorted(args.directory.rglob("perfetto_trace.json.gz"))
    ]
    if not summaries:
        raise ValueError(f"no JAX Perfetto traces found in {args.directory}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2) + "\n")
    for summary in summaries:
        count = len(summary["steps"])
        print(
            summary["trace"],
            {
                key: round(sum(step[key] for step in summary["steps"]) / count, 4)
                for key in summary["steps"][0]
            },
        )


if __name__ == "__main__":
    main()
