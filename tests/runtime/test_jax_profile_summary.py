from __future__ import annotations

import gzip
import json

import pytest

from tools.validation.summarize_jax_profile import summarize


def test_overlapping_streams_and_host_allocations_are_not_double_counted(tmp_path):
    events = [
        {
            "ph": "M",
            "pid": 1,
            "name": "process_name",
            "args": {"name": "/device:GPU:0"},
        },
        {"ph": "X", "pid": 2, "name": "step", "ts": 0, "dur": 1_000_000},
        {"ph": "X", "pid": 1, "name": "gemm", "ts": 100_000, "dur": 600_000},
        {"ph": "X", "pid": 1, "name": "ncclAllGather", "ts": 400_000, "dur": 400_000},
        {
            "ph": "X",
            "pid": 2,
            "name": "Build buffer allocations",
            "ts": 50_000,
            "dur": 200_000,
        },
        {"ph": "X", "pid": 1, "name": "gemm", "ts": 2_000_000, "dur": 100_000},
    ]
    trace = tmp_path / "trace.json.gz"
    with gzip.open(trace, "wt") as output:
        json.dump({"traceEvents": events}, output)
    report = summarize(trace, "step")
    step = report["steps"][0]
    assert step["gpu_busy_union_seconds"] == pytest.approx(0.7)
    assert step["gpu_idle_seconds"] == pytest.approx(0.3)
    assert step["host_buffer_allocation_seconds"] == pytest.approx(0.2)
    assert step["gpu_idle_during_host_allocation_seconds"] == pytest.approx(0.05)
    assert report["gpu_kernel_count"] == 2
    assert report["gpu_kernel_seconds_by_category"]["gemm"] == pytest.approx(0.6)


def test_missing_device_trace_is_not_reported_as_idle_gpu(tmp_path):
    trace = tmp_path / "trace.json.gz"
    with gzip.open(trace, "wt") as output:
        json.dump(
            {"traceEvents": [{"ph": "X", "pid": 2, "name": "step", "ts": 0, "dur": 1}]},
            output,
        )
    with pytest.raises(ValueError, match="no GPU streams"):
        summarize(trace, "step")
