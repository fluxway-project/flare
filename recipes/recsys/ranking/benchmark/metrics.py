from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def summarize_sink_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    input_workers: int,
    reader_workers: int,
    samples_per_input_step: int,
    global_batch_size: int,
    steps: int,
    wall_seconds: float,
) -> dict[str, Any]:
    if any(
        type(value) is not int or value <= 0
        for value in (
            input_workers,
            reader_workers,
            samples_per_input_step,
            global_batch_size,
            steps,
        )
    ):
        raise ValueError(
            "worker counts, samples per input step, global batch and steps "
            "must be positive integers"
        )
    if global_batch_size != samples_per_input_step * input_workers:
        raise ValueError(
            "global_batch_size must derive from samples_per_input_step and input workers"
        )
    if reader_workers % input_workers:
        raise ValueError("Reader count must be divisible by Input count")
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("wall duration must be finite and positive")
    if (
        len(reports) != input_workers
        or any(type(report["rank"]) is not int for report in reports)
        or {report["rank"] for report in reports} != set(range(input_workers))
    ):
        raise ValueError("sink reports must cover every rank exactly once")
    intervals: list[float] = []
    total_samples = 0
    window_samples = 0
    durations: list[float] = []
    for report in reports:
        if (
            type(report["world_size"]) is not int
            or type(report["graph_steps"]) is not int
            or report["world_size"] != input_workers
            or report["graph_steps"] != steps
        ):
            raise ValueError(
                "sink report topology or completed steps do not match request"
            )
        counts = report["step_valid_samples"]
        if len(counts) != steps or any(
            type(count) is not int or count <= 0 for count in counts
        ):
            raise ValueError("sink must report positive valid samples for every step")
        if any(count != samples_per_input_step for count in counts):
            raise ValueError("sink valid samples do not match samples_per_input_step")
        if sum(counts) != report["valid_samples"]:
            raise ValueError("sink total disagrees with per-step counts")
        gaps = report["step_intervals"]
        if len(gaps) != steps - 1 or any(
            not math.isfinite(gap) or gap <= 0 for gap in gaps
        ):
            raise ValueError("sink arrival intervals are invalid or incomplete")
        duration = report["measured_duration_seconds"]
        if not math.isfinite(duration) or not math.isclose(
            duration, sum(gaps), rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("sink duration disagrees with arrival intervals")
        durations.append(duration)
        intervals.extend(gaps)
        total_samples += sum(counts)
        window_samples += sum(counts[1:])
    end_to_end = total_samples / wall_seconds
    window_seconds = max(durations)
    window_rate = window_samples / window_seconds if steps > 1 else None
    return {
        "reader_workers": reader_workers,
        "input_workers": input_workers,
        "sink_workers": input_workers,
        "samples_per_input_step": samples_per_input_step,
        "global_batch_size": global_batch_size,
        "reader_interleave_factor": reader_workers // input_workers,
        "graph_steps": steps,
        "valid_samples": total_samples,
        "wall_seconds": wall_seconds,
        "end_to_end_aggregate_samples_per_second": end_to_end,
        "end_to_end_per_input_samples_per_second": end_to_end / input_workers,
        "arrival_window_samples": window_samples,
        "arrival_window_seconds": window_seconds,
        "arrival_window_aggregate_samples_per_second": window_rate,
        "arrival_window_per_input_samples_per_second": (
            None if window_rate is None else window_rate / input_workers
        ),
        "p50_step_seconds": float(np.percentile(intervals, 50)) if intervals else None,
        "p95_step_seconds": float(np.percentile(intervals, 95)) if intervals else None,
    }


__all__ = ["summarize_sink_reports"]
