from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import make_dataclass
from pathlib import Path
from typing import Any

import jax.tree_util as jtu
import numpy as np
from fluxway.common.bind import bind_spmd
from fluxway.common.decorators import data_type, requires_worker_process
from fluxway.common.types.tensor import TransportTensor
from fluxway.placement import ResourceProfile

logger = logging.getLogger(__name__)

_sink_out_fields: list[tuple[str, type]] = []
ThroughputSinkOutput = make_dataclass("ThroughputSinkOutput", _sink_out_fields)
ThroughputSinkOutput = data_type()(ThroughputSinkOutput)
jtu.register_dataclass(ThroughputSinkOutput, data_fields=[], meta_fields=[])


@requires_worker_process
class ThroughputSinkSPMD:
    rank: int
    world_size: int
    ctx: Any
    rank0_address: str
    ray_component_limits: Any

    def __init__(
        self,
        max_steps: int | None = None,
        reader_workers: int = 8,
        report_output_path: str | None = None,
    ) -> None:
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if reader_workers <= 0:
            raise ValueError(f"reader_workers must be positive, got {reader_workers}")
        self.max_steps = None if max_steps is None else int(max_steps)
        self.reader_workers = int(reader_workers)
        self.report_output_path = report_output_path

        self._current_step = 0
        self._total_valid_samples = 0
        self._step_timestamps: list[float] = []
        self._step_intervals: list[float] = []
        self._step_valid_samples: list[int] = []
        self._reported = False
        self._start_time: float | None = None

    def consume(self, **kwargs: Any) -> ThroughputSinkOutput:
        count_tensor = kwargs["samples_valid_count"]
        if not isinstance(count_tensor, TransportTensor):
            raise TypeError("samples_valid_count must be a TransportTensor")
        count_values = np.asarray(count_tensor.flat_values).reshape(-1)
        if count_values.dtype != np.int32 or count_values.size != 1:
            raise ValueError("samples_valid_count must be one int32 scalar")
        valid_samples = int(count_values[0])
        if valid_samples <= 0:
            raise ValueError("throughput sink requires positive valid samples")
        now = time.perf_counter()
        if self._start_time is None:
            self._start_time = now

        if self._step_timestamps:
            interval = now - self._step_timestamps[-1]
            self._step_intervals.append(interval)
        self._step_timestamps.append(now)

        self._step_valid_samples.append(valid_samples)
        self._total_valid_samples += valid_samples
        self._current_step += 1

        if self.max_steps is not None and self._current_step >= self.max_steps:
            self._report(scope="step_cap")

        return ThroughputSinkOutput()

    async def on_shutdown(self) -> None:
        if not self._reported:
            self._report(scope="shutdown")

    def _report(self, *, scope: str) -> None:
        if self._reported:
            return
        self._reported = True

        total_steps = self._current_step
        total_samples = self._total_valid_samples
        if total_steps == 0 or self._start_time is None:
            logger.warning("[ThroughputSink rank=%d] No steps recorded.", self.rank)
            return

        end_time = self._step_timestamps[-1]
        elapsed = end_time - self._start_time
        intervals = self._step_intervals
        if intervals:
            median_step_s = float(np.median(intervals))
            p50_step_s = float(np.percentile(intervals, 50))
            p95_step_s = float(np.percentile(intervals, 95))
        else:
            median_step_s = elapsed
            p50_step_s = elapsed
            p95_step_s = elapsed

        arrival_window_samples = sum(self._step_valid_samples[1:])
        arrival_window_throughput = (
            float(arrival_window_samples) / elapsed
            if intervals and elapsed > 0.0
            else None
        )

        metrics_payload = {
            "rank": self.rank,
            "world_size": self.world_size,
            "scope": scope,
            "graph_steps": total_steps,
            "valid_samples": total_samples,
            "measured_duration_seconds": elapsed,
            "median_step_seconds": median_step_s,
            "p50_step_seconds": p50_step_s,
            "p95_step_seconds": p95_step_s,
            "arrival_window_valid_samples": arrival_window_samples,
            "arrival_window_per_input_samples_per_second": arrival_window_throughput,
            "step_intervals": intervals,
            "step_valid_samples": self._step_valid_samples,
        }

        print(
            f"[ThroughputSink rank={self.rank}/{self.world_size}] "
            f"scope={scope} steps={total_steps} samples={total_samples} "
            f"duration={elapsed:.3f}s median_step={median_step_s * 1000:.2f}ms "
            f"p95_step={p95_step_s * 1000:.2f}ms "
            f"arrival_window_per_input_throughput="
            f"{arrival_window_throughput if arrival_window_throughput is not None else 'n/a'} samples/s",
            flush=True,
        )

        if self.report_output_path is not None:
            output_file = Path(self.report_output_path)
            target_file = (
                output_file.parent
                / f"{output_file.stem}_rank_{self.rank}{output_file.suffix}"
            )
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(json.dumps(metrics_payload, indent=2))


def get_throughput_sink_spmd_class(
    *,
    batch_fields: Sequence[str] = (
        "side_info",
        "labels",
        "group_id",
        "samples_valid_count",
    ),
    runtime_env: dict[str, Any] | None = None,
    resource_profile: Any = None,
    dp_size: int = 8,
    heartbeat_timeout_s: float = 300.0,
    heartbeat_interval_s: float = 5.0,
) -> type:
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if resource_profile is None:
        resource_profile = ResourceProfile(base={"CPU": 1})

    input_shards: dict[str, Any] = {}
    for f in batch_fields:
        if f == "samples_valid_count":
            input_shards[f] = {"type": "static", "axis": []}
        else:
            input_shards[f] = {"type": "static", "axis": ["dp"]}

    return bind_spmd(
        ThroughputSinkSPMD,
        resource_profile=resource_profile,
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(dp_size,),
        runtime_env=runtime_env,
        checkpoint_components=(),
        heartbeat_timeout_s=heartbeat_timeout_s,
        heartbeat_interval_s=heartbeat_interval_s,
        consume={
            "input_sharding": input_shards,
            "output_sharding": {},
        },
    )


__all__ = [
    "ThroughputSinkOutput",
    "ThroughputSinkSPMD",
    "get_throughput_sink_spmd_class",
]
