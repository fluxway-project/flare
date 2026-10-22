from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fluxway import ResourceProfile, start
from fluxway.common.types.checkpoint import CheckpointRetentionPolicy
from fluxway.common.types.configs import AsyncStepConfig, HeartbeatConfig
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.profiler.config import ProfilerConfig
from sparse_embedding.checkpoint import PartitionSpec

from flare.data.pipeline.consumption.bootstrap.factory import STANDARD_SOURCE_FACTORY
from flare.runtime.fluxway.resources.resolution import ResourceProfiles
from recipes.recsys.ranking.benchmark.config import (
    RankingThroughputBenchmarkConfigBase,
)
from recipes.recsys.ranking.benchmark.metrics import summarize_sink_reports
from recipes.recsys.ranking.benchmark.pipeline import (
    build_throughput_pipeline_inputs,
)
from recipes.recsys.ranking.benchmark.placement import build_data_throughput_placement
from recipes.recsys.ranking.benchmark.throughput_sink import (
    get_throughput_sink_spmd_class,
)


def validate_reader_execution_audit(
    *, audit_path: Path, reader_workers: int, steps: int, global_batch_size: int
) -> dict[str, Any]:
    if audit_path.exists():
        raise FileExistsError(audit_path)
    rank_paths = tuple(
        audit_path.parent / f"{audit_path.stem}_rank_{rank}{audit_path.suffix}"
        for rank in range(reader_workers)
    )
    if any(not path.is_file() for path in rank_paths):
        raise FileNotFoundError("Reader audit rank files are incomplete")
    records: list[dict[str, Any]] = []
    for rank, path in enumerate(rank_paths):
        payload = json.loads(path.read_text())
        if payload.get("rank") != rank or not isinstance(payload.get("records"), list):
            raise ValueError(f"malformed Reader audit file {path}")
        records.extend(payload["records"])
    by_step: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        step = record["step"]
        if type(step) is not int or step not in range(steps):
            raise ValueError(f"unexpected Reader audit step {step!r}")
        if record["global_batch_size"] != global_batch_size:
            raise ValueError("Reader audit global batch size mismatch")
        by_step.setdefault(step, []).append(record)
    if set(by_step) != set(range(steps)):
        raise ValueError("Reader audit does not cover every step")
    summary = {"reader_workers": reader_workers, "steps": steps, "records": records}
    audit_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


@dataclass(frozen=True)
class ThroughputBenchmarkTarget:
    model_builder: Any
    side_info_seq: Sequence[str]
    batch_transformer_factory: Callable[[Any], Callable[[Any], Mapping[str, Any]]]
    label_column: str
    metadata_cols: Sequence[str]
    group_id_column: str
    extra_label_cols: Sequence[str]
    null_index_sentinel: int
    include_all_physical_columns: bool


async def run_recsys_throughput_benchmark(
    args: RankingThroughputBenchmarkConfigBase,
    target: ThroughputBenchmarkTarget,
    *,
    build_pipeline_nodes_fn: Callable[..., Any],
    build_worker_runtime_env_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if not args.table_identifier:
        raise ValueError(
            "--table-identifier (FLUXWAY_GRAPH_TABLE_IDENTIFIER) is required"
        )
    if args.reader_workers <= 0 or args.dp <= 0:
        raise ValueError("reader_workers and dp must be positive")
    if args.reader_repeat_first_batch:
        raise ValueError("throughput benchmark requires fresh source batches")
    if args.iceberg_snapshot_id is None:
        raise ValueError(
            "throughput benchmark requires an explicit iceberg_snapshot_id"
        )
    if type(args.steps) is not int or args.steps < 2:
        raise ValueError("throughput benchmark requires at least two steps")

    xla_cache_dir = os.path.abspath(args.xla_cache_dir)
    cpu_env = build_worker_runtime_env_fn(
        {
            "env_vars": {
                "JAX_PLATFORMS": "cpu",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "JAX_COMPILATION_CACHE_DIR": xla_cache_dir,
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            }
        }
    )

    GiB = 1024**3
    reader_mem = 1 * GiB if args.reader_workers > 8 else 2 * GiB
    input_mem = 2 * GiB if args.reader_workers > 8 else 3 * GiB
    benchmark_profiles = ResourceProfiles(
        {
            "reader": ResourceProfile(base={"CPU": 1, "memory": reader_mem}),
            "input": ResourceProfile(base={"CPU": 1, "memory": input_mem}),
            "metrics": ResourceProfile(base={"CPU": 1, "memory": 1 * GiB}),
            "compute": ResourceProfile(base={"CPU": 1, "memory": 1 * GiB}),
            "table": ResourceProfile(base={"CPU": 1, "memory": 1 * GiB}),
        }
    )

    common = build_pipeline_nodes_fn(
        target.model_builder,
        args,
        source_factory=STANDARD_SOURCE_FACTORY,
        manifest_source_enabled=True,
        runtime_env=cpu_env,
        optimizer_enabled=False,
        load_optimizer_state=False,
        retention_policy=CheckpointRetentionPolicy(retain_last=1),
        resource_profiles=benchmark_profiles,
        sparse_partition_assignment_spec=PartitionSpec.modulo(),
        key_dtype="uint64",
    )

    resources = common.resources
    BoundThroughputSinkSPMD = get_throughput_sink_spmd_class(
        batch_fields=(
            "side_info",
            "labels",
            "group_id",
            "samples_valid_count",
        ),
        runtime_env=cpu_env,
        resource_profile=resources.get_role("metrics"),
        dp_size=args.dp,
    )

    run_id = (
        f"reader{args.reader_workers}_input{args.dp}"
        f"_samples_per_input_step{args.samples_per_input_step}_{time.time_ns()}"
    )
    report_dir = Path("tmp/throughput_benchmark_reports").resolve() / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = str(report_dir / "throughput_report.json")

    with FunctionalGraph("FlareJax_Throughput_Benchmark") as graph:
        pipeline = build_throughput_pipeline_inputs(
            mb=target.model_builder,
            side_info_seq=target.side_info_seq,
            args=args,
            common=common,
            sink_spmd=BoundThroughputSinkSPMD,
            repeat_first_batch=args.reader_repeat_first_batch,
            execution_tracer=None,
            shuffle=False,
            num_epochs=1,
            batch_transformer_factory=target.batch_transformer_factory,
            report_output_path=report_path,
            include_all_physical_columns=target.include_all_physical_columns,
            metadata_cols=target.metadata_cols,
            label_column=target.label_column,
            group_id_column=target.group_id_column,
            extra_label_cols=target.extra_label_cols,
            null_index_sentinel=target.null_index_sentinel,
            execution_audit_path=(
                str(Path(args.reader_execution_audit_path).expanduser().resolve())
                if args.reader_execution_audit_path is not None
                else None
            ),
        )

        placement_recipe = build_data_throughput_placement(
            recipe_id="flare-jax-throughput-placement",
            reader_spmd=pipeline.reader,
            input_spmd=pipeline.input_node,
            sink_spmd=pipeline.sink_node,
        )
        task_graph = graph.build()

    print("[Fluxway Benchmark] Graph compiled successfully.")
    print(f"[Fluxway Benchmark] Nodes: {list(task_graph.nodes)}")
    print(f"[Fluxway Benchmark] Dependencies count: {len(task_graph.dependency_specs)}")

    profiler_config = None
    if args.profiler_trace_dir and args.profiler_run_id:
        profiler_config = ProfilerConfig(
            trace_dir=args.profiler_trace_dir,
            run_id=args.profiler_run_id,
            buffer_bytes_per_process=args.profiler_buffer_mib * 1024 * 1024,
            clock_sync_rounds=args.profiler_clock_sync_rounds,
            max_clock_uncertainty_ns=args.profiler_max_clock_uncertainty_us * 1000,
            output_file=args.profiler_output_file,
            diagnostics=frozenset(args.profiler_diagnostics),
            skip_initial_steps=args.profiler_skip_initial_steps,
            steady_state_only=args.profiler_steady_state_only,
        )

    t_start = time.perf_counter()
    execution = await start(
        task_graph,
        placement_recipe=placement_recipe,
        async_step_config=AsyncStepConfig(max_steps=args.steps),
        runtime_env=cpu_env,
        profiler_config=profiler_config,
        heartbeat_config=HeartbeatConfig(
            heartbeat_timeout=300.0, heartbeat_interval=5.0
        ),
        resume=None,
    )
    print("[Fluxway Benchmark] Allocation Summary:", execution.allocation)

    await execution.wait()
    t_end = time.perf_counter()
    wall_duration = t_end - t_start

    print(
        f"[Fluxway Benchmark] Execution finished in {wall_duration:.3f}s. Collecting metrics..."
    )

    stem = Path(report_path).stem
    rank_files = []
    for _ in range(50):
        rank_files = sorted(report_dir.glob(f"{stem}_rank_*.json"))
        if len(rank_files) == args.dp:
            break
        await asyncio.sleep(0.1)

    if len(rank_files) != args.dp:
        raise RuntimeError(
            f"Expected exactly {args.dp} worker reports matching "
            f"{stem}_rank_*.json in {report_dir}, found {len(rank_files)}"
        )

    worker_reports = []
    for rf in rank_files:
        worker_reports.append(json.loads(rf.read_text()))

    summary_data = summarize_sink_reports(
        worker_reports,
        input_workers=args.dp,
        reader_workers=args.reader_workers,
        samples_per_input_step=args.samples_per_input_step,
        global_batch_size=args.global_batch_size,
        steps=args.steps,
        wall_seconds=wall_duration,
    )
    summary_data["table_identifier"] = args.table_identifier
    summary_data["iceberg_snapshot_id"] = args.iceberg_snapshot_id
    summary_data["reader_execution_cycle"] = (
        task_graph.require_compiled_graph()
        .spmd("Reader")
        .operator("read")
        .execution_layout.cycle
    )
    compiled_graph = task_graph.require_compiled_graph()
    for summary_key, spmd_name, operator_name in (
        ("reader_async_policy", "Reader", "read"),
        ("input_async_policy", "Input", "process"),
    ):
        policy = (
            compiled_graph.spmd(spmd_name).operator(operator_name).spec.runtime_policy
        )
        summary_data[summary_key] = {
            "mode": policy.mode,
            "pipeline_depth": policy.pipeline_depth,
            "max_inflight_steps": policy.max_inflight_steps,
            "state_access": policy.state_access,
        }
    summary_data["jax_platform"] = args.jax_platform
    if args.reader_execution_audit_path is not None:
        validate_reader_execution_audit(
            audit_path=Path(args.reader_execution_audit_path).expanduser().resolve(),
            reader_workers=args.reader_workers,
            steps=args.steps,
            global_batch_size=args.global_batch_size,
        )
    summary_json_path = report_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_data, indent=2))
    print(json.dumps(summary_data, indent=2))
    print(f"[Fluxway Benchmark] Summary written to {summary_json_path}")
    return summary_data


__all__ = [
    "ThroughputBenchmarkTarget",
    "run_recsys_throughput_benchmark",
]
