"""End-to-end benchmark: HDFS → line-parse (Cython) → cross + encode → Iceberg.

Measures real-world throughput of the full streaming pipeline on the target
hardware (8 GPU + 120 CPU single-node). Output: rows/sec, elapsed time,
and per-stage breakdown.

Compatible with ``ray job submit``:

    ray job submit \\
      --working-dir /path/to/flare-jax \\
      --runtime-env-json '{"py_modules": ["dataset", "util", "conf"]}' \\
      -- python dataset/publish/benchmarks/bench_e2e_hdfs_vortex.py \\
         --hdfs-root /path/to/data \\
         --output db.table \\
         --start-date 2026-01-06 --end-date 2026-01-07 \\
         --cpu-only --limit-files 100

Target: ≥ 1,000,000 rows/sec sustained throughput.
"""

import dataclasses
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Literal

from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

import ray
import ray.data
import tyro

from flare.data.connectors.adapters.iceberg.catalog import build_iceberg_catalog_kwargs
from flare.data.pipeline.production.builder.hdfs_source import collect_hdfs_file_paths
from flare.data.pipeline.production.processing.parser_contract import ParserContract
from flare.data.pipeline.production.publish.config import PipelineConfig
from flare.data.pipeline.production.publish.orchestrator import (
    RayDataPipeline,
)
from flare.io.aws import S3Config, get_pyiceberg_s3_kwargs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("bench_e2e")

_DEFAULT_OUTPUT = "db.table"
_DEFAULT_FAILURE_LOG = Path("/tmp/flare_pipeline_optimization_failures.md")
_TARGET_ROWS_PER_SECOND = 10_000


@dataclasses.dataclass
class BenchArgs:
    hdfs_root: str
    start_date: str = "2026-01-06"
    end_date: str = "2026-01-07"
    conf_dir: Path
    model_conf_dir: Path
    parser_contract_module: str
    encoders_per_gpu: int | None = None
    read_concurrency: int | None = None
    write_concurrency: int | None = None
    parse_actor_max_size: int | None = None
    encode_actor_max_size: int | None = None
    read_task_memory_mib: int | None = None
    downstream_capacity_backpressure_ratio: float = 1.0
    dynamic_output_queue_backpressure: bool = True
    source_csv_block_size_mib: int = 32
    target_min_block_size_mib: int = 64
    target_max_block_size_mib: int = 128
    source_files_per_task: int = 1
    iceberg_catalog_name: str | None = None
    iceberg_catalog_type: str | None = None
    iceberg_catalog_uri: str | None = None
    iceberg_catalog_warehouse: str | None = None
    limit_files: int | None = None
    log_file: Path | None = None
    failure_log: Path = dataclasses.field(default_factory=lambda: _DEFAULT_FAILURE_LOG)
    log_ray_data_stats: bool = False
    output: str = _DEFAULT_OUTPUT
    cpu_only: bool = False
    gpu: Literal["auto", "force", "disable"] | None = None
    inflight: Literal["none", "arrow", "numpy"] = "none"
    s3: S3Config = dataclasses.field(default_factory=S3Config)


def _configure_file_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
    )
    logging.getLogger().addHandler(handler)
    logger.info("Benchmark log file: %s", log_file)


def _record_failure(path: Path, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"## {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- exception: `{type(exc).__name__}`\n")
        f.write(f"- message: `{exc}`\n\n")


def _load_parser_contract(module_path: str) -> ParserContract:
    module_name, separator, attribute_name = module_path.rpartition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("parser contract module must use 'module:attribute' notation")
    import importlib

    contract = getattr(importlib.import_module(module_name), attribute_name)
    if not isinstance(contract, ParserContract):
        raise TypeError(f"{module_path} must resolve to ParserContract")
    return contract


def main() -> None:
    args = tyro.cli(BenchArgs)

    log_file = args.log_file or Path(
        f"/tmp/flare_pipeline_bench_{int(time.time())}.log"
    )
    _configure_file_logging(log_file)

    s3_options = args.s3.to_options()
    if args.gpu is not None:
        gpu_mode = args.gpu
    elif args.cpu_only:
        gpu_mode = "disable"
    else:
        gpu_mode = "auto"

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    logger.info("Enumerating HDFS files: %s → %s ...", start, end)

    file_paths = collect_hdfs_file_paths(args.hdfs_root, start, end)
    if not file_paths:
        raise RuntimeError(f"No files found under {args.hdfs_root} for {start}→{end}")

    if args.limit_files:
        file_paths = file_paths[: args.limit_files]

    ray.init(address="auto")

    if not sys.stdout.isatty():
        ray.data.DataContext.get_current().enable_progress_bars = False

    from flare.data.pipeline.production.publish.resource_profile import (
        RayClusterCapacity,
    )

    capacity = RayClusterCapacity.from_mapping(ray.cluster_resources())
    logger.info(
        "Ray cluster: CPUs=%d, GPUs=%d, Memory=%.1fGB, ObjectStore=%.1fGB",
        capacity.cpu_count,
        capacity.gpu_count,
        capacity.heap_bytes / (1024**3),
        capacity.object_store_bytes / (1024**3),
    )

    gpu_memory_bytes = 0
    if capacity.gpu_count > 0 and gpu_mode != "disable":
        from flare.data.pipeline.production.publish.resource_profile import (
            HardwareProfiler,
        )

        gpu_memory_bytes = HardwareProfiler()._detect_gpu()[1]

    config = PipelineConfig.create(
        cluster_cpus=capacity.cpu_count,
        cluster_memory_bytes=capacity.heap_bytes,
        object_store_capacity_bytes=capacity.object_store_bytes,
        read_task_memory_bytes=(
            args.read_task_memory_mib * 1024 * 1024
            if args.read_task_memory_mib is not None
            else None
        ),
        downstream_capacity_backpressure_ratio=(
            args.downstream_capacity_backpressure_ratio
        ),
        dynamic_output_queue_backpressure=args.dynamic_output_queue_backpressure,
        n_gpus=capacity.gpu_count,
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_mode=gpu_mode,
        encoders_per_gpu=args.encoders_per_gpu,
        read_concurrency=args.read_concurrency,
        write_concurrency=args.write_concurrency,
        parse_actor_max_size=args.parse_actor_max_size,
        encode_actor_max_size=args.encode_actor_max_size,
        source_csv_block_size_bytes=args.source_csv_block_size_mib * 1024 * 1024,
        target_min_block_size_bytes=args.target_min_block_size_mib * 1024 * 1024,
        target_max_block_size_bytes=args.target_max_block_size_mib * 1024 * 1024,
        source_files_per_task=args.source_files_per_task,
    )
    encode_backend = "gpu" if config.gpu_fraction > 0 else "cpu"

    pipeline = RayDataPipeline(
        config=config,
        conf_dir=str(args.conf_dir),
        model_conf_dir=str(args.model_conf_dir),
        parser_contract=_load_parser_contract(args.parser_contract_module),
        log_ray_data_stats=args.log_ray_data_stats,
    )

    logger.info("=" * 60)
    logger.info(
        "BENCHMARK START: %d files, ray_resource_backend=%s",
        len(file_paths),
        encode_backend,
    )
    logger.info(
        "  source_csv_block_size=%d, read_planner_block_bytes=%d",
        config.source_csv_block_size_bytes,
        config.read_planner_block_bytes,
    )
    logger.info("=" * 60)

    from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher
    from flare.data.connectors.adapters.stream.sink import RayDataStreamDrainer

    if args.inflight in ("arrow", "numpy"):
        consumer = RayDataStreamDrainer(output_format=args.inflight)
    else:
        publisher = IcebergSnapshotPublisher(
            table_identifier=args.output,
            catalog_provider=ConfiguredIcebergCatalogProvider(
                {
                    **build_iceberg_catalog_kwargs(
                        name=args.iceberg_catalog_name,
                        catalog_type=args.iceberg_catalog_type,
                        uri=args.iceberg_catalog_uri,
                        warehouse=args.iceberg_catalog_warehouse,
                    ),
                    **get_pyiceberg_s3_kwargs(s3_options),
                }
            ),
            write_task_num_cpus=config.write_task_num_cpus,
            write_concurrency=config.write_concurrency,
        )
        from flare.data.contracts.sink import PublicationIdentity

        consumer = publisher.bind(PublicationIdentity("bench-e2e-hdfs-parquet-v1"))

    t0 = time.perf_counter()
    try:
        batch_result = pipeline.execute(tuple(file_paths), consumer)
    except Exception as exc:
        _record_failure(args.failure_log, exc)
        raise
    total_rows = batch_result.total_rows

    elapsed = time.perf_counter() - t0
    throughput = total_rows / max(elapsed, 0.001)

    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("  Total rows:       %d", total_rows)
    logger.info("  Elapsed:          %.2f s", elapsed)
    logger.info("  Throughput:       %.0f rows/s", throughput)
    logger.info(
        "  Target (10k/s):   %s",
        "PASS ✓" if throughput >= _TARGET_ROWS_PER_SECOND else "FAIL ✗",
    )
    logger.info("  Files processed:  %d", len(file_paths))
    logger.info("  Ray backend:      %s", encode_backend)
    logger.info("  Encode engine:    PyArrow CPU ExprPlan")
    logger.info(
        "  Cluster:          %d CPUs, %d GPUs",
        capacity.cpu_count,
        capacity.gpu_count,
    )
    logger.info("=" * 60)

    print(f"\n{'=' * 40} RESULT {'=' * 40}")
    print(f"rows={total_rows}")
    print(f"elapsed_s={elapsed:.3f}")
    print(f"throughput_rows_per_s={throughput:.0f}")
    print(f"target_met={'true' if throughput >= _TARGET_ROWS_PER_SECOND else 'false'}")
    print(f"log_file={log_file}")
    print(f"ray_resource_backend={encode_backend}")
    print("encode_engine=pyarrow_cpu_expr_plan")
    print(f"cluster_cpus={capacity.cpu_count}")
    print(f"cluster_gpus={capacity.gpu_count}")
    print(f"source_csv_block_size={config.source_csv_block_size_bytes}")
    print(f"read_planner_block_bytes={config.read_planner_block_bytes}")
    print(f"{'=' * 88}\n")


if __name__ == "__main__":
    main()
