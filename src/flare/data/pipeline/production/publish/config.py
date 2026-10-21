"""flare.data.pipeline.production.publish.config — Pipeline configuration.

Stage-split pipeline: Datasource → ParseUDF → EncodeUDF → Consumer.

Concurrency derivation is delegated to
:class:`flare.data.pipeline.production.publish.resource_profile.StageTopologyPlanner` so the
benchmark CLI and the production orchestrator share one rule set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .resource_profile import (
    StageTopologyPlanner,
    StreamingMemoryPlanner,
    WorkerMemoryPlanner,
)

logger = logging.getLogger(__name__)

_DEFAULT_READ_PLANNER_BLOCK_MULTIPLIER = 1
DEFAULT_ACTOR_MAX_TASKS_IN_FLIGHT = 2
DEFAULT_ENCODED_BATCH_TARGET_BYTES = 64 * 1024 * 1024
DEFAULT_STREAMING_GENERATOR_MAX_BLOCKS = 2


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for the Ray Data streaming pipeline.

    Attributes:
        encoders_per_gpu: EncodeUDF instances per physical GPU.
        gpu_fraction: ``num_gpus=`` fraction each EncodeUDF requests.
    """

    encoders_per_gpu: int
    gpu_fraction: float
    object_store_memory_bytes: int
    downstream_capacity_backpressure_ratio: float
    dynamic_output_queue_backpressure: bool
    streaming_generator_max_blocks: int
    eager_free: bool
    read_profile: bool
    stage_profile: bool
    fuse_parse_encode: bool
    concurrent_pipelines: int
    source_csv_block_size_bytes: int
    target_min_block_size_bytes: int
    target_max_block_size_bytes: int
    read_planner_block_bytes: int
    read_task_memory_bytes: int
    worker_heap_budget_bytes: int
    parse_actor_memory_bytes: int
    encode_actor_memory_bytes: int
    read_concurrency: int | None
    parse_actor_min_size: int
    parse_actor_max_size: int
    encode_actor_min_size: int
    encode_actor_max_size: int
    write_concurrency: int | None
    read_task_num_cpus: float
    map_actor_num_cpus: float
    write_task_num_cpus: float
    datasource_prefetch_max_buffers: int
    datasource_prefetch_buffer_size_bytes: int
    source_files_per_task: int
    actor_max_tasks_in_flight: int
    encoded_batch_target_bytes: int

    @classmethod
    def create(
        cls,
        *,
        cluster_cpus: int,
        cluster_memory_bytes: int,
        object_store_capacity_bytes: int,
        read_task_memory_bytes: int | None = None,
        n_gpus: int = 0,
        gpu_memory_bytes: int = 0,
        gpu_mode: str = "auto",
        encoders_per_gpu: int | None = None,
        concurrent_pipelines: int = 1,
        object_store_memory_fraction: float = 0.7,
        downstream_capacity_backpressure_ratio: float = 1.0,
        dynamic_output_queue_backpressure: bool = True,
        streaming_generator_max_blocks: int = DEFAULT_STREAMING_GENERATOR_MAX_BLOCKS,
        eager_free: bool = True,
        read_profile: bool = False,
        stage_profile: bool = False,
        fuse_parse_encode: bool = False,
        source_csv_block_size_bytes: int | None = 128 * 1024 * 1024,
        target_min_block_size_bytes: int | None = 256 * 1024 * 1024,
        target_max_block_size_bytes: int | None = 512 * 1024 * 1024,
        read_planner_block_bytes: int | None = None,
        read_concurrency: int | None = None,
        read_task_num_cpus: float | None = None,
        write_concurrency: int | None = None,
        write_task_num_cpus: float | None = None,
        parse_actor_max_size: int | None = None,
        encode_actor_max_size: int | None = None,
        actor_max_tasks_in_flight: int = DEFAULT_ACTOR_MAX_TASKS_IN_FLIGHT,
        datasource_prefetch_max_buffers: int = 2,
        datasource_prefetch_buffer_size_bytes: int | None = None,
        source_files_per_task: int = 1,
        encoded_batch_target_bytes: int = DEFAULT_ENCODED_BATCH_TARGET_BYTES,
    ) -> PipelineConfig:
        """Create config from cluster resources via :class:`StageTopologyPlanner`.

        Args:
            cluster_cpus: Total CPU cores visible to Ray.
            n_gpus: Total GPUs visible to Ray.
            gpu_memory_bytes: Per-GPU VRAM in bytes (drives encoders_per_gpu auto).
            gpu_mode: ``'auto'`` / ``'force'`` / ``'disable'``.
            encoders_per_gpu: Manual override; ``None`` auto-infers from VRAM.
        """
        requested_auto_sizing = any(
            value is None
            for value in (
                source_csv_block_size_bytes,
                target_min_block_size_bytes,
                target_max_block_size_bytes,
            )
        )
        if datasource_prefetch_buffer_size_bytes is None and not requested_auto_sizing:
            datasource_prefetch_buffer_size_bytes = source_csv_block_size_bytes
        for name, value in (
            ("source_csv_block_size_bytes", source_csv_block_size_bytes),
            (
                "datasource_prefetch_buffer_size_bytes",
                datasource_prefetch_buffer_size_bytes,
            ),
            ("target_min_block_size_bytes", target_min_block_size_bytes),
            ("target_max_block_size_bytes", target_max_block_size_bytes),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in (
            ("read_concurrency", read_concurrency),
            ("write_concurrency", write_concurrency),
            ("parse_actor_max_size", parse_actor_max_size),
            ("encode_actor_max_size", encode_actor_max_size),
            ("read_planner_block_bytes", read_planner_block_bytes),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")
        if (
            target_min_block_size_bytes is not None
            and target_max_block_size_bytes is not None
            and target_min_block_size_bytes > target_max_block_size_bytes
        ):
            raise ValueError(
                "target_min_block_size_bytes must be <= target_max_block_size_bytes, "
                f"got {target_min_block_size_bytes} > {target_max_block_size_bytes}"
            )
        if datasource_prefetch_max_buffers <= 0:
            raise ValueError(
                f"datasource_prefetch_max_buffers must be positive, got {datasource_prefetch_max_buffers}"
            )
        if source_files_per_task <= 0:
            raise ValueError(
                f"source_files_per_task must be positive, got {source_files_per_task}"
            )
        if downstream_capacity_backpressure_ratio <= 0:
            raise ValueError(
                "downstream_capacity_backpressure_ratio must be positive, "
                f"got {downstream_capacity_backpressure_ratio}"
            )
        if (
            not isinstance(streaming_generator_max_blocks, int)
            or isinstance(streaming_generator_max_blocks, bool)
            or not 1 <= streaming_generator_max_blocks <= 3
        ):
            raise ValueError(
                "streaming_generator_max_blocks must be an integer in [1, 3], "
                f"got {streaming_generator_max_blocks!r}"
            )
        if not isinstance(eager_free, bool):
            raise TypeError(f"eager_free must be bool, got {type(eager_free).__name__}")
        if not isinstance(read_profile, bool):
            raise TypeError(
                f"read_profile must be bool, got {type(read_profile).__name__}"
            )
        if not isinstance(stage_profile, bool):
            raise TypeError(
                f"stage_profile must be bool, got {type(stage_profile).__name__}"
            )
        if not isinstance(fuse_parse_encode, bool):
            raise TypeError(
                "fuse_parse_encode must be bool, "
                f"got {type(fuse_parse_encode).__name__}"
            )
        if read_task_memory_bytes is None:
            raise ValueError(
                "read_task_memory_bytes is required because compressed-source "
                "worker heap cannot be derived from block and prefetch sizes"
            )
        if read_task_memory_bytes <= 0:
            raise ValueError(
                f"read_task_memory_bytes must be positive, got {read_task_memory_bytes}"
            )
        if actor_max_tasks_in_flight <= 0:
            raise ValueError(
                "actor_max_tasks_in_flight must be positive, "
                f"got {actor_max_tasks_in_flight}"
            )
        if encoded_batch_target_bytes <= 0:
            raise ValueError(
                "encoded_batch_target_bytes must be positive, "
                f"got {encoded_batch_target_bytes}"
            )

        if gpu_mode not in {"auto", "force", "disable"}:
            raise ValueError(
                f"gpu_mode must be 'auto', 'force', or 'disable', got {gpu_mode!r}"
            )
        if gpu_mode == "force" and n_gpus <= 0:
            raise ValueError("gpu_mode='force' requires at least one GPU")
        use_gpu = gpu_mode == "force" or (gpu_mode == "auto" and n_gpus > 0)
        if use_gpu and gpu_memory_bytes <= 0:
            raise ValueError(
                f"GPU execution requires positive gpu_memory_bytes, got {gpu_memory_bytes}"
            )
        backend = "gpu" if use_gpu else "cpu"
        if (
            read_planner_block_bytes is None
            and target_max_block_size_bytes is not None
            and not requested_auto_sizing
        ):
            read_planner_block_bytes = (
                target_max_block_size_bytes * _DEFAULT_READ_PLANNER_BLOCK_MULTIPLIER
            )

        topology = StageTopologyPlanner().plan(
            backend=backend,
            cluster_cpus=cluster_cpus,
            n_gpus=n_gpus if use_gpu else 0,
            object_store_capacity_bytes=object_store_capacity_bytes,
            encoders_per_gpu=encoders_per_gpu,
            gpu_memory_bytes=gpu_memory_bytes,
            concurrent_pipelines=concurrent_pipelines,
            object_store_memory_fraction=object_store_memory_fraction,
            read_concurrency=read_concurrency,
            read_task_num_cpus=read_task_num_cpus,
            write_concurrency=write_concurrency,
            write_task_num_cpus=write_task_num_cpus,
            parse_actor_max_size=parse_actor_max_size,
            encode_actor_max_size=encode_actor_max_size,
        )
        if requested_auto_sizing or datasource_prefetch_buffer_size_bytes is None:
            if topology.object_store_memory_bytes <= 0:
                raise ValueError(
                    "automatic streaming block sizing requires Ray object-store "
                    "capacity; pass explicit block sizes or configure Ray Plasma"
                )
            memory_plan = StreamingMemoryPlanner().plan(
                object_store_memory_bytes=topology.object_store_memory_bytes,
                read_concurrency=topology.read_concurrency or 1,
                datasource_prefetch_max_buffers=datasource_prefetch_max_buffers,
                source_csv_block_size_bytes=source_csv_block_size_bytes,
                target_min_block_size_bytes=target_min_block_size_bytes,
                target_max_block_size_bytes=target_max_block_size_bytes,
                datasource_prefetch_buffer_size_bytes=datasource_prefetch_buffer_size_bytes,
            )
            source_csv_block_size_bytes = memory_plan.source_csv_block_size_bytes
            target_min_block_size_bytes = memory_plan.target_min_block_size_bytes
            target_max_block_size_bytes = memory_plan.target_max_block_size_bytes
            datasource_prefetch_buffer_size_bytes = (
                memory_plan.datasource_prefetch_buffer_size_bytes
            )
            if read_planner_block_bytes is None:
                read_planner_block_bytes = target_max_block_size_bytes

        assert source_csv_block_size_bytes is not None
        assert target_min_block_size_bytes is not None
        assert target_max_block_size_bytes is not None
        assert datasource_prefetch_buffer_size_bytes is not None
        assert read_planner_block_bytes is not None

        worker_memory = WorkerMemoryPlanner().plan(
            cluster_cpus=cluster_cpus,
            cluster_memory_bytes=cluster_memory_bytes,
            read_concurrency=topology.read_concurrency or 1,
            parse_actor_max_size=topology.parse_actor_max_size,
            encode_actor_max_size=topology.encode_actor_max_size,
            read_task_memory_bytes=read_task_memory_bytes,
            source_block_bytes=source_csv_block_size_bytes,
            target_block_bytes=target_max_block_size_bytes,
            read_concurrency_explicit=read_concurrency is not None,
            parse_actor_max_explicit=parse_actor_max_size is not None,
            encode_actor_max_explicit=encode_actor_max_size is not None,
        )

        logger.info(
            "PipelineConfig: gpu_fraction=%.2f, "
            "read_concurrency=%s, "
            "write_concurrency=%s, parse_actor_pool=%d..%d, encode_actor_pool=%d..%d, backend=%s",
            topology.gpu_fraction,
            worker_memory.read_concurrency,
            topology.write_concurrency,
            worker_memory.parse_actor_max_size,
            worker_memory.parse_actor_max_size,
            worker_memory.encode_actor_max_size,
            worker_memory.encode_actor_max_size,
            backend,
        )
        actor_memory_limit = (
            min(
                worker_memory.parse_actor_memory_bytes,
                worker_memory.encode_actor_memory_bytes,
            )
            // 4
        )
        if encoded_batch_target_bytes > actor_memory_limit:
            raise ValueError(
                "encoded_batch_target_bytes exceeds the per-actor safety limit: "
                f"{encoded_batch_target_bytes} > {actor_memory_limit}"
            )

        return cls(
            encoders_per_gpu=topology.encoders_per_gpu,
            gpu_fraction=topology.gpu_fraction,
            object_store_memory_bytes=topology.object_store_memory_bytes,
            downstream_capacity_backpressure_ratio=downstream_capacity_backpressure_ratio,
            dynamic_output_queue_backpressure=dynamic_output_queue_backpressure,
            streaming_generator_max_blocks=streaming_generator_max_blocks,
            eager_free=eager_free,
            read_profile=read_profile,
            stage_profile=stage_profile,
            fuse_parse_encode=fuse_parse_encode,
            concurrent_pipelines=concurrent_pipelines,
            source_csv_block_size_bytes=source_csv_block_size_bytes,
            target_min_block_size_bytes=target_min_block_size_bytes,
            target_max_block_size_bytes=target_max_block_size_bytes,
            read_planner_block_bytes=read_planner_block_bytes,
            read_task_memory_bytes=read_task_memory_bytes,
            worker_heap_budget_bytes=worker_memory.heap_budget_bytes,
            parse_actor_memory_bytes=worker_memory.parse_actor_memory_bytes,
            encode_actor_memory_bytes=worker_memory.encode_actor_memory_bytes,
            read_concurrency=worker_memory.read_concurrency,
            parse_actor_min_size=worker_memory.parse_actor_max_size,
            parse_actor_max_size=worker_memory.parse_actor_max_size,
            encode_actor_min_size=worker_memory.encode_actor_max_size,
            encode_actor_max_size=worker_memory.encode_actor_max_size,
            write_concurrency=topology.write_concurrency,
            read_task_num_cpus=topology.read_task_num_cpus,
            map_actor_num_cpus=topology.map_actor_num_cpus,
            write_task_num_cpus=topology.write_task_num_cpus,
            datasource_prefetch_max_buffers=datasource_prefetch_max_buffers,
            datasource_prefetch_buffer_size_bytes=datasource_prefetch_buffer_size_bytes,
            source_files_per_task=source_files_per_task,
            actor_max_tasks_in_flight=actor_max_tasks_in_flight,
            encoded_batch_target_bytes=encoded_batch_target_bytes,
        )
