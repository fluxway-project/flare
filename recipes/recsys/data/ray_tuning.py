"""Schema-aware CPU Ray Data tuning for Iceberg production.

The profile selects a feasible operator shape from the compiled output schema.
It does not benchmark the source or claim to choose the faster execution engine.
Explicit CLI resource settings always take precedence over profile defaults.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from flare.data.pipeline.production.publish.config import PipelineConfig
from flare.data.pipeline.production.publish.resource_profile import (
    WorkerMemoryCapacityError,
    estimate_arrow_row_bytes,
)

logger = logging.getLogger(__name__)

MIB = 1024 * 1024
PROFILE_VERSION = "recipe-cpu-auto-v1"
_WIDE_ROW_BYTES = 32 * 1024
_DEFAULT_READ_MEMORY_BYTES = 512 * MIB
_FRACTIONAL_IO_TASK_CPUS = 0.25


def resolve_auto_tune(
    *,
    requested: bool,
    execution_engine: str,
    output_format: str,
    gpu_mode: str,
    gpu_count: int,
    read_profile: bool,
    stage_profile: bool,
) -> bool:
    return (
        requested
        and execution_engine == "ray-data"
        and output_format == "iceberg"
        and gpu_mode != "force"
        and not read_profile
        and not stage_profile
        and (gpu_mode == "disable" or gpu_count == 0)
    )


@dataclass(frozen=True, slots=True)
class RayTuningResult:
    config: PipelineConfig
    estimated_row_bytes: int
    profile: str = PROFILE_VERSION


def build_cpu_iceberg_config(
    *,
    options: Mapping[str, Any],
    output_schema: pa.Schema,
) -> RayTuningResult:
    """Choose a memory-valid CPU Iceberg profile without overriding inputs.

    Wide schemas start at a 256 MiB encoded batch and narrower schemas at
    128 MiB. Only an automatically selected batch may step down when the
    generic worker-memory planner rejects its per-actor safety limit.
    """
    row_bytes = estimate_arrow_row_bytes(output_schema)
    supplied_batch = options.get("encoded_batch_target_bytes")
    batches = (256 * MIB, 128 * MIB, 64 * MIB)
    if row_bytes < _WIDE_ROW_BYTES:
        batches = batches[1:]
    if supplied_batch is not None:
        batches = (supplied_batch,)

    for batch_bytes in batches:
        candidate = dict(options)
        candidate["fuse_parse_encode"] = True
        candidate.setdefault("source_csv_block_size_bytes", None)
        candidate.setdefault("datasource_prefetch_buffer_size_bytes", None)
        for key, default in (
            ("read_task_memory_bytes", _DEFAULT_READ_MEMORY_BYTES),
            ("read_task_num_cpus", _FRACTIONAL_IO_TASK_CPUS),
            ("write_task_num_cpus", _FRACTIONAL_IO_TASK_CPUS),
            ("actor_max_tasks_in_flight", 1),
            ("encoded_batch_target_bytes", batch_bytes),
            ("target_min_block_size_bytes", max(16 * MIB, batch_bytes // 2)),
            (
                "target_max_block_size_bytes",
                batch_bytes * 2 if row_bytes >= _WIDE_ROW_BYTES else batch_bytes,
            ),
        ):
            if candidate.get(key) is None:
                candidate[key] = default
        try:
            config = PipelineConfig.create(**candidate)
        except WorkerMemoryCapacityError:
            if supplied_batch is None and batch_bytes != batches[-1]:
                logger.info(
                    "Ray Data auto tuning: %d MiB batch/block profile exceeds worker memory; trying smaller profile",
                    batch_bytes // MIB,
                )
                continue
            raise
        logger.info(
            "Ray Data %s: row_width=%d bytes, batch=%d MiB, blocks=%d..%d MiB, "
            "read/write CPU=%.2f/%.2f, read/parse/encode/write concurrency=%s/%d/%d/%s",
            PROFILE_VERSION,
            row_bytes,
            config.encoded_batch_target_bytes // MIB,
            config.target_min_block_size_bytes // MIB,
            config.target_max_block_size_bytes // MIB,
            config.read_task_num_cpus,
            config.write_task_num_cpus,
            config.read_concurrency,
            config.parse_actor_max_size,
            config.encode_actor_max_size,
            config.write_concurrency,
        )
        return RayTuningResult(config=config, estimated_row_bytes=row_bytes)

    raise AssertionError("the last tuning candidate must succeed or raise")
