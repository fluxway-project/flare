"""Ray Data streaming pipeline orchestrator.

    HdfsZstdDatasource          (IO + zstd decompress)
      → map_batches(ParseUDF)   (text → raw Arrow features)
      → map_batches(EncodeUDF)  (ExprPlan: fea_cross → encode → fid_cross)
      → terminal.finalize(ds)    (bound transactional publish / stream consume)

The terminal stage is pluggable via
:class:`flare.data.execution.plan.terminal.DatasetTerminal`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import ray.data

from flare.data.contracts.assets import AssetRef
from flare.data.execution.plan.physical import PhysicalPlan, PhysicalStage
from flare.data.execution.plan.terminal import DatasetTerminal
from flare.data.execution.ray.stage_runtime import apply_physical_plan
from flare.data.pipeline.production.processing.artifact import (
    FeatureProcessingArtifact,
    ProcessingArtifactCompiler,
)
from flare.data.pipeline.production.processing.parser_contract import ParserContract
from flare.data.pipeline.production.publish.config import PipelineConfig
from flare.data.pipeline.production.publish.datasource import HdfsZstdDatasource
from flare.data.pipeline.production.publish.partitioning import append_partition_fields
from flare.data.pipeline.production.publish.read_profile import (
    collect_read_profile_summary,
    create_read_profile_collector,
)
from flare.data.pipeline.production.publish.resource_profile import (
    estimate_arrow_row_bytes,
)
from flare.data.pipeline.production.publish.source import PartitionedSourceProvider
from flare.data.pipeline.production.publish.stage_profile import (
    collect_stage_profile_summary,
    create_stage_profile_collector,
)
from flare.data.pipeline.production.publish.stages import EncodeUDF, ParseUDF
from flare.data.pipeline.production.publish.stages.fused_parse_encode import (
    FusedParseEncodeUDF,
)

logger = logging.getLogger(__name__)


_MAX_AUTO_ENCODED_BLOCK_BYTES = 64 * 1024 * 1024


@dataclass
class BatchResult:
    """Aggregate stats from one ``execute()`` call."""

    total_rows: int
    elapsed_seconds: float
    outputs: tuple[AssetRef, ...] = ()


class RayDataPipeline:
    """Stage-split Ray Data pipeline with a swappable terminal consumer."""

    def __init__(
        self,
        config: PipelineConfig,
        conf_dir: str | None = None,
        model_conf_dir: str | None = None,
        parser_contract: ParserContract | None = None,
        artifact: FeatureProcessingArtifact | None = None,
        category_values: Mapping[str, Mapping[int, int]] | None = None,
        auto_category_features: frozenset[str] = frozenset(),
        log_ray_data_stats: bool = False,
        source_provider: PartitionedSourceProvider | None = None,
    ):
        self._config = config
        self._source_provider = source_provider
        self._log_ray_data_stats = log_ray_data_stats

        # Disable Ray Data progress bars to prevent messy logs from worker nodes
        ray.data.DataContext.get_current().enable_progress_bars = False

        if artifact is not None:
            self._artifact = artifact
        else:
            if conf_dir is None or model_conf_dir is None or parser_contract is None:
                raise ValueError(
                    "Either artifact or (conf_dir, model_conf_dir, parser_contract) must be provided"
                )
            self._artifact = ProcessingArtifactCompiler().compile(
                conf_dir=Path(conf_dir),
                model_conf_dir=Path(model_conf_dir),
                parser_contract=parser_contract,
                category_values=category_values,
                auto_category_features=auto_category_features,
            )
        self._schema = self._artifact.compiled_pipeline.output_schema
        self._output_schema = append_partition_fields(self._schema)
        row_bytes = estimate_arrow_row_bytes(self._output_schema)
        self._max_rows_per_batch = max(
            1, config.encoded_batch_target_bytes // row_bytes
        )
        logger.info(
            "RayDataPipeline: gpu_frac=%.3f, source_csv_block_size=%d, "
            "read_planner_block_bytes=%d, read_concurrency=%d, "
            "parse_max=%d, encode_max=%d, worker_heap_budget=%d",
            config.gpu_fraction,
            config.source_csv_block_size_bytes,
            config.read_planner_block_bytes,
            config.read_concurrency or 0,
            config.parse_actor_max_size,
            config.encode_actor_max_size,
            config.worker_heap_budget_bytes,
        )
        logger.info(
            "RayDataPipeline: artifact_digest=%s, output_row_bytes=%d, max_rows_per_batch=%d",
            self._artifact.compiled_pipeline.digest,
            row_bytes,
            self._max_rows_per_batch,
        )

    # ------------------------------------------------------------------
    # ExprPlan + manifest construction
    # ------------------------------------------------------------------

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def write_feature_manifest(self, out_dir: Path) -> Path:
        target = self._artifact.feature_catalog.write_manifest(out_dir)
        logger.info("Wrote feature manifest → %s", target)
        return target

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        file_paths: tuple[str, ...],
        terminal: DatasetTerminal,
    ) -> BatchResult:
        """Execute one finite source manifest as one Ray Data execution."""
        t0 = time.perf_counter()

        self._configure_execution_context()
        read_profile_collector = (
            create_read_profile_collector() if self._config.read_profile else None
        )
        stage_profile_collector = (
            create_stage_profile_collector() if self._config.stage_profile else None
        )
        parse_constructor_kwargs = {
            "parser_spec": self._artifact.parser_spec,
            "metadata_maps": {
                name: dict(values)
                for name, values in self._artifact.metadata_maps.items()
            },
        }
        encode_constructor_kwargs = {
            "encoded_plan": self._artifact.encoded_plan,
            "output_schema": self._schema,
        }
        if stage_profile_collector is not None:
            parse_constructor_kwargs["stage_profile_collector"] = (
                stage_profile_collector
            )
            encode_constructor_kwargs["stage_profile_collector"] = (
                stage_profile_collector
            )

        datasource = HdfsZstdDatasource(
            source_provider=self._source_provider,
            file_paths=file_paths,
            source_csv_block_size_bytes=self._config.source_csv_block_size_bytes,
            read_planner_block_bytes=self._config.read_planner_block_bytes,
            datasource_prefetch_max_buffers=self._config.datasource_prefetch_max_buffers,
            datasource_prefetch_buffer_size_bytes=self._config.datasource_prefetch_buffer_size_bytes,
            source_files_per_task=self._config.source_files_per_task,
            read_profile=self._config.read_profile,
            read_profile_collector=read_profile_collector,
        )

        read_task_count = min(
            len(file_paths),
            self._config.read_concurrency or len(file_paths),
        )
        ds = ray.data.read_datasource(
            datasource,
            num_cpus=self._config.read_task_num_cpus,
            memory=self._config.read_task_memory_bytes,
            concurrency=self._config.read_concurrency,
            override_num_blocks=read_task_count,
        )

        if self._config.fuse_parse_encode:
            physical_plan = PhysicalPlan(
                stages=(
                    PhysicalStage(
                        stage_id="parse-encode",
                        transform=FusedParseEncodeUDF,
                        constructor_kwargs={
                            **parse_constructor_kwargs,
                            **encode_constructor_kwargs,
                        },
                        min_actors=(
                            self._config.parse_actor_min_size
                            + self._config.encode_actor_min_size
                        ),
                        max_actors=(
                            self._config.parse_actor_max_size
                            + self._config.encode_actor_max_size
                        ),
                        tasks_in_flight=self._config.actor_max_tasks_in_flight,
                        num_cpus=self._config.map_actor_num_cpus,
                        # Both UDF working sets can coexist before the parsed
                        # table is released, so reserve their combined bounds.
                        memory_bytes=(
                            self._config.parse_actor_memory_bytes
                            + self._config.encode_actor_memory_bytes
                        ),
                        batch_size=self._max_rows_per_batch,
                    ),
                )
            )
        else:
            physical_plan = PhysicalPlan(
                stages=(
                    PhysicalStage(
                        stage_id="parse",
                        transform=ParseUDF,
                        constructor_kwargs=parse_constructor_kwargs,
                        min_actors=self._config.parse_actor_min_size,
                        max_actors=self._config.parse_actor_max_size,
                        tasks_in_flight=self._config.actor_max_tasks_in_flight,
                        num_cpus=self._config.map_actor_num_cpus,
                        memory_bytes=self._config.parse_actor_memory_bytes,
                        batch_size=self._max_rows_per_batch,
                    ),
                    PhysicalStage(
                        stage_id="feature-transform",
                        transform=EncodeUDF,
                        constructor_kwargs=encode_constructor_kwargs,
                        min_actors=self._config.encode_actor_min_size,
                        max_actors=self._config.encode_actor_max_size,
                        tasks_in_flight=self._config.actor_max_tasks_in_flight,
                        num_cpus=self._config.map_actor_num_cpus,
                        memory_bytes=self._config.encode_actor_memory_bytes,
                        batch_size=self._max_rows_per_batch,
                    ),
                )
            )
        ds = apply_physical_plan(ds, physical_plan)

        result = terminal.finalize(
            ds,
            schema=self._output_schema,
        )
        if read_profile_collector is not None:
            summary = collect_read_profile_summary(read_profile_collector)
            logger.info(
                "Read profile aggregate: files=%d compressed_bytes=%d total=%.3fs "
                "open=%.3fs fetch_wall=%.3fs fetch_cpu=%.3fs "
                "prefetch_wait=%.3fs csv_decode=%.3fs arrow_materialize=%.3fs "
                "yield_wait=%.3fs stream_setup=%.3fs csv_open=%.3fs",
                summary.files,
                summary.compressed_bytes,
                summary.total_seconds,
                summary.open_seconds,
                summary.fetch_wall_seconds,
                summary.fetch_cpu_seconds,
                summary.prefetch_wait_seconds,
                summary.csv_decode_seconds,
                summary.arrow_materialize_seconds,
                summary.yield_wait_seconds,
                summary.stream_setup_seconds,
                summary.csv_open_seconds,
            )
        if stage_profile_collector is not None:
            for stage in ("parse", "encode"):
                summary = collect_stage_profile_summary(stage_profile_collector, stage)
                logger.info(
                    "Stage profile aggregate: stage=%s calls=%d input_rows=%d output_rows=%d wall=%.3fs cpu=%.3fs",
                    summary.stage,
                    summary.calls,
                    summary.input_rows,
                    summary.output_rows,
                    summary.wall_seconds,
                    summary.cpu_seconds,
                )
        if self._log_ray_data_stats:
            stats = ds.stats()
            if stats:
                logger.info("Ray Data stats:\n%s", stats)
            else:
                logger.info("Ray Data stats are empty after pipeline execution")

        elapsed = time.perf_counter() - t0
        logger.info(
            "Publication complete: %d files, %d rows, %.1fs (%.0f rows/s)",
            len(file_paths),
            result.total_rows,
            elapsed,
            result.total_rows / max(elapsed, 0.001),
        )
        return BatchResult(
            total_rows=result.total_rows,
            elapsed_seconds=elapsed,
            outputs=result.outputs,
        )

    def _configure_execution_context(self) -> None:
        ctx = ray.data.DataContext.get_current()
        ctx.target_min_block_size = self._config.target_min_block_size_bytes
        ctx.target_max_block_size = self._config.target_max_block_size_bytes
        ctx.execution_options.resource_limits = ray.data.ExecutionResources(
            object_store_memory=self._config.object_store_memory_bytes
        )
        ctx.downstream_capacity_backpressure_ratio = (
            self._config.downstream_capacity_backpressure_ratio
        )
        ctx.enable_dynamic_output_queue_size_backpressure = (
            self._config.dynamic_output_queue_backpressure
        )
        # Ray Core blocks a task generator after this many unconsumed blocks.
        # Keep the finite admission limit in PipelineConfig rather than relying
        # on Ray's mutable process-global default.
        ctx._max_num_blocks_in_streaming_gen_buffer = (
            self._config.streaming_generator_max_blocks
        )
        ctx.eager_free = self._config.eager_free
        logger.info(
            "Ray Data execution context: downstream_backpressure_ratio=%.2f, "
            "dynamic_output_queue_backpressure=%s, "
            "streaming_generator_max_blocks=%d, eager_free=%s",
            ctx.downstream_capacity_backpressure_ratio,
            ctx.enable_dynamic_output_queue_size_backpressure,
            ctx._max_num_blocks_in_streaming_gen_buffer,
            self._config.eager_free,
        )

    def shutdown(self) -> None:
        """No-op: Ray Data manages actor lifecycle automatically."""

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    @property
    def output_schema(self) -> pa.Schema:
        return self._output_schema

    @property
    def feature_catalog(self):
        return self._artifact.feature_catalog
