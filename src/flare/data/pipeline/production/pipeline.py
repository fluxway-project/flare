from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ray
from ray.data import DataContext

from flare.data.connectors.adapters.iceberg.catalog import build_iceberg_catalog_kwargs
from flare.data.contracts.pipeline import (
    DataPipelineBuilder,
    DataPipelineSpec,
    SentinelConfig,
    StageConfig,
)
from flare.data.pipeline.production.builder.resume_state_manager import (
    ResumeStateManager,
)
from flare.data.pipeline.production.publish.config import PipelineConfig
from flare.data.pipeline.production.publish.orchestrator import RayDataPipeline
from flare.data.pipeline.production.publish.resource_profile import (
    HardwareProfiler,
    RayClusterCapacity,
)
from flare.data.pipeline.production.publish.runner import (
    CompositeProgressReporter,
    LoggingProgressReporter,
    RayGaugeProgressReporter,
    _build_output_publisher,
    _initialize_ray,
    _reconcile_in_flight,
    _resolve_feature_manifest_dir,
    _resolve_output_path,
    _resolve_resume_state_path,
    _validate_output_arguments,
    discover_publication_manifest,
    publish_manifest,
)
from flare.runtime.ray import RayConfig

logger = logging.getLogger(__name__)


class DataPipeline:
    """Universal pipeline executor for DataPipelineSpec."""

    def __init__(self, spec: DataPipelineSpec):
        self._spec = spec

    @property
    def spec(self) -> DataPipelineSpec:
        return self._spec

    def run(
        self,
        *,
        output_format: str,
        output: str | None = None,
        input: str | None = None,
        hdfs_root: str | None = None,
        overwrite: bool = False,
        start_date: str | None = None,
        end_date: str | None = None,
        limit_files: int | None = None,
        publication_window_files: int | None = None,
        resume_state_path: str | None = None,
        iceberg_catalog_name: str | None = None,
        iceberg_catalog_type: str | None = None,
        iceberg_catalog_uri: str | None = None,
        iceberg_catalog_warehouse: str | None = None,
        iceberg_catalog_kwargs: dict[str, Any] | None = None,
        iceberg_writer_target_size_mib: int | None = None,
        iceberg_commit_layout_observation: bool = False,
        storage_options: dict[str, Any] | None = None,
        ray_num_cpus: int | None = None,
        ray_object_store_memory_mib: int | None = None,
        ray_config: RayConfig | None = None,
        gpu_mode: str = "auto",
        encoders_per_gpu: int | None = None,
        object_store_memory_fraction: float = 0.7,
        downstream_capacity_backpressure_ratio: float = 1.0,
        dynamic_output_queue_backpressure: bool = True,
        streaming_generator_max_blocks: int = 2,
        eager_free: bool = True,
        read_profile: bool = False,
        stage_profile: bool = False,
        fuse_parse_encode: bool = False,
        config: PipelineConfig | None = None,
        conf_dir: Path | None = None,
        model_conf_dir: Path | None = None,
    ) -> None:
        if output is None:
            output = self._spec.sink
        if input is None and hdfs_root is None:
            if not isinstance(self._spec.source, str):
                raise TypeError("Pipeline execution requires a source path")
            if self._spec.source.startswith("hdfs://"):
                hdfs_root = self._spec.source
            else:
                input = self._spec.source
        if input is not None and hdfs_root is not None:
            raise ValueError("input and hdfs_root are mutually exclusive")
        if hdfs_root is not None and (start_date is None or end_date is None):
            raise ValueError("HDFS execution requires start_date and end_date")
        if output is None:
            raise ValueError(
                "Pipeline execution requires an output destination or sink"
            )

        args = SimpleNamespace(
            output=output,
            output_format=output_format,
            input=input,
            hdfs_root=hdfs_root,
            overwrite=overwrite,
            start_date=start_date,
            end_date=end_date,
            limit_files=limit_files,
            publication_window_files=publication_window_files,
            resume_state_path=resume_state_path,
            iceberg_catalog_name=iceberg_catalog_name,
            iceberg_catalog_type=iceberg_catalog_type,
            iceberg_catalog_uri=iceberg_catalog_uri,
            iceberg_catalog_warehouse=iceberg_catalog_warehouse,
            iceberg_writer_target_size_mib=iceberg_writer_target_size_mib,
            iceberg_commit_layout_observation=iceberg_commit_layout_observation,
            ray_num_cpus=ray_num_cpus,
            ray_object_store_memory_mib=ray_object_store_memory_mib,
            ray=ray_config,
        )

        storage_options = {} if storage_options is None else storage_options
        _validate_output_arguments(args, storage_options)
        args.output = _resolve_output_path(args.output, args.output_format)
        args.resume_state_path = _resolve_resume_state_path(args)

        if iceberg_catalog_kwargs is None:
            iceberg_catalog_kwargs = build_iceberg_catalog_kwargs(
                name=args.iceberg_catalog_name,
                catalog_type=args.iceberg_catalog_type,
                uri=args.iceberg_catalog_uri,
                warehouse=args.iceberg_catalog_warehouse,
            )
        from flare.data.connectors.adapters.iceberg.access import (
            ConfiguredIcebergCatalogProvider,
        )
        from flare.io.aws import get_pyiceberg_s3_kwargs

        if args.output_format == "iceberg":
            options = dict(iceberg_catalog_kwargs)
            options.update(get_pyiceberg_s3_kwargs(storage_options))
            catalog_provider = ConfiguredIcebergCatalogProvider(options)
        else:
            catalog_provider = None

        state_manager = ResumeStateManager(args.resume_state_path, storage_options)
        if args.overwrite:
            state_manager.clear()
        else:
            _, in_flight = state_manager.load()
            if in_flight:
                _reconcile_in_flight(
                    args,
                    state_manager,
                    in_flight,
                    catalog_provider,
                )

        manifest = discover_publication_manifest(args, state_manager)
        if manifest is None:
            return
        if not ray.is_initialized():
            _initialize_ray(args)

        DataContext.get_current().enable_progress_bars = False

        cluster_resources = ray.cluster_resources()
        capacity = RayClusterCapacity.from_mapping(cluster_resources)

        gpu_memory_bytes = 0
        if capacity.gpu_count > 0 and gpu_mode != "disable":
            gpu_memory_bytes = HardwareProfiler()._detect_gpu()[1]

        pipeline_config = config
        if pipeline_config is None:
            pipeline_config = PipelineConfig.create(
                cluster_cpus=capacity.cpu_count,
                cluster_memory_bytes=capacity.heap_bytes,
                object_store_capacity_bytes=capacity.object_store_bytes,
                n_gpus=capacity.gpu_count,
                gpu_memory_bytes=gpu_memory_bytes,
                gpu_mode=gpu_mode,
                encoders_per_gpu=encoders_per_gpu,
                object_store_memory_fraction=object_store_memory_fraction,
                downstream_capacity_backpressure_ratio=downstream_capacity_backpressure_ratio,
                dynamic_output_queue_backpressure=dynamic_output_queue_backpressure,
                streaming_generator_max_blocks=streaming_generator_max_blocks,
                eager_free=eager_free,
                read_profile=read_profile,
                stage_profile=stage_profile,
                fuse_parse_encode=fuse_parse_encode,
                read_task_memory_bytes=self._spec.stage_config.source_block_size_bytes
                * 2
                if self._spec.stage_config.source_block_size_bytes
                else 256 * 1024 * 1024,
                source_csv_block_size_bytes=self._spec.stage_config.source_block_size_bytes,
                target_min_block_size_bytes=self._spec.stage_config.target_block_size_bytes,
                read_concurrency=self._spec.stage_config.read_concurrency,
                write_concurrency=self._spec.stage_config.write_concurrency,
                parse_actor_max_size=self._spec.stage_config.parse_concurrency,
                encode_actor_max_size=self._spec.stage_config.encode_concurrency,
            )

        artifact = self._spec.metadata.get("artifact")
        pipeline = RayDataPipeline(
            config=pipeline_config,
            conf_dir=str(conf_dir) if conf_dir else None,
            model_conf_dir=str(model_conf_dir) if model_conf_dir else None,
            parser_contract=self._spec.parser,
            artifact=artifact,
        )

        output_publisher = _build_output_publisher(
            args,
            catalog_provider,
            pipeline,
        )
        pipeline.write_feature_manifest(
            Path(_resolve_feature_manifest_dir(args.output, args.output_format))
        )

        reporter = CompositeProgressReporter(
            [RayGaugeProgressReporter(), LoggingProgressReporter(logger)]
        )
        publish_manifest(
            manifest,
            args,
            state_manager,
            pipeline,
            output_publisher,
            reporter,
        )


__all__ = [
    "DataPipeline",
    "DataPipelineBuilder",
    "DataPipelineSpec",
    "SentinelConfig",
    "StageConfig",
]
