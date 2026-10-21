"""flare.data.run — Universal Recipe Loader and Ray Data streaming pipeline entry point.

Usage:
    # Generic Recipe execution:
    python -m flare.data.run \\
        --recipe recipes.my_recipe.data.pipeline.pipeline.MyPipeline \\
        --input dataset/test/data/test_data_batch.csv \\
        --output data/local_train \\
        --output-format parquet
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tyro

from flare.config.iceberg import IcebergSettings
from flare.data.contracts.pipeline import DataPipelineBuilder, StageConfig
from flare.data.pipeline.production.pipeline import DataPipeline
from flare.data.pipeline.production.processing.parser_contract import ParserContract
from flare.data.pipeline.production.publish.runner import (
    _validate_output_arguments,
)
from flare.io.aws import S3Config
from flare.runtime.ray import RayConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    output: str
    """Iceberg table identifier or local Parquet dataset directory"""
    output_format: str
    """Persistent publication format: 'iceberg' or 'parquet'"""

    recipe: str | None = None
    """Dynamic recipe builder class path, e.g. 'recipes.<business>.data.pipeline.Builder'"""
    hdfs_root: str | None = None
    """HDFS root path containing ds=*/hour=*/part-*.zst partitions"""
    input: str | None = None
    """Local plain-text file (no decompression, for testing); mutually exclusive with hdfs_root"""

    overwrite: bool = False
    resume_state_path: str | None = None
    start_date: str = "2025-12-03"
    end_date: str = "2026-01-07"
    conf_dir: Path | None = None
    model_conf_dir: Path | None = None
    """Directory containing model_config.json."""
    parser_contract_module: str | None = None
    read_concurrency: int | None = None
    read_task_num_cpus: float | None = None
    write_concurrency: int | None = None
    write_task_num_cpus: float | None = None
    parse_actor_max_size: int | None = None
    encode_actor_max_size: int | None = None
    actor_max_tasks_in_flight: int = 2
    gpu_mode: str = "auto"
    encoders_per_gpu: int | None = None
    object_store_memory_fraction: float = 0.7
    downstream_capacity_backpressure_ratio: float = 1.0
    dynamic_output_queue_backpressure: bool = True
    streaming_generator_max_blocks: int = 2
    eager_free: bool = True
    read_profile: bool = False
    stage_profile: bool = False
    fuse_parse_encode: bool = False
    read_task_memory_mib: int | None = None
    ray_num_cpus: int | None = None
    ray_object_store_memory_mib: int | None = None
    source_csv_block_size_mib: int | None = None
    target_min_block_size_mib: int | None = None
    target_max_block_size_mib: int | None = None
    encoded_batch_target_mib: int = 64
    datasource_prefetch_max_buffers: int = 2
    datasource_prefetch_buffer_size_mib: int | None = None
    source_files_per_task: int = 1
    iceberg_writer_target_size_mib: int | None = None
    iceberg_commit_layout_observation: bool = False
    """Read committed Iceberg manifests after each publication for file-layout metrics."""
    iceberg_catalog_name: str | None = None
    iceberg_catalog_type: str | None = None
    iceberg_catalog_uri: str | None = None
    iceberg_catalog_warehouse: str | None = None
    iceberg_config: Path | None = None
    limit_files: int | None = None
    publication_window_files: int | None = None
    ray: RayConfig = field(default_factory=RayConfig)
    s3: S3Config = field(default_factory=S3Config)

    def __post_init__(self) -> None:
        if self.input is not None and self.hdfs_root is not None:
            raise ValueError("--input and --hdfs-root are mutually exclusive")
        if self.input is None and self.hdfs_root is None:
            raise ValueError("one of --input or --hdfs-root is required")
        if self.recipe is None and self.conf_dir is None:
            raise ValueError("--conf-dir is required when recipe is not specified")
        if self.recipe is None and self.model_conf_dir is None:
            raise ValueError(
                "--model-conf-dir is required when recipe is not specified"
            )
        if self.output_format not in {"iceberg", "parquet"}:
            raise ValueError(
                f"--output-format must be 'iceberg' or 'parquet', got {self.output_format!r}"
            )
        if self.gpu_mode not in {"auto", "force", "disable"}:
            raise ValueError(
                f"--gpu-mode must be 'auto', 'force', or 'disable', got {self.gpu_mode!r}"
            )
        if self.hdfs_root is not None and self.publication_window_files is None:
            raise ValueError("--publication-window-files is required for HDFS mode")
        if (
            self.publication_window_files is not None
            and self.publication_window_files <= 0
        ):
            raise ValueError(
                "--publication-window-files must be positive, "
                f"got {self.publication_window_files}"
            )


def load_recipe_builder(recipe_path: str) -> type[DataPipelineBuilder]:
    """Dynamically load a DataPipelineBuilder class from a module path."""
    if ":" in recipe_path:
        module_name, class_name = recipe_path.split(":", 1)
    else:
        parts = recipe_path.rsplit(".", 1)
        if len(parts) == 2:
            module_name, class_name = parts
        else:
            raise ValueError(f"Invalid recipe path format: {recipe_path!r}")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def _resolve_model_conf_dir(args: Any) -> Path:
    if getattr(args, "model_conf_dir", None) is None:
        raise ValueError("--model-conf-dir is required")
    return args.model_conf_dir


def _load_parser_contract(args: Any) -> ParserContract:
    contract_module = getattr(args, "parser_contract_module", None)
    if not contract_module:
        raise ValueError(
            "--parser-contract-module is required; the application recipe must "
            "provide the source layout and metadata maps"
        )
    module_name, separator, attribute_name = contract_module.rpartition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("parser contract module must use 'module:attribute' notation")
    module = importlib.import_module(module_name)
    contract = getattr(module, attribute_name)
    if not isinstance(contract, ParserContract):
        raise TypeError(
            f"{contract_module} must resolve to flare.data.pipeline.production.processing.ParserContract"
        )
    return contract


def main() -> None:
    args = tyro.cli(RunConfig)
    iceberg_settings = (
        IcebergSettings.from_file(args.iceberg_config)
        if args.iceberg_config is not None
        else None
    )
    if iceberg_settings is not None and args.output_format != "iceberg":
        raise ValueError("--iceberg-config requires --output-format iceberg")
    if iceberg_settings is not None and any(
        value is not None
        for value in (
            args.iceberg_catalog_name,
            args.iceberg_catalog_type,
            args.iceberg_catalog_uri,
            args.iceberg_catalog_warehouse,
        )
    ):
        raise ValueError(
            "--iceberg-config cannot be combined with individual Iceberg catalog arguments"
        )
    storage_options = (
        dict(iceberg_settings.object_store.options)
        if iceberg_settings is not None and args.output_format == "iceberg"
        else args.s3.to_options()
        if args.output_format == "iceberg"
        else {}
    )
    _validate_output_arguments(args, storage_options)

    if args.recipe:
        recipe_cls = load_recipe_builder(args.recipe)
        builder = recipe_cls()
        stage_config = StageConfig(
            source_block_size_bytes=(
                args.source_csv_block_size_mib * 1024 * 1024
                if args.source_csv_block_size_mib
                else 64 * 1024 * 1024
            ),
            target_block_size_bytes=(
                args.target_min_block_size_mib * 1024 * 1024
                if args.target_min_block_size_mib
                else 128 * 1024 * 1024
            ),
            read_concurrency=args.read_concurrency,
            write_concurrency=args.write_concurrency or 1,
            parse_concurrency=args.parse_actor_max_size,
            encode_concurrency=args.encode_actor_max_size,
        )
        spec = builder.build_pipeline_spec(
            source=args.input or args.hdfs_root,
            sink=args.output,
            stage_config=stage_config,
        )
        pipeline = DataPipeline(spec)
        pipeline.run(
            output_format=args.output_format,
            overwrite=args.overwrite,
            resume_state_path=args.resume_state_path,
            start_date=args.start_date,
            end_date=args.end_date,
            publication_window_files=args.publication_window_files,
            limit_files=args.limit_files,
            gpu_mode=args.gpu_mode,
            ray_config=args.ray,
            storage_options=storage_options,
            iceberg_catalog_name=args.iceberg_catalog_name,
            iceberg_catalog_type=args.iceberg_catalog_type,
            iceberg_catalog_uri=args.iceberg_catalog_uri,
            iceberg_catalog_warehouse=args.iceberg_catalog_warehouse,
            iceberg_catalog_kwargs=(
                iceberg_settings.to_pyiceberg_kwargs()
                if iceberg_settings is not None
                else None
            ),
            iceberg_writer_target_size_mib=args.iceberg_writer_target_size_mib,
            iceberg_commit_layout_observation=args.iceberg_commit_layout_observation,
        )
        logger.info("Pipeline execution complete.")
        return

    raise ValueError(
        "--recipe is required for the generic Flare data entry point; "
        "business recipes must be invoked through their own recipe entry point"
    )


if __name__ == "__main__":
    main()
