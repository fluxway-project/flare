"""Reusable pipeline execution and publication utilities for Ray Data."""

from __future__ import annotations

import hashlib
import itertools
import logging
import time
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any

from flare.data.connectors.adapters.iceberg.recovery import reconcile_iceberg_in_flight
from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher
from flare.data.connectors.adapters.iceberg.table import (
    DEFAULT_WRITER_TARGET_SIZE_BYTES,
)
from flare.data.connectors.adapters.parquet.sink import ParquetDatasetPublisher
from flare.data.pipeline.production.builder.hdfs_source import iter_hdfs_file_paths
from flare.data.pipeline.production.builder.resume_state_manager import (
    ResumeStateManager,
)
from flare.data.pipeline.production.publication.contract import (
    PublicationPendingStore,
    PublicationProgressReporter,
    PublicationProtocol,
)
from flare.data.pipeline.production.publication.manifest import PublicationManifest
from flare.data.pipeline.production.publication.session import (
    ManifestExecutor,
    execute_publication,
)
from flare.data.pipeline.production.publication.window import (
    partition_publication_manifest,
)
from flare.data.pipeline.production.publish.resource_profile import (
    estimate_arrow_row_bytes,
)
from flare.runtime.ray import LocalRayInitOptions, initialize_ray_runtime

logger = logging.getLogger(__name__)


class RayGaugeProgressReporter:
    def __init__(self, metric_name: str = "dataset_pipeline_overall_progress"):
        from ray.util.metrics import Gauge

        self._gauge = Gauge(
            metric_name,
            description="Overall progress percentage of the dataset pipeline",
        )

    def update(
        self, current: int, total: int, rows_processed: int, elapsed_batch: float
    ) -> None:
        if total > 0:
            self._gauge.set((current / total) * 100)


class LoggingProgressReporter:
    def __init__(self, logger_instance: logging.Logger):
        self._logger = logger_instance

    def update(
        self, current: int, total: int, rows_processed: int, elapsed_batch: float
    ) -> None:
        progress_pct = (current / total) * 100 if total > 0 else None
        if rows_processed > 0 or elapsed_batch > 0.0:
            if progress_pct is None:
                self._logger.info(
                    "Batch [%d] complete: %d rows, %.1fs. Overall Progress: unknown",
                    current,
                    rows_processed,
                    elapsed_batch,
                )
            else:
                self._logger.info(
                    "Batch [%d/%d] complete: %d rows, %.1fs. Overall Progress: %.1f%%",
                    current,
                    total,
                    rows_processed,
                    elapsed_batch,
                    progress_pct,
                )


class CompositeProgressReporter:
    def __init__(self, reporters: list[PublicationProgressReporter]):
        self._reporters = reporters

    def update(
        self, current: int, total: int, rows_processed: int, elapsed_batch: float
    ) -> None:
        for r in self._reporters:
            r.update(current, total, rows_processed, elapsed_batch)


def _resolve_output_path(output: str, output_format: str) -> str:
    if output_format == "iceberg":
        if not output or "/" in output:
            raise ValueError("Iceberg output must be a non-empty table identifier")
        return output
    if output_format != "parquet":
        raise ValueError(f"unsupported output format: {output_format!r}")
    parsed = urllib.parse.urlparse(output)
    if parsed.scheme not in {"", "file"}:
        raise ValueError("Parquet output must be a local filesystem path")
    path = Path(parsed.path if parsed.scheme == "file" else output).resolve()
    if path == Path(path.anchor):
        raise ValueError("Parquet output must not be a filesystem root")
    return str(path)


def _validate_output_arguments(args: Any, storage_options: dict[str, Any]) -> None:
    if args.output_format == "iceberg":
        return
    if args.output_format != "parquet":
        raise ValueError(f"unsupported output format: {args.output_format!r}")
    iceberg_arguments = {
        "--iceberg-catalog-name": getattr(args, "iceberg_catalog_name", None),
        "--iceberg-catalog-type": getattr(args, "iceberg_catalog_type", None),
        "--iceberg-catalog-uri": getattr(args, "iceberg_catalog_uri", None),
        "--iceberg-catalog-warehouse": getattr(args, "iceberg_catalog_warehouse", None),
    }
    configured = [
        name for name, value in iceberg_arguments.items() if value is not None
    ]
    if configured:
        raise ValueError(
            "Parquet output does not accept Iceberg catalog arguments: "
            + ", ".join(configured)
        )
    if storage_options:
        raise ValueError("Parquet output does not accept S3 configuration")


def _resolve_writer_target_size_bytes(args: Any) -> int:
    writer_target_mib = getattr(args, "iceberg_writer_target_size_mib", None)
    if writer_target_mib is not None:
        if writer_target_mib <= 0:
            raise ValueError(
                "--iceberg-writer-target-size-mib must be positive, "
                f"got {writer_target_mib}"
            )
        return writer_target_mib * 1024 * 1024
    return DEFAULT_WRITER_TARGET_SIZE_BYTES


def _build_output_publisher(
    args: Any,
    catalog_kwargs: dict[str, Any],
    storage_options: dict[str, Any],
    pipeline: Any,
) -> Any:
    if args.output_format == "parquet":
        target_size_bytes = _resolve_writer_target_size_bytes(args)
        row_bytes = estimate_arrow_row_bytes(pipeline.output_schema)
        return ParquetDatasetPublisher(
            args.output,
            max_rows_per_file=max(1, target_size_bytes // row_bytes),
            overwrite=getattr(args, "overwrite", False),
        )
    if args.output_format != "iceberg":
        raise ValueError(f"unsupported output format: {args.output_format!r}")
    return IcebergSnapshotPublisher(
        table_identifier=args.output,
        catalog_kwargs=catalog_kwargs,
        write_task_num_cpus=pipeline.config.write_task_num_cpus,
        write_concurrency=pipeline.config.write_concurrency,
        storage_options=storage_options,
        writer_target_size_bytes=_resolve_writer_target_size_bytes(args),
        observe_file_layout=args.iceberg_commit_layout_observation,
    )


def _resolve_resume_state_path(args: Any) -> str:
    if getattr(args, "resume_state_path", None):
        return args.resume_state_path
    identity = f"{args.output_format}:{args.output}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:24]
    return str((Path("data") / "_resume" / f"{digest}.json").resolve())


def _resolve_feature_manifest_dir(output_uri: str, output_format: str) -> str:
    if output_format == "parquet":
        return str(Path(output_uri).resolve())
    if output_format != "iceberg":
        raise ValueError(f"unsupported output format: {output_format!r}")
    parsed = urllib.parse.urlparse(output_uri)
    if parsed.scheme and parsed.scheme not in {"file"}:
        raise ValueError("Iceberg output must be a table identifier")
    return str((Path("data") / "_manifests" / output_uri).resolve())


def _iter_local_input_paths(input_path: str):
    root = Path(input_path).resolve()
    if root.is_file():
        yield str(root)
        return
    elif root.is_dir():
        suffixes = {".txt", ".zst", ".zstd", ".csv", ".parquet"}
        for path in root.rglob("*"):
            if path.is_file() and any(
                str(path).endswith(suffix) for suffix in suffixes
            ):
                yield str(path)
        return
    else:
        raise FileNotFoundError(f"Local input path does not exist: {input_path}")


def _bind_manifest_terminal(output: Any, manifest: PublicationManifest) -> Any:
    protocol = output.publication_protocol
    if protocol is PublicationProtocol.ICEBERG_SNAPSHOT:
        return output.bind(manifest.identity)
    if protocol is PublicationProtocol.NON_TRANSACTIONAL:
        return output
    raise ValueError(f"unsupported publication protocol: {protocol!r}")


def _build_local_ray_init_kwargs(args: Any) -> LocalRayInitOptions:
    kwargs: LocalRayInitOptions = {}
    ray_num_cpus = getattr(args, "ray_num_cpus", None)
    if ray_num_cpus is not None:
        if ray_num_cpus <= 0:
            raise ValueError(f"--ray-num-cpus must be positive, got {ray_num_cpus}")
        kwargs["num_cpus"] = ray_num_cpus
    ray_object_store_memory_mib = getattr(args, "ray_object_store_memory_mib", None)
    if ray_object_store_memory_mib is not None:
        if ray_object_store_memory_mib <= 0:
            raise ValueError(
                "--ray-object-store-memory-mib must be positive, "
                f"got {ray_object_store_memory_mib}"
            )
        kwargs["object_store_memory"] = ray_object_store_memory_mib * 1024 * 1024
    return kwargs


def _initialize_ray(args: Any) -> None:
    local_kwargs = _build_local_ray_init_kwargs(args)
    ray_config = getattr(args, "ray", None)
    address = getattr(ray_config, "ray_address", "auto") if ray_config else "auto"
    node_ip_address = (
        getattr(ray_config, "ray_node_ip_address", None) if ray_config else None
    )
    namespace = (
        getattr(ray_config, "ray_namespace", "flare-data-pipeline")
        if ray_config
        else "flare-data-pipeline"
    )
    initialize_ray_runtime(
        address=address,
        node_ip_address=node_ip_address,
        namespace=namespace,
        local_init_kwargs=local_kwargs,
    )


def _reconcile_in_flight(
    args: Any,
    state_manager: ResumeStateManager,
    in_flight: Any,
    iceberg_catalog_kwargs: dict[str, Any],
) -> None:
    if args.output_format != "iceberg":
        raise RuntimeError(
            "Cannot reconcile in-flight publications for output format: "
            f"{args.output_format}"
        )
    reconcile_iceberg_in_flight(
        state_manager,
        in_flight,
        catalog_kwargs=iceberg_catalog_kwargs,
        table_identifier=args.output,
    )
    logger.info("Reconciled %d in-flight publications.", len(in_flight))


def _run_local_mode(
    args: Any,
    pipeline: Any,
    output: Any,
) -> None:
    """Publish one finite local source manifest through one execution."""
    input_path = getattr(args, "input", None)
    if input_path is None:
        raise ValueError("local mode requires input path")
    logger.info("Local mode: processing %s", input_path)
    file_paths = _iter_local_input_paths(input_path)
    limit_files = getattr(args, "limit_files", None)
    if limit_files is not None:
        file_paths = itertools.islice(file_paths, limit_files)
    manifest = PublicationManifest.from_files(file_paths)
    result = pipeline.execute(
        manifest.files,
        _bind_manifest_terminal(output, manifest),
    )
    if result.total_rows == 0:
        raise FileNotFoundError(f"No supported local input files under: {input_path}")
    logger.info(
        "Local publication complete: %d files, %d rows.",
        len(manifest.files),
        result.total_rows,
    )


def _run_hdfs_mode(
    args: Any,
    state_manager: PublicationPendingStore,
    pipeline: ManifestExecutor,
    output: Any,
    progress_reporter: PublicationProgressReporter,
) -> None:
    """Publish one finite HDFS source manifest through one Ray execution."""
    hdfs_root = getattr(args, "hdfs_root", None)
    if hdfs_root is None:
        raise RuntimeError("HDFS mode requires hdfs_root")
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    logger.info("Enumerating HDFS partitions %s → %s ...", start, end)

    discovered_paths = iter_hdfs_file_paths(hdfs_root, start, end)
    limit_files = getattr(args, "limit_files", None)
    if limit_files is not None:
        discovered_paths = itertools.islice(discovered_paths, limit_files)
        logger.info("Limiting streaming discovery to %d files.", limit_files)

    manifest = PublicationManifest.from_files(
        path for path in discovered_paths if state_manager.is_pending(path)
    )
    publication_window_files = getattr(args, "publication_window_files", None)
    if publication_window_files is None:
        raise RuntimeError("HDFS publication window size was not resolved")
    windows = partition_publication_manifest(manifest, publication_window_files)
    completed_files = 0
    total_rows = 0
    for index, window in enumerate(windows, start=1):
        started = time.perf_counter()
        result = execute_publication(
            manifest=window,
            executor=pipeline,
            terminal=_bind_manifest_terminal(output, window),
            protocol=output.publication_protocol,
            lease_store=state_manager,
        )
        completed_files += len(window.files)
        total_rows += result.total_rows
        progress_reporter.update(
            current=completed_files,
            total=len(manifest.files),
            rows_processed=result.total_rows,
            elapsed_batch=time.perf_counter() - started,
        )
        logger.info(
            "HDFS publication window %d/%d complete: %d files, %d rows, publication=%s.",
            index,
            len(windows),
            len(window.files),
            result.total_rows,
            window.identity.value,
        )
    logger.info(
        "HDFS publication complete: %d files, %d rows, windows=%d.",
        len(manifest.files),
        total_rows,
        len(windows),
    )


__all__ = [
    "CompositeProgressReporter",
    "LoggingProgressReporter",
    "RayGaugeProgressReporter",
    "_bind_manifest_terminal",
    "_build_local_ray_init_kwargs",
    "_build_output_publisher",
    "_initialize_ray",
    "_iter_local_input_paths",
    "_reconcile_in_flight",
    "_resolve_feature_manifest_dir",
    "_resolve_output_path",
    "_resolve_resume_state_path",
    "_resolve_writer_target_size_bytes",
    "_run_hdfs_mode",
    "_run_local_mode",
    "_validate_output_arguments",
]
