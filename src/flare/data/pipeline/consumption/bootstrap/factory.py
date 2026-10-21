from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flare.config.iceberg import IcebergSettings
from flare.data.connectors.adapters.file.source import resolve_file_paths
from flare.data.connectors.adapters.file.scan import FileListSourceScanAdapter
from flare.data.connectors.adapters.file.source import FileListSourceAdapter
from flare.data.connectors.adapters.iceberg.catalog import (
    build_iceberg_catalog_kwargs,
    normalize_iceberg_catalog_kwargs,
)
from flare.data.connectors.adapters.iceberg.modes import PinnedSnapshot
from flare.data.connectors.adapters.iceberg.source import (
    IcebergSourcePlanner,
    IcebergSourceReader,
)
from flare.data.connectors.adapters.iceberg.scan import IcebergSourceScanAdapter
from flare.data.connectors.adapters.parquet.range_reader import (
    ParquetInputFileReader,
)
from flare.data.connectors.adapters.parquet.source import ParquetSourceAdapter
from flare.data.connectors.adapters.vortex.source import VortexSourceAdapter
from flare.data.pipeline.consumption.source.contracts.source import (
    ControlledSourceScanAdapter,
    SourceFactory,
    SourceReader,
)


def iceberg_catalog_kwargs(options: Mapping[str, Any]) -> dict[str, Any]:
    configured = options.get("catalog_kwargs")
    if configured is not None:
        if not isinstance(configured, dict):
            raise TypeError("catalog_kwargs must be a dictionary")
        return normalize_iceberg_catalog_kwargs(configured)
    config_path = options.get("iceberg_config")
    if config_path is not None:
        if not isinstance(config_path, (str, Path)):
            raise TypeError("iceberg_config must be a path")
        if any(
            options.get(name) is not None
            for name in ("catalog_name", "catalog_type", "catalog_uri", "warehouse")
        ):
            raise ValueError(
                "iceberg_config cannot be combined with individual catalog arguments"
            )
        return normalize_iceberg_catalog_kwargs(
            IcebergSettings.from_file(Path(config_path)).to_pyiceberg_kwargs()
        )
    return build_iceberg_catalog_kwargs(
        name=options.get("catalog_name"),
        catalog_type=options.get("catalog_type"),
        uri=options.get("catalog_uri"),
        warehouse=options.get("warehouse"),
        extra={"token": options["catalog_token"]}
        if options.get("catalog_token") is not None
        else None,
    )


def _iceberg_source_configuration(options: Mapping[str, Any]) -> dict[str, Any]:
    table_identifier = options.get("table_identifier")
    if not isinstance(table_identifier, str) or not table_identifier:
        raise ValueError("manifest source requires table_identifier")
    projection = options.get("projection")
    if not isinstance(projection, (tuple, list)) or not projection:
        raise TypeError("manifest source requires a non-empty projection")
    catalog_kw = iceberg_catalog_kwargs(options)
    if options.get("iceberg_snapshot_id") is not None:
        mode = PinnedSnapshot(int(options["iceberg_snapshot_id"]))
    else:
        mode = options.get("mode")
    return {
        "table_identifier": table_identifier,
        "catalog_kwargs": catalog_kw,
        "projection": tuple(projection),
        "mode": mode,
    }


def iceberg_max_open_files(options: Mapping[str, Any]) -> int:
    configured = options.get("iceberg_max_open_files")
    value = options.get("source_read_concurrency") if configured is None else configured
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("iceberg_max_open_files must be a positive integer")
    return value


def _resolve_file_source_paths(
    options: Mapping[str, Any], format_id: str
) -> tuple[str, ...]:
    files = options.get("files")
    if files is not None:
        if isinstance(files, (list, tuple)):
            paths = tuple(str(f) for f in files if f)
            if paths:
                return paths
        raise ValueError(f"{format_id.capitalize()} Reader requires valid files list")
    dataset_path = options.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError(f"{format_id.capitalize()} Reader requires dataset_path")
    return resolve_file_paths(
        dataset_path,
        format_id,
        storage_options=options.get("file_storage_options"),
    )


@dataclass(frozen=True, slots=True)
class StandardSourceFactory:
    def create_scan_planner(
        self, options: Mapping[str, Any]
    ) -> ControlledSourceScanAdapter:
        kind = options.get("storage_format")
        if kind == "iceberg":
            return IcebergSourceScanAdapter(
                IcebergSourcePlanner(**_iceberg_source_configuration(options))
            )
        if kind in {"parquet", "vortex"}:
            reader = self.create_reader(options)
            if not isinstance(reader, FileListSourceAdapter):
                raise TypeError(
                    f"file source scan requires FileListSourceAdapter, got {type(reader).__name__}"
                )
            return FileListSourceScanAdapter(reader)
        raise ValueError(f"unsupported controlled source scan format: {kind!r}")

    def create_reader(self, options: Mapping[str, Any]) -> SourceReader:
        kind = options.get("storage_format")
        if kind == "iceberg":
            return IcebergSourceReader(
                **_iceberg_source_configuration(options),
                format_readers={
                    "parquet": ParquetInputFileReader(
                        max_open_files=iceberg_max_open_files(options)
                    )
                },
            )
        if kind not in {"parquet", "vortex"}:
            raise ValueError(f"unsupported manifest source format: {kind!r}")
        projection = options.get("projection")
        if not isinstance(projection, (tuple, list)) or not projection:
            raise TypeError("manifest source reader requires a non-empty projection")

        files = _resolve_file_source_paths(options, str(kind))
        if kind == "parquet":
            return ParquetSourceAdapter(
                files=files,
                projection=tuple(projection),
                storage_options=options.get("file_storage_options"),
            )
        record_counts = options.get("record_counts", {})
        return VortexSourceAdapter(
            files=files,
            projection=tuple(projection),
            record_counts=record_counts,
            storage_options=options.get("file_storage_options"),
        )


STANDARD_SOURCE_FACTORY: SourceFactory = StandardSourceFactory()
