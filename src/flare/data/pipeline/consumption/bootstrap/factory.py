from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from flare.config.iceberg import IcebergSettings
from flare.data.connectors.adapters.file.scan import FileListSourceScanAdapter
from flare.data.connectors.adapters.file.source import resolve_file_paths
from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
    IcebergCatalogProvider,
)
from flare.data.connectors.adapters.iceberg.catalog import (
    build_iceberg_catalog_kwargs,
    normalize_iceberg_catalog_kwargs,
)
from flare.data.connectors.adapters.iceberg.modes import PinnedSnapshot
from flare.data.connectors.adapters.iceberg.scan import IcebergSourceScanAdapter
from flare.data.connectors.adapters.iceberg.source import (
    IcebergSourcePlanner,
    IcebergSourceReader,
)
from flare.data.connectors.adapters.parquet.range_reader import (
    ParquetInputFileReader,
)
from flare.data.connectors.adapters.parquet.source import ParquetSourceAdapter
from flare.data.connectors.adapters.vortex.source import VortexSourceAdapter
from flare.data.contracts.assets import AssetRef
from flare.data.pipeline.consumption.source.contracts.source import (
    ControlledSourceScanAdapter,
    SourceFactory,
    SourceReadSession,
)
from flare.data.pipeline.consumption.source.execution.materialization import (
    DEFAULT_SOURCE_READ_CONCURRENCY,
    validate_source_read_concurrency,
)
from flare.data.pipeline.consumption.source.execution.record_ranges import (
    RecordRangeReadSession,
)
from flare.data.schema.virtual_pad import ArrowVirtualPadBatchFactory


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


def _iceberg_source_configuration(
    options: Mapping[str, Any], provider: IcebergCatalogProvider
) -> dict[str, Any]:
    table_identifier = options.get("table_identifier")
    if not isinstance(table_identifier, str) or not table_identifier:
        raise ValueError("manifest source requires table_identifier")
    projection = options.get("projection")
    if not isinstance(projection, (tuple, list)) or not projection:
        raise TypeError("manifest source requires a non-empty projection")
    if options.get("iceberg_snapshot_id") is not None:
        mode = PinnedSnapshot(int(options["iceberg_snapshot_id"]))
    else:
        mode = options.get("mode")
    return {
        "table_identifier": table_identifier,
        "catalog_provider": provider,
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


_STANDARD_OPTION_NAMES = frozenset(
    {
        "projection",
        "source_read_concurrency",
        "pad_batch_factory",
        "dataset_path",
        "storage_format",
        "file_storage_options",
        "files",
        "record_counts",
        "catalog_name",
        "catalog_type",
        "catalog_uri",
        "catalog_token",
        "warehouse",
        "catalog_kwargs",
        "iceberg_config",
        "iceberg_snapshot_id",
        "mode",
        "table_identifier",
        "iceberg_max_open_files",
    }
)
_CATALOG_OPTION_NAMES = (
    "catalog_name",
    "catalog_type",
    "catalog_uri",
    "catalog_token",
    "warehouse",
    "catalog_kwargs",
    "iceberg_config",
)


@dataclass(frozen=True, slots=True)
class StandardSourceFactory:
    iceberg_catalog_provider: IcebergCatalogProvider | None = None

    def prepare_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(options, Mapping):
            raise TypeError("source options must be a mapping")
        unknown = set(options).difference(_STANDARD_OPTION_NAMES)
        if unknown:
            raise ValueError(
                f"standard source has unknown options: {sorted(unknown)!r}"
            )
        prepared = dict(options)
        kind = prepared.get("storage_format")
        if kind not in {"iceberg", "parquet", "vortex"}:
            raise ValueError(f"unsupported standard source format: {kind!r}")
        projection = prepared.get("projection")
        if not isinstance(projection, (tuple, list)) or not projection:
            raise TypeError("standard source requires a non-empty projection")
        prepared["projection"] = tuple(projection)
        prepared["source_read_concurrency"] = validate_source_read_concurrency(
            prepared.get("source_read_concurrency", DEFAULT_SOURCE_READ_CONCURRENCY)
        )
        pad_factory = prepared.get("pad_batch_factory")
        if pad_factory is not None and not callable(pad_factory):
            raise TypeError("pad_batch_factory must be callable")
        if kind == "iceberg":
            if self.iceberg_catalog_provider is not None and any(
                prepared.get(name) is not None for name in _CATALOG_OPTION_NAMES
            ):
                raise ValueError(
                    "injected catalog provider conflicts with catalog options"
                )
            config_path = prepared.get("iceberg_config")
            if config_path is not None:
                if not isinstance(config_path, (str, Path)):
                    raise TypeError("iceberg_config must be a path")
                resolved = Path(config_path).expanduser().resolve(strict=True)
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                prepared["iceberg_config"] = str(resolved)
        return MappingProxyType(prepared)

    def _catalog_provider(self, options: Mapping[str, Any]) -> IcebergCatalogProvider:
        if self.iceberg_catalog_provider is not None:
            return self.iceberg_catalog_provider
        return ConfiguredIcebergCatalogProvider(iceberg_catalog_kwargs(options))

    def _file_reader(
        self, options: Mapping[str, Any]
    ) -> ParquetSourceAdapter | VortexSourceAdapter:
        kind = options["storage_format"]
        files = _resolve_file_source_paths(options, kind)
        if kind == "parquet":
            return ParquetSourceAdapter(
                files=files,
                projection=options["projection"],
                storage_options=options.get("file_storage_options"),
            )
        return VortexSourceAdapter(
            files=files,
            projection=options["projection"],
            record_counts=options.get("record_counts", {}),
            storage_options=options.get("file_storage_options"),
        )

    def create_scan_planner(
        self, options: Mapping[str, Any]
    ) -> ControlledSourceScanAdapter:
        options = self.prepare_options(options)
        if options["storage_format"] == "iceberg":
            return IcebergSourceScanAdapter(
                IcebergSourcePlanner(
                    **_iceberg_source_configuration(
                        options, self._catalog_provider(options)
                    )
                )
            )
        return FileListSourceScanAdapter(self._file_reader(options))

    def create_reader(
        self, options: Mapping[str, Any], *, include_lineage: bool = False
    ) -> SourceReadSession:
        if not isinstance(include_lineage, bool):
            raise TypeError("include_lineage must be bool")
        options = self.prepare_options(options)
        asset = None
        if options["storage_format"] == "iceberg":
            provider = self._catalog_provider(options)
            reader = IcebergSourceReader(
                **_iceberg_source_configuration(options, provider),
                format_readers={
                    "parquet": ParquetInputFileReader(
                        max_open_files=iceberg_max_open_files(options)
                    )
                },
            )
            if include_lineage:
                asset = AssetRef(provider.identity, options["table_identifier"])
        else:
            reader = self._file_reader(options)
            if include_lineage:
                asset = AssetRef(
                    f"file-list:{options['storage_format']}", reader.source_identity
                )
        return RecordRangeReadSession(
            reader=reader,
            asset=asset,
            projection=options["projection"],
            max_concurrent_reads=options["source_read_concurrency"],
            pad_batch_factory=options.get(
                "pad_batch_factory", ArrowVirtualPadBatchFactory.create_batch
            ),
        )


STANDARD_SOURCE_FACTORY: SourceFactory = StandardSourceFactory()
