"""Format-neutral source plan used by reader configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from flare.data.schema.virtual_pad import (
    ArrowVirtualPadBatchFactory,
    VirtualPadBatchFactory,
)
from flare.runtime.fluxway.source import WorkItemScheduledDatasetSourceDescriptor
from fluxway.reader.ingestion_layout import IngestionReadSession

_SOURCE_OPTION_NAMES = frozenset(
    {
        "projection",
        "source_read_concurrency",
        "dataset_path",
        "storage_format",
        "file_storage_options",
        "catalog_name",
        "catalog_type",
        "catalog_uri",
        "catalog_token",
        "warehouse",
        "iceberg_config",
        "iceberg_snapshot_id",
        "mode",
        "table_identifier",
        "iceberg_max_open_files",
    }
)


@dataclass(frozen=True, slots=True)
class ReaderSource:
    """Compiled source plan; callers must construct it through the builder."""

    descriptor: WorkItemScheduledDatasetSourceDescriptor | None
    options: Mapping[str, Any]
    read_session: IngestionReadSession | None
    read_chunk_size: int | None
    pad_batch_factory: VirtualPadBatchFactory = ArrowVirtualPadBatchFactory.create_batch

    def __post_init__(self) -> None:
        if not callable(self.pad_batch_factory):
            raise TypeError("reader source requires a pad batch factory")
        assigned = self.descriptor is not None
        direct = self.read_session is not None
        if assigned == direct:
            raise ValueError("reader source must be assigned or direct, never both")
        if assigned:
            if self.read_chunk_size is not None:
                raise ValueError("assigned reader source cannot define read_chunk_size")
            missing = {"projection", "source_read_concurrency"}.difference(self.options)
            if missing:
                raise ValueError(
                    f"assigned reader source is missing options: {sorted(missing)!r}"
                )
            unknown = set(self.options).difference(_SOURCE_OPTION_NAMES)
            if unknown:
                raise ValueError(
                    f"assigned reader source has unknown options: {sorted(unknown)!r}"
                )
        else:
            if self.options:
                raise ValueError("direct reader source cannot define source options")
            if not callable(getattr(self.read_session, "read", None)):
                raise TypeError("direct reader source requires a read session")
            if not callable(getattr(self.read_session, "close", None)):
                raise TypeError("direct reader source requires a closable read session")
            if type(self.read_chunk_size) is not int or self.read_chunk_size <= 0:
                raise ValueError("direct read_chunk_size must be a positive integer")


def build_reader_source(
    *,
    descriptor: WorkItemScheduledDatasetSourceDescriptor | None = None,
    projection: Sequence[str] = (),
    source_read_concurrency: int | None = None,
    storage_options: Mapping[str, Any] | None = None,
    read_session: IngestionReadSession | None = None,
    read_chunk_size: int | None = None,
) -> ReaderSource:
    if (descriptor is None) == (read_session is None):
        raise ValueError("build_reader_source requires exactly one source mode")
    if descriptor is not None:
        if not projection:
            raise ValueError("assigned reader source requires a non-empty projection")
        if source_read_concurrency is None:
            raise ValueError("assigned reader source requires source_read_concurrency")
        if not isinstance(storage_options, Mapping):
            raise TypeError("assigned reader source storage_options must be a mapping")
        options = dict(storage_options)
        config_path = options.get("iceberg_config")
        if config_path is not None:
            if not isinstance(config_path, (str, Path)):
                raise TypeError("iceberg_config must be a path")
            resolved_config_path = Path(config_path).expanduser().resolve(strict=True)
            if not resolved_config_path.is_file():
                raise FileNotFoundError(resolved_config_path)
            options["iceberg_config"] = str(resolved_config_path)
        options.update(
            {
                "projection": tuple(projection),
                "source_read_concurrency": source_read_concurrency,
            }
        )
        return ReaderSource(
            descriptor,
            MappingProxyType(options),
            None,
            None,
        )
    if storage_options is not None:
        raise ValueError("direct reader source cannot define storage_options")
    return ReaderSource(None, MappingProxyType({}), read_session, read_chunk_size)


__all__ = ["ReaderSource", "build_reader_source"]
