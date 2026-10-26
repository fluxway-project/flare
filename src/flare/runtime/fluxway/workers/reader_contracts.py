"""Format-neutral source plan used by reader configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from flare.data.schema.virtual_pad import (
    VirtualPadBatchFactory,
)
from flare.runtime.fluxway.source import WorkItemScheduledDatasetSourceDescriptor
from fluxway.reader.ingestion_layout import IngestionReadSession


@dataclass(frozen=True, slots=True)
class ReaderSource:
    """Compiled source plan; callers must construct it through the builder."""

    descriptor: WorkItemScheduledDatasetSourceDescriptor | None
    options: Mapping[str, Any]
    read_session: IngestionReadSession | None
    read_chunk_size: int | None
    pad_batch_factory: VirtualPadBatchFactory | None = None

    def __post_init__(self) -> None:
        if self.pad_batch_factory is not None and not callable(self.pad_batch_factory):
            raise TypeError("reader source requires a pad batch factory")
        if not isinstance(self.options, Mapping):
            raise TypeError("reader source options must be a mapping")
        assigned = self.descriptor is not None
        direct = self.read_session is not None
        if assigned == direct:
            raise ValueError("reader source must be assigned or direct, never both")
        if assigned:
            if self.read_chunk_size is not None:
                raise ValueError("assigned reader source cannot define read_chunk_size")
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
        if storage_options is not None and not isinstance(storage_options, Mapping):
            raise TypeError("assigned reader source storage_options must be a mapping")
        options: dict[str, Any] = (
            {} if storage_options is None else dict(storage_options)
        )
        if projection:
            options["projection"] = tuple(projection)
        if source_read_concurrency is not None:
            options["source_read_concurrency"] = source_read_concurrency
        return ReaderSource(
            descriptor,
            descriptor.source_factory.prepare_options(options),
            None,
            None,
        )
    if storage_options is not None:
        raise ValueError("direct reader source cannot define storage_options")
    return ReaderSource(None, MappingProxyType({}), read_session, read_chunk_size)


__all__ = ["ReaderSource", "build_reader_source"]
