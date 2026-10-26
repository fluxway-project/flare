from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from typing import Any

import pyarrow as pa

from flare.data.connectors.adapters.file.source import FileListSourceAdapter
from flare.data.connectors.adapters.parquet.range_reader import (
    count_parquet_records,
    iter_counted_parquet_range_batches,
    read_parquet_range,
    read_parquet_schema,
)
from flare.data.pipeline.consumption.source.contracts.model import SourceRecordRange


class ParquetSourceAdapter(FileListSourceAdapter):
    def __init__(
        self,
        *,
        files: Sequence[str],
        projection: Sequence[str],
        storage_options: Mapping[str, Any] | None = None,
        record_counts: Mapping[str, int] | None = None,
    ) -> None:
        options = dict(storage_options or {})
        self._storage_options = options
        counts = dict(record_counts or {})

        def count(path: str) -> int:
            if path in counts:
                return counts[path]
            return count_parquet_records(path, storage_options=options)

        super().__init__(
            format_id="parquet",
            files=files,
            projection=projection,
            range_reader=partial(read_parquet_range, storage_options=options),
            record_counter=count,
        )

    @property
    def schema(self) -> pa.Schema:
        schema = read_parquet_schema(
            self._files[0], storage_options=self._storage_options
        )
        return pa.schema(
            [schema.field(name) for name in self._projection], metadata=schema.metadata
        )

    def read_batches(
        self,
        record_range: SourceRecordRange,
        projection: Sequence[str],
        *,
        batch_size: int,
    ) -> Iterator[pa.RecordBatch]:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("file source batch_size must be a positive integer")
        if record_range.source_identity != self.source_identity:
            raise ValueError("record range belongs to another file-list source")
        if record_range.record_format != "parquet":
            raise ValueError("record range format does not match Parquet source")
        if tuple(projection) != self._projection:
            raise ValueError(
                "read projection does not match immutable source projection"
            )
        return (
            batch
            for batch, _ in iter_counted_parquet_range_batches(
                record_range.locator.canonical(),
                projection=self._projection,
                row_start=record_range.record_start,
                row_count=record_range.record_count,
                storage_options=self._storage_options,
                batch_size=batch_size,
            )
        )
