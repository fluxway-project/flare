from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import pyarrow as pa

from flare.data.connectors.adapters.file.source import resolve_file_paths
from flare.data.connectors.adapters.parquet.source import ParquetSourceAdapter


class ParquetRangeBatchLoader:
    """Read static Parquet ranges and apply one caller-owned batch transform."""

    def __init__(
        self,
        *,
        dataset_path: str,
        projection: Sequence[str],
        batch_size: int,
        batch_transformer: Callable[[pa.Table], object],
        shard_index: int = 0,
        shard_count: int = 1,
        num_epochs: int = 1,
    ) -> None:
        if not isinstance(dataset_path, str) or not dataset_path:
            raise ValueError("dataset_path must be a non-empty string")
        if not projection:
            raise ValueError("projection must not be empty")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not callable(batch_transformer):
            raise TypeError("batch_transformer must be callable")
        for value, name in (
            (shard_index, "shard_index"),
            (shard_count, "shard_count"),
            (num_epochs, "num_epochs"),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if shard_count <= 0 or not 0 <= shard_index < shard_count:
            raise ValueError("shard_index must be within positive shard_count")
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        files = resolve_file_paths(dataset_path, "parquet")
        self._adapter = ParquetSourceAdapter(
            files=files,
            projection=tuple(projection),
        )
        self._projection = tuple(projection)
        self._batch_size = batch_size
        self._batch_transformer = batch_transformer
        self._shard_index = shard_index
        self._shard_count = shard_count
        self._num_epochs = num_epochs

    def __iter__(self) -> Iterator[object]:
        ranges = tuple(
            record_range
            for index, record_range in enumerate(self._adapter.record_ranges())
            if index % self._shard_count == self._shard_index
        )
        for _ in range(self._num_epochs):
            for record_range in ranges:
                for batch in self._adapter.read_batches(
                    record_range,
                    self._projection,
                    batch_size=self._batch_size,
                ):
                    yield self._batch_transformer(pa.Table.from_batches([batch]))


__all__ = ["ParquetRangeBatchLoader"]
