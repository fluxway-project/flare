from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from threading import Condition, RLock
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.fs import FileSystem

from flare.data.schema.feature_catalog import validate_storage_columns
from flare.io.aws import get_pyarrow_s3_fs

STREAMING_READ_BATCH_ROWS = 65_536


class _CountingRandomAccessFile:
    def __init__(self, source: Any) -> None:
        self._source = source
        self.bytes_read = 0

    @property
    def closed(self) -> bool:
        return self._source.closed

    def close(self) -> None:
        self._source.close()

    def read(self, size: int = -1) -> bytes:
        value = self._source.read(size)
        self.bytes_read += len(value)
        return value

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._source.seek(offset, whence)

    def tell(self) -> int:
        return self._source.tell()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False


class CountedParquetFile:
    def __init__(self, path: str, storage_options: Mapping[str, Any] | None) -> None:
        raw = _open_parquet_input_file(path, storage_options)
        self._initialize(raw)

    @classmethod
    def from_input_file(cls, input_file: Any) -> CountedParquetFile:
        instance = cls.__new__(cls)
        instance._initialize(input_file.open())
        return instance

    def _initialize(self, raw: Any) -> None:
        self._counter = _CountingRandomAccessFile(raw)
        self._stream = pa.PythonFile(self._counter, mode="r")
        self.parquet = pq.ParquetFile(self._stream)
        self._reported_bytes = 0
        self._closed = False

    @property
    def bytes_read(self) -> int:
        return self._counter.bytes_read

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.close()

    def consume_bytes_delta(self) -> int:
        current = self.bytes_read
        delta = current - self._reported_bytes
        self._reported_bytes = current
        return delta


class ParquetInputFileReadSession:
    def __init__(self, *, source_identity: str, max_open_files: int) -> None:
        if not source_identity:
            raise ValueError("source_identity must not be empty")
        if (
            not isinstance(max_open_files, int)
            or isinstance(max_open_files, bool)
            or max_open_files <= 0
        ):
            raise ValueError("max_open_files must be a positive integer")
        self._source_identity = source_identity
        self._max_open_files = max_open_files
        self._opened_files: OrderedDict[tuple[str, str], CountedParquetFile] = (
            OrderedDict()
        )
        self._file_locks: dict[tuple[str, str], RLock] = {}
        self._opening_files: set[tuple[str, str]] = set()
        self._active_reads: dict[tuple[str, str], int] = {}
        self._state = Condition()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def cache_key(self, revision_identity: str, locator: str) -> tuple[str, str, str]:
        if not revision_identity or not locator:
            raise ValueError("revision_identity and locator must not be empty")
        return self._source_identity, revision_identity, locator

    def ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Parquet input file read session is closed")

    def _acquire_source(
        self,
        revision_identity: str,
        locator: str,
        input_file_factory: Callable[[], Any],
    ) -> tuple[CountedParquetFile, RLock, bool]:
        key = (revision_identity, locator)
        while True:
            with self._state:
                self.ensure_open()
                source = self._opened_files.get(key)
                if source is not None:
                    self._opened_files.move_to_end(key)
                    lock = self._file_locks[key]
                    self._active_reads[key] = self._active_reads.get(key, 0) + 1
                    return source, lock, False
                if key in self._opening_files:
                    self._state.wait()
                    continue
                self._evict_idle_file_if_full()
                self._opening_files.add(key)
                self._active_reads[key] = self._active_reads.get(key, 0) + 1
                break
        try:
            source = CountedParquetFile.from_input_file(input_file_factory())
        except BaseException:
            with self._state:
                self._opening_files.remove(key)
                self._release_source_locked(key)
                self._state.notify_all()
            raise
        with self._state:
            self._opening_files.remove(key)
            self._opened_files[key] = source
            lock = RLock()
            self._file_locks[key] = lock
            self._state.notify_all()
            return source, lock, True

    def _release_source(self, key: tuple[str, str]) -> None:
        with self._state:
            self._release_source_locked(key)

    def _release_source_locked(self, key: tuple[str, str]) -> None:
        active = self._active_reads[key] - 1
        if active == 0:
            del self._active_reads[key]
            self._state.notify_all()
            return
        self._active_reads[key] = active

    def _evict_idle_file_if_full(self) -> None:
        if len(self._opened_files) + len(self._opening_files) < self._max_open_files:
            return
        for key, source in self._opened_files.items():
            lock = self._file_locks[key]
            if self._active_reads.get(key, 0) != 0:
                continue
            if not lock.acquire(blocking=False):
                continue
            try:
                source.close()
                del self._opened_files[key]
                del self._file_locks[key]
                return
            finally:
                lock.release()
        raise RuntimeError("Parquet input file session reached max_open_files")

    def read_batches(
        self,
        *,
        revision_identity: str,
        locator: str,
        input_file_factory: Callable[[], Any],
        projection: Sequence[str],
        row_start: int,
        row_count: int,
        batch_size: int = STREAMING_READ_BATCH_ROWS,
    ):
        source, lock, _opened = self._acquire_source(
            revision_identity,
            locator,
            input_file_factory,
        )
        key = (revision_identity, locator)
        try:
            with lock:
                for batch, _physical_bytes in iter_counted_parquet_file_range_batches(
                    source,
                    projection=projection,
                    row_start=row_start,
                    row_count=row_count,
                    batch_size=batch_size,
                ):
                    yield batch
        finally:
            self._release_source(key)

    def close(self) -> None:
        with self._state:
            if self._closed:
                return
            self._closed = True
            while self._active_reads:
                self._state.wait()
            if self._opening_files:
                raise RuntimeError("closed Parquet session still has opening files")
            opened = tuple(
                (self._file_locks[key], source)
                for key, source in self._opened_files.items()
            )
            self._opened_files.clear()
            self._file_locks.clear()
        for lock, source in opened:
            with lock:
                source.close()

    def __enter__(self) -> Self:
        self.ensure_open()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def iter_counted_parquet_file_range_batches(
    source: CountedParquetFile,
    *,
    projection: Sequence[str],
    row_start: int,
    row_count: int,
    batch_size: int = STREAMING_READ_BATCH_ROWS,
) -> Iterator[tuple[pa.RecordBatch, int]]:
    if source.closed:
        raise RuntimeError("counted Parquet file is closed")
    for batch in _iter_parquet_file_range_batches(
        source.parquet,
        projection=projection,
        row_start=row_start,
        row_count=row_count,
        batch_size=batch_size,
    ):
        yield batch, source.consume_bytes_delta()


def iter_counted_parquet_range_batches(
    path: str,
    *,
    projection: Sequence[str],
    row_start: int,
    row_count: int,
    storage_options: Mapping[str, Any] | None = None,
    batch_size: int = STREAMING_READ_BATCH_ROWS,
) -> Iterator[tuple[pa.RecordBatch, int]]:
    source = CountedParquetFile(path, storage_options)
    try:
        yield from iter_counted_parquet_file_range_batches(
            source,
            projection=projection,
            row_start=row_start,
            row_count=row_count,
            batch_size=batch_size,
        )
    finally:
        source.close()


def iter_parquet_range_batches(
    path: str,
    *,
    projection: Sequence[str],
    row_start: int,
    row_count: int,
    storage_options: Mapping[str, Any] | None = None,
    batch_size: int = STREAMING_READ_BATCH_ROWS,
) -> Iterator[pa.RecordBatch]:
    """Yield a range without collecting the selected row groups into a table."""
    return _iter_parquet_file_range_batches(
        _open_parquet_file(path, storage_options),
        projection=projection,
        row_start=row_start,
        row_count=row_count,
        batch_size=batch_size,
    )


def _iter_parquet_file_range_batches(
    parquet: pq.ParquetFile,
    *,
    projection: Sequence[str],
    row_start: int,
    row_count: int,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    requested = validate_storage_columns(projection)
    if row_start < 0 or row_count <= 0 or batch_size <= 0:
        raise ValueError("Parquet range and batch_size must be positive")
    missing = sorted(set(requested) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"Parquet source is missing columns: {missing}")
    offsets = [0]
    for row_group_index in range(parquet.metadata.num_row_groups):
        offsets.append(
            offsets[-1] + int(parquet.metadata.row_group(row_group_index).num_rows)
        )
    row_end = row_start + row_count
    if row_end > offsets[-1]:
        raise ValueError(
            f"record range exceeds source rows: start={row_start} count={row_count}"
        )
    first_group = bisect_right(offsets, row_start) - 1
    last_group = bisect_right(offsets, row_end - 1) - 1
    skip = row_start - offsets[first_group]
    remaining = row_count
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        row_groups=tuple(range(first_group, last_group + 1)),
        columns=tuple(requested),
        use_threads=True,
        use_pandas_metadata=False,
    ):
        if not isinstance(batch, pa.RecordBatch):
            raise TypeError("Parquet iter_batches must yield RecordBatch")
        if skip >= batch.num_rows:
            skip -= batch.num_rows
            continue
        start = skip
        take = min(batch.num_rows - start, remaining)
        selected = batch.slice(start, take)
        if tuple(selected.schema.names) != requested:
            raise ValueError("Parquet batch projection differs from requested columns")
        yield selected
        remaining -= take
        skip = 0
        if remaining == 0:
            return
    raise ValueError(
        f"record range exceeds source rows: start={row_start} count={row_count}"
    )


def read_parquet_range(
    path: str,
    *,
    projection: Sequence[str],
    row_start: int,
    row_count: int,
    storage_options: Mapping[str, Any] | None = None,
) -> pa.Table:
    return _read_parquet_file_range(
        _open_parquet_file(path, storage_options),
        projection=projection,
        row_start=row_start,
        row_count=row_count,
    )


def count_parquet_records(
    path: str,
    *,
    storage_options: Mapping[str, Any] | None = None,
) -> int:
    return int(_open_parquet_file(path, storage_options).metadata.num_rows)


def read_parquet_schema(
    path: str,
    *,
    storage_options: Mapping[str, Any] | None = None,
) -> pa.Schema:
    with _open_parquet_file(path, storage_options) as parquet:
        return parquet.schema_arrow


def read_parquet_input_file_range(
    input_file: Any,
    *,
    projection: Sequence[str],
    row_start: int,
    row_count: int,
) -> pa.Table:
    stream = input_file.open()
    try:
        return _read_parquet_file_range(
            pq.ParquetFile(stream),
            projection=projection,
            row_start=row_start,
            row_count=row_count,
        )
    finally:
        stream.close()


def iter_parquet_input_file_range_batches(
    input_file: Any,
    *,
    projection: Sequence[str],
    row_start: int,
    row_count: int,
    batch_size: int = STREAMING_READ_BATCH_ROWS,
):
    stream = input_file.open()
    parquet = pq.ParquetFile(stream)
    try:
        yield from _iter_parquet_file_range_batches(
            parquet,
            projection=projection,
            row_start=row_start,
            row_count=row_count,
            batch_size=batch_size,
        )
    finally:
        stream.close()


class ParquetInputFileReader:
    def __init__(self, *, max_open_files: int) -> None:
        if (
            not isinstance(max_open_files, int)
            or isinstance(max_open_files, bool)
            or max_open_files <= 0
        ):
            raise ValueError("max_open_files must be a positive integer")
        self._max_open_files = max_open_files

    def create_session(self, *, source_identity: str) -> ParquetInputFileReadSession:
        return ParquetInputFileReadSession(
            source_identity=source_identity,
            max_open_files=self._max_open_files,
        )


def _open_parquet_file(
    path: str, storage_options: Mapping[str, Any] | None
) -> pq.ParquetFile:
    kwargs: dict[str, Any] = {}
    path_to_open = path
    if path.startswith("s3://"):
        filesystem = get_pyarrow_s3_fs(dict(storage_options or {}))
        if filesystem is None:
            raise ValueError("S3 Parquet input requires complete storage_options")
        kwargs["filesystem"] = filesystem
        path_to_open = path[5:]
    return pq.ParquetFile(path_to_open, **kwargs)


def _open_parquet_input_file(
    path: str, storage_options: Mapping[str, Any] | None
) -> Any:
    if path.startswith("s3://"):
        filesystem = get_pyarrow_s3_fs(dict(storage_options or {}))
        if filesystem is None:
            raise ValueError("S3 Parquet input requires complete storage_options")
        return filesystem.open_input_file(path[5:])
    filesystem, resolved_path = FileSystem.from_uri(
        str(Path(path).expanduser().resolve())
    )
    return filesystem.open_input_file(resolved_path)


def _read_parquet_file_range(
    parquet: pq.ParquetFile,
    *,
    projection: Sequence[str],
    row_start: int,
    row_count: int,
) -> pa.Table:
    requested = validate_storage_columns(projection)
    missing = sorted(set(requested) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"Parquet source is missing columns: {missing}")
    offsets = [0]
    for row_group_index in range(parquet.metadata.num_row_groups):
        offsets.append(
            offsets[-1] + int(parquet.metadata.row_group(row_group_index).num_rows)
        )
    row_end = row_start + row_count
    if row_end > offsets[-1]:
        raise ValueError(
            f"record range exceeds source rows: start={row_start} count={row_count}"
        )
    first_group = bisect_right(offsets, row_start) - 1
    last_group = bisect_right(offsets, row_end - 1) - 1
    table = parquet.read_row_groups(
        list(range(first_group, last_group + 1)),
        columns=list(requested),
        use_threads=True,
        use_pandas_metadata=False,
    )
    local_start = row_start - offsets[first_group]
    table = table.slice(local_start, row_count)
    if tuple(table.column_names) != requested:
        raise ValueError(
            "Parquet range did not return the requested columns: "
            f"requested={requested!r} actual={tuple(table.column_names)!r}"
        )
    return table
