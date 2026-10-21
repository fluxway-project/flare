"""flare.data.pipeline.production.publish.datasource — Ray Data Datasource: HDFS zstd → raw text lines.

Stage-split design (Datasource → ParseUDF → EncodeUDF → Consumer):
    WebHDFS download → zstd stream decompress → yield Arrow blocks of raw lines

Each ReadTask:
    - Streams one deterministic striped shard of HDFS files
    - Decompresses + chunks via PyArrow CSV byte-sized blocks
    - Yields Arrow Tables with schema: {line: large_string}, one row per text line

The line-parse step (lines → raw_features) lives in ParseUDF, a separate
``ray.data.map_batches`` stage with its own actor pool. This decouples IO
parallelism (one task per file) from CPU parsing throughput (parse pool sized
by ``StageTopologyPlanner.n_parse``).
"""

from __future__ import annotations

import io
import logging
import os
import queue
import threading
import time
import typing
from collections.abc import Sequence
from functools import partial

import pyarrow as pa
from ray.data import Datasource, ReadTask
from ray.data.block import BlockMetadata

from flare.data.pipeline.production.builder.hdfs_source import (
    ResilientWebHdfsReader,
    _base_url,
    _full_path,
)
from flare.data.pipeline.production.publish.partitioning import (
    PARTITION_COLUMNS,
    source_partition_values,
)
from flare.data.pipeline.production.publish.read_profile import (
    ReadProfile,
    report_read_profile,
)

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_S = 30
_READ_TIMEOUT_S = 600


def _url_resolver(hdfs_path: str) -> str:
    """Construct WebHDFS OPEN URL for a given HDFS path."""
    full = _full_path(hdfs_path)
    return f"{_base_url(full)}&path={full}&op=OPEN"


class HdfsZstdDatasource(Datasource):
    """Ray Data Datasource: HDFS zstd → decompress → yield raw text lines.

    Each task processes its files sequentially because an individual zstd stream
    is not byte-range splittable.
    Downstream stages (``ParseUDF`` + ``EncodeUDF``) run as independent
    actor pools so IO bandwidth, parse CPU, and encode GPU all scale
    independently.
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        *,
        source_csv_block_size_bytes: int,
        read_planner_block_bytes: int,
        datasource_prefetch_max_buffers: int,
        datasource_prefetch_buffer_size_bytes: int,
        source_files_per_task: int,
        read_profile: bool = False,
        read_profile_collector: typing.Any | None = None,
    ):
        self._file_paths = tuple(file_paths)
        if source_csv_block_size_bytes <= 0:
            raise ValueError(
                f"source_csv_block_size_bytes must be positive, got {source_csv_block_size_bytes}"
            )
        if read_planner_block_bytes <= 0:
            raise ValueError(
                f"read_planner_block_bytes must be positive, got {read_planner_block_bytes}"
            )
        if datasource_prefetch_max_buffers <= 0:
            raise ValueError(
                f"datasource_prefetch_max_buffers must be positive, got {datasource_prefetch_max_buffers}"
            )
        if datasource_prefetch_buffer_size_bytes <= 0:
            raise ValueError(
                "datasource_prefetch_buffer_size_bytes must be positive, "
                f"got {datasource_prefetch_buffer_size_bytes}"
            )
        if source_files_per_task <= 0:
            raise ValueError(
                f"source_files_per_task must be positive, got {source_files_per_task}"
            )
        if not isinstance(read_profile, bool):
            raise TypeError(
                f"read_profile must be bool, got {type(read_profile).__name__}"
            )
        if read_profile != (read_profile_collector is not None):
            raise ValueError(
                "read profile and read profile collector must be configured together"
            )
        self._source_csv_block_size_bytes = source_csv_block_size_bytes
        self._read_planner_block_bytes = read_planner_block_bytes
        self._datasource_prefetch_max_buffers = datasource_prefetch_max_buffers
        self._datasource_prefetch_buffer_size_bytes = (
            datasource_prefetch_buffer_size_bytes
        )
        self._source_files_per_task = source_files_per_task
        self._read_profile = read_profile
        self._read_profile_collector = read_profile_collector

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
        data_context: typing.Any | None = None,
    ) -> list[ReadTask]:
        """Create a bounded, deterministic set of streaming source tasks."""
        if parallelism <= 0:
            raise ValueError(f"parallelism must be positive, got {parallelism}")
        shards = _plan_source_task_file_groups(
            self._file_paths,
            self._source_files_per_task,
        )
        tasks = []

        for paths in shards:
            metadata = BlockMetadata(
                num_rows=None,
                size_bytes=self._read_planner_block_bytes * len(paths),
                input_files=paths,
                exec_stats=None,
            )

            def read_fn(
                shard_paths=paths,
                source_csv_block_size_bytes=self._source_csv_block_size_bytes,
                p_max=self._datasource_prefetch_max_buffers,
                prefetch_buffer_size=self._datasource_prefetch_buffer_size_bytes,
                read_profile=self._read_profile,
                read_profile_collector=self._read_profile_collector,
            ):
                factory = partial(
                    default_prefetch_reader_factory,
                    buffer_size=prefetch_buffer_size,
                    max_buffers=p_max,
                )
                for path in shard_paths:
                    yield from _read_decompress_file(
                        path,
                        source_csv_block_size_bytes,
                        reader_factory=factory,
                        read_profile=read_profile,
                        read_profile_collector=read_profile_collector,
                    )

            tasks.append(ReadTask(read_fn, metadata))
        return tasks

    def estimate_inmemory_data_size(self) -> int | None:
        return None


def _plan_source_task_file_groups(
    file_paths: tuple[str, ...], files_per_task: int
) -> tuple[tuple[str, ...], ...]:
    """Create bounded source tasks without changing publication membership."""
    if files_per_task <= 0:
        raise ValueError(f"files_per_task must be positive, got {files_per_task}")
    if not file_paths:
        raise ValueError("HdfsZstdDatasource requires at least one file")
    groups = tuple(
        file_paths[start : start + files_per_task]
        for start in range(0, len(file_paths), files_per_task)
    )
    if any(not group for group in groups):
        raise RuntimeError("source task planner produced an empty task")
    if tuple(path for group in groups for path in group) != file_paths:
        raise RuntimeError("source task planner did not preserve input membership")
    return groups


class PrefetchReader(io.RawIOBase):
    """Wraps any RawIOBase with a background thread to prefetch data.

    Overlaps Network IO wait with CPU-bound decompression/parsing.
    Employs memoryview zero-copy slicing and short-read fast paths.
    """

    def __init__(self, inner: io.RawIOBase, max_buffers: int, buffer_size: int):
        self._inner = inner
        self._buffer_size = buffer_size
        self._queue = queue.Queue(maxsize=max_buffers)
        self._error = None
        self._current_chunk = memoryview(b"")
        self._stop_event = threading.Event()
        self._eof = False

        self._t = threading.Thread(target=self._fetch_loop, daemon=True)
        self._t.start()

    def _fetch_loop(self):
        try:
            while not self._stop_event.is_set():
                chunk = self._inner.read(self._buffer_size)
                if not chunk:
                    self._queue.put(b"")
                    break
                self._queue.put(chunk)
        except Exception as e:  # noqa: BLE001
            self._error = e
            self._queue.put(b"")

    def readable(self) -> bool:
        return True

    def _load_next_chunk(self) -> bool:
        """Blocks until the next chunk is available. Returns True if data loaded, False on EOF."""
        if self._current_chunk:
            return True

        if self._error:
            raise self._error

        if self._eof:
            return False

        chunk = self._queue.get()
        if not chunk:
            self._eof = True
            if self._error:
                raise self._error
            return False

        self._current_chunk = memoryview(chunk)
        return True

    def readinto(self, b) -> int:
        """Zero-copy fast path: read directly into caller's pre-allocated buffer."""
        if not self._load_next_chunk():
            return 0

        available = len(self._current_chunk)
        size = len(b)
        read_size = min(size, available)

        # Memoryview assignment copies directly from source bytes to target C-buffer
        b[:read_size] = self._current_chunk[:read_size]
        self._current_chunk = self._current_chunk[read_size:]
        return read_size

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunks = []
            while self._load_next_chunk():
                chunks.append(self._current_chunk.tobytes())
                self._current_chunk = memoryview(b"")
            return b"".join(chunks)

        if not self._load_next_chunk():
            return b""

        available = len(self._current_chunk)
        read_size = min(size, available)

        result = self._current_chunk[:read_size].tobytes()
        self._current_chunk = self._current_chunk[read_size:]
        return result

    def close(self):
        self._stop_event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._inner.close()
        super().close()


class ProfiledPrefetchReader(PrefetchReader):
    def __init__(
        self,
        inner: io.RawIOBase,
        max_buffers: int,
        buffer_size: int,
        profile: ReadProfile,
    ):
        self._profile = profile
        super().__init__(inner, max_buffers=max_buffers, buffer_size=buffer_size)

    def _fetch_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                wall_started = time.perf_counter()
                cpu_started = time.thread_time()
                chunk = self._inner.read(self._buffer_size)
                self._profile.record_fetch(
                    len(chunk),
                    time.perf_counter() - wall_started,
                    time.thread_time() - cpu_started,
                )
                if not chunk:
                    self._queue.put(b"")
                    break
                self._queue.put(chunk)
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            self._queue.put(b"")

    def _load_next_chunk(self) -> bool:
        if self._current_chunk:
            return True
        if self._error:
            raise self._error
        if self._eof:
            return False
        started = time.perf_counter()
        chunk = self._queue.get()
        self._profile.record_prefetch_wait(time.perf_counter() - started)
        if not chunk:
            self._eof = True
            if self._error:
                raise self._error
            return False
        self._current_chunk = memoryview(chunk)
        return True


def default_prefetch_reader_factory(
    hdfs_path: str,
    buffer_size: int,
    max_buffers: int,
    profile: ReadProfile | None = None,
) -> io.RawIOBase:
    reader = ResilientWebHdfsReader(
        hdfs_path,
        url_resolver=_url_resolver,
        max_retries=5,
        connect_timeout=_CONNECT_TIMEOUT_S,
        read_timeout=_READ_TIMEOUT_S,
    )
    if profile is None:
        return PrefetchReader(reader, max_buffers=max_buffers, buffer_size=buffer_size)
    return ProfiledPrefetchReader(
        reader,
        max_buffers=max_buffers,
        buffer_size=buffer_size,
        profile=profile,
    )


def _constant_string_array(value: str, rows: int) -> pa.Array:
    return pa.repeat(pa.scalar(value, type=pa.string()), rows)


def _read_decompress_file(
    hdfs_path: str,
    source_csv_block_size_bytes: int,
    reader_factory: typing.Callable[..., io.RawIOBase],
    read_profile: bool = False,
    read_profile_collector: typing.Any | None = None,
):
    """ReadTask body: stream → decompress → yield raw-line blocks.

    Pure IO + decompress: NO parsing, NO pickle. Each yielded Arrow Table
    contains chunks with a single ``line`` column (large_string).
    Line parsing happens in the downstream ``ParseUDF`` actor pool.
    """
    import pyarrow.csv as pacsv

    if not isinstance(read_profile, bool):
        raise TypeError(f"read_profile must be bool, got {type(read_profile).__name__}")
    if read_profile and read_profile_collector is None:
        raise ValueError("read profile requires a read profile collector")
    partition_values = source_partition_values(hdfs_path)

    t0 = time.perf_counter()
    total_lines = 0
    chunks_produced = 0

    read_options = pacsv.ReadOptions(
        column_names=["line"],
        block_size=source_csv_block_size_bytes,
        use_threads=False,
    )
    # The datasource contract is one non-null raw text column.  Explicitly
    # declaring it avoids Arrow's first-block type inference, while retaining
    # the large-string schema previously produced by the per-batch cast.
    convert_options = pacsv.ConvertOptions(
        column_types={"line": pa.large_string()},
        strings_can_be_null=False,
    )
    # Use '\x01' as a fake delimiter to read entire line as a single string column.
    parse_options = pacsv.ParseOptions(
        delimiter="\x01", quote_char=False, escape_char=False
    )

    def stream_csv(stream, profile: ReadProfile | None):
        nonlocal total_lines, chunks_produced
        csv_open_started = time.perf_counter()
        csv_reader = pacsv.open_csv(
            stream,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        if profile is not None:
            profile.record_csv_open(time.perf_counter() - csv_open_started)
        with csv_reader:
            while True:
                decode_started = time.perf_counter()
                try:
                    batch = csv_reader.read_next_batch()
                except StopIteration:
                    break
                if profile is not None:
                    profile.record_csv_decode(time.perf_counter() - decode_started)
                materialize_started = time.perf_counter()
                table = pa.Table.from_batches([batch])
                if table.num_rows > 0:
                    for name in PARTITION_COLUMNS:
                        values = _constant_string_array(
                            partition_values[name], table.num_rows
                        )
                        table = table.append_column(name, values)
                    if profile is not None:
                        profile.record_arrow_materialize(
                            time.perf_counter() - materialize_started
                        )
                        yield_started = time.perf_counter()
                        yield table
                        profile.record_yield_wait(time.perf_counter() - yield_started)
                    else:
                        yield table
                    total_lines += table.num_rows
                    chunks_produced += 1

    if os.path.exists(hdfs_path):
        logger.info("Reading local file instead of HDFS: %s", hdfs_path)
        with pa.OSFile(hdfs_path, "rb") as f:
            if hdfs_path.endswith((".zst", ".zstd")):
                with pa.CompressedInputStream(f, "zstd") as stream:
                    yield from stream_csv(stream, None)
            else:
                yield from stream_csv(f, None)

        elapsed = time.perf_counter() - t0
        logger.info(
            "ReadTask Local %s: %d chunks, %d lines, %.1fs (%.0f lines/s)",
            hdfs_path,
            chunks_produced,
            total_lines,
            elapsed,
            total_lines / max(elapsed, 0.001),
        )
        return

    profile = ReadProfile() if read_profile else None
    open_started = time.perf_counter()
    stream = (
        reader_factory(hdfs_path, profile=profile)
        if profile is not None
        else reader_factory(hdfs_path)
    )
    if profile is not None:
        profile.record_open(time.perf_counter() - open_started)

    try:
        setup_started = time.perf_counter()
        with (
            pa.PythonFile(stream, mode="r") as f,
            pa.CompressedInputStream(f, "zstd") as zstream,
        ):
            if profile is not None:
                profile.record_stream_setup(time.perf_counter() - setup_started)
            yield from stream_csv(zstream, profile)
    finally:
        stream.close()

    elapsed = time.perf_counter() - t0
    logger.info(
        "ReadTask %s: %d chunks, %d lines, %.1fs (%.0f lines/s)",
        hdfs_path,
        chunks_produced,
        total_lines,
        elapsed,
        total_lines / max(elapsed, 0.001),
    )
    if profile is not None:
        snapshot = profile.snapshot()
        report_read_profile(read_profile_collector, snapshot, elapsed)
        logger.info(
            "ReadTask profile %s: compressed_bytes=%d open=%.3fs fetch_wall=%.3fs "
            "fetch_cpu=%.3fs prefetch_wait=%.3fs csv_decode=%.3fs "
            "arrow_materialize=%.3fs yield_wait=%.3fs stream_setup=%.3fs "
            "csv_open=%.3fs total=%.3fs",
            hdfs_path,
            snapshot.compressed_bytes,
            snapshot.open_seconds,
            snapshot.fetch_wall_seconds,
            snapshot.fetch_cpu_seconds,
            snapshot.prefetch_wait_seconds,
            snapshot.csv_decode_seconds,
            snapshot.arrow_materialize_seconds,
            snapshot.yield_wait_seconds,
            snapshot.stream_setup_seconds,
            snapshot.csv_open_seconds,
            elapsed,
        )
