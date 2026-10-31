"""Shared pipeline contracts, independent of recipe configurations."""

from __future__ import annotations

import pytest


class TestDatasourcePlanning:
    def test_profiled_prefetch_reader_reports_network_and_consumer_metrics(self):
        import io

        from flare.data.pipeline.production.publish.datasource import (
            ProfiledPrefetchReader,
            ReadProfile,
        )

        class BytesReader(io.RawIOBase):
            def __init__(self, payload: bytes) -> None:
                self._payload = memoryview(payload)

            def readable(self) -> bool:
                return True

            def read(self, size: int = -1) -> bytes:
                if size < 0:
                    size = len(self._payload)
                result = self._payload[:size].tobytes()
                self._payload = self._payload[len(result) :]
                return result

        profile = ReadProfile()
        reader = ProfiledPrefetchReader(
            BytesReader(b"profiled-prefetch"),
            max_buffers=1,
            buffer_size=4,
            profile=profile,
        )
        assert reader.read() == b"profiled-prefetch"
        reader.close()

        snapshot = profile.snapshot()
        assert snapshot.compressed_bytes == len(b"profiled-prefetch")
        assert snapshot.fetch_wall_seconds >= 0.0
        assert snapshot.fetch_cpu_seconds >= 0.0
        assert snapshot.prefetch_wait_seconds >= 0.0
        assert snapshot.yield_wait_seconds >= 0.0

    def test_read_profile_preserves_zstd_source_semantics(self):
        import io
        from unittest.mock import patch

        import pyarrow as pa

        from flare.data.pipeline.production.publish.datasource import (
            ProfiledPrefetchReader,
            ReadProfile,
            _read_decompress_file,
        )

        class BytesReader(io.RawIOBase):
            def __init__(self, payload: bytes) -> None:
                self._payload = memoryview(payload)

            def readable(self) -> bool:
                return True

            def read(self, size: int = -1) -> bytes:
                if size < 0:
                    size = len(self._payload)
                result = self._payload[:size].tobytes()
                self._payload = self._payload[len(result) :]
                return result

        sink = pa.BufferOutputStream()
        with pa.CompressedOutputStream(sink, "zstd") as stream:
            stream.write(b"first\nsecond\n")
        compressed = sink.getvalue().to_pybytes()
        profiles: list[ReadProfile] = []

        def reader_factory(_: str, *, profile: ReadProfile) -> ProfiledPrefetchReader:
            profiles.append(profile)
            return ProfiledPrefetchReader(
                BytesReader(compressed), max_buffers=1, buffer_size=4, profile=profile
            )

        reports = []
        with patch(
            "flare.data.pipeline.production.publish.datasource.report_read_profile",
            side_effect=lambda collector, snapshot, elapsed: reports.append(
                (collector, snapshot, elapsed)
            ),
        ):
            tables = list(
                _read_decompress_file(
                    "/virtual/ds=2026-01-06/hour=00/part-00000.zst",
                    1024,
                    reader_factory=reader_factory,
                    read_profile=True,
                    read_profile_collector="collector",
                )
            )

        assert sum(table.num_rows for table in tables) == 2
        assert len(profiles) == 1
        snapshot = profiles[0].snapshot()
        assert snapshot.compressed_bytes == len(compressed)
        assert snapshot.csv_decode_seconds >= 0.0
        assert snapshot.arrow_materialize_seconds >= 0.0
        assert snapshot.yield_wait_seconds >= 0.0
        assert snapshot.stream_setup_seconds >= 0.0
        assert snapshot.csv_open_seconds >= 0.0
        assert len(reports) == 1

    def test_csv_source_declares_large_string_without_type_inference(self, tmp_path):
        import pyarrow as pa

        from flare.data.pipeline.production.publish.datasource import (
            _read_decompress_file,
        )

        source = tmp_path / "ds=2026-01-06" / "hour=00" / "part-00000.zst"
        source.parent.mkdir(parents=True)
        sink = pa.BufferOutputStream()
        with pa.CompressedOutputStream(sink, "zstd") as stream:
            stream.write(b"one\n2\nnull\n")
        source.write_bytes(sink.getvalue().to_pybytes())

        tables = list(
            _read_decompress_file(
                str(source),
                1024,
                reader_factory=lambda _: (_ for _ in ()).throw(AssertionError()),
            )
        )

        table = pa.concat_tables(tables)
        assert table.schema == pa.schema(
            [
                ("line", pa.large_string()),
                ("_ds", pa.string()),
                ("_hour", pa.string()),
            ]
        )
        assert table["line"].to_pylist() == ["one", "2", "null"]

    def test_hdfs_read_failure_is_not_retried(self):
        import io
        from collections.abc import Buffer

        import pytest

        from flare.data.pipeline.production.publish.datasource import (
            _read_decompress_file,
        )

        readers = []

        class InvalidZstdReader(io.RawIOBase):
            def __init__(self) -> None:
                self._payload = memoryview(b"not-a-zstd-frame")

            def readable(self) -> bool:
                return True

            def readinto(self, buffer: Buffer, /) -> int:
                target = memoryview(buffer)
                length = min(len(target), len(self._payload))
                target[:length] = self._payload[:length]
                self._payload = self._payload[length:]
                return length

        def reader_factory(_: str) -> InvalidZstdReader:
            reader = InvalidZstdReader()
            readers.append(reader)
            return reader

        with pytest.raises(OSError, match="ZSTD decompress failed"):
            list(
                _read_decompress_file(
                    "/missing/ds=2026-01-06/hour=00/part-00000.zst",
                    1024,
                    reader_factory=reader_factory,
                )
            )

        assert len(readers) == 1
        assert readers[0].closed

    def test_hdfs_datasource_separates_csv_block_and_planner_bytes(self):
        from flare.data.pipeline.production.publish.datasource import HdfsZstdDatasource

        datasource = HdfsZstdDatasource(
            file_paths=["/tmp/part-00000.zst"],
            source_csv_block_size_bytes=32 * 1024 * 1024,
            read_planner_block_bytes=128 * 1024 * 1024,
            datasource_prefetch_max_buffers=2,
            datasource_prefetch_buffer_size_bytes=32 * 1024 * 1024,
            source_files_per_task=1,
        )
        task = datasource.get_read_tasks(1)[0]
        assert task.metadata.size_bytes == 128 * 1024 * 1024
        assert datasource._source_csv_block_size_bytes == 32 * 1024 * 1024

    def test_hdfs_datasource_uses_explicit_prefetch_buffer_size(self):
        from flare.data.pipeline.production.publish.datasource import HdfsZstdDatasource

        datasource = HdfsZstdDatasource(
            file_paths=["/tmp/part-00000.zst"],
            source_csv_block_size_bytes=32 * 1024 * 1024,
            read_planner_block_bytes=128 * 1024 * 1024,
            datasource_prefetch_max_buffers=2,
            datasource_prefetch_buffer_size_bytes=8 * 1024 * 1024,
            source_files_per_task=1,
        )
        assert datasource._datasource_prefetch_buffer_size_bytes == 8 * 1024 * 1024

    def test_hdfs_datasource_read_profile_is_explicit(self):
        from flare.data.pipeline.production.publish.datasource import HdfsZstdDatasource

        with pytest.raises(ValueError, match="collector"):
            HdfsZstdDatasource(
                file_paths=["/tmp/part-00000.zst"],
                source_csv_block_size_bytes=32 * 1024 * 1024,
                read_planner_block_bytes=128 * 1024 * 1024,
                datasource_prefetch_max_buffers=2,
                datasource_prefetch_buffer_size_bytes=8 * 1024 * 1024,
                source_files_per_task=1,
                read_profile=True,
            )
        datasource = HdfsZstdDatasource(
            file_paths=["/tmp/part-00000.zst"],
            source_csv_block_size_bytes=32 * 1024 * 1024,
            read_planner_block_bytes=128 * 1024 * 1024,
            datasource_prefetch_max_buffers=2,
            datasource_prefetch_buffer_size_bytes=8 * 1024 * 1024,
            source_files_per_task=1,
            read_profile=True,
            read_profile_collector=object(),
        )

        assert datasource._read_profile
        with pytest.raises(TypeError, match="read_profile"):
            HdfsZstdDatasource(
                file_paths=["/tmp/part-00000.zst"],
                source_csv_block_size_bytes=32 * 1024 * 1024,
                read_planner_block_bytes=128 * 1024 * 1024,
                datasource_prefetch_max_buffers=2,
                datasource_prefetch_buffer_size_bytes=8 * 1024 * 1024,
                source_files_per_task=1,
                read_profile=1,  # pyrefly: ignore[bad-argument-type]
                read_profile_collector=object(),
            )

    def test_hdfs_datasource_bounds_each_task_and_preserves_source_membership(self):
        from flare.data.pipeline.production.publish.datasource import HdfsZstdDatasource

        paths = tuple(f"/tmp/part-{index:02d}.zst" for index in range(7))
        datasource = HdfsZstdDatasource(
            file_paths=paths,
            source_csv_block_size_bytes=32 * 1024 * 1024,
            read_planner_block_bytes=128 * 1024 * 1024,
            datasource_prefetch_max_buffers=2,
            datasource_prefetch_buffer_size_bytes=8 * 1024 * 1024,
            source_files_per_task=1,
        )

        tasks = datasource.get_read_tasks(3)

        assert len(tasks) == len(paths)
        input_files = [task.metadata.input_files for task in tasks]
        assert all(files is not None for files in input_files)
        flattened = tuple(
            path
            for files in input_files
            for path in files  # pyrefly: ignore[not-iterable]
        )
        assert sorted(flattened) == list(paths)
        assert flattened == paths
        assert [task.metadata.size_bytes for task in tasks] == [
            128 * 1024 * 1024,
            128 * 1024 * 1024,
            128 * 1024 * 1024,
            128 * 1024 * 1024,
            128 * 1024 * 1024,
            128 * 1024 * 1024,
            128 * 1024 * 1024,
        ]

    def test_hdfs_datasource_groups_files_deterministically(self):
        from flare.data.pipeline.production.publish.datasource import HdfsZstdDatasource

        paths = tuple(f"/tmp/part-{index:02d}.zst" for index in range(5))
        datasource = HdfsZstdDatasource(
            file_paths=paths,
            source_csv_block_size_bytes=32 * 1024 * 1024,
            read_planner_block_bytes=128 * 1024 * 1024,
            datasource_prefetch_max_buffers=2,
            datasource_prefetch_buffer_size_bytes=8 * 1024 * 1024,
            source_files_per_task=2,
        )

        tasks = datasource.get_read_tasks(3)

        assert [task.metadata.input_files for task in tasks] == [
            paths[:2],
            paths[2:4],
            paths[4:],
        ]
        assert [task.metadata.size_bytes for task in tasks] == [
            2 * 128 * 1024 * 1024,
            2 * 128 * 1024 * 1024,
            128 * 1024 * 1024,
        ]

    def test_hdfs_datasource_constant_partition_columns_are_plain_string(self):
        import pyarrow as pa

        from flare.data.pipeline.production.publish.datasource import (
            _constant_string_array,
        )

        arr = _constant_string_array("2025-12-03", 4)

        assert pa.types.is_string(arr.type)
        assert arr.to_pylist() == ["2025-12-03"] * 4

    def test_hdfs_datasource_empty_constant_partition_column(self):
        import pyarrow as pa

        from flare.data.pipeline.production.publish.datasource import (
            _constant_string_array,
        )

        arr = _constant_string_array("00", 0)

        assert pa.types.is_string(arr.type)
        assert len(arr) == 0
