from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from flare.data.connectors.adapters.parquet.local_cache import LocalParquetFileCache
from flare.data.connectors.adapters.parquet.range_reader import (
    ParquetInputFileReader,
)


def test_cached_row_group_serves_later_ranges_without_reading_file_again(tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({"id": list(range(100)), "value": [f"v{i}" for i in range(100)]}),
        path,
        row_group_size=100,
    )
    session = ParquetInputFileReader(max_open_files=2).create_session(
        source_identity="test"
    )
    arguments = {
        "revision_identity": "snapshot-1",
        "locator": str(path),
        "input_file_factory": lambda: SimpleNamespace(open=lambda: path.open("rb")),
        "projection": ("id", "value"),
    }
    try:
        first = list(session.read_batches(**arguments, row_start=0, row_count=10))
        source = session._opened_files[("snapshot-1", str(path))]
        bytes_after_first = source.bytes_read
        second = list(session.read_batches(**arguments, row_start=40, row_count=10))
        assert [
            value for batch in first for value in batch.column("id").to_pylist()
        ] == list(range(10))
        assert [
            value for batch in second for value in batch.column("id").to_pylist()
        ] == list(range(40, 50))
        assert bytes_after_first > 0
        assert source.bytes_read == bytes_after_first
        changed = list(
            session.read_batches(
                **{**arguments, "projection": ("value",)}, row_start=90, row_count=10
            )
        )
        assert [
            value for batch in changed for value in batch.column("value").to_pylist()
        ] == [f"v{i}" for i in range(90, 100)]
        assert source.bytes_read > bytes_after_first
    finally:
        session.close()


def test_remote_locator_uses_node_local_cache_without_a_user_option(
    tmp_path, monkeypatch
):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"id": list(range(100))}), path, row_group_size=100)
    cache = LocalParquetFileCache(1024 * 1024, tmp_path / "cache")
    monkeypatch.setattr(
        "flare.data.connectors.adapters.parquet.range_reader.LocalParquetFileCache",
        lambda _capacity: cache,
    )
    args = {
        "revision_identity": "snapshot-1",
        "locator": "s3://bucket/data.parquet",
        "projection": ("id",),
        "row_count": 10,
    }
    first = ParquetInputFileReader(max_open_files=2).create_session(
        source_identity="test"
    )
    try:
        batches = list(
            first.read_batches(
                **args,
                row_start=0,
                input_file_factory=lambda: SimpleNamespace(
                    open=lambda: path.open("rb")
                ),
            )
        )
        assert [
            value for batch in batches for value in batch.column("id").to_pylist()
        ] == list(range(10))
    finally:
        first.close()

    second = ParquetInputFileReader(max_open_files=2).create_session(
        source_identity="test"
    )
    try:
        batches = list(
            second.read_batches(
                **args,
                row_start=40,
                input_file_factory=lambda: (_ for _ in ()).throw(
                    AssertionError("cached file must not reopen the remote source")
                ),
            )
        )
        assert [
            value for batch in batches for value in batch.column("id").to_pylist()
        ] == list(range(40, 50))
    finally:
        second.close()
