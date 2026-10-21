import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flare.data.connectors.adapters.file.source import FileListSourceAdapter


def test_file_list_parquet_uses_bounded_range_reader() -> None:
    calls = []

    def bounded_reader(path, **kwargs):
        calls.append((path, kwargs))
        return pa.table({"value": [1, 2]})

    adapter = FileListSourceAdapter(
        format_id="parquet",
        files=["data.parquet"],
        projection=["value"],
        range_reader=bounded_reader,
        record_counter=lambda path: 2,
    )
    ranges = adapter.record_ranges()
    assert len(ranges) == 1
    assert not hasattr(adapter, "discover")
    assert not hasattr(adapter, "validate_revision")
    assert not hasattr(adapter, "serialize_cursor")
    record_range = ranges[0]
    adapter.read(record_range, ["value"])

    assert calls == [
        (
            "data.parquet",
            {
                "projection": ("value",),
                "row_start": 0,
                "row_count": 2,
            },
        )
    ]


def test_file_list_vortex_uses_bounded_range_reader() -> None:
    calls = []

    def bounded_reader(path, **kwargs):
        calls.append((path, kwargs))
        return pa.table({"value": [1, 2]})

    adapter = FileListSourceAdapter(
        format_id="vortex",
        files=["data.vortex"],
        projection=["value"],
        range_reader=bounded_reader,
        record_counter=lambda path: 2,
    )
    record_range = adapter.record_ranges()[0]
    adapter.read(record_range, ["value"])

    assert calls[0][0] == "data.vortex"
    assert calls[0][1]["row_start"] == 0
    assert calls[0][1]["row_count"] == 2


def test_parquet_range_rejects_missing_columns(tmp_path) -> None:
    from flare.data.connectors.adapters.parquet.range_reader import read_parquet_range

    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"physical_name": [1, 2]}), path)

    with pytest.raises(ValueError, match="missing columns"):
        read_parquet_range(
            str(path),
            projection=("missing_column",),
            row_start=0,
            row_count=1,
        )


@pytest.mark.parametrize("value", [0, -1, True])
def test_source_read_concurrency_is_strict(value) -> None:
    from flare.data.pipeline.consumption.source.execution.materialization import (
        validate_source_read_concurrency,
    )

    with pytest.raises(ValueError, match="positive integer"):
        validate_source_read_concurrency(value)
