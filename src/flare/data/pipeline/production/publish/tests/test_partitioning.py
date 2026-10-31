from __future__ import annotations

import pyarrow as pa
import pytest

from flare.data.pipeline.production.publish.partitioning import (
    PARTITION_COLUMNS,
    append_partition_fields,
    parquet_partition_directory,
    require_partition_fields,
    source_partition_values,
)


def test_partition_contract_maps_hdfs_to_storage_columns(tmp_path) -> None:
    assert PARTITION_COLUMNS == ("_ds", "_hour")
    assert source_partition_values(
        "/warehouse/table/ds=2025-12-03/hour=07/part-0.zst"
    ) == {"_ds": "2025-12-03", "_hour": "07"}
    assert (
        parquet_partition_directory(
            tmp_path,
            ("2025-12-03", "07"),
        )
        == tmp_path / "_ds=2025-12-03" / "_hour=07"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/warehouse/table/hour=07/part-0.zst",
        "/warehouse/table/ds=2025-12-03/part-0.zst",
        "/warehouse/table/ds=/hour=07/part-0.zst",
    ],
)
def test_partition_contract_has_no_unknown_value_fallback(path) -> None:
    with pytest.raises(ValueError, match="must contain exactly one non-empty"):
        source_partition_values(path)


def test_partition_schema_is_exact_and_typed() -> None:
    schema = append_partition_fields(pa.schema([pa.field("fid_101", pa.int64())]))

    fields = require_partition_fields(schema)

    assert tuple(field.name for field in fields) == PARTITION_COLUMNS
    assert all(field.type == pa.string() for field in fields)


def test_daily_partition_requires_explicit_hour_and_preserves_real_hours() -> None:
    assert source_partition_values(
        "/ds=2026-01-01/part-0.zst", defaults={"_hour": "unknown"}
    ) == {
        "_ds": "2026-01-01",
        "_hour": "unknown",
    }
    assert (
        source_partition_values(
            "/ds=2026-01-01/hour=07/part-0.zst", defaults={"_hour": "unknown"}
        )["_hour"]
        == "07"
    )
    for path in [
        "/part-0.zst",
        "/ds=2026-01-01/hour=/part-0.zst",
        "/ds=2026-01-01/hour=01/hour=02/part-0.zst",
    ]:
        with pytest.raises(ValueError):
            source_partition_values(path, defaults={"_hour": "unknown"})
    for value in ["", "../unknown", "a\\b"]:
        with pytest.raises(ValueError):
            source_partition_values(
                "/ds=2026-01-01/part-0.zst", defaults={"_hour": value}
            )


def test_partition_defaults_are_generic_and_unknown_columns_are_rejected():
    assert source_partition_values(
        "/hour=07/part-0.zst", defaults={"_ds": "2000-01-01"}
    ) == {"_ds": "2000-01-01", "_hour": "07"}
    with pytest.raises(ValueError, match="unknown partition default columns"):
        source_partition_values(
            "/ds=2000-01-01/hour=07/part-0.zst", defaults={"typo": "x"}
        )


def test_daily_zstd_read_retains_rows_and_explicit_partition(tmp_path) -> None:
    from flare.data.pipeline.production.publish.datasource import HdfsZstdDatasource

    path = tmp_path / "ds=2026-01-01" / "part-0.zst"
    path.parent.mkdir()
    path.write_bytes(pa.compress(b"first row\nsecond row\n", codec="zstd"))
    source = HdfsZstdDatasource(
        [str(path)],
        source_csv_block_size_bytes=1024,
        read_planner_block_bytes=1024,
        datasource_prefetch_max_buffers=1,
        datasource_prefetch_buffer_size_bytes=1024,
        source_files_per_task=1,
        source_partition_defaults={"_hour": "unknown"},
    )
    blocks = list(source.get_read_tasks(1)[0]())
    assert pa.concat_tables(blocks).to_pydict() == {
        "line": ["first row", "second row"],
        "_ds": ["2026-01-01", "2026-01-01"],
        "_hour": ["unknown", "unknown"],
    }
