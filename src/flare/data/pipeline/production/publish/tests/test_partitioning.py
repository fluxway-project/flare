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
