from unittest.mock import MagicMock, call

import pyarrow as pa
import pytest
from pyiceberg.catalog.memory import InMemoryCatalog

from flare.data.connectors.adapters.iceberg.table import (
    create_iceberg_table,
    validate_iceberg_table,
)


def test_explicit_table_creation_uses_static_schema_and_business_partitions() -> None:
    schema = pa.schema(
        [
            pa.field("fid_101", pa.int64()),
            pa.field("_ds", pa.string()),
            pa.field("_hour", pa.string()),
        ]
    )
    table = MagicMock()
    update = table.update_spec.return_value.__enter__.return_value
    catalog = MagicMock()
    catalog.create_table.return_value = table

    create_iceberg_table(
        catalog=catalog,
        table_identifier="db.tbl",
        schema=schema,
        writer_target_size_bytes=1024,
    )

    catalog.create_namespace.assert_not_called()
    catalog.create_table.assert_called_once_with(
        "db.tbl",
        schema=schema,
        properties={
            "write.format.default": "parquet",
            "write.parquet.compression-codec": "zstd",
            "write.parquet.compression-level": "3",
            "write.parquet.version": "2.6",
            "write.metadata.metrics.default": "none",
            "write.target-file-size-bytes": "1024",
        },
    )
    assert update.add_identity.call_args_list == [call("_ds"), call("_hour")]


def test_existing_table_must_match_schema_and_partition_contract(tmp_path) -> None:
    schema = pa.schema(
        [
            pa.field("fid_101", pa.int64(), nullable=False),
            pa.field("_ds", pa.string(), nullable=False),
            pa.field("_hour", pa.string(), nullable=False),
        ]
    )
    catalog = InMemoryCatalog("test", warehouse=str(tmp_path))
    catalog.create_namespace("db")
    table = create_iceberg_table(
        catalog=catalog,
        table_identifier="db.tbl",
        schema=schema,
        writer_target_size_bytes=1024,
    )

    validate_iceberg_table(
        table,
        expected_schema=schema,
        expected_writer_target_size_bytes=1024,
    )

    incompatible = schema.set(0, pa.field("fid_101", pa.int32(), nullable=False))
    with pytest.raises(ValueError, match="schema does not match"):
        validate_iceberg_table(
            table,
            expected_schema=incompatible,
            expected_writer_target_size_bytes=1024,
        )


def test_schema_mismatch_reports_fields_without_rendering_entire_schema(
    tmp_path,
) -> None:
    schema = pa.schema(
        [
            pa.field("dense", pa.int32(), nullable=False),
            pa.field("sparse", pa.int64(), nullable=False),
            pa.field("_ds", pa.string(), nullable=False),
            pa.field("_hour", pa.string(), nullable=False),
        ]
    )
    catalog = InMemoryCatalog("test", warehouse=str(tmp_path))
    catalog.create_namespace("db")
    table = create_iceberg_table(
        catalog=catalog,
        table_identifier="db.tbl",
        schema=schema,
        writer_target_size_bytes=1024,
    )
    incompatible = schema.set(0, pa.field("dense", pa.int64(), nullable=False))

    with pytest.raises(ValueError) as error:
        validate_iceberg_table(
            table,
            expected_schema=incompatible,
            expected_writer_target_size_bytes=1024,
        )

    message = str(error.value)
    assert "dense" in message
    assert "('long', 'int')" in message
    assert "Schema(" not in message
