from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.schema import assign_fresh_schema_ids

from flare.data.pipeline.production.publish.partitioning import require_partition_fields

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_VERSION = "2.6"
ICEBERG_METRICS_MODE = "none"
DEFAULT_WRITER_TARGET_SIZE_BYTES = 256 * 1024 * 1024
IcebergFormatVersion = Literal[1, 2, 3]


def require_iceberg_format_version(value: object) -> IcebergFormatVersion:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Iceberg format version must be an integer")
    if value not in (1, 2, 3):
        raise ValueError(f"unsupported Iceberg format version: {value}")
    return value


@lru_cache(maxsize=16)
def _iceberg_schema_for_arrow(
    serialized_schema: bytes,
    format_version: IcebergFormatVersion,
):
    arrow_schema = pa.ipc.read_schema(pa.BufferReader(serialized_schema))
    return assign_fresh_schema_ids(
        Catalog._convert_schema_if_needed(
            arrow_schema,
            format_version=format_version,
        )
    )


def iceberg_table_properties(writer_target_size_bytes: int) -> dict[str, str]:
    if writer_target_size_bytes <= 0:
        raise ValueError("writer_target_size_bytes must be positive")
    return {
        "write.format.default": "parquet",
        "write.parquet.compression-codec": PARQUET_COMPRESSION,
        "write.parquet.compression-level": str(PARQUET_COMPRESSION_LEVEL),
        "write.parquet.version": PARQUET_VERSION,
        "write.metadata.metrics.default": ICEBERG_METRICS_MODE,
        "write.target-file-size-bytes": str(writer_target_size_bytes),
    }


def validate_iceberg_table_properties(
    table_properties: dict[str, str] | None,
    *,
    expected_writer_target_size_bytes: int,
) -> None:
    if table_properties is None:
        raise ValueError("Iceberg snapshot publisher requires table_properties")
    expected = iceberg_table_properties(expected_writer_target_size_bytes)
    mismatches = {
        key: (expected_value, table_properties.get(key))
        for key, expected_value in expected.items()
        if table_properties.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Iceberg table properties do not match: {mismatches}")


def validate_iceberg_table(
    table: Any,
    *,
    expected_schema: pa.Schema,
    expected_writer_target_size_bytes: int,
) -> Any:
    if not isinstance(expected_schema, pa.Schema):
        raise TypeError("expected_schema must be a PyArrow schema")
    validate_iceberg_table_properties(
        dict(table.properties),
        expected_writer_target_size_bytes=expected_writer_target_size_bytes,
    )
    expected_iceberg_schema = _iceberg_schema_for_arrow(
        expected_schema.serialize().to_pybytes(),
        require_iceberg_format_version(table.metadata.format_version),
    )
    actual_iceberg_schema = table.schema()
    if actual_iceberg_schema != expected_iceberg_schema:
        expected_fields = {
            field.name: field.field_type for field in expected_iceberg_schema.fields
        }
        actual_fields = {
            field.name: field.field_type for field in actual_iceberg_schema.fields
        }
        missing = sorted(expected_fields.keys() - actual_fields.keys())
        extra = sorted(actual_fields.keys() - expected_fields.keys())
        type_mismatches = {
            name: (
                str(expected_fields[name]),
                str(actual_fields[name]),
            )
            for name in expected_fields.keys() & actual_fields.keys()
            if expected_fields[name] != actual_fields[name]
        }
        raise ValueError(
            "Iceberg table schema does not match the producer schema: "
            f"missing={missing!r} extra={extra!r} "
            f"type_mismatches={type_mismatches!r}"
        )

    expected_partitions = tuple(
        (field.name, field.name, "identity")
        for field in require_partition_fields(expected_schema)
    )
    actual_partitions = tuple(
        (
            actual_iceberg_schema.find_column_name(field.source_id),
            field.name,
            str(field.transform),
        )
        for field in table.spec().fields
    )
    if actual_partitions != expected_partitions:
        raise ValueError(
            "Iceberg table partition spec does not match the producer schema: "
            f"expected={expected_partitions!r} actual={actual_partitions!r}"
        )
    return actual_iceberg_schema


def create_iceberg_table(
    *,
    catalog: Any,
    table_identifier: str,
    schema: pa.Schema,
    writer_target_size_bytes: int = DEFAULT_WRITER_TARGET_SIZE_BYTES,
) -> Any:
    if not table_identifier:
        raise ValueError("table_identifier must not be empty")
    if not isinstance(schema, pa.Schema):
        raise TypeError("Iceberg table creation requires a PyArrow schema")
    partition_fields = require_partition_fields(schema)
    table = catalog.create_table(
        table_identifier,
        schema=schema,
        properties=iceberg_table_properties(writer_target_size_bytes),
    )
    with table.update_spec() as update:
        for field in partition_fields:
            update.add_identity(field.name)
    return table
