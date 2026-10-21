from __future__ import annotations

from pathlib import Path

import pyarrow as pa

PARTITION_COLUMNS = ("_ds", "_hour")
_SOURCE_SEGMENTS = {"_ds": "ds", "_hour": "hour"}


def append_partition_fields(schema: pa.Schema) -> pa.Schema:
    overlap = sorted(set(schema.names) & set(PARTITION_COLUMNS))
    if overlap:
        raise ValueError(f"schema already contains partition columns: {overlap}")
    for name in PARTITION_COLUMNS:
        schema = schema.append(pa.field(name, pa.string()))
    return schema


def require_partition_fields(schema: pa.Schema) -> tuple[pa.Field, ...]:
    missing = [name for name in PARTITION_COLUMNS if name not in schema.names]
    if missing:
        raise ValueError(f"output schema is missing partition columns: {missing}")
    fields = tuple(schema.field(name) for name in PARTITION_COLUMNS)
    invalid = [field.name for field in fields if field.type != pa.string()]
    if invalid:
        raise TypeError(f"partition columns must be strings: {invalid}")
    return fields


def source_partition_values(path: str) -> dict[str, str]:
    segments = path.replace("\\", "/").split("/")
    values: dict[str, str] = {}
    for column, source_name in _SOURCE_SEGMENTS.items():
        prefix = f"{source_name}="
        matches = [
            segment[len(prefix) :] for segment in segments if segment.startswith(prefix)
        ]
        if len(matches) != 1 or not matches[0]:
            raise ValueError(
                f"source path must contain exactly one non-empty {source_name}= partition: {path}"
            )
        values[column] = matches[0]
    return values


def parquet_partition_directory(
    root: Path,
    values: tuple[str, ...],
) -> Path:
    if len(values) != len(PARTITION_COLUMNS):
        raise ValueError("partition values do not match the partition contract")
    directory = root
    for name, value in zip(PARTITION_COLUMNS, values, strict=True):
        if not value or "/" in value or "\\" in value:
            raise ValueError(f"invalid Parquet partition value {name}={value!r}")
        directory /= f"{name}={value}"
    return directory
