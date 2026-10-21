from __future__ import annotations

import pyarrow as pa
import pytest

from flare.data.schema import (
    DEFAULT_VALIDITY_COLUMN,
    ArrowVirtualPadBatchFactory,
)


def test_create_batch_with_defaults() -> None:
    schema = pa.schema(
        [
            pa.field("user_id", pa.int64(), nullable=True),
            pa.field("category", pa.string(), nullable=True),
            pa.field("samples_valid_count", pa.int32(), nullable=False),
        ]
    )
    batch = ArrowVirtualPadBatchFactory.create_batch(
        schema=schema,
        projection=["user_id", "category", "samples_valid_count"],
        count=3,
    )
    assert batch.num_rows == 3
    assert batch.column("user_id").to_pylist() == [None, None, None]
    assert batch.column("category").to_pylist() == [None, None, None]
    assert batch.column("samples_valid_count").to_pylist() == [0, 0, 0]


def test_create_batch_with_explicit_pad_values() -> None:
    schema = pa.schema(
        [
            pa.field("user_id", pa.int64(), nullable=False),
            pa.field("category", pa.string(), nullable=True),
        ]
    )
    batch = ArrowVirtualPadBatchFactory.create_batch(
        schema=schema,
        projection=["user_id", "category"],
        count=2,
        pad_values={"user_id": 12345, "category": "UNKNOWN"},
    )
    assert batch.num_rows == 2
    assert batch.column("user_id").to_pylist() == [12345, 12345]
    assert batch.column("category").to_pylist() == ["UNKNOWN", "UNKNOWN"]


def test_create_batch_missing_pad_value_for_non_nullable_raises() -> None:
    schema = pa.schema(
        [
            pa.field("user_id", pa.int64(), nullable=False),
        ]
    )
    with pytest.raises(
        ValueError,
        match="requires explicit pad value for non-nullable column 'user_id'",
    ):
        ArrowVirtualPadBatchFactory.create_batch(
            schema=schema,
            projection=["user_id"],
            count=1,
        )


def test_create_batch_invalid_count_raises() -> None:
    schema = pa.schema([pa.field("col", pa.int32(), nullable=True)])
    with pytest.raises(
        ValueError, match="Virtual pad batch count must be positive, got 0"
    ):
        ArrowVirtualPadBatchFactory.create_batch(
            schema=schema,
            projection=["col"],
            count=0,
        )
    with pytest.raises(
        ValueError, match="Virtual pad batch count must be positive, got -1"
    ):
        ArrowVirtualPadBatchFactory.create_batch(
            schema=schema,
            projection=["col"],
            count=-1,
        )


def test_create_batch_custom_validity_column() -> None:
    schema = pa.schema(
        [
            pa.field("custom_valid_count", pa.int32(), nullable=False),
            pa.field("samples_valid_count", pa.int32(), nullable=True),
        ]
    )
    batch = ArrowVirtualPadBatchFactory.create_batch(
        schema=schema,
        projection=["custom_valid_count", "samples_valid_count"],
        count=2,
        validity_column="custom_valid_count",
    )
    assert batch.column("custom_valid_count").to_pylist() == [0, 0]
    assert batch.column("samples_valid_count").to_pylist() == [None, None]


def test_create_batch_pad_values_override_validity_column() -> None:
    schema = pa.schema(
        [
            pa.field("samples_valid_count", pa.int32(), nullable=False),
        ]
    )
    batch = ArrowVirtualPadBatchFactory.create_batch(
        schema=schema,
        projection=["samples_valid_count"],
        count=2,
        pad_values={"samples_valid_count": 99},
    )
    assert batch.column("samples_valid_count").to_pylist() == [99, 99]


def test_default_validity_column_constant() -> None:
    assert DEFAULT_VALIDITY_COLUMN == "samples_valid_count"


def test_create_batch_requires_an_explicit_complete_schema() -> None:
    with pytest.raises(TypeError, match="explicit Arrow schema"):
        ArrowVirtualPadBatchFactory.create_batch(None, ["feature"], 2)
    with pytest.raises(ValueError, match="missing projected columns"):
        ArrowVirtualPadBatchFactory.create_batch(
            pa.schema([pa.field("other", pa.int64())]), ["feature"], 2
        )


def test_create_batch_preserves_nested_schema_and_nonnullable_field_contract() -> None:
    schema = pa.schema(
        [
            pa.field(
                "feature",
                pa.list_(pa.int64()),
                nullable=False,
                metadata={b"kind": b"fid"},
            )
        ],
        metadata={b"schema": b"nested"},
    )
    batch = ArrowVirtualPadBatchFactory.create_batch(
        schema, ["feature"], 2, pad_values={"feature": [-3]}, validity_column=None
    )
    assert batch.schema.equals(schema, check_metadata=True)
    assert batch.column("feature").to_pylist() == [[-3], [-3]]
    with pytest.raises(ValueError, match="cannot be null"):
        ArrowVirtualPadBatchFactory.create_batch(
            schema, ["feature"], 2, pad_values={"feature": None}, validity_column=None
        )
