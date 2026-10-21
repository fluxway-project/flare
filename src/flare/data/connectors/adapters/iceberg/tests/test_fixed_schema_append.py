from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from ray.data import SaveMode

from flare.data.connectors.adapters.iceberg.fixed_schema_append import (
    FixedSchemaAppendIcebergDatasink,
    require_fixed_output_schemas,
)


def test_require_fixed_output_schemas_accepts_validated_schema() -> None:
    schema = pa.schema([pa.field("id", pa.int64())])

    require_fixed_output_schemas([schema, schema], schema)


def test_require_fixed_output_schemas_checks_each_distinct_schema_once() -> None:
    from flare.data.connectors.adapters.iceberg.fixed_schema_append import (
        require_fixed_output_schemas,
    )

    class SchemaProbe:
        def __init__(self, matches: bool) -> None:
            self.matches = matches
            self.calls = 0

        def equals(self, _: object) -> bool:
            self.calls += 1
            return self.matches

    expected = SchemaProbe(True)
    matching = SchemaProbe(True)
    require_fixed_output_schemas([matching, matching, matching], expected)  # type: ignore[arg-type]
    assert matching.calls == 1

    mismatch = SchemaProbe(False)
    with pytest.raises(RuntimeError, match="mismatched_blocks=\\[1, 2\\]"):
        require_fixed_output_schemas([matching, mismatch, mismatch], expected)  # type: ignore[arg-type]
    assert mismatch.calls == 1


def test_require_fixed_output_schemas_rejects_mismatch() -> None:
    expected_schema = pa.schema([pa.field("id", pa.int64())])
    mismatched_schema = pa.schema([pa.field("id", pa.string())])

    with pytest.raises(RuntimeError, match="mismatched_blocks=\\[0\\]"):
        require_fixed_output_schemas([mismatched_schema], expected_schema)


def test_fixed_schema_append_commits_validated_append_without_schema_update() -> None:
    schema = pa.schema([pa.field("id", pa.int64())])
    # Construct only the completion callback state: this isolates the private
    # Ray commit boundary without starting a Ray writer or an Iceberg service.
    datasink = object.__new__(FixedSchemaAppendIcebergDatasink)
    table = MagicMock()
    transaction = table.transaction.return_value
    expected_iceberg_schema = object()
    datasink._expected_schema = schema
    datasink._expected_iceberg_schema = expected_iceberg_schema
    datasink._table = table
    datasink._mode = SaveMode.APPEND
    table.schema.return_value = expected_iceberg_schema
    write_result = SimpleNamespace(
        write_returns=[
            SimpleNamespace(data_files=["data-file"], schemas=[schema]),
        ]
    )

    with patch.object(datasink, "_append_and_commit") as append_and_commit:
        datasink.on_write_complete(write_result)

    append_and_commit.assert_called_once_with(transaction, ["data-file"])
    transaction.update_schema.assert_not_called()


def test_fixed_schema_append_rejects_table_schema_mismatch() -> None:
    expected_schema = pa.schema([pa.field("id", pa.int64())])
    # See the preceding test: direct construction keeps this contract test
    # deterministic while exercising the callback's fail-closed behavior.
    datasink = object.__new__(FixedSchemaAppendIcebergDatasink)
    datasink._expected_schema = expected_schema
    datasink._expected_iceberg_schema = object()
    datasink._table = MagicMock()
    datasink._mode = SaveMode.APPEND
    datasink._table.schema.return_value = object()
    write_result = SimpleNamespace(
        write_returns=[
            SimpleNamespace(data_files=["data-file"], schemas=[expected_schema]),
        ]
    )

    with pytest.raises(RuntimeError, match="table schema differs"):
        datasink.on_write_complete(write_result)


def test_fixed_schema_append_rejects_non_append_mode() -> None:
    schema = pa.schema([pa.field("id", pa.int64())])

    with pytest.raises(RuntimeError, match="requires APPEND"):
        FixedSchemaAppendIcebergDatasink(
            expected_schema=schema,
            expected_iceberg_schema=object(),
            table_identifier="db.tbl",
            mode=SaveMode.OVERWRITE,
        )
