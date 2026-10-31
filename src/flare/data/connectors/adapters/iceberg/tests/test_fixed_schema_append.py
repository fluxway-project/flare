from __future__ import annotations

from contextlib import nullcontext
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


def test_writer_bundles_small_blocks_without_losing_partition_values() -> None:
    blocks = [
        pa.table({"id": [i, i + 1], "_ds": [day, day]})
        for i, day in [(0, "2026-01-01"), (2, "2026-01-02"), (4, "2026-01-02")]
    ]
    sink = object.__new__(FixedSchemaAppendIcebergDatasink)
    sink._expected_schema = blocks[0].schema
    sink._coalesce_target_bytes = blocks[0].nbytes * 2
    sink._min_rows_per_write = 4
    result = list(sink._coalesced_blocks(blocks))
    assert [block.num_rows for block in result] == [4, 2]
    assert pa.concat_tables(result).equals(pa.concat_tables(blocks))
    assert (
        result[0].column("id").chunk(0).buffers()[1].address
        == blocks[0].column("id").chunk(0).buffers()[1].address
    )
    assert sink.min_rows_per_write == 4


def test_writer_bundling_validates_schema_before_writing() -> None:
    sink = object.__new__(FixedSchemaAppendIcebergDatasink)
    sink._expected_schema = pa.schema([pa.field("id", pa.int64())])
    sink._coalesce_target_bytes = 1024
    with pytest.raises(RuntimeError, match="schemas differ"):
        list(sink._coalesced_blocks([pa.table({"id": ["bad"]})]))


def test_generic_writer_does_not_require_feature_width_metadata():
    schema = pa.schema([pa.field("tags", pa.list_(pa.string()))])
    sink = FixedSchemaAppendIcebergDatasink(
        expected_schema=schema,
        expected_iceberg_schema=object(),
        catalog_provider=MagicMock(),
        table_identifier="test.tags",
        coalesce_target_bytes=1024,
    )
    assert sink.min_rows_per_write is None
    block = pa.table({"tags": [["a", "b"], []]}, schema=schema)
    assert pa.concat_tables(
        list(sink._coalesced_blocks([block, block]))
    ).to_pydict() == {
        "tags": [["a", "b"], [], ["a", "b"], []],
    }


def test_coalesced_parquet_append_roundtrip_across_daily_partitions(tmp_path) -> None:
    from pyiceberg.catalog.sql import SqlCatalog

    from flare.data.connectors.adapters.iceberg.access import (
        ConfiguredIcebergCatalogProvider,
    )
    from flare.data.connectors.adapters.iceberg.table import create_iceberg_table

    options = {
        "name": "bundle-test",
        "type": "sql",
        "uri": f"sqlite:///{tmp_path / 'catalog.db'}",
        "warehouse": tmp_path.as_uri(),
    }
    catalog = SqlCatalog(**options)
    catalog.create_namespace("test")
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("_ds", pa.string()),
            pa.field("_hour", pa.string()),
        ]
    )
    table = create_iceberg_table(
        catalog=catalog,
        table_identifier="test.daily",
        schema=schema,
        writer_target_size_bytes=1024 * 1024,
    )
    sink = FixedSchemaAppendIcebergDatasink(
        expected_schema=schema,
        expected_iceberg_schema=table.schema(),
        catalog_provider=ConfiguredIcebergCatalogProvider(options),
        table_identifier="test.daily",
        coalesce_target_bytes=1024 * 1024,
        snapshot_properties={"flare.publication-id": "bundle-roundtrip"},
    )
    blocks = [
        pa.table(
            {"id": list(range(i, i + 4)), "_ds": [day] * 4, "_hour": ["unknown"] * 4},
            schema=schema,
        )
        for i, day in [
            (0, "2026-01-01"),
            (4, "2026-01-02"),
            (8, "2026-01-01"),
            (12, "2026-01-02"),
        ]
    ]
    sink.on_write_start(schema)
    result = sink.write(blocks, None)
    assert len(result.data_files) == 2
    sink.on_write_complete(SimpleNamespace(write_returns=[result]))
    table.refresh()
    actual = table.scan().to_arrow().sort_by("id")
    assert actual.to_pydict() == pa.concat_tables(blocks).sort_by("id").to_pydict()
    assert table.current_snapshot().summary["total-records"] == "16"
    catalog.close()


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
    datasink._table_scope = lambda: nullcontext()
    datasink._table = table
    datasink._mode = SaveMode.APPEND
    datasink._snapshot_properties = {}
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
    datasink = FixedSchemaAppendIcebergDatasink(
        expected_schema=expected_schema,
        expected_iceberg_schema=object(),
        table_identifier="test.events",
        catalog_provider=MagicMock(),
    )
    table = MagicMock()
    table.schema.return_value = object()
    with (
        patch(
            "flare.data.connectors.adapters.iceberg.fixed_schema_append.IcebergTableSession",
            return_value=SimpleNamespace(table=lambda: table, close=lambda: None),
        ),
        patch.object(datasink, "_append_and_commit") as append,
        pytest.raises(RuntimeError, match="table schema differs"),
    ):
        datasink.on_write_complete(SimpleNamespace(write_returns=[]))
    append.assert_not_called()


def test_fixed_schema_append_rejects_non_append_mode() -> None:
    schema = pa.schema([pa.field("id", pa.int64())])

    with pytest.raises(RuntimeError, match="requires APPEND"):
        FixedSchemaAppendIcebergDatasink(
            expected_schema=schema,
            expected_iceberg_schema=object(),
            table_identifier="db.tbl",
            catalog_provider=MagicMock(),
            mode=SaveMode.OVERWRITE,
        )
