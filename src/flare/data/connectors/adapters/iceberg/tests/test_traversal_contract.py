from types import SimpleNamespace

import pytest

from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
)
from flare.data.connectors.adapters.iceberg.scan_cursor import (
    IcebergScanCursor,
    ManifestEntryOffset,
    ManifestRef,
)
from flare.data.connectors.adapters.iceberg.scan_result import (
    FileScanRange,
    FileScanResult,
)
from flare.data.connectors.adapters.iceberg.source import (
    IcebergSourcePlanner,
    IcebergSourceReader,
)
from flare.data.connectors.adapters.iceberg.tests.catalog_fixtures import (
    MemoryCatalogProvider,
)


def test_scan_next_forwards_all_bounds_without_snapshot_collection() -> None:
    class Snapshot:
        snapshot_id = 7
        summary = None
        manifest_list = "list.avro"

    class Input:
        def __len__(self):
            return 1

    class IO:
        def new_input(self, path):
            assert path == "list.avro"
            return Input()

    class Table:
        class Metadata:
            table_uuid = "table-uuid"

        metadata = Metadata()
        io = IO()

        def refresh(self):
            return self

        def current_snapshot(self):
            return Snapshot()

        def snapshot_by_id(self, snapshot_id):
            assert snapshot_id == 7
            return Snapshot()

        def scan(self, **kwargs):
            raise AssertionError("scan_next must not call table.scan")

    class Traversal:
        def scan_next(self, table, **kwargs):
            assert isinstance(table, Table)
            assert kwargs["max_records"] == 2
            assert kwargs["max_units"] == 1
            assert kwargs["max_metadata_block_bytes"] == 64 * 1024 * 1024
            cursor = kwargs["cursor"]
            return FileScanResult(
                files=(
                    FileScanRange(
                        "file.parquet",
                        2,
                        "parquet",
                        IcebergScanCursor(
                            cursor.state.with_manifest(
                                ManifestRef.create(
                                    path="manifest.avro",
                                    object_length=1,
                                    content="0",
                                    sequence=1,
                                    added_snapshot_id=7,
                                    partition_spec_id=0,
                                ),
                                ManifestEntryOffset(0, 0, 0),
                            )
                        ),
                    ),
                ),
                next_cursor=IcebergScanCursor(
                    cursor.state.with_manifest(
                        ManifestRef.create(
                            path="manifest.avro",
                            object_length=1,
                            content="0",
                            sequence=1,
                            added_snapshot_id=7,
                            partition_spec_id=0,
                        ),
                        ManifestEntryOffset(0, 0, 1),
                    )
                ),
                snapshot_exhausted=False,
            )

    planner = IcebergSourcePlanner(
        table_identifier="catalog.db.table",
        catalog_provider=MemoryCatalogProvider(
            lambda _identifier: Table(),
            identity=(
                ConfiguredIcebergCatalogProvider(
                    {"name": "test", "type": "rest", "uri": "http://catalog"}
                )
            ).identity,
        ),
        projection=("id",),
        traversal=Traversal(),
    )
    cursor = planner.initial_scan_cursor()
    scan_result = planner.scan_next(
        cursor,
        max_records=2,
        max_units=1,
    )
    assert [(f.path, f.record_start, f.record_count) for f in scan_result.files] == [
        ("file.parquet", 0, 2)
    ]
    assert scan_result.next_cursor.is_after(cursor)


def test_metadata_budgets_are_strictly_bounded() -> None:
    with pytest.raises(ValueError, match="metadata_scan_records"):
        IcebergSourcePlanner(
            table_identifier="catalog.db.table",
            catalog_provider=ConfiguredIcebergCatalogProvider(
                {"name": "test", "type": "rest", "uri": "http://catalog"}
            ),
            projection=("id",),
            metadata_scan_records=0,
        )


def test_planner_rejects_table_uuid_change_before_scan_output() -> None:
    class Snapshot:
        snapshot_id = 7
        summary = None
        manifest_list = "list.avro"

    class Table:
        class Metadata:
            table_uuid = "replacement-uuid"

        metadata = Metadata()

        def refresh(self):
            return self

        def current_snapshot(self):
            return Snapshot()

        def snapshot_by_id(self, snapshot_id):
            return Snapshot() if snapshot_id == 7 else None

    class OriginalTable(Table):
        class Metadata:
            table_uuid = "original-uuid"

        metadata = Metadata()

        def refresh(self):
            return Table()

    tables = iter((OriginalTable(), Table()))
    planner = IcebergSourcePlanner(
        table_identifier="catalog.db.table",
        catalog_provider=MemoryCatalogProvider(
            lambda _identifier: next(tables),
            identity=(
                ConfiguredIcebergCatalogProvider(
                    {"name": "test", "type": "rest", "uri": "http://catalog"}
                )
            ).identity,
        ),
        projection=("id",),
    )
    _ = planner.binding_identity
    with pytest.raises(ValueError, match="table UUID changed"):
        planner.initial_scan_cursor()


def test_reader_rejects_table_uuid_change_before_file_materialization() -> None:
    class Snapshot:
        def __init__(self, snapshot_id):
            self.snapshot_id = snapshot_id

    class ReplacementTable:
        class Metadata:
            table_uuid = "replacement-uuid"

        metadata = Metadata()

    class OriginalTable:
        class Metadata:
            table_uuid = "original-uuid"

        metadata = Metadata()

        def refresh(self):
            return ReplacementTable()

    class FormatReader:
        def create_session(self, **kwargs):
            raise AssertionError("UUID validation must precede session creation")

    tables = iter((OriginalTable(), ReplacementTable()))
    reader = IcebergSourceReader(
        table_identifier="catalog.db.table",
        catalog_provider=MemoryCatalogProvider(
            lambda _identifier: next(tables),
            identity=(
                ConfiguredIcebergCatalogProvider(
                    {"name": "test", "type": "rest", "uri": "http://catalog"}
                )
            ).identity,
        ),
        projection=("id",),
        format_readers={"parquet": FormatReader()},
    )
    _ = reader.binding_identity
    reader._materialization_revision = "1"
    record_range = SimpleNamespace(
        revision_identity="2",
        locator=SimpleNamespace(canonical=lambda: "file.parquet"),
    )
    with pytest.raises(ValueError, match="table UUID changed"):
        reader._input_file(record_range)
