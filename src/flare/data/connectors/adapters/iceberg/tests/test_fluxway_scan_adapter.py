from dataclasses import replace

import pytest
from fluxway.common.ingestion import (
    OpaqueSourceCursor,
    SourceScanBudget,
    SourceScanStatus,
)
from fluxway.common.metadata import unpack_metadata
from pyiceberg.table.snapshots import Operation

from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
)
from flare.data.connectors.adapters.iceberg.modes import ContinuousAppend
from flare.data.connectors.adapters.iceberg.scan import (
    IcebergSourceScanAdapter,
    iceberg_scan_lineage_identity,
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
from flare.data.connectors.adapters.iceberg.source import IcebergSourcePlanner
from flare.data.connectors.adapters.iceberg.tests.catalog_fixtures import (
    MemoryCatalogProvider,
)


def test_continuous_empty_table_has_no_initial_cursor():
    class Table:
        class Metadata:
            table_uuid = "empty-table"

        metadata = Metadata()

        def refresh(self):
            return self

        def current_snapshot(self):
            return None

        def snapshot_by_name(self, branch):
            assert branch == "main"

    planner = IcebergSourcePlanner(
        table_identifier="catalog.db.empty",
        catalog_provider=MemoryCatalogProvider(
            lambda _identifier: Table(),
            identity=(
                ConfiguredIcebergCatalogProvider(
                    {"name": "test", "type": "rest", "uri": "http://catalog"}
                )
            ).identity,
        ),
        projection=("id",),
        mode=ContinuousAppend(),
    )
    assert planner.initial_scan_cursor() is None
    assert planner.binding_identity["native_identity"] == "empty-table"

    finite = IcebergSourcePlanner(
        table_identifier="catalog.db.empty",
        catalog_provider=MemoryCatalogProvider(
            lambda _identifier: Table(),
            identity=(
                ConfiguredIcebergCatalogProvider(
                    {"name": "test", "type": "rest", "uri": "http://catalog"}
                )
            ).identity,
        ),
        projection=("id",),
    )
    with pytest.raises(ValueError, match="no snapshot"):
        finite.initial_scan_cursor()


def test_continuous_empty_binding_discovers_append_snapshots_in_order():
    class Input:
        def __len__(self):
            return 1

    class IO:
        def new_input(self, path):
            assert path.startswith("list-")
            return Input()

    class Summary:
        def __init__(self, operation):
            self.operation = operation

    class Snapshot:
        def __init__(self, snapshot_id, parent_snapshot_id, operation):
            self.snapshot_id = snapshot_id
            self.parent_snapshot_id = parent_snapshot_id
            self.summary = Summary(operation)
            self.manifest_list = f"list-{snapshot_id}.avro"

    class Table:
        class Metadata:
            table_uuid = "continuous-table"

        metadata = Metadata()
        io = IO()

        def __init__(self):
            self.snapshots = {}
            self.branch_snapshot = None

        def refresh(self):
            return self

        def snapshot_by_id(self, snapshot_id):
            return self.snapshots.get(snapshot_id)

        def snapshot_by_name(self, branch):
            assert branch == "main"
            return self.branch_snapshot

    class Traversal:
        def scan_next(self, table, **kwargs):
            cursor = kwargs["cursor"]
            return FileScanResult(
                files=(),
                next_cursor=cursor,
                snapshot_exhausted=True,
            )

    table = Table()
    planner = IcebergSourcePlanner(
        table_identifier="catalog.db.continuous",
        catalog_provider=MemoryCatalogProvider(
            lambda _identifier: table,
            identity=(
                ConfiguredIcebergCatalogProvider(
                    {"name": "test", "type": "rest", "uri": "http://catalog"}
                )
            ).identity,
        ),
        projection=("id",),
        mode=ContinuousAppend(),
        traversal=Traversal(),
    )
    adapter = IcebergSourceScanAdapter(planner)

    assert planner.initial_scan_cursor() is None
    table.snapshots[1] = Snapshot(1, None, Operation.APPEND)
    table.branch_snapshot = table.snapshots[1]
    first_native = planner.initial_scan_cursor()
    assert first_native is not None
    first = OpaqueSourceCursor.from_payload(first_native.encode())
    first_native = IcebergScanCursor.decode(first.payload)
    assert first_native.snapshot_id == 1
    assert first_native.state.additions_only is False
    binding_identity = dict(planner.binding_identity)

    table.snapshots[2] = Snapshot(2, 1, Operation.APPEND)
    table.branch_snapshot = table.snapshots[2]
    transition = adapter.scan_next(first, SourceScanBudget(1, 1))
    assert transition.status is SourceScanStatus.PROGRESS
    assert transition.records == 0
    second_native = IcebergScanCursor.decode(transition.next_cursor.payload)
    assert second_native.snapshot_id == 2
    assert second_native.state.additions_only is True
    assert dict(planner.binding_identity) == binding_identity

    table.snapshots[3] = Snapshot(3, 2, Operation.REPLACE)
    table.branch_snapshot = table.snapshots[3]
    with pytest.raises(ValueError, match="requires append snapshots"):
        adapter.scan_next(transition.next_cursor, SourceScanBudget(1, 1))


def test_iceberg_scan_result_adapter_translates_one_bounded_scan_result() -> None:
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
            raise AssertionError("scan result adapter must not call table.scan")

    class Traversal:
        def scan_next(self, table, **kwargs):
            cursor = kwargs["cursor"]
            return FileScanResult(
                files=(
                    FileScanRange(
                        "file.parquet",
                        3,
                        "parquet",
                        IcebergScanCursor(
                            replace(
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
                                ),
                                file_record_offset=5,
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
    initial = planner.initial_scan_cursor()
    adapter = IcebergSourceScanAdapter(planner)
    scan_result = adapter.scan_next(
        OpaqueSourceCursor.from_payload(initial.encode()),
        SourceScanBudget(
            max_records=3,
            max_units=1,
        ),
    )
    assert scan_result.status is SourceScanStatus.READY
    assert scan_result.records == 3
    assert scan_result.unit_count == 1
    assert scan_result.next_cursor.payload != initial.encode()
    record_range = unpack_metadata(scan_result.units[0].opaque_token)
    assert scan_result.units[0].logical_start == record_range["record_start"] == 5
    assert record_range["record_count"] == 3
    advanced = IcebergScanCursor.decode(scan_result.next_cursor.payload)
    assert iceberg_scan_lineage_identity(advanced) == scan_result.lineage_digest
