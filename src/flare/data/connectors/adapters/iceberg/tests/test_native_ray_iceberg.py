from __future__ import annotations

import pyarrow as pa
import pytest
import ray
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.manifest import ManifestEntryStatus
from pyiceberg.table import TableProperties

from flare.data.connectors.adapters.iceberg.manifest_stream import (
    IcebergManifestTraversal,
)
from flare.data.connectors.adapters.iceberg.modes import ContinuousAppend
from flare.data.connectors.adapters.iceberg.scan_cursor import (
    IcebergScanCursor,
    ManifestListRef,
)
from flare.data.connectors.adapters.iceberg.source import IcebergSourcePlanner


@pytest.fixture(scope="module")
def native_table(tmp_path_factory):
    location = tmp_path_factory.mktemp("native-ray-iceberg")
    catalog_config = {
        "name": "native",
        "type": "sql",
        "uri": f"sqlite:///{location / 'catalog.db'}",
        "warehouse": location.as_uri(),
    }
    catalog = SqlCatalog(**catalog_config)
    catalog.create_namespace("test")
    table = catalog.create_table(
        "test.events",
        schema=pa.schema([pa.field("id", pa.int64())]),
        properties={
            TableProperties.MANIFEST_MERGE_ENABLED: "true",
            TableProperties.MANIFEST_MIN_MERGE_COUNT: "1",
        },
    )
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    try:
        ray.data.from_arrow(pa.table({"id": list(range(8))})).write_iceberg(
            table_identifier="test.events", catalog_kwargs=catalog_config
        )
        table.refresh()
        first_snapshot = table.current_snapshot()
        assert first_snapshot is not None
        yield table, first_snapshot.snapshot_id, catalog_config
    finally:
        ray.shutdown()


def _read_scan_results(table, snapshot_id, monkeypatch):
    snapshot = table.snapshot_by_id(snapshot_id)
    manifests = snapshot.manifests(table.io)
    expected = {
        entry.data_file.file_path: entry.data_file.record_count
        for manifest in manifests
        for entry in manifest.fetch_manifest_entry(table.io)
    }
    original_new_input = table.io.new_input
    opened = []

    def metadata_input(location):
        assert location not in expected, "Source planning opened a training data file"
        opened.append(location)
        return original_new_input(location)

    monkeypatch.setattr(table.io, "new_input", metadata_input)
    cursor = IcebergScanCursor.create(
        snapshot_id=snapshot_id,
        contract_digest="a" * 64,
        manifest_list_ref=ManifestListRef.create(
            snapshot_id=snapshot_id,
            path=snapshot.manifest_list,
            object_length=len(table.io.new_input(snapshot.manifest_list)),
        ),
    )
    actual = []
    for _ in range(6 * len(expected) + 8):
        scan_result = IcebergManifestTraversal().scan_next(
            table,
            cursor=cursor,
            max_records=sum(expected.values()),
            max_units=1,
            max_metadata_block_bytes=1024 * 1024,
            snapshot_id=snapshot_id,
            projection=("id",),
            row_filter=None,
        )
        actual.extend((value.path, value.record_count) for value in scan_result.files)
        cursor = IcebergScanCursor.decode(scan_result.next_cursor.encode())
        if scan_result.snapshot_exhausted:
            break
    else:
        pytest.fail("Metadata traversal failed to terminate after all file entries")
    assert len(actual) == len(expected)
    assert dict(actual) == expected
    assert opened


def test_native_ray_initial_snapshot_uses_metadata_only(native_table, monkeypatch):
    table, first_id, _catalog_config = native_table
    _read_scan_results(table, first_id, monkeypatch)


def test_native_ray_merged_manifest_reads_existing_files(native_table, monkeypatch):
    table, first_id, catalog_config = native_table
    with monkeypatch.context() as metadata_only:
        _read_scan_results(table, first_id, metadata_only)
    ray.data.from_arrow(pa.table({"id": list(range(8, 13))})).write_iceberg(
        table_identifier="test.events", catalog_kwargs=catalog_config
    )
    table.refresh()
    snapshot = table.current_snapshot()
    assert snapshot is not None
    assert snapshot.parent_snapshot_id == first_id
    entries = [
        entry
        for manifest in snapshot.manifests(table.io)
        for entry in manifest.fetch_manifest_entry(table.io)
    ]
    assert any(entry.status is ManifestEntryStatus.EXISTING for entry in entries)
    assert sum(entry.data_file.record_count for entry in entries) == 13
    _read_scan_results(table, snapshot.snapshot_id, monkeypatch)


def test_native_continuous_source_resumes_after_idle_with_no_writer_metadata(
    native_table, monkeypatch
):
    table, _first_id, catalog_config = native_table
    table.refresh()
    initial_count = int(table.current_snapshot().summary["total-records"])
    planner = IcebergSourcePlanner(
        table_identifier="test.events",
        catalog_kwargs=catalog_config,
        projection=("id",),
        mode=ContinuousAppend(),
    )
    cursor = planner.initial_scan_cursor()
    original_new_input = type(table.io).new_input
    metadata_reads = []
    planning = False

    def metadata_only_input(self, location):
        if planning:
            assert not location.endswith(".parquet"), "Planning opened training data"
            metadata_reads.append(location)
        return original_new_input(self, location)

    monkeypatch.setattr(type(table.io), "new_input", metadata_only_input)

    def scan_until_idle():
        nonlocal cursor, planning
        files = []
        planning = True
        try:
            for _ in range(32):
                result = planner.scan_next(cursor, max_records=128, max_units=1)
                files.extend(result.files)
                cursor = IcebergScanCursor.decode(result.next_cursor.encode())
                assert not planner.is_finite
                if result.snapshot_exhausted and not result.files:
                    return files
            pytest.fail("Continuous source failed to reach idle")
        finally:
            planning = False

    initial_files = scan_until_idle()
    assert sum(value.record_count for value in initial_files) == initial_count
    assert scan_until_idle() == []
    ray.data.from_arrow(pa.table({"id": list(range(13, 17))})).write_iceberg(
        table_identifier="test.events", catalog_kwargs=catalog_config
    )
    appended_files = scan_until_idle()
    assert sum(value.record_count for value in appended_files) == 4
    assert not {value.path for value in initial_files} & {
        value.path for value in appended_files
    }
    assert scan_until_idle() == []
    assert metadata_reads
    table.refresh()
    assert (
        not {"campaign", "generation", "append_sequence", "append_index_locator"}
        & table.current_snapshot().summary.additional_properties.keys()
    )


def test_native_ray_consumption_bookmark_replans_with_fresh_source(
    native_table, monkeypatch
):
    from fluxway.common.ingestion import (
        OpaqueSourceCursor,
        SourceScanBudget,
        SourceScanStatus,
    )
    from fluxway.common.metadata import unpack_metadata

    from flare.data.connectors.adapters.iceberg.modes import PinnedSnapshot
    from flare.data.connectors.adapters.iceberg.scan import IcebergSourceScanAdapter

    table, first_id, config = native_table
    original_input = type(table.io).new_input
    computing_bookmark = False

    def metadata_only_input(self, location):
        assert not computing_bookmark, "Bookmark calculation must not read metadata"
        assert not location.endswith(".parquet"), "Planning must not read training data"
        return original_input(self, location)

    monkeypatch.setattr(type(table.io), "new_input", metadata_only_input)

    def new_source():
        planner = IcebergSourcePlanner(
            table_identifier="test.events",
            catalog_kwargs=config,
            projection=("id",),
            mode=PinnedSnapshot(first_id),
        )
        return planner, IcebergSourceScanAdapter(planner)

    def next_data(adapter, cursor, budget):
        for _ in range(32):
            result = adapter.scan_next(cursor, budget)
            if result.status is not SourceScanStatus.PROGRESS:
                return result
            cursor = result.next_cursor
        pytest.fail("Native traversal did not reach data or termination")

    def positions(result):
        for unit in result.units:
            interval = unpack_metadata(unit.opaque_token)
            for row in range(
                interval["record_start"],
                interval["record_start"] + interval["record_count"],
            ):
                yield unit.physical_identity, row

    planner, adapter = new_source()
    cursor = OpaqueSourceCursor.from_payload(planner.initial_scan_cursor().encode())
    initial = next_data(adapter, cursor, SourceScanBudget(8, 8))
    assert initial.records == 8
    expected = tuple(positions(initial))[5:]
    computing_bookmark = True
    try:
        bookmark = initial.cursor_at(5, adapter.advance_cursor)
    finally:
        computing_bookmark = False

    _, restored = new_source()
    cursor = OpaqueSourceCursor.from_checkpoint(bookmark.to_checkpoint())
    remaining = []
    for _ in range(16):
        result = next_data(restored, cursor, SourceScanBudget(2, 1))
        remaining.extend(positions(result))
        if result.status is SourceScanStatus.TERMINAL:
            break
        cursor = result.next_cursor
    else:
        pytest.fail("Restored native traversal did not terminate")
    assert tuple(remaining) == expected


@pytest.mark.asyncio
async def test_native_ray_joint_source_checkpoint_reassembles_file_ranges(
    native_table, monkeypatch
):
    from fluxway.common.ingestion import (
        OpaqueSourceCursor,
        SourceScanBudget,
        SourceWorkAcknowledgement,
    )
    from fluxway.common.types import CheckpointCut
    from fluxway.common.types.configs import SourceControlConfig
    from fluxway.control_plane.components import ControlPlaneComponentRegistry
    from fluxway.control_plane.ingestion import (
        AssignmentPolicy,
        IngestionLayoutPlanningComponent,
        RankDemand,
        ReaderGroupDemand,
        ReaderTopologyDescriptor,
        TailLayoutMode,
    )
    from fluxway.control_plane.source_control import SourceControlRuntime

    from flare.data.connectors.adapters.iceberg.modes import PinnedSnapshot
    from flare.data.connectors.adapters.iceberg.source import IcebergSourceReader
    from flare.data.connectors.adapters.parquet.range_reader import (
        ParquetInputFileReader,
    )
    from flare.data.connectors.adapters.iceberg.scan import IcebergSourceScanAdapter
    from flare.runtime.fluxway.source import AssignedSourceBatchReader

    table, first_id, config = native_table
    original_input = type(table.io).new_input

    def metadata_only_input(self, path):
        assert not path.endswith(".parquet")
        return original_input(self, path)

    monkeypatch.setattr(type(table.io), "new_input", metadata_only_input)

    def source():
        return IcebergSourcePlanner(
            table_identifier="test.events",
            catalog_kwargs=config,
            projection=("id",),
            mode=PinnedSnapshot(first_id),
        )

    def topology(demands):
        return ReaderTopologyDescriptor(
            sum(demands),
            tuple(
                ReaderGroupDemand(str(i), (RankDemand(str(i), count),))
                for i, count in enumerate(demands)
            ),
        )

    def runtime(demands):
        planning = IngestionLayoutPlanningComponent(
            "planning", topology=topology(demands)
        )
        control = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=4))
        registry = ControlPlaneComponentRegistry()
        registry.register(planning)
        registry.register(control)
        return planning, control, registry

    before, control, registry = runtime((2, 2))
    after, restored_control, restored_registry = runtime((4,))
    native = source()
    before_adapter = IcebergSourceScanAdapter(native)

    def before_initial_cursor_factory():
        return OpaqueSourceCursor.from_payload(native.initial_scan_cursor().encode())

    try:
        await before.plan_source_scan(
            before_adapter,
            before_initial_cursor_factory,
            control.work_ledger,
            OpaqueSourceCursor.from_payload(native.initial_scan_cursor().encode()),
            budget=SourceScanBudget(8, 8),
            topology=topology((2, 2)),
            tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=0,
        )
        for window in before.windows:
            for step in range(window.work_epoch.start_step, window.work_epoch.end_step):
                segments = tuple(
                    item
                    for layout in window.worker_layouts
                    for item in layout.segments
                    if item.step == step
                )
                leases = await control.work_ledger.acquire(
                    window.work_epoch.epoch, segments
                )
                await control.work_ledger.acknowledge(
                    tuple(
                        SourceWorkAcknowledgement(lease.item.key, lease.lease_id)
                        for lease in leases
                    )
                )
        await control.work_ledger.commit_cut(1)
        cut = CheckpointCut.for_position("cut", 1)
        restored_registry.restore_state(registry.checkpoint_state(cut), cut)
        assert after.windows == ()
        assert restored_control.work_ledger.committed_cut == 1
        restored_native = source()
        after_adapter = IcebergSourceScanAdapter(restored_native)

        def after_initial_cursor_factory():
            return OpaqueSourceCursor.from_payload(
                restored_native.initial_scan_cursor().encode()
            )

        await after.plan_source_scan(
            after_adapter,
            after_initial_cursor_factory,
            restored_control.work_ledger,
            after.resume_cursor,
            budget=SourceScanBudget(2, 1),
            topology=topology((4,)),
            tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=1,
            max_steps=1,
        )
        monkeypatch.setattr(type(table.io), "new_input", original_input)
        reader = AssignedSourceBatchReader(
            reader=IcebergSourceReader(
                table_identifier="test.events",
                catalog_kwargs=config,
                projection=("id",),
                mode=PinnedSnapshot(first_id),
                format_readers={"parquet": ParquetInputFileReader(max_open_files=2)},
            ),
            projection=("id",),
        )
        restored_rows = []
        try:
            for window in after.windows:
                for step in range(
                    window.work_epoch.start_step, window.work_epoch.end_step
                ):
                    segments = tuple(
                        segment
                        for layout in window.worker_layouts
                        for segment in layout.segments
                        if segment.step == step
                    )
                    leases = await restored_control.work_ledger.acquire(
                        window.work_epoch.epoch, segments
                    )
                    work = tuple(
                        sorted(
                            (lease.item for lease in leases),
                            key=lambda item: item.global_start,
                        )
                    )
                    batch = await reader.read(step, work_items=work)
                    restored_rows.extend(batch["id"].to_pylist())
        finally:
            await reader.close()
        assert restored_rows == [4, 5, 6, 7]
    finally:
        await before.close()
        await after.close()
