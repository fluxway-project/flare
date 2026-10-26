from types import SimpleNamespace

import pytest
from pyiceberg.avro.file import AvroOutputFile
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import (
    MANIFEST_ENTRY_SCHEMAS,
    MANIFEST_LIST_FILE_SCHEMAS,
    DataFile,
    DataFileContent,
    FileFormat,
    ManifestContent,
    ManifestEntry,
    ManifestEntryStatus,
    ManifestFile,
)

from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
)
from flare.data.connectors.adapters.iceberg.manifest_stream import (
    IcebergManifestTraversal,
)
from flare.data.connectors.adapters.iceberg.scan_cursor import (
    IcebergScanCursor,
    ManifestListRef,
)
from flare.data.connectors.adapters.iceberg.source import IcebergSourcePlanner
from flare.data.connectors.adapters.iceberg.tests.catalog_fixtures import (
    MemoryCatalogProvider,
)

FIXTURE_TABLE_UUID = "fixture-table-uuid"


def _scan_until_data(scan, *args, **kwargs):
    for _ in range(32):
        result = scan(*args, **kwargs)
        result.validate_transition(kwargs["cursor"])
        if result.files or result.snapshot_exhausted:
            return result
        kwargs["cursor"] = IcebergScanCursor.decode(result.next_cursor.encode())
    pytest.fail("Fixture traversal did not finish")


def _write_metadata(tmp_path, counts=(2, 2), snapshot_ids=None, entries_per_block=None):
    io = PyArrowFileIO()
    entries = tuple(
        ManifestEntry.from_args(
            status=ManifestEntryStatus.ADDED,
            snapshot_id=7 if snapshot_ids is None else snapshot_ids[index],
            sequence_number=index + 1,
            file_sequence_number=index + 1,
            data_file=DataFile.from_args(
                content=DataFileContent.DATA,
                file_path=str(tmp_path / f"data-{index}.parquet"),
                file_format=FileFormat.PARQUET,
                partition={},
                record_count=counts[index],
                file_size_in_bytes=10,
            ),
        )
        for index in range(len(counts))
    )
    manifest_path = tmp_path / "manifest.avro"
    with AvroOutputFile(
        io.new_output(str(manifest_path)),
        MANIFEST_ENTRY_SCHEMAS[2],
        "manifest_entry",
        metadata={"avro.codec": "null"},
    ) as writer:
        block_size = len(entries) if entries_per_block is None else entries_per_block
        for start in range(0, len(entries), block_size):
            writer.write_block(list(entries[start : start + block_size]))
    manifest = ManifestFile.from_args(
        manifest_path=str(manifest_path),
        manifest_length=manifest_path.stat().st_size,
        partition_spec_id=0,
        content=ManifestContent.DATA,
        sequence_number=1,
        min_sequence_number=1,
        added_snapshot_id=7,
        added_files_count=len(counts),
        existing_files_count=0,
        deleted_files_count=0,
        added_rows_count=sum(counts),
        existing_rows_count=0,
        deleted_rows_count=0,
        partitions=[],
    )
    manifest_list_path = tmp_path / "manifest-list.avro"
    with AvroOutputFile(
        io.new_output(str(manifest_list_path)),
        MANIFEST_LIST_FILE_SCHEMAS[2],
        "manifest_file",
        metadata={"avro.codec": "null"},
    ) as writer:
        writer.write_block([manifest])
    return io, manifest_list_path, manifest_path


def _initial_cursor(io, manifest_list_path):
    ref = ManifestListRef.create(
        snapshot_id=7,
        path=str(manifest_list_path),
        object_length=len(io.new_input(str(manifest_list_path))),
    )
    return IcebergScanCursor.create(
        snapshot_id=7, contract_digest="a" * 64, manifest_list_ref=ref
    )


def test_manifest_traversal_resumes_inside_manifest_without_rescanning_data(
    tmp_path,
) -> None:
    io, manifest_list_path, manifest_path = _write_metadata(tmp_path)
    snapshot = SimpleNamespace(snapshot_id=7, manifest_list=str(manifest_list_path))
    table = SimpleNamespace(
        io=io,
        metadata=SimpleNamespace(format_version=2, table_uuid=FIXTURE_TABLE_UUID),
        snapshot_by_id=lambda snapshot_id: snapshot if snapshot_id == 7 else None,
    )
    traversal = IcebergManifestTraversal()
    initial = _initial_cursor(io, manifest_list_path)
    first = _scan_until_data(
        traversal.scan_next,
        table,
        cursor=initial,
        max_records=2,
        max_units=1,
        max_metadata_block_bytes=64 * 1024,
        snapshot_id=7,
        projection=("id",),
        row_filter=None,
    )
    assert first.files and not first.snapshot_exhausted
    assert first.files[0].path == str(tmp_path / "data-0.parquet")
    assert first.next_cursor.state.manifest is not None
    assert first.next_cursor.state.manifest.path == str(manifest_path)
    second = _scan_until_data(
        traversal.scan_next,
        table,
        cursor=first.next_cursor,
        max_records=2,
        max_units=1,
        max_metadata_block_bytes=64 * 1024,
        snapshot_id=7,
        projection=("id",),
        row_filter=None,
    )
    assert second.files and not second.snapshot_exhausted
    assert second.files[0].path == str(tmp_path / "data-1.parquet")
    terminal = _scan_until_data(
        traversal.scan_next,
        table,
        cursor=second.next_cursor,
        max_records=2,
        max_units=1,
        max_metadata_block_bytes=64 * 1024,
        snapshot_id=7,
        projection=("id",),
        row_filter=None,
    )
    assert terminal.snapshot_exhausted
    assert terminal.files == ()


def test_manifest_traversal_rejects_filters_and_oversized_blocks(tmp_path) -> None:
    io, manifest_list_path, _manifest_path = _write_metadata(tmp_path)
    snapshot = SimpleNamespace(snapshot_id=7, manifest_list=str(manifest_list_path))
    table = SimpleNamespace(
        io=io,
        metadata=SimpleNamespace(format_version=2, table_uuid=FIXTURE_TABLE_UUID),
        snapshot_by_id=lambda snapshot_id: snapshot if snapshot_id == 7 else None,
    )
    traversal = IcebergManifestTraversal()
    cursor = _initial_cursor(io, manifest_list_path)
    with pytest.raises(ValueError, match="row filters"):
        _scan_until_data(
            traversal.scan_next,
            table,
            cursor=cursor,
            max_records=1,
            max_units=1,
            max_metadata_block_bytes=64 * 1024,
            snapshot_id=7,
            projection=("id",),
            row_filter={"id": 1},
        )
    with pytest.raises(ValueError, match="block"):
        _scan_until_data(
            traversal.scan_next,
            table,
            cursor=cursor,
            max_records=1,
            max_units=1,
            max_metadata_block_bytes=1,
            snapshot_id=7,
            projection=("id",),
            row_filter=None,
        )


def test_manifest_traversal_rejects_manifest_identity_change(tmp_path) -> None:
    io, manifest_list_path, _manifest_path = _write_metadata(tmp_path)
    snapshot = SimpleNamespace(snapshot_id=7, manifest_list=str(manifest_list_path))
    table = SimpleNamespace(
        io=io,
        metadata=SimpleNamespace(format_version=2, table_uuid=FIXTURE_TABLE_UUID),
        snapshot_by_id=lambda snapshot_id: snapshot if snapshot_id == 7 else None,
    )
    traversal = IcebergManifestTraversal()
    wrong_ref = ManifestListRef.create(
        snapshot_id=7, path="other-list.avro", object_length=1
    )
    cursor = IcebergScanCursor.create(
        snapshot_id=7, contract_digest="a" * 64, manifest_list_ref=wrong_ref
    )
    with pytest.raises(ValueError, match="manifest-list identity"):
        _scan_until_data(
            traversal.scan_next,
            table,
            cursor=cursor,
            max_records=1,
            max_units=1,
            max_metadata_block_bytes=64 * 1024,
            snapshot_id=7,
            projection=("id",),
            row_filter=None,
        )


def test_manifest_traversal_rejects_manifest_list_length_change(tmp_path) -> None:
    io, manifest_list_path, _manifest_path = _write_metadata(tmp_path)
    snapshot = SimpleNamespace(snapshot_id=7, manifest_list=str(manifest_list_path))
    table = SimpleNamespace(
        io=io,
        metadata=SimpleNamespace(format_version=2, table_uuid=FIXTURE_TABLE_UUID),
        snapshot_by_id=lambda snapshot_id: snapshot if snapshot_id == 7 else None,
    )
    ref = ManifestListRef.create(
        snapshot_id=7,
        path=str(manifest_list_path),
        object_length=len(io.new_input(str(manifest_list_path))) + 1,
    )
    cursor = IcebergScanCursor.create(
        snapshot_id=7, contract_digest="a" * 64, manifest_list_ref=ref
    )
    with pytest.raises(ValueError, match="manifest-list identity"):
        _scan_until_data(
            IcebergManifestTraversal().scan_next,
            table,
            cursor=cursor,
            max_records=1,
            max_units=1,
            max_metadata_block_bytes=64 * 1024,
            snapshot_id=7,
            projection=("id",),
            row_filter=None,
        )


def test_default_iceberg_planner_uses_bounded_manifest_provider(tmp_path) -> None:
    io, manifest_list_path, _manifest_path = _write_metadata(tmp_path)
    snapshot = SimpleNamespace(snapshot_id=7, manifest_list=str(manifest_list_path))
    table = SimpleNamespace(
        io=io,
        metadata=SimpleNamespace(format_version=2, table_uuid=FIXTURE_TABLE_UUID),
        refresh=lambda: table,
        current_snapshot=lambda: snapshot,
        snapshot_by_id=lambda snapshot_id: snapshot if snapshot_id == 7 else None,
    )
    planner = IcebergSourcePlanner(
        table_identifier="catalog.db.table",
        catalog_provider=MemoryCatalogProvider(
            lambda _identifier: table,
            identity=(
                ConfiguredIcebergCatalogProvider(
                    {"name": "test", "type": "rest", "uri": "http://catalog"}
                )
            ).identity,
        ),
        projection=("id",),
    )
    scan_result = _scan_until_data(
        planner.scan_next,
        cursor=planner.initial_scan_cursor(),
        max_records=2,
        max_units=1,
    )
    assert scan_result.files[0].path == str(tmp_path / "data-0.parquet")


def test_file_row_bookmark_replans_without_reading_data(tmp_path, monkeypatch):
    io, manifest_list_path, _ = _write_metadata(tmp_path, counts=(8,))
    original = io.new_input

    def metadata_only(path):
        assert not path.endswith(".parquet")
        return original(path)

    monkeypatch.setattr(io, "new_input", metadata_only)
    snapshot = SimpleNamespace(snapshot_id=7, manifest_list=str(manifest_list_path))
    table = SimpleNamespace(
        io=io,
        metadata=SimpleNamespace(format_version=2, table_uuid=FIXTURE_TABLE_UUID),
        snapshot_by_id=lambda snapshot_id: snapshot if snapshot_id == 7 else None,
    )

    def scan(cursor):
        result = _scan_until_data(
            IcebergManifestTraversal().scan_next,
            table,
            cursor=IcebergScanCursor.decode(cursor.encode()),
            max_records=3,
            max_units=1,
            max_metadata_block_bytes=64 * 1024,
            snapshot_id=7,
            projection=("id",),
            row_filter=None,
        )
        result.validate_budget(max_records=3, max_units=1)
        result.validate_transition(cursor)
        return result

    first = scan(_initial_cursor(io, manifest_list_path))
    second = scan(first.next_cursor)
    third = scan(second.next_cursor)
    assert [
        (r.files[0].record_start, r.record_count) for r in (first, second, third)
    ] == [
        (0, 3),
        (3, 3),
        (6, 2),
    ]
    restored = scan(first.files[0].cursor_after(2))
    assert (restored.files[0].record_start, restored.record_count) == (2, 3)
    assert scan(third.files[0].cursor_after(2)).snapshot_exhausted
    assert scan(third.next_cursor).snapshot_exhausted
    with pytest.raises(ValueError, match="within"):
        first.files[0].cursor_after(4)


def test_historical_only_blocks_advance_without_scanning_remaining_history(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from flare.data.connectors.adapters.iceberg.avro_blocks import BoundedAvroBlocks

    io, list_path, _ = _write_metadata(
        tmp_path, counts=(2, 2, 2, 2), snapshot_ids=(6, 6, 6, 7), entries_per_block=1
    )
    snapshot = SimpleNamespace(snapshot_id=7, manifest_list=str(list_path))
    table = SimpleNamespace(
        io=io,
        metadata=SimpleNamespace(format_version=2, table_uuid=FIXTURE_TABLE_UUID),
        snapshot_by_id=lambda snapshot_id: snapshot if snapshot_id == 7 else None,
    )
    cursor = IcebergScanCursor(
        replace(_initial_cursor(io, list_path).state, additions_only=True)
    )
    read_blocks = []
    original_next = BoundedAvroBlocks.__next__

    def observe_next(reader):
        block = original_next(reader)
        read_blocks.append(block)
        return block

    monkeypatch.setattr(BoundedAvroBlocks, "__next__", observe_next)
    files = []
    empty_blocks = 0
    for _ in range(16):
        read_blocks.clear()
        result = IcebergManifestTraversal().scan_next(
            table,
            cursor=cursor,
            max_records=4,
            max_units=2,
            max_metadata_block_bytes=64 * 1024,
            snapshot_id=7,
            projection=("id",),
            row_filter=None,
        )
        assert len(read_blocks) <= 1
        result.validate_transition(cursor)
        if cursor.state.manifest is not None and read_blocks and not result.files:
            empty_blocks += 1
            assert not result.files and not result.snapshot_exhausted
        files.extend(result.files)
        cursor = IcebergScanCursor.decode(result.next_cursor.encode())
        if result.snapshot_exhausted:
            break
    else:
        pytest.fail("Metadata traversal failed to terminate")
    assert empty_blocks == 3
    assert [(f.path, f.record_count) for f in files] == [
        (str(tmp_path / "data-3.parquet"), 2)
    ]


def test_cross_repo_consumption_bookmark_resumes_inside_second_file(
    tmp_path, monkeypatch
):
    from fluxway.common.ingestion import (
        OpaqueSourceCursor,
        SourceScanBudget,
        SourceScanStatus,
    )
    from fluxway.common.metadata import unpack_metadata

    from flare.data.connectors.adapters.iceberg.scan import IcebergSourceScanAdapter

    io, list_path, _ = _write_metadata(tmp_path, counts=(8, 6))
    original_input = io.new_input

    def metadata_input(path):
        assert not path.endswith(".parquet"), "Planning must not read training data"
        return original_input(path)

    monkeypatch.setattr(io, "new_input", metadata_input)
    snapshot = SimpleNamespace(snapshot_id=7, manifest_list=str(list_path))
    table = SimpleNamespace(
        io=io,
        metadata=SimpleNamespace(format_version=2, table_uuid=FIXTURE_TABLE_UUID),
        refresh=lambda: table,
        current_snapshot=lambda: snapshot,
        snapshot_by_id=lambda snapshot_id: snapshot if snapshot_id == 7 else None,
    )
    planner = IcebergSourcePlanner(
        table_identifier="catalog.db.table",
        catalog_provider=MemoryCatalogProvider(
            lambda _identifier: table,
            identity=(
                ConfiguredIcebergCatalogProvider(
                    {"name": "test", "type": "rest", "uri": "http://catalog"}
                )
            ).identity,
        ),
        projection=("id",),
    )
    adapter = IcebergSourceScanAdapter(planner)

    def next_data(cursor, limit):
        for _ in range(32):
            result = adapter.scan_next(cursor, SourceScanBudget(limit, 2))
            if result.status is not SourceScanStatus.PROGRESS:
                return result
            cursor = result.next_cursor
        pytest.fail("Fixture failed to finish")

    start = OpaqueSourceCursor.from_payload(planner.initial_scan_cursor().encode())
    result = next_data(start, 10)
    assert [unit.cardinality for unit in result.units] == [8, 2]

    def forbidden_io(path):
        raise AssertionError("Bookmark calculation must not perform source IO")

    with monkeypatch.context() as no_io:
        no_io.setattr(io, "new_input", forbidden_io)
        bookmark = result.cursor_at(9, adapter.advance_cursor)
    native = IcebergScanCursor.decode(bookmark.payload)
    assert native.state.file_record_offset == 1

    remaining = []
    cursor = OpaqueSourceCursor.from_checkpoint(bookmark.to_checkpoint())
    for _ in range(8):
        resumed = next_data(cursor, 3)
        for unit in resumed.units:
            interval = unpack_metadata(unit.opaque_token)
            assert interval["record_start"] >= 1
            remaining.extend(
                range(
                    interval["record_start"],
                    interval["record_start"] + interval["record_count"],
                )
            )
        if resumed.status is SourceScanStatus.TERMINAL:
            break
        cursor = resumed.next_cursor
    else:
        pytest.fail("Resumed traversal failed to finish")
    assert remaining == [1, 2, 3, 4, 5]
