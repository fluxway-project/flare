from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyiceberg.manifest import FileFormat

from flare.data.connectors.adapters.iceberg.modes import PinnedSnapshot
from flare.data.connectors.adapters.iceberg.source import (
    IcebergSourcePlanner,
    IcebergSourceReader,
    _normalize_record_format,
)
from flare.data.connectors.adapters.parquet.range_reader import (
    ParquetInputFileReader,
)
from flare.data.pipeline.consumption.source.contracts.model import (
    FileLocator,
    SourceRecordRange,
)

FORMAT_READERS = {"parquet": ParquetInputFileReader(max_open_files=2)}


def test_iceberg_planner_and_reader_share_binding_identity(monkeypatch) -> None:
    table = SimpleNamespace(
        metadata=SimpleNamespace(table_uuid="table-uuid"),
        io=SimpleNamespace(),
        current_snapshot=lambda: SimpleNamespace(snapshot_id=7),
        snapshot_by_id=lambda snapshot_id: (
            SimpleNamespace(snapshot_id=7) if snapshot_id == 7 else None
        ),
    )
    planner = IcebergSourcePlanner(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog"},
        projection=("fid_101",),
        mode=PinnedSnapshot(7),
    )
    planner._table = table
    monkeypatch.setattr(
        "flare.data.connectors.adapters.iceberg.source.load_iceberg_table",
        lambda *_args, **_kwargs: table,
    )
    reader = IcebergSourceReader(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog"},
        projection=("fid_101",),
        format_readers=FORMAT_READERS,
        mode=PinnedSnapshot(7),
    )

    assert dict(planner.binding_identity) == dict(reader.binding_identity)

    changed = IcebergSourceReader(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog"},
        projection=("fid_101",),
        format_readers=FORMAT_READERS,
    )
    assert dict(changed.binding_identity) != dict(planner.binding_identity)


def test_iceberg_reader_validates_assignment_identity_before_output(
    monkeypatch,
) -> None:
    original_table = SimpleNamespace(
        metadata=SimpleNamespace(table_uuid="table-uuid"),
        io=SimpleNamespace(),
        current_snapshot=lambda: SimpleNamespace(snapshot_id=7),
        snapshot_by_id=lambda snapshot_id: (
            SimpleNamespace(snapshot_id=7) if snapshot_id == 7 else None
        ),
    )
    planner = IcebergSourcePlanner(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog"},
        projection=("fid_101",),
        mode=PinnedSnapshot(7),
    )
    planner._table = original_table
    expected_source_identity = planner.binding_identity["source_identity"]
    replacement_table = SimpleNamespace(
        metadata=SimpleNamespace(table_uuid="replacement-uuid"),
        io=SimpleNamespace(),
    )
    monkeypatch.setattr(
        "flare.data.connectors.adapters.iceberg.source.load_iceberg_table",
        lambda *_args, **_kwargs: replacement_table,
    )
    reader = IcebergSourceReader(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog"},
        projection=("fid_101",),
        format_readers=FORMAT_READERS,
        mode=PinnedSnapshot(7),
    )

    reader.bind_source_identity(expected_source_identity)
    with pytest.raises(ValueError, match="differs from assignment"):
        reader._ensure_table_bound()
    with pytest.raises(ValueError, match="changed during session"):
        reader.bind_source_identity("another-source")


def test_iceberg_adapter_normalizes_declared_file_format() -> None:
    assert _normalize_record_format(FileFormat.PARQUET) == "parquet"
    with pytest.raises(ValueError, match="no declared file format"):
        _normalize_record_format(None)


def test_iceberg_checkpoint_identity_hashes_catalog_and_projection() -> None:
    original = IcebergSourcePlanner(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog-a"},
        projection=("fid_1",),
    )
    changed = IcebergSourcePlanner(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog-b"},
        projection=("fid_2",),
    )
    assert original.source_identity != changed.source_identity
    assert "http://catalog-a" not in original.source_identity
    assert original.schema_digest != changed.schema_digest


def test_iceberg_checkpoint_identity_excludes_catalog_credentials() -> None:
    common = {
        "name": "rest",
        "type": "rest",
        "uri": "http://catalog-a",
        "token": "token-a",
    }
    original = IcebergSourcePlanner(
        table_identifier="ns.table",
        catalog_kwargs=common,
        projection=("fid_1",),
    )
    rotated = IcebergSourcePlanner(
        table_identifier="ns.table",
        catalog_kwargs={**common, "token": "token-b"},
        projection=("fid_1",),
    )
    assert original.source_identity == rotated.source_identity
    assert "token-a" not in original.source_identity


def test_iceberg_adapter_rejects_logical_names_as_physical_projection() -> None:
    with pytest.raises(ValueError, match="identifier naming rules"):
        IcebergSourcePlanner(
            table_identifier="ns.table",
            catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog"},
            projection=("con_city-expect_city_fid",),
        )


def test_iceberg_adapter_reads_with_catalog_vended_file_io(monkeypatch) -> None:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"fid_101": pa.array([11, 12, 13], pa.int64())}), sink)
    payload = sink.getvalue()

    class InputFile:
        def open(self):
            return pa.BufferReader(payload)

    class FileIO:
        def __init__(self) -> None:
            self.locations: list[str] = []

        def new_input(self, location: str):
            self.locations.append(location)
            return InputFile()

    file_io = FileIO()
    table = SimpleNamespace(
        io=file_io,
        metadata=SimpleNamespace(table_uuid="table-uuid"),
        refresh=lambda: table,
        current_snapshot=lambda: SimpleNamespace(snapshot_id=7),
        snapshot_by_id=lambda snapshot_id: (
            SimpleNamespace(snapshot_id=7) if snapshot_id == 7 else None
        ),
    )
    monkeypatch.setattr(
        "flare.data.connectors.adapters.iceberg.source.load_iceberg_table",
        lambda *_args, **_kwargs: table,
    )
    reader = IcebergSourceReader(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog"},
        projection=("fid_101",),
        format_readers=FORMAT_READERS,
    )
    reader._ensure_table_bound()
    record_range = SourceRecordRange(
        source_identity=reader.source_identity,
        revision_identity="7",
        locator=FileLocator("s3://hidden-bucket/data.parquet"),
        record_start=1,
        record_count=2,
        schema_digest=reader.schema_digest,
        record_format="parquet",
    )

    result = reader.read(record_range, ("fid_101",))

    assert result["fid_101"].to_pylist() == [12, 13]
    assert file_io.locations == ["s3://hidden-bucket/data.parquet"]


def test_iceberg_reader_reuses_one_vended_input_per_snapshot_file(monkeypatch) -> None:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"fid_101": pa.array([11, 12, 13], pa.int64())}), sink)
    payload = sink.getvalue()

    class InputFile:
        def open(self):
            return pa.BufferReader(payload)

    class FileIO:
        def __init__(self) -> None:
            self.locations: list[str] = []

        def new_input(self, location: str):
            self.locations.append(location)
            return InputFile()

    file_io = FileIO()
    table = SimpleNamespace(
        io=file_io,
        metadata=SimpleNamespace(table_uuid="table-uuid"),
        refresh=lambda: table,
        current_snapshot=lambda: SimpleNamespace(snapshot_id=7),
        snapshot_by_id=lambda snapshot_id: (
            SimpleNamespace(snapshot_id=7) if snapshot_id == 7 else None
        ),
    )
    monkeypatch.setattr(
        "flare.data.connectors.adapters.iceberg.source.load_iceberg_table",
        lambda *_args, **_kwargs: table,
    )
    reader = IcebergSourceReader(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog"},
        projection=("fid_101",),
        format_readers=FORMAT_READERS,
    )
    reader._ensure_table_bound()
    first = SourceRecordRange(
        reader.source_identity,
        "7",
        FileLocator("s3://hidden-bucket/data.parquet"),
        0,
        1,
        reader.schema_digest,
        "parquet",
    )
    second = SourceRecordRange(
        reader.source_identity,
        "7",
        FileLocator("s3://hidden-bucket/data.parquet"),
        1,
        2,
        reader.schema_digest,
        "parquet",
    )

    assert reader.read(first, ("fid_101",))["fid_101"].to_pylist() == [11]
    assert reader.read(second, ("fid_101",))["fid_101"].to_pylist() == [12, 13]
    assert file_io.locations == ["s3://hidden-bucket/data.parquet"]


def test_iceberg_reader_does_not_reuse_file_between_snapshots(monkeypatch) -> None:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"fid_101": pa.array([11, 12, 13], pa.int64())}), sink)
    payload = sink.getvalue()

    class InputFile:
        def open(self):
            return pa.BufferReader(payload)

    class FileIO:
        def __init__(self) -> None:
            self.locations: list[str] = []

        def new_input(self, location: str):
            self.locations.append(location)
            return InputFile()

    file_io = FileIO()
    snapshots = {7: SimpleNamespace(snapshot_id=7), 8: SimpleNamespace(snapshot_id=8)}
    table = SimpleNamespace(
        io=file_io,
        metadata=SimpleNamespace(table_uuid="table-uuid"),
        refresh=lambda: table,
        current_snapshot=lambda: snapshots[8],
        snapshot_by_id=snapshots.get,
    )
    monkeypatch.setattr(
        "flare.data.connectors.adapters.iceberg.source.load_iceberg_table",
        lambda *_args, **_kwargs: table,
    )
    reader = IcebergSourceReader(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog"},
        projection=("fid_101",),
        format_readers=FORMAT_READERS,
    )
    reader._ensure_table_bound()
    for snapshot_id in ("7", "8"):
        record_range = SourceRecordRange(
            reader.source_identity,
            snapshot_id,
            FileLocator("s3://hidden-bucket/data.parquet"),
            0,
            1,
            reader.schema_digest,
            "parquet",
        )
        assert reader.read(record_range, ("fid_101",))["fid_101"].to_pylist() == [11]

    assert file_io.locations == [
        "s3://hidden-bucket/data.parquet",
        "s3://hidden-bucket/data.parquet",
    ]


def test_iceberg_reader_rejects_reads_after_close(monkeypatch) -> None:
    table = SimpleNamespace(io=object())
    monkeypatch.setattr(
        "flare.data.connectors.adapters.iceberg.source.load_iceberg_table",
        lambda *_args, **_kwargs: table,
    )
    reader = IcebergSourceReader(
        table_identifier="ns.table",
        catalog_kwargs={"name": "rest", "type": "rest", "uri": "http://catalog"},
        projection=("fid_101",),
        format_readers=FORMAT_READERS,
    )
    reader.close()
    record_range = SourceRecordRange(
        reader.source_identity,
        "7",
        FileLocator("s3://hidden-bucket/data.parquet"),
        0,
        1,
        reader.schema_digest,
        "parquet",
    )
    with pytest.raises(RuntimeError, match="reader is closed"):
        reader.read_batches(record_range, ("fid_101",), batch_size=1)
