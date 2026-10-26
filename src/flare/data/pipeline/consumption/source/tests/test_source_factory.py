from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fluxway.common.ingestion import (
    SourceScanBudget,
    SourceScanStatus,
    SourceWorkAssignment,
    SourceWorkItem,
)

from flare.data.connectors.adapters.file.scan import FileListSourceScanAdapter
from flare.data.connectors.adapters.file.source import FileListSourceAdapter
from flare.data.connectors.adapters.iceberg.scan import IcebergSourceScanAdapter
from flare.data.connectors.adapters.parquet.source import ParquetSourceAdapter
from flare.data.connectors.adapters.vortex.source import VortexSourceAdapter
from flare.data.contracts.assets import AssetRef
from flare.data.contracts.lineage import LineageConfig, LineageContext
from flare.data.pipeline.consumption.bootstrap.factory import StandardSourceFactory
from flare.data.pipeline.consumption.source.contracts.source import (
    SourceFactory,
    SourceReadSession,
)
from flare.data.pipeline.consumption.source.execution.invocation import (
    read_source_batch,
)
from flare.runtime.fluxway.source import WorkItemScheduledDatasetSourceDescriptor


def test_standard_factory_exposes_the_controlled_scan_entrypoint() -> None:
    factory = StandardSourceFactory()

    assert isinstance(factory, SourceFactory)
    assert not hasattr(factory, "create_planner")
    planner = factory.create_scan_planner(
        {
            "storage_format": "iceberg",
            "table_identifier": "namespace.table",
            "projection": ("id",),
            "catalog_kwargs": {
                "name": "test",
                "type": "in-memory",
                "warehouse": "/tmp/iceberg",
            },
        }
    )

    assert isinstance(planner, IcebergSourceScanAdapter)


def test_standard_factory_uses_controlled_scan_for_parquet(tmp_path) -> None:
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"id": pa.array([1, 2, 3, 4])}), path)

    adapter = StandardSourceFactory().create_scan_planner(
        {
            "storage_format": "parquet",
            "dataset_path": str(path),
            "projection": ("id",),
        }
    )

    assert isinstance(adapter, FileListSourceScanAdapter)
    initial = adapter.initial_scan_cursor()
    assert initial is not None
    first = adapter.scan_next(initial, SourceScanBudget(2, 1))
    assert first.status is SourceScanStatus.READY
    assert first.records == 2
    assert first.units[0].cardinality == 2
    assert first.next_cursor != initial

    restored = adapter.scan_next(initial, SourceScanBudget(2, 1))
    assert restored.units[0].unit_id == first.units[0].unit_id

    terminal = adapter.scan_next(first.next_cursor, SourceScanBudget(2, 1))
    assert terminal.status is SourceScanStatus.TERMINAL
    assert terminal.records == 2

    advanced = adapter.advance_cursor(initial, 3)
    tail = adapter.scan_next(advanced, SourceScanBudget(2, 1))
    assert tail.status is SourceScanStatus.TERMINAL
    assert tail.records == 1


def test_standard_factory_rejects_unsupported_scan_format() -> None:
    with pytest.raises(ValueError, match="unsupported standard source format"):
        StandardSourceFactory().create_scan_planner({"storage_format": "csv"})


def test_file_source_contract_provides_batched_reads() -> None:
    source = FileListSourceAdapter(
        format_id="vortex",
        files=("memory://part.vortex",),
        projection=("id",),
        range_reader=lambda *_args, **_kwargs: pa.table({"id": pa.array([1, 2, 3, 4])}),
        record_counter=lambda _path: 4,
    )
    record_range = source.record_ranges()[0]

    batches = tuple(source.read_batches(record_range, ("id",), batch_size=2))

    assert [batch.num_rows for batch in batches] == [2, 2]


def test_standard_factory_creates_vortex_source_reader() -> None:
    factory = StandardSourceFactory()
    reader = factory.create_reader(
        {
            "storage_format": "vortex",
            "files": ["memory://data.vortex"],
            "projection": ["id"],
            "record_counts": {"memory://data.vortex": 5},
        }
    )
    assert isinstance(reader, SourceReadSession)
    assert isinstance(reader._reader, VortexSourceAdapter)

    reader._reader._range_reader = lambda *_args, **_kwargs: pa.table(
        {"id": pa.array([10, 20, 30, 40, 50])}
    )
    record_range = reader._reader.record_ranges()[0]
    batches = list(reader._reader.read_batches(record_range, ("id",), batch_size=2))
    assert len(batches) == 3
    assert [batch.num_rows for batch in batches] == [2, 2, 1]
    assert batches[0].column("id").to_pylist() == [10, 20]
    assert batches[1].column("id").to_pylist() == [30, 40]
    assert batches[2].column("id").to_pylist() == [50]


def test_standard_factory_creates_parquet_source_reader(tmp_path) -> None:
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"id": pa.array([1, 2, 3, 4, 5])}), path)
    factory = StandardSourceFactory()
    reader = factory.create_reader(
        {
            "storage_format": "parquet",
            "files": [str(path)],
            "projection": ["id"],
        }
    )
    assert isinstance(reader, SourceReadSession)
    assert isinstance(reader._reader, ParquetSourceAdapter)
    record_range = reader._reader.record_ranges()[0]
    batches = list(reader._reader.read_batches(record_range, ("id",), batch_size=2))
    assert len(batches) == 3
    assert [batch.num_rows for batch in batches] == [2, 2, 1]
    assert batches[0].column("id").to_pylist() == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_lineage", [False, True])
async def test_standard_source_only_builds_input_assets_when_lineage_is_configured(
    tmp_path, monkeypatch, with_lineage
) -> None:
    class ExporterFactory:
        def create(self, context):
            raise AssertionError("creating a read session must not open an exporter")

    def unexpected_asset(*args, **kwargs):
        raise AssertionError("unconfigured lineage must not build input assets")

    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"id": [1, 2, 3, 4]}), path)
    options = {"storage_format": "parquet", "files": [str(path)], "projection": ["id"]}
    factory = StandardSourceFactory()
    planner = factory.create_scan_planner(options)
    scan = planner.scan_next(planner.initial_scan_cursor(), SourceScanBudget(4, 1))
    unit = scan.units[0]
    assignment = SourceWorkAssignment.create(
        epoch=0,
        step=0,
        source_identity=scan.source_identity,
        view_identity=scan.view_identity,
        schema_digest=scan.schema_digest,
        lineage_digest=scan.lineage_digest,
        descriptor=unit,
        global_start=0,
        global_batch_size=4,
        global_valid_count=4,
    )
    descriptor = WorkItemScheduledDatasetSourceDescriptor(
        source_factory=factory,
        global_batch_size=4,
        completion_policy="drop",
        lineage=(
            LineageConfig(ExporterFactory(), LineageContext("run", "read"))
            if with_lineage
            else None
        ),
    )
    if not with_lineage:
        for module in (
            "flare.data.pipeline.consumption.bootstrap.factory",
            "flare.data.pipeline.consumption.source.execution.record_ranges",
        ):
            monkeypatch.setattr(f"{module}.AssetRef", unexpected_asset)
    session = await descriptor.create_session(options)
    try:
        result = await read_source_batch(
            session,
            0,
            work_items=(
                SourceWorkItem(
                    assignment,
                    unit.logical_start,
                    unit.logical_start + unit.cardinality,
                ),
            ),
        )
        assert result.payload["id"].to_pylist() == [1, 2, 3, 4]
        assert result.inputs == (
            (
                AssetRef(
                    "file-list:parquet", scan.source_identity, unit.revision_identity
                ),
            )
            if with_lineage
            else ()
        )
    finally:
        await session.close()
        planner.close()
