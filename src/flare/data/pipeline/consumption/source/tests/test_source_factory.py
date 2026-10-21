from __future__ import annotations

import pyarrow as pa
import pytest
import pyarrow.parquet as pq

from flare.data.connectors.adapters.file.scan import FileListSourceScanAdapter
from flare.data.connectors.adapters.file.source import FileListSourceAdapter
from flare.data.connectors.adapters.iceberg.scan import IcebergSourceScanAdapter
from flare.data.connectors.adapters.parquet.source import ParquetSourceAdapter
from flare.data.connectors.adapters.vortex.source import VortexSourceAdapter
from flare.data.pipeline.consumption.bootstrap.factory import StandardSourceFactory
from flare.data.pipeline.consumption.source.contracts.source import (
    SourceFactory,
    SourceReader,
)
from fluxway.common.ingestion import SourceScanBudget, SourceScanStatus


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
    with pytest.raises(ValueError, match="unsupported controlled source scan format"):
        StandardSourceFactory().create_scan_planner({"storage_format": "csv"})


def test_file_source_contract_provides_batched_reads() -> None:
    source = FileListSourceAdapter(
        format_id="vortex",
        files=("memory://part.vortex",),
        projection=("id",),
        range_reader=lambda *_args, **_kwargs: pa.table(
            {"id": pa.array([1, 2, 3, 4])}
        ),
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
    assert isinstance(reader, SourceReader)
    assert isinstance(reader, VortexSourceAdapter)
    assert hasattr(reader, "read_batches")

    reader._range_reader = lambda *_args, **_kwargs: pa.table(
        {"id": pa.array([10, 20, 30, 40, 50])}
    )
    record_range = reader.record_ranges()[0]
    batches = list(reader.read_batches(record_range, ("id",), batch_size=2))
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
    assert isinstance(reader, SourceReader)
    assert isinstance(reader, ParquetSourceAdapter)
    record_range = reader.record_ranges()[0]
    batches = list(reader.read_batches(record_range, ("id",), batch_size=2))
    assert len(batches) == 3
    assert [batch.num_rows for batch in batches] == [2, 2, 1]
    assert batches[0].column("id").to_pylist() == [1, 2]

