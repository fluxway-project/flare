from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import pyarrow as pa

from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
)
from flare.data.connectors.adapters.iceberg.observation import (
    IcebergCommitObservation,
    IcebergFileLayout,
)
from flare.data.connectors.adapters.iceberg.sink import IcebergSnapshotPublisher
from flare.data.contracts.sink import PublicationIdentity


def _publisher(*, observe_file_layout: bool) -> IcebergSnapshotPublisher:
    publisher = object.__new__(IcebergSnapshotPublisher)
    publisher._table = object()
    publisher._writer_target_size_bytes = 128
    publisher._snapshot_properties = {}
    publisher._catalog_kwargs = {
        "name": "catalog",
        "type": "rest",
        "uri": "http://catalog",
    }
    publisher._asset_namespace = ConfiguredIcebergCatalogProvider(
        publisher._catalog_kwargs
    ).identity
    table = SimpleNamespace(metadata=SimpleNamespace(table_uuid=UUID(int=1)))
    publisher._catalog_provider = SimpleNamespace(
        identity=publisher._asset_namespace,
        open=lambda: nullcontext(SimpleNamespace(load_table=lambda _: table)),
    )
    publisher._table_identifier = "db.table"
    publisher._observe_file_layout = observe_file_layout
    publisher._write_task_num_cpus = 1.0
    publisher._write_concurrency = 1
    return publisher


def _observation(publication_id: str) -> IcebergCommitObservation:
    return IcebergCommitObservation(
        snapshot_id=9,
        publication_id=publication_id,
        added_records=12,
        write_seconds=1.0,
        snapshot_observation_seconds=0.1,
        file_layout=IcebergFileLayout(
            data_file_count=1,
            data_file_bytes=10,
            smallest_file_bytes=10,
            p50_file_bytes=10,
            p95_file_bytes=10,
            largest_file_bytes=10,
        ),
    )


def test_publish_observes_only_after_the_single_datasink_write() -> None:
    publisher = _publisher(observe_file_layout=True)
    events: list[str] = []
    dataset = SimpleNamespace(
        write_datasink=lambda *args, **kwargs: events.append("write")
    )

    def observe(*, publication_id: str, **kwargs):
        events.append("observe")
        assert publication_id == "publication-1"
        assert publisher._observe_file_layout
        return _observation(publication_id)

    with (
        patch(
            "flare.data.connectors.adapters.iceberg.sink.validate_iceberg_table",
            return_value=object(),
        ),
        patch.object(publisher, "_observe_committed_publication", side_effect=observe),
        patch.object(publisher, "_log_commit_observation"),
    ):
        result = publisher.publish(
            dataset,
            schema=pa.schema([pa.field("id", pa.int64())]),
            publication=PublicationIdentity("publication-1"),
        )

    assert events == ["write", "observe"]
    assert result.total_rows == 12


def test_publisher_passes_default_disabled_layout_flag_to_observer() -> None:
    publisher = _publisher(observe_file_layout=False)
    calls = []

    with patch(
        "flare.data.connectors.adapters.iceberg.sink.observe_committed_publication",
        side_effect=lambda **kwargs: (
            calls.append(kwargs) or _observation("publication-1")
        ),
    ):
        publisher._observe_committed_publication(
            publication_id="publication-1",
            write_seconds=0.0,
        )

    assert calls[0]["include_file_layout"] is False
