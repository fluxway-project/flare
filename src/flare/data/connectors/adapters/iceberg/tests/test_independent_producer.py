from __future__ import annotations

import json
import subprocess
import sys

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog

from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
)
from flare.data.connectors.adapters.iceberg.modes import ContinuousAppend
from flare.data.connectors.adapters.iceberg.source import IcebergSourcePlanner

_PRODUCER = """
import json
import sys

import pyarrow as pa
import ray

catalog_config, rows = (json.loads(value) for value in sys.argv[1:])
ray.init(address="local", num_cpus=2, include_dashboard=False)
try:
    ray.data.from_arrow(pa.table({"id": rows})).write_iceberg(
        table_identifier="test.events", catalog_kwargs=catalog_config
    )
finally:
    ray.shutdown()
"""


def test_consumer_discovers_independent_ray_producers(tmp_path, monkeypatch):
    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    producer_directory = tmp_path / "producer"
    producer_directory.mkdir()
    consumer_directory = tmp_path / "consumer"
    consumer_directory.mkdir()
    catalog_config = {
        "name": "independent",
        "type": "sql",
        "uri": f"sqlite:///{warehouse / 'catalog.db'}",
        "warehouse": warehouse.as_uri(),
    }
    catalog = SqlCatalog(**catalog_config)
    catalog.create_namespace("test")
    table = catalog.create_table(
        "test.events", schema=pa.schema([pa.field("id", pa.int64())])
    )

    def append(rows):
        log_path = producer_directory / f"write-{rows[0]}.log"
        with log_path.open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _PRODUCER,
                    json.dumps(catalog_config),
                    json.dumps(rows),
                ],
                cwd=producer_directory,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )

    append([0, 1, 2, 3])
    monkeypatch.chdir(consumer_directory)
    planner = IcebergSourcePlanner(
        table_identifier="test.events",
        catalog_provider=ConfiguredIcebergCatalogProvider(catalog_config),
        projection=("id",),
        mode=ContinuousAppend(),
    )
    input_type = type(planner._current_table().io)
    new_input = input_type.new_input
    opened = []

    def metadata_only_input(self, location):
        assert not location.endswith(".parquet"), "Planning opened training data"
        assert str(producer_directory) not in location, (
            "Consumer accessed producer state"
        )
        opened.append(location)
        return new_input(self, location)

    monkeypatch.setattr(input_type, "new_input", metadata_only_input)
    cursor = planner.initial_scan_cursor()

    def scan_until_idle():
        nonlocal cursor
        files = []
        for _ in range(32):
            result = planner.scan_next(cursor, max_records=2, max_units=1)
            files.extend(result.files)
            cursor = result.next_cursor
            if result.snapshot_exhausted and not result.files:
                return files
        raise AssertionError("Consumer did not finish the visible native metadata")

    initial = scan_until_idle()
    assert sum(item.record_count for item in initial) == 4
    assert scan_until_idle() == []

    append([4, 5])
    added = scan_until_idle()
    assert sum(item.record_count for item in added) == 2
    assert not {item.path for item in initial} & {item.path for item in added}
    assert scan_until_idle() == []
    assert opened
    assert list(consumer_directory.iterdir()) == []
    table.refresh()
    for snapshot in table.snapshots():
        assert snapshot.summary is not None
        assert not any(
            key.startswith(("flare.", "flare_jax.", "fluxway."))
            for key in snapshot.summary.additional_properties
        )
