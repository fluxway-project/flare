"""Validate a remote Iceberg table through Flare's bounded scan and reader ports."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

from flare.config.iceberg import IcebergSettings
from flare.data.connectors.adapters.iceberg.access import (
    ConfiguredIcebergCatalogProvider,
)
from flare.data.connectors.adapters.iceberg.catalog import (
    build_iceberg_catalog_kwargs,
)
from flare.data.connectors.adapters.iceberg.modes import PinnedSnapshot
from flare.data.connectors.adapters.iceberg.source import (
    IcebergSourcePlanner,
    IcebergSourceReader,
)
from flare.data.connectors.adapters.parquet.range_reader import (
    ParquetInputFileReader,
)

_PROJECTION = ("row_id", "payload")
_ARROW_SCHEMA = pa.schema(
    (
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("payload", pa.string(), nullable=False),
    )
)
_ICEBERG_SCHEMA = Schema(
    NestedField(id=1, name="row_id", type=LongType(), required=True),
    NestedField(id=2, name="payload", type=StringType(), required=True),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iceberg-config", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--table-prefix", default="remote_scan_e2e")
    parser.add_argument("--keep-table", action="store_true")
    return parser.parse_args()


def _arrow_batch(start: int, values: tuple[str, str]) -> pa.Table:
    return pa.Table.from_arrays(
        [pa.array([start, start + 1], type=pa.int64()), pa.array(values)],
        schema=_ARROW_SCHEMA,
    )


def _scan_files(
    planner: IcebergSourcePlanner,
    cursor: Any,
) -> tuple[Any, ...]:
    results = []
    while True:
        result = planner.scan_next(cursor, max_records=16, max_units=16)
        results.append(result)
        if result.files or result.snapshot_exhausted:
            return tuple(item for result in results for item in result.files)
        cursor = result.next_cursor
        if len(results) > 32:
            raise RuntimeError(
                "bounded scan exceeded 32 metadata blocks without a file"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = IcebergSettings.from_file(args.iceberg_config)
    catalog_kwargs = build_iceberg_catalog_kwargs(settings=settings)
    catalog = load_catalog(catalog_kwargs.pop("name"), **catalog_kwargs)
    if not catalog.namespace_exists(args.namespace):
        raise RuntimeError(f"target namespace does not exist: {args.namespace!r}")
    table_name = (
        f"{args.table_prefix}_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    identifier = f"{args.namespace}.{table_name}"
    table = catalog.create_table(identifier, schema=_ICEBERG_SCHEMA)
    try:
        table.append(
            _arrow_batch(1, ("first-1", "first-2")),
            snapshot_properties={"validation": "first"},
        )
        first_snapshot = int(table.current_snapshot().snapshot_id)
        table.append(
            _arrow_batch(3, ("second-1", "second-2")),
            snapshot_properties={"validation": "second"},
        )
        second_snapshot = int(
            catalog.load_table(identifier).current_snapshot().snapshot_id
        )
        planner = IcebergSourcePlanner(
            table_identifier=identifier,
            catalog_provider=ConfiguredIcebergCatalogProvider(
                settings.to_pyiceberg_kwargs()
            ),
            projection=_PROJECTION,
            mode=PinnedSnapshot(first_snapshot),
        )
        cursor = planner.initial_scan_cursor()
        if cursor is None:
            raise RuntimeError("planner returned no initial cursor")
        files = _scan_files(planner, cursor)
        reader = IcebergSourceReader(
            table_identifier=identifier,
            catalog_provider=ConfiguredIcebergCatalogProvider(
                settings.to_pyiceberg_kwargs()
            ),
            projection=_PROJECTION,
            format_readers={"parquet": ParquetInputFileReader(max_open_files=2)},
            mode=PinnedSnapshot(first_snapshot),
        )
        try:
            reader.bind_source_identity(planner.source_identity)
            rows = []
            for item in files:
                record_range = planner.record_range_for_file(
                    snapshot_id=item.cursor.snapshot_id,
                    path=item.path,
                    record_start=item.record_start,
                    record_count=item.record_count,
                    record_format=item.record_format,
                )
                rows.extend(reader.read(record_range, _PROJECTION).to_pylist())
            expected = [
                {"row_id": 1, "payload": "first-1"},
                {"row_id": 2, "payload": "first-2"},
            ]
            if sorted(rows, key=lambda row: row["row_id"]) != expected:
                raise AssertionError(f"unexpected Flare rows: {rows!r}")
            try:
                reader.bind_source_identity(f"{planner.source_identity}:replacement")
            except ValueError:
                pass
            else:
                raise AssertionError("replacement source identity was accepted")
        finally:
            reader.close()
        return {
            "table": identifier,
            "table_uuid": planner.table_uuid,
            "first_snapshot": first_snapshot,
            "second_snapshot": second_snapshot,
            "scan_file_count": len(files),
            "scan_record_count": sum(item.record_count for item in files),
            "rows": rows,
            "source_identity_bound": True,
        }
    finally:
        if not args.keep_table:
            catalog.purge_table(identifier)


def main() -> None:
    result = run(_parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
